//! Qwen3.5 full (gated) attention: fused graph, GPU scoring, and decode step.

use super::persistent::{
    PersistentKvCache, PersistentScoringContext, decode_scoring_gpu_persistent,
};
use super::projection::{FullAttentionDims, PreparedAttention, project_and_prepare_qkv};
use super::shared::{
    FlashAttentionConfig, RopeParams, apply_neox_rope_in_place, apply_optional_per_head_norm,
    host_attention_scoring, run_flash_attention_pipeline, validate_gqa_heads,
};
use crate::e2e::error::{E2eError, GgmlResultExt};
use crate::e2e::numeric::{checked_mul, sigmoid_scalar};
use crate::e2e::plan::Qwen35FullAttentionLayerPlan;
use crate::e2e::state::Qwen35FullAttentionState;
use crate::e2e::tensor_ops::{
    ProjectionSpec, build_batch_projections, project_sequence_graph, upload_weight,
};
use ggml_rs::{Backend, Context, Shape2D, Shape4D};

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

pub(in crate::e2e) fn qwen35_full_attention_inference(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    qwen35_full_attention_core(
        attention,
        input,
        sequence_length,
        rms_norm_eps,
        attn_norm_weight,
        None,
        backend,
    )
}

/// Prefill variant: computes full attention AND stores post-RoPE K + raw V in `state`.
pub(in crate::e2e) fn qwen35_full_attention_prefill(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    state: &mut Qwen35FullAttentionState,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    qwen35_full_attention_core(
        attention,
        input,
        sequence_length,
        rms_norm_eps,
        attn_norm_weight,
        Some(state),
        backend,
    )
}

fn qwen35_full_attention_core(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    state: Option<&mut Qwen35FullAttentionState>,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    fully_fused_attention_graph(
        attention,
        input,
        sequence_length,
        rms_norm_eps,
        attn_norm_weight,
        state,
        backend,
    )
}

// ---------------------------------------------------------------------------
// Fully fused attention graph
// ---------------------------------------------------------------------------

/// Fully fused attention graph: layer norm + projection + deinterleave + norm + RoPE + scoring.
///
/// Accepts **un-normed** hidden state and applies layer pre-norm as the first
/// graph operation, eliminating the host↔device round-trip for normalization.
///
///   rms_norm(X, eps) * attn_norm_weight
///   → mul_mat(W_q/W_k/W_v, X_normed) → strided deinterleave Q/gate
///   → rms_norm + weight → rope_ext (NeoX mode=2)
///   → permute → cont → flash_attn_ext → sigmoid(gate) → mul
///   → reshape_2d → mul_mat(W_out)
///
/// When `state` is Some, reads back post-RoPE K and raw V for KV cache capture.
fn fully_fused_attention_graph(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    t: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    state: Option<&mut Qwen35FullAttentionState>,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    use crate::e2e::numeric::build_causal_mask_f16_bytes;
    use ggml_rs::{Dims, Length, RopeExtParams, Type};

    let dims = FullAttentionDims::new(attention)?;
    let FullAttentionDims {
        d,
        h,
        hkv,
        hidden,
        qf,
        qf2,
        kvf,
    } = dims;

    let expected_input = checked_mul(hidden, t)?;
    if input.len() != expected_input {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input,
            actual: input.len(),
        });
    }

    let ctx = Context::new_no_alloc_bytes(dims.estimate_memory(t))
        .ggml_ctx("Context::new(fully_fused_attn)")?;

    // --- Input tensors ---
    let x_raw = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden, t))
        .ggml_ctx("new<X>")?;

    // Layer pre-norm weight: [hidden_features]
    let attn_norm_w = ctx
        .new_tensor_1d::<f32>(Length::new(hidden))
        .ggml_ctx("new<attn_norm_w>")?;

    // In-graph layer pre-norm: rms_norm(X, eps) * attn_norm_weight
    let x_normed = ctx
        .rms_norm(&x_raw, rms_norm_eps)
        .ggml_ctx("rms_norm(X_layer)")?;
    let x = ctx
        .mul(&x_normed, &attn_norm_w)
        .ggml_ctx("mul(X_layer_norm)")?;

    // QKV projections via shared builder (Q carries interleaved gate, hence qf2).
    let qkv = build_batch_projections(
        &ctx,
        &x,
        hidden,
        &[
            ProjectionSpec {
                weight_label: "new<W_q>",
                matmul_label: "mul_mat(Q)",
                out_features: qf2,
            },
            ProjectionSpec {
                weight_label: "new<W_k>",
                matmul_label: "mul_mat(K)",
                out_features: kvf,
            },
            ProjectionSpec {
                weight_label: "new<W_v>",
                matmul_label: "mul_mat(V)",
                out_features: kvf,
            },
        ],
    )?;
    let (w_q, q_full) = (&qkv[0].w, &qkv[0].y);
    let (w_k, k_proj) = (&qkv[1].w, &qkv[1].y);
    let (w_v, v_proj) = (&qkv[2].w, &qkv[2].y);

    // Output projection weight.
    let w_out = ctx
        .new_tensor_2d::<f32>(Shape2D::new(qf, hidden))
        .ggml_ctx("new<W_out>")?;

    // Norm weight tensors: [D] each, broadcast across H and T.
    let q_norm_w = ctx
        .new_tensor_1d::<f32>(Length::new(d))
        .ggml_ctx("new<q_norm>")?;
    let k_norm_w = ctx
        .new_tensor_1d::<f32>(Length::new(d))
        .ggml_ctx("new<k_norm>")?;

    // Position tensor for RoPE: [T] i32
    let positions = ctx
        .new_tensor_1d::<i32>(Length::new(t))
        .ggml_ctx("new<positions>")?;

    // Causal mask as f16: [T, T, 1, 1]
    let mask = ctx
        .new_tensor(Type::F16, Dims::new([t, t, 1, 1]))
        .ggml_ctx("new<mask>")?;

    // --- Deinterleave Q and Gate via strided view ---
    // Q_full layout per token: [Q_h0(D), G_h0(D), Q_h1(D), G_h1(D), ...]
    // Reshape to [D, 2*H, T] then take every-other slice along dim 1.
    let q_full_3d = ctx
        .reshape_3d(q_full, d, 2 * h, t)
        .ggml_ctx("reshape_3d(Q_full)")?;

    let elem = std::mem::size_of::<f32>();
    let stride_h = 2 * d * elem; // skip one Q + one gate head
    let stride_t = 2 * h * d * elem; // full token

    // Q: even-indexed heads (offset 0)
    let q_view = ctx
        .view_3d(&q_full_3d, d, h, t, stride_h, stride_t, 0)
        .ggml_ctx("view_3d(Q)")?;
    let q_cont = ctx.cont(&q_view).ggml_ctx("cont(Q_deinterleave)")?; // [D, H, T]

    // Gate: odd-indexed heads (offset D*sizeof(f32))
    let gate_view = ctx
        .view_3d(&q_full_3d, d, h, t, stride_h, stride_t, d * elem)
        .ggml_ctx("view_3d(Gate)")?;
    let gate_cont = ctx.cont(&gate_view).ggml_ctx("cont(Gate_deinterleave)")?; // [D, H, T]

    // --- Per-head RMS norm (unconditional for Qwen3.5) ---
    let k_3d = ctx
        .reshape_3d(k_proj, d, hkv, t)
        .ggml_ctx("reshape_3d(K)")?;

    let q_scaled = apply_optional_per_head_norm(
        &ctx,
        q_cont,
        Some(&q_norm_w),
        rms_norm_eps,
        "rms_norm(Q)",
        "mul(Q_norm)",
    )?;
    let k_scaled = apply_optional_per_head_norm(
        &ctx,
        k_3d,
        Some(&k_norm_w),
        rms_norm_eps,
        "rms_norm(K)",
        "mul(K_norm)",
    )?;

    // --- NeoX RoPE (mode=2) ---
    let rope_params = RopeExtParams {
        n_dims: attention.rope_n_dims as i32,
        mode: 2, // NeoX: rotate first half, keep second half
        n_ctx_orig: 0,
        freq_base: attention.rope_freq_base,
        freq_scale: attention.rope_freq_scale,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 0.0,
        beta_slow: 0.0,
    };

    let q_rope = ctx
        .rope_ext_with_i32_positions(&q_scaled, &positions, None, rope_params)
        .ggml_ctx("rope_ext(Q)")?; // [D, H, T]
    let k_rope = ctx
        .rope_ext_with_i32_positions(&k_scaled, &positions, None, rope_params)
        .ggml_ctx("rope_ext(K)")?; // [D, Hkv, T]

    // --- Flash attention pipeline (shared helper) ---
    let flash_cfg = FlashAttentionConfig {
        d,
        h,
        hkv,
        t,
        qf,
        attention_scale: attention.attention_scale,
    };
    let output = run_flash_attention_pipeline(
        &ctx,
        &flash_cfg,
        (&q_rope, &k_rope, v_proj),
        Some(&mask),
        Some(&gate_cont),
        &w_out,
    )?;

    // --- Build, allocate, write, compute ---
    let mut graph = ctx.new_graph().ggml_ctx("new_graph(fully_fused)")?;
    graph.build_forward_expand(&output);
    // Also include K_rope and V_proj in the graph for intermediate readback.
    graph.build_forward_expand(&k_rope);
    graph.build_forward_expand(v_proj);

    let _buffer = ctx
        .allocate_tensors(backend)
        .ggml_ctx("allocate_tensors(fully_fused)")?;

    // Write input data (un-normed).
    upload_weight(&x_raw, input, "write<X>")?;

    // Layer pre-norm weight.
    upload_weight(&attn_norm_w, attn_norm_weight, "write<attn_norm_w>")?;

    // Write weight data.
    upload_weight(w_q, &attention.q_weight_values, "write<W_q>")?;
    upload_weight(w_k, &attention.k_weight_values, "write<W_k>")?;
    upload_weight(w_v, &attention.v_weight_values, "write<W_v>")?;
    upload_weight(&w_out, &attention.output_weight_values, "write<W_out>")?;

    // Norm weights.
    upload_weight(&q_norm_w, &attention.q_norm_values, "write<q_norm>")?;
    upload_weight(&k_norm_w, &attention.k_norm_values, "write<k_norm>")?;

    // Position indices: [0, 1, 2, ..., T-1]
    let pos_data: Vec<i32> = (0..t as i32).collect();
    positions
        .write_data_backend(&pos_data)
        .ggml_ctx("write<positions>")?;

    // Causal mask as f16.
    let mask_bytes = build_causal_mask_f16_bytes(t)?;
    mask.write_bytes_backend(&mask_bytes)
        .ggml_ctx("write<mask>")?;

    backend
        .compute(&mut graph)
        .ggml_ctx("compute(fully_fused)")?;

    // Read back post-RoPE K and raw V for KV cache state capture.
    if let Some(state) = state {
        let k_rope_data: Vec<f32> = k_rope.read_data_backend().ggml_ctx("read<K_rope>")?;
        let v_data: Vec<f32> = v_proj.read_data_backend().ggml_ctx("read<V_proj>")?;
        state.append_batch(&k_rope_data, &v_data, t)?;
    }

    output.read_data_backend().ggml_ctx("read<output>")
}

// ---------------------------------------------------------------------------
// GPU-accelerated decode scoring (ephemeral KV upload)
// ---------------------------------------------------------------------------

/// GPU-accelerated attention scoring via `flash_attn_ext`.
///
/// Builds a temporary ggml graph each decode step that:
/// 1. Uploads Q, live KV cache prefix, and gate to the backend
/// 2. Permutes K/V from host cache layout `[D, Hkv, T]` → flash `[D, T, Hkv]`
/// 3. Runs `flash_attn_ext` (Q·K scoring + softmax + V aggregation)
/// 4. Applies sigmoid gating
///
/// Returns the gated head outputs matching host scoring loop semantics.
/// On any failure the caller should fall back to the host scoring loop.
fn decode_scoring_gpu(
    q_values: &[f32],
    q_gate: &[f32],
    state: &Qwen35FullAttentionState,
    total_tokens: usize,
    attention: &Qwen35FullAttentionLayerPlan,
    query_features: usize,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    let d = attention.head_dimension;
    let h = attention.head_count;
    let hkv = attention.kv_head_count;
    let t = total_tokens;

    // Metadata-only context (backend manages data); 2 MB is generous for ~10 tensors + 1 graph.
    let ctx = Context::new_no_alloc(2 * 1024 * 1024).ggml_ctx("ctx(scoring_gpu)")?;

    // Input tensors — shapes follow flash_attn_ext convention.
    // Q: [D, T_q=1, H, 1]
    let q = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, 1, h, 1))
        .ggml_ctx("q_tensor")?;
    // K/V in host cache layout [D, Hkv, T, 1]; will be permuted to [D, T, Hkv, 1].
    let k_raw = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, hkv, t, 1))
        .ggml_ctx("k_tensor")?;
    let v_raw = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, hkv, t, 1))
        .ggml_ctx("v_tensor")?;
    // Gate: [D, H, 1, 1] — matches flash_attn_ext output shape.
    let gate = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, h, 1, 1))
        .ggml_ctx("gate_tensor")?;

    // Permute K/V from [D, Hkv, T, 1] → [D, T, Hkv, 1] + make contiguous.
    let k_perm = ctx.permute(&k_raw, 0, 2, 1, 3).ggml_ctx("permute(K)")?;
    let k = ctx.cont(&k_perm).ggml_ctx("cont(K)")?;
    let v_perm = ctx.permute(&v_raw, 0, 2, 1, 3).ggml_ctx("permute(V)")?;
    let v = ctx.cont(&v_perm).ggml_ctx("cont(V)")?;

    // flash_attn_ext: Q·K scoring + softmax + V aggregation.
    // No causal mask for single-query decode (token attends to itself + all past).
    let attn = ctx
        .flash_attn_ext(&q, &k, &v, None, attention.attention_scale, 0.0, 0.0)
        .ggml_ctx("flash_attn_ext(decode)")?;
    // Output: [D, H, 1, 1]

    // Gating: sigmoid(gate) ⊙ attn
    let gate_sig = ctx.sigmoid(&gate).ggml_ctx("sigmoid(gate)")?;
    let gated = ctx.mul(&attn, &gate_sig).ggml_ctx("mul(attn,gate)")?;

    // Build graph → allocate → upload → compute → readback.
    let mut graph = ctx.new_graph().ggml_ctx("new_graph(scoring)")?;
    graph.build_forward_expand(&gated);

    let _buffer = ctx
        .allocate_tensors(backend)
        .ggml_ctx("allocate(scoring)")?;

    let kv_prefix_len = t * state.kv_features;
    upload_weight(&q, q_values, "write(Q)")?;
    upload_weight(&k_raw, &state.k_cache[..kv_prefix_len], "write(K)")?;
    upload_weight(&v_raw, &state.v_cache[..kv_prefix_len], "write(V)")?;
    upload_weight(&gate, q_gate, "write(gate)")?;

    backend.compute(&mut graph).ggml_ctx("compute(scoring)")?;

    let outputs = gated.read_data_backend().ggml_ctx("read(gated)")?;

    if outputs.len() != query_features {
        return Err(E2eError::BufferLengthMismatch {
            expected: query_features,
            actual: outputs.len(),
        });
    }

    Ok(outputs)
}

// ---------------------------------------------------------------------------
// Decode core + step
// ---------------------------------------------------------------------------

/// Core full attention decode logic: RoPE → KV cache append → scoring → gating.
///
/// Takes prepared QKV projections and returns gated head outputs (before output
/// projection). The caller is responsible for projecting the output.
///
/// When `persistent_kv` is `Some`, uses persistent backend-resident KV cache
/// (O(1) per-step upload). Otherwise falls back to `decode_scoring_gpu` (O(T)
/// upload) or host scoring loop.
pub(in crate::e2e) fn full_attention_decode_core(
    prepared: PreparedAttention,
    attention: &Qwen35FullAttentionLayerPlan,
    state: &mut Qwen35FullAttentionState,
    backend: Option<&Backend>,
    persistent_kv: Option<&PersistentKvCache<'static>>,
    scoring_ctx: Option<&mut PersistentScoringContext>,
) -> Result<Vec<f32>, E2eError> {
    let PreparedAttention {
        mut q_values,
        mut k_values,
        v_proj,
        q_gate,
        hidden_features: _,
        query_features,
    } = prepared;

    validate_gqa_heads(attention.head_count, attention.kv_head_count)?;

    let hd = attention.head_dimension;

    let rope = RopeParams {
        n_rot: attention.rope_n_dims,
        freq_base: attention.rope_freq_base,
        freq_scale: attention.rope_freq_scale,
        position_offset: state.token_count(),
    };

    apply_neox_rope_in_place(
        &mut q_values,
        1,
        attention.head_count,
        attention.head_dimension,
        &rope,
    )?;
    apply_neox_rope_in_place(
        &mut k_values,
        1,
        attention.kv_head_count,
        attention.head_dimension,
        &rope,
    )?;

    state.append_batch(&k_values, &v_proj, 1)?;
    let total_tokens = state.token_count();

    // Also append to persistent backend KV cache (O(1) upload).
    if let Some(kv) = persistent_kv {
        // cached_len was incremented by append_batch, so the new token is at
        // position `total_tokens - 1`.
        let _ = kv.append_token(&k_values, &v_proj, total_tokens - 1);
    }

    // Try persistent GPU scoring first (O(1) upload, O(T) on-device permute).
    if let (Some(kv), Some(backend)) = (persistent_kv, backend)
        && !state.gpu_scoring_failed
    {
        if let Ok(outputs) = decode_scoring_gpu_persistent(
            &q_values,
            &q_gate,
            total_tokens,
            attention,
            kv,
            backend,
            scoring_ctx,
        ) {
            return Ok(outputs);
        }
        state.gpu_scoring_failed = true;
    }

    // Fallback: try ephemeral GPU scoring (O(T) upload).
    if let Some(backend) = backend
        && !state.gpu_scoring_failed
    {
        if let Ok(outputs) = decode_scoring_gpu(
            &q_values,
            &q_gate,
            state,
            total_tokens,
            attention,
            query_features,
            backend,
        ) {
            return Ok(outputs);
        }
        // First GPU failure — disable for future decode steps.
        state.gpu_scoring_failed = true;
    }

    // Host-side attention scoring (shared with standard attention fallback).
    let mut head_outputs = host_attention_scoring(
        &q_values,
        attention.head_count,
        attention.kv_head_count,
        hd,
        attention.attention_scale,
        total_tokens,
        state,
    );

    // Per-head sigmoid gating (Qwen3.5-specific).
    for head in 0..attention.head_count {
        let gate = &q_gate[head * hd..(head + 1) * hd];
        let dst = &mut head_outputs[head * hd..(head + 1) * hd];
        for (d, g) in dst.iter_mut().zip(gate.iter()) {
            *d *= sigmoid_scalar(*g);
        }
    }

    Ok(head_outputs)
}

/// Processes one token using the KV cache accumulated during prefill (and
/// previous decode steps). The new K/V are appended to `state` BEFORE attention
/// so the token attends to itself.
pub(in crate::e2e) fn qwen35_full_attention_decode_step(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    rms_norm_eps: f32,
    state: &mut Qwen35FullAttentionState,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    let prepared = project_and_prepare_qkv(attention, input, 1, rms_norm_eps, Some(backend))?;
    let hidden_features = prepared.hidden_features;
    let query_features = prepared.query_features;
    let head_outputs =
        full_attention_decode_core(prepared, attention, state, Some(backend), None, None)?;

    project_sequence_graph(
        &head_outputs,
        1,
        query_features,
        hidden_features,
        &attention.output_weight_values,
        backend,
    )
}
