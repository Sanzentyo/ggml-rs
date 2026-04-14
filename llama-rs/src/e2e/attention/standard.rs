//! Standard (non-gated) attention: prefill, inference, and decode step.

use super::shared::{
    FlashAttentionConfig, RopeParams, apply_neox_rope_in_place, apply_optional_per_head_norm,
    graph_norm_input, host_attention_scoring, run_flash_attention_pipeline, validate_gqa_heads,
};
use crate::e2e::error::{E2eError, GgmlResultExt};
use crate::e2e::numeric::checked_mul;
use crate::e2e::plan::StandardAttentionLayerPlan;
use crate::e2e::state::StandardAttentionState;
use crate::e2e::tensor_ops::{
    ProjectionSpec, build_batch_projections, per_head_rms_norm, project_sequence_graph,
    upload_weight,
};
use crate::inference::{AttentionMaskPolicy, RotaryEmbedding};
use ggml_rs::{Backend, Bytes, Context, Length, Shape2D};

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Prefill standard attention: computes full-sequence attention via a fused
/// ggml graph and captures post-RoPE K / raw V into the state for subsequent
/// incremental decode steps.
///
/// Graph topology: `rms_norm(X) → mul_mat(W_q/W_k/W_v) → [optional per-head
/// norm] → RoPE → flash_attn_ext → mul_mat(W_out)`.
///
/// `input` is **un-normed** (layer pre-norm is fused into the graph).
pub(in crate::e2e) fn standard_attention_prefill(
    attention: &StandardAttentionLayerPlan,
    input: &[f32],
    seq_len: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    state: &mut StandardAttentionState,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    standard_attention_graph(
        attention,
        input,
        seq_len,
        rms_norm_eps,
        attn_norm_weight,
        Some(state),
        backend,
    )
}

/// Stateless standard attention inference (no state capture).
pub(in crate::e2e) fn standard_attention_inference(
    attention: &StandardAttentionLayerPlan,
    input: &[f32],
    seq_len: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    standard_attention_graph(
        attention,
        input,
        seq_len,
        rms_norm_eps,
        attn_norm_weight,
        None,
        backend,
    )
}

// ---------------------------------------------------------------------------
// Fused graph
// ---------------------------------------------------------------------------

/// Fused graph implementation shared by inference and prefill paths.
fn standard_attention_graph(
    attention: &StandardAttentionLayerPlan,
    input: &[f32],
    t: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    state: Option<&mut StandardAttentionState>,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    use crate::e2e::numeric::build_causal_mask_f16_bytes;
    use ggml_rs::{Dims, RopeExtParams, Type};

    let config = &attention.weights.config;
    let layout = config.layout;
    let d = layout.head_dimension();
    let h = layout.query_head_count();
    let hkv = layout.kv_head_count();
    let hidden = layout.hidden_features();
    let qf = checked_mul(h, d)?;
    let kvf = checked_mul(hkv, d)?;

    validate_gqa_heads(h, hkv)?;

    let expected_input = checked_mul(hidden, t)?;
    if input.len() != expected_input {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input,
            actual: input.len(),
        });
    }

    // Memory estimate (conservative): weights + IO + intermediates.
    let elem = std::mem::size_of::<f32>();
    let weight_bytes =
        (checked_mul(hidden, qf)? + 2 * checked_mul(hidden, kvf)? + checked_mul(qf, hidden)?)
            * elem;
    let io_bytes =
        (checked_mul(hidden, t)? + checked_mul(qf, t)? + 2 * checked_mul(kvf, t)?) * elem;
    let mask_bytes_est = t * t * 2; // f16 mask
    let overhead = 64 * 1024 * 1024; // 64 MB slack for intermediates
    let total_mem = Bytes::new(weight_bytes + io_bytes + mask_bytes_est + overhead);

    let ctx = Context::new_no_alloc_bytes(total_mem).ggml_ctx("Context::new(std_attn)")?;

    let ni = graph_norm_input(&ctx, hidden, t, rms_norm_eps)?;

    // QKV projections.
    let qkv = build_batch_projections(
        &ctx,
        &ni.x,
        hidden,
        &[
            ProjectionSpec {
                weight_label: "new<W_q>",
                matmul_label: "mul_mat(Q)",
                out_features: qf,
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
    let (_w_q, q_proj) = (&qkv[0].w, &qkv[0].y);
    let (_w_k, k_proj) = (&qkv[1].w, &qkv[1].y);
    let (_w_v, v_proj) = (&qkv[2].w, &qkv[2].y);

    // Output projection weight.
    let w_out = ctx
        .new_tensor_2d::<f32>(Shape2D::new(qf, hidden))
        .ggml_ctx("new<W_out>")?;

    // Reshape Q, K to 3D for per-head operations.
    let q_3d = ctx.reshape_3d(q_proj, d, h, t).ggml_ctx("reshape_3d(Q)")?;
    let k_3d = ctx
        .reshape_3d(k_proj, d, hkv, t)
        .ggml_ctx("reshape_3d(K)")?;

    // Optional per-head Q/K RMS norm.
    let q_norm_w = attention
        .weights
        .q_norm_values()
        .map(|_| {
            ctx.new_tensor_1d::<f32>(Length::new(d))
                .ggml_ctx("new<q_norm>")
        })
        .transpose()?;
    let k_norm_w = attention
        .weights
        .k_norm_values()
        .map(|_| {
            ctx.new_tensor_1d::<f32>(Length::new(d))
                .ggml_ctx("new<k_norm>")
        })
        .transpose()?;

    let q_after_norm = apply_optional_per_head_norm(
        &ctx,
        q_3d,
        q_norm_w.as_ref(),
        rms_norm_eps,
        "rms_norm(Q)",
        "mul(Q_norm)",
    )?;
    let k_after_norm = apply_optional_per_head_norm(
        &ctx,
        k_3d,
        k_norm_w.as_ref(),
        rms_norm_eps,
        "rms_norm(K)",
        "mul(K_norm)",
    )?;

    // RoPE (if configured).
    let positions = if matches!(config.rotary, RotaryEmbedding::Llama(_)) {
        Some(
            ctx.new_tensor_1d::<i32>(Length::new(t))
                .ggml_ctx("new<positions>")?,
        )
    } else {
        None
    };

    let (q_final, k_final) = if let RotaryEmbedding::Llama(ref rope_config) = config.rotary {
        let rope_params = RopeExtParams {
            n_dims: rope_config.dimensions.get() as i32,
            mode: 2, // NeoX
            n_ctx_orig: 0,
            freq_base: rope_config.base,
            freq_scale: rope_config.scale,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 0.0,
            beta_slow: 0.0,
        };
        let q_rope = ctx
            .rope_ext_with_i32_positions(
                &q_after_norm,
                positions.as_ref().unwrap(),
                None,
                rope_params,
            )
            .ggml_ctx("rope_ext(Q)")?;
        let k_rope = ctx
            .rope_ext_with_i32_positions(
                &k_after_norm,
                positions.as_ref().unwrap(),
                None,
                rope_params,
            )
            .ggml_ctx("rope_ext(K)")?;
        (q_rope, k_rope)
    } else {
        (q_after_norm, k_after_norm)
    };

    // Causal mask as f16.
    let mask = if matches!(config.mask, AttentionMaskPolicy::Causal { .. }) {
        Some(
            ctx.new_tensor(Type::F16, Dims::new([t, t, 1, 1]))
                .ggml_ctx("new<mask>")?,
        )
    } else {
        None
    };

    // --- Flash attention pipeline (shared helper, no gating for standard) ---
    let flash_cfg = FlashAttentionConfig {
        d,
        h,
        hkv,
        t,
        qf,
        attention_scale: config.attention_scale,
    };
    let output = run_flash_attention_pipeline(
        &ctx,
        &flash_cfg,
        (&q_final, &k_final, v_proj),
        mask.as_ref(),
        None, // no gating
        &w_out,
    )?;

    // Build graph with K/V readback nodes for state capture.
    let mut graph = ctx.new_graph().ggml_ctx("new_graph(std_attn)")?;
    graph.build_forward_expand(&output);
    if state.is_some() {
        graph.build_forward_expand(&k_final);
        graph.build_forward_expand(v_proj);
    }

    let _buffer = ctx
        .allocate_tensors(backend)
        .ggml_ctx("allocate_tensors(std_attn)")?;

    // Write data.
    upload_weight(&ni.x_raw, input, "write<X>")?;
    upload_weight(&ni.norm_w, attn_norm_weight, "write<attn_norm_w>")?;
    upload_weight(&qkv[0].w, attention.weights.q_values(), "write<W_q>")?;
    upload_weight(&qkv[1].w, attention.weights.k_values(), "write<W_k>")?;
    upload_weight(&qkv[2].w, attention.weights.v_values(), "write<W_v>")?;
    upload_weight(&w_out, attention.weights.o_values(), "write<W_out>")?;

    if let (Some(nw), Some(values)) = (&q_norm_w, attention.weights.q_norm_values()) {
        upload_weight(nw, values, "write<q_norm>")?;
    }
    if let (Some(nw), Some(values)) = (&k_norm_w, attention.weights.k_norm_values()) {
        upload_weight(nw, values, "write<k_norm>")?;
    }

    if let Some(ref pos) = positions {
        let pos_data: Vec<i32> = (0..t as i32).collect();
        pos.write_data_backend(&pos_data)
            .ggml_ctx("write<positions>")?;
    }
    if let Some(ref m) = mask {
        let mask_bytes = build_causal_mask_f16_bytes(t)?;
        m.write_bytes_backend(&mask_bytes).ggml_ctx("write<mask>")?;
    }

    backend.compute(&mut graph).ggml_ctx("compute(std_attn)")?;

    // Capture KV state for incremental decode.
    if let Some(state) = state {
        let k_data: Vec<f32> = k_final.read_data_backend().ggml_ctx("read<K>")?;
        let v_data: Vec<f32> = v_proj.read_data_backend().ggml_ctx("read<V>")?;
        state.append_batch(&k_data, &v_data, t)?;
    }

    output.read_data_backend().ggml_ctx("read<output>")
}

// ---------------------------------------------------------------------------
// Decode step
// ---------------------------------------------------------------------------

/// Processes one token using the KV cache accumulated during prefill (and
/// previous decode steps). The new K/V are appended to `state` BEFORE attention
/// so the token attends to itself.
///
/// `input` is **pre-normed** (DecodeStrategy applies host-side RMS norm).
pub(in crate::e2e) fn standard_attention_decode_step(
    attention: &StandardAttentionLayerPlan,
    input: &[f32],
    rms_norm_eps: f32,
    state: &mut StandardAttentionState,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    let config = &attention.weights.config;
    let layout = config.layout;
    let d = layout.head_dimension();
    let h = layout.query_head_count();
    let hkv = layout.kv_head_count();
    let hidden = layout.hidden_features();
    let qf = checked_mul(h, d)?;
    let kvf = checked_mul(hkv, d)?;

    validate_gqa_heads(h, hkv)?;

    if input.len() != hidden {
        return Err(E2eError::BufferLengthMismatch {
            expected: hidden,
            actual: input.len(),
        });
    }

    // Project Q, K, V for single token (graph or host).
    let q_values =
        project_sequence_graph(input, 1, hidden, qf, attention.weights.q_values(), backend)?;
    let mut k_values =
        project_sequence_graph(input, 1, hidden, kvf, attention.weights.k_values(), backend)?;
    let v_values =
        project_sequence_graph(input, 1, hidden, kvf, attention.weights.v_values(), backend)?;

    // Optional per-head Q/K RMS norm (host-side for single token).
    let mut q_values = if let Some(q_norm) = attention.weights.q_norm_values() {
        per_head_rms_norm(&q_values, 1, h, d, q_norm, rms_norm_eps)?
    } else {
        q_values
    };
    if let Some(k_norm) = attention.weights.k_norm_values() {
        k_values = per_head_rms_norm(&k_values, 1, hkv, d, k_norm, rms_norm_eps)?;
    }

    // RoPE with position_offset = cached token count.
    if let RotaryEmbedding::Llama(ref rope_config) = config.rotary {
        let rope = RopeParams {
            n_rot: rope_config.dimensions.get(),
            freq_base: rope_config.base,
            freq_scale: rope_config.scale,
            position_offset: state.token_count(),
        };
        apply_neox_rope_in_place(&mut q_values, 1, h, d, &rope)?;
        apply_neox_rope_in_place(&mut k_values, 1, hkv, d, &rope)?;
    }

    // Append to KV cache (token attends to itself).
    state.append_batch(&k_values, &v_values, 1)?;
    let total_tokens = state.token_count();

    // Host-side attention scoring (shared with Qwen3.5 full attention fallback).
    let head_outputs = host_attention_scoring(
        &q_values,
        h,
        hkv,
        d,
        config.attention_scale,
        total_tokens,
        state,
    );

    // Output projection.
    project_sequence_graph(
        &head_outputs,
        1,
        qf,
        hidden,
        attention.weights.o_values(),
        backend,
    )
}
