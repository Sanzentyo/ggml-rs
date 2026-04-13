//! Qwen3.5 full (standard-style) attention with gated Q and NeoX RoPE.
//!
//! Provides full-sequence inference, prefill (capturing KV cache), and
//! single-token decode step using cached state.

use super::error::E2eError;
use super::numeric::{checked_mul, dot, sigmoid_scalar, softmax_prefix};
use super::plan::Qwen35FullAttentionLayerPlan;
use super::state::Qwen35FullAttentionState;
use super::tensor_ops::{PROJECTION_SLACK_BYTES, per_head_rms_norm, project_sequence};
use ggml_rs::{Backend, Bytes, Context, Shape2D};

pub(super) fn qwen35_full_attention_inference(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    qwen35_full_attention_core(
        attention,
        input,
        sequence_length,
        rms_norm_eps,
        None,
        backend,
    )
}

/// Prefill variant: computes full attention AND stores post-RoPE K + raw V in `state`.
pub(super) fn qwen35_full_attention_prefill(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    state: &mut Qwen35FullAttentionState,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    qwen35_full_attention_core(
        attention,
        input,
        sequence_length,
        rms_norm_eps,
        Some(state),
        backend,
    )
}

/// Projected and normalized Q, K, V + gate vectors (pre-RoPE).
struct PreparedAttention {
    q_values: Vec<f32>,
    k_values: Vec<f32>,
    v_proj: Vec<f32>,
    q_gate: Vec<f32>,
    hidden_features: usize,
    query_features: usize,
}

/// Estimate the backend memory needed for attention QKV projections.
///
/// Sums the memory for three independent matmuls (Q, K, V) sharing the same
/// input tensor, plus a slack constant for graph/tensor overhead.
fn recommended_qkv_projection_memory(
    hidden_features: usize,
    query_features_x2: usize,
    kv_features: usize,
    sequence_length: usize,
) -> Result<Bytes, E2eError> {
    let q_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, query_features_x2),
        Shape2D::new(hidden_features, sequence_length),
    )
    .map_err(|source| E2eError::ggml("recommended_backend_matmul_memory(Q)", source))?;
    let k_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, kv_features),
        Shape2D::new(hidden_features, sequence_length),
    )
    .map_err(|source| E2eError::ggml("recommended_backend_matmul_memory(K)", source))?;
    let v_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, kv_features),
        Shape2D::new(hidden_features, sequence_length),
    )
    .map_err(|source| E2eError::ggml("recommended_backend_matmul_memory(V)", source))?;
    let total = q_mem
        .get()
        .checked_add(k_mem.get())
        .and_then(|v| v.checked_add(v_mem.get()))
        .and_then(|v| v.checked_add(PROJECTION_SLACK_BYTES))
        .ok_or(E2eError::MemorySizeOverflow)?;
    Ok(Bytes::new(total))
}

/// Compute Q, K, V projections using a single ggml compute graph.
///
/// Batches three `mul_mat` operations sharing the same input tensor into one
/// graph execution. Returns `(q_full, k_proj, v_proj)` as host vectors.
fn project_qkv_graph(
    input: &[f32],
    sequence_length: usize,
    hidden_features: usize,
    query_features_x2: usize,
    kv_features: usize,
    q_weights: &[f32],
    k_weights: &[f32],
    v_weights: &[f32],
    backend: &Backend,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), E2eError> {
    let ctx_size = recommended_qkv_projection_memory(
        hidden_features,
        query_features_x2,
        kv_features,
        sequence_length,
    )?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| E2eError::ggml("Context::new_no_alloc_bytes(QKV)", source))?;

    let w_q = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, query_features_x2))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_Q>", source))?;
    let w_k = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, kv_features))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_K>", source))?;
    let w_v = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, kv_features))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_V>", source))?;
    let x = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, sequence_length))
        .map_err(|source| E2eError::ggml("new_tensor_2d<X>", source))?;

    let q_out = ctx
        .mul_mat(&w_q, &x)
        .map_err(|source| E2eError::ggml("mul_mat(Q)", source))?;
    let k_out = ctx
        .mul_mat(&w_k, &x)
        .map_err(|source| E2eError::ggml("mul_mat(K)", source))?;
    let v_out = ctx
        .mul_mat(&w_v, &x)
        .map_err(|source| E2eError::ggml("mul_mat(V)", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(QKV)", source))?;
    graph.build_forward_expand(&q_out);
    graph.build_forward_expand(&k_out);
    graph.build_forward_expand(&v_out);

    let _buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("allocate_tensors(QKV)", source))?;

    w_q.write_data_backend(q_weights)
        .map_err(|source| E2eError::ggml("write_data_backend<W_Q>", source))?;
    w_k.write_data_backend(k_weights)
        .map_err(|source| E2eError::ggml("write_data_backend<W_K>", source))?;
    w_v.write_data_backend(v_weights)
        .map_err(|source| E2eError::ggml("write_data_backend<W_V>", source))?;
    x.write_data_backend(input)
        .map_err(|source| E2eError::ggml("write_data_backend<X>", source))?;

    backend
        .compute(&mut graph)
        .map_err(|source| E2eError::ggml("compute(QKV)", source))?;

    let q_full = q_out
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read_data_backend<Q>", source))?;
    let k_proj = k_out
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read_data_backend<K>", source))?;
    let v_proj = v_out
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read_data_backend<V>", source))?;
    Ok((q_full, k_proj, v_proj))
}

/// Shared projection + deinterleave + per-head RMS norm for both core and
/// decode paths. The caller applies RoPE with its own position_offset.
///
/// When `backend` is `Some`, uses ggml compute graphs for projections
/// (prefill/inference path). When `None`, falls back to host-side scalar
/// dot products (decode path, where graph overhead exceeds benefit).
fn project_and_prepare_qkv(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    backend: Option<&Backend>,
) -> Result<PreparedAttention, E2eError> {
    let total_output_elements = attention
        .head_count
        .checked_mul(attention.head_dimension)
        .and_then(|qf| attention.output_weight_values.len().checked_div(qf))
        .filter(|&h| {
            h > 0
                && h * attention.head_count * attention.head_dimension
                    == attention.output_weight_values.len()
        })
        .ok_or(E2eError::BufferLengthMismatch {
            expected: 1,
            actual: 0,
        })?;
    let hidden_features = total_output_elements;
    let expected_input_len = checked_mul(hidden_features, sequence_length)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }

    let query_features = checked_mul(attention.head_count, attention.head_dimension)?;
    let kv_features = checked_mul(attention.kv_head_count, attention.head_dimension)?;
    let query_features_x2 = checked_mul(query_features, 2)?;

    let (q_full, k_proj, v_proj) = if let Some(backend) = backend {
        project_qkv_graph(
            input,
            sequence_length,
            hidden_features,
            query_features_x2,
            kv_features,
            &attention.q_weight_values,
            &attention.k_weight_values,
            &attention.v_weight_values,
            backend,
        )?
    } else {
        let q_full = project_sequence(
            input,
            sequence_length,
            hidden_features,
            query_features_x2,
            &attention.q_weight_values,
        )?;
        let k_proj = project_sequence(
            input,
            sequence_length,
            hidden_features,
            kv_features,
            &attention.k_weight_values,
        )?;
        let v_proj = project_sequence(
            input,
            sequence_length,
            hidden_features,
            kv_features,
            &attention.v_weight_values,
        )?;
        (q_full, k_proj, v_proj)
    };

    let (q_values, q_gate) = deinterleave_q_gate(
        &q_full,
        sequence_length,
        attention.head_count,
        attention.head_dimension,
    )?;

    let q_values = per_head_rms_norm(
        &q_values,
        sequence_length,
        attention.head_count,
        attention.head_dimension,
        &attention.q_norm_values,
        rms_norm_eps,
    )?;
    let k_values = per_head_rms_norm(
        &k_proj,
        sequence_length,
        attention.kv_head_count,
        attention.head_dimension,
        &attention.k_norm_values,
        rms_norm_eps,
    )?;

    Ok(PreparedAttention {
        q_values,
        k_values,
        v_proj,
        q_gate,
        hidden_features,
        query_features,
    })
}

fn qwen35_full_attention_core(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    state: Option<&mut Qwen35FullAttentionState>,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    let PreparedAttention {
        mut q_values,
        mut k_values,
        v_proj,
        q_gate,
        hidden_features,
        query_features: _,
    } = project_and_prepare_qkv(
        attention,
        input,
        sequence_length,
        rms_norm_eps,
        Some(backend),
    )?;

    // Apply NeoX-style RoPE to Q and K after normalization.
    apply_neox_rope_in_place(
        &mut q_values,
        sequence_length,
        attention.head_count,
        attention.head_dimension,
        attention.rope_n_dims,
        attention.rope_freq_base,
        attention.rope_freq_scale,
        0,
    )?;
    apply_neox_rope_in_place(
        &mut k_values,
        sequence_length,
        attention.kv_head_count,
        attention.head_dimension,
        attention.rope_n_dims,
        attention.rope_freq_base,
        attention.rope_freq_scale,
        0,
    )?;

    // Capture post-RoPE K and raw V into the KV cache if we are in prefill mode.
    if let Some(state) = state {
        state.append_batch(&k_values, &v_proj, sequence_length)?;
    }

    // Fused attention scoring + gating + output projection via flash_attn_ext.
    fused_attention_scoring_graph(
        &q_values,
        &k_values,
        &v_proj,
        &q_gate,
        &attention.output_weight_values,
        attention.head_dimension,
        sequence_length,
        attention.head_count,
        attention.kv_head_count,
        hidden_features,
        attention.attention_scale,
        backend,
    )
}

/// De-interleave ggml's `[Q_h0, G_h0, Q_h1, G_h1, ...]` layout into
/// separate Q and gate buffers.
///
/// Both call sites (prefill multi-token and decode single-token) go through
/// this function so validation is unified.
fn deinterleave_q_gate(
    q_full: &[f32],
    sequence_length: usize,
    head_count: usize,
    head_dimension: usize,
) -> Result<(Vec<f32>, Vec<f32>), E2eError> {
    let query_features = checked_mul(head_count, head_dimension)?;
    let per_token_qg = checked_mul(query_features, 2)?;
    let expected_len = checked_mul(sequence_length, per_token_qg)?;
    if q_full.len() != expected_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_len,
            actual: q_full.len(),
        });
    }

    let total_out = checked_mul(sequence_length, query_features)?;
    let mut q_values = vec![0.0_f32; total_out];
    let mut q_gate = vec![0.0_f32; total_out];

    for ((src_token, q_dst_token), g_dst_token) in q_full
        .chunks_exact(per_token_qg)
        .zip(q_values.chunks_exact_mut(query_features))
        .zip(q_gate.chunks_exact_mut(query_features))
    {
        for head in 0..head_count {
            let hd = head_dimension;
            q_dst_token[head * hd..(head + 1) * hd]
                .copy_from_slice(&src_token[head * 2 * hd..head * 2 * hd + hd]);
            g_dst_token[head * hd..(head + 1) * hd]
                .copy_from_slice(&src_token[head * 2 * hd + hd..(head + 1) * 2 * hd]);
        }
    }

    Ok((q_values, q_gate))
}

/// Apply NeoX-style rotary position embedding in-place.
///
/// For each token at position `position_offset + pos`, rotates dimension pairs
/// `(x[k], x[k + n_rot/2])` for `k` in `0..n_rot/2` using angle
/// `theta_k = pos * freq_base^(-2k / n_rot)`.
/// Dimensions beyond `n_rot` are left unchanged.
///
/// `position_offset` shifts the starting position (0 for prefill, prompt_len for decode).
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_neox_rope_in_place(
    values: &mut [f32],
    sequence_length: usize,
    head_count: usize,
    head_dimension: usize,
    n_rot: usize,
    freq_base: f32,
    freq_scale: f32,
    position_offset: usize,
) -> Result<(), E2eError> {
    let total_features = checked_mul(head_count, head_dimension)?;
    let expected_len = checked_mul(sequence_length, total_features)?;
    if values.len() != expected_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_len,
            actual: values.len(),
        });
    }
    debug_assert!(n_rot <= head_dimension && n_rot.is_multiple_of(2));

    let half_rot = n_rot / 2;
    let theta_scale = freq_base.powf(-2.0 / n_rot as f32);

    let cache_size = checked_mul(sequence_length, half_rot)?;
    let mut cos_cache = vec![0.0_f32; cache_size];
    let mut sin_cache = vec![0.0_f32; cache_size];
    for pos in 0..sequence_length {
        let mut theta = (position_offset + pos) as f32;
        for k in 0..half_rot {
            let cache_idx = pos * half_rot + k;
            let angle = theta * freq_scale;
            cos_cache[cache_idx] = angle.cos();
            sin_cache[cache_idx] = angle.sin();
            theta *= theta_scale;
        }
    }

    for pos in 0..sequence_length {
        let token_base = pos * total_features;
        for head in 0..head_count {
            let head_base = token_base + head * head_dimension;
            for k in 0..half_rot {
                let cache_idx = pos * half_rot + k;
                let cos_t = cos_cache[cache_idx];
                let sin_t = sin_cache[cache_idx];
                let idx0 = head_base + k;
                let idx1 = head_base + k + half_rot;
                let x0 = values[idx0];
                let x1 = values[idx1];
                values[idx0] = x0 * cos_t - x1 * sin_t;
                values[idx1] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    Ok(())
}

/// Fused attention scoring + gating + output projection using flash_attn_ext.
///
/// Replaces the host-side O(T²·H·D) scoring loop with a single ggml graph:
///   permute(Q/K/V) → flash_attn_ext → sigmoid(gate) → mul → reshape → mul_mat(output)
///
/// All input vectors use `[T, H, D]` host layout (= ggml `[D, H, T, 1]`).
/// The causal mask is built as f16 per ggml CPU kernel requirements.
#[allow(clippy::too_many_arguments)]
fn fused_attention_scoring_graph(
    q_values: &[f32],
    k_values: &[f32],
    v_proj: &[f32],
    q_gate: &[f32],
    output_weight: &[f32],
    d: usize,
    t: usize,
    h: usize,
    hkv: usize,
    hidden: usize,
    scale: f32,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    use super::numeric::build_causal_mask_f16_bytes;
    use ggml_rs::{Dims, Shape4D, Type};

    if !h.is_multiple_of(hkv) {
        return Err(E2eError::BufferLengthMismatch {
            expected: 0,
            actual: h % hkv,
        });
    }

    let qf = h * d;
    let kvf = hkv * d;
    let mask_elems = t * t;

    // Memory: inputs + intermediates + mask + output weight + graph overhead.
    let data_bytes = (qf * t + kvf * t * 2 + qf * t + mask_elems + qf * hidden + hidden * t) * 4;
    let mem = Bytes::new(data_bytes * 4 + 524_288);
    let ctx = Context::new_no_alloc_bytes(mem)
        .map_err(|source| E2eError::ggml("Context::new_no_alloc_bytes(fused_attn)", source))?;

    // Input tensors in host [T, H, D] layout = ggml [D, H, T, 1].
    let q = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, h, t, 1))
        .map_err(|source| E2eError::ggml("new_tensor_4d<Q>", source))?;
    let k = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, hkv, t, 1))
        .map_err(|source| E2eError::ggml("new_tensor_4d<K>", source))?;
    let v = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, hkv, t, 1))
        .map_err(|source| E2eError::ggml("new_tensor_4d<V>", source))?;
    let gate = ctx
        .new_tensor_4d::<f32>(Shape4D::new(d, h, t, 1))
        .map_err(|source| E2eError::ggml("new_tensor_4d<gate>", source))?;

    // Causal mask as f16: [T, T, 1, 1].
    let mask = ctx
        .new_tensor(Type::F16, Dims::new([t, t, 1, 1]))
        .map_err(|source| E2eError::ggml("new_tensor<mask>", source))?;

    // Output weight: [H*D, hidden].
    let w_out = ctx
        .new_tensor_2d::<f32>(Shape2D::new(qf, hidden))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_out>", source))?;

    // Permute Q/K/V from [D, H, T, 1] to [D, T, H, 1] for flash_attn_ext.
    let q_perm = ctx
        .permute(&q, 0, 2, 1, 3)
        .map_err(|source| E2eError::ggml("permute(Q)", source))?;
    let k_perm = ctx
        .permute(&k, 0, 2, 1, 3)
        .map_err(|source| E2eError::ggml("permute(K)", source))?;
    let v_perm = ctx
        .permute(&v, 0, 2, 1, 3)
        .map_err(|source| E2eError::ggml("permute(V)", source))?;

    // flash_attn_ext may need contiguous inputs on some backends.
    let q_cont = ctx
        .cont(&q_perm)
        .map_err(|source| E2eError::ggml("cont(Q)", source))?;
    let k_cont = ctx
        .cont(&k_perm)
        .map_err(|source| E2eError::ggml("cont(K)", source))?;
    let v_cont = ctx
        .cont(&v_perm)
        .map_err(|source| E2eError::ggml("cont(V)", source))?;

    // Flash attention → [D, H, T, 1] (permuted output).
    let attn = ctx
        .flash_attn_ext(&q_cont, &k_cont, &v_cont, Some(&mask), scale, 0.0, 0.0)
        .map_err(|source| E2eError::ggml("flash_attn_ext", source))?;

    // Gate: sigmoid(gate) in [D, H, T, 1] — same layout as flash output.
    let gate_sig = ctx
        .sigmoid(&gate)
        .map_err(|source| E2eError::ggml("sigmoid(gate)", source))?;

    // Element-wise gating: attn * sigmoid(gate).
    let gated = ctx
        .mul(&attn, &gate_sig)
        .map_err(|source| E2eError::ggml("mul(attn, sigmoid_gate)", source))?;

    // Reshape for output projection: [D, H, T, 1] → [H*D, T].
    // [D, H, T, 1] in memory is d-fastest, h-next, t-slowest — same as [H*D, T].
    let gated_2d = ctx
        .reshape_2d(&gated, qf, t)
        .map_err(|source| E2eError::ggml("reshape_2d(gated)", source))?;

    // Output projection: mul_mat(W_out, gated) → [hidden, T].
    let output = ctx
        .mul_mat(&w_out, &gated_2d)
        .map_err(|source| E2eError::ggml("mul_mat(output)", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(fused_attn)", source))?;
    graph.build_forward_expand(&output);

    let _buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("allocate_tensors(fused_attn)", source))?;

    // Write input data.
    q.write_data_backend(q_values)
        .map_err(|source| E2eError::ggml("write<Q>", source))?;
    k.write_data_backend(k_values)
        .map_err(|source| E2eError::ggml("write<K>", source))?;
    v.write_data_backend(v_proj)
        .map_err(|source| E2eError::ggml("write<V>", source))?;
    gate.write_data_backend(q_gate)
        .map_err(|source| E2eError::ggml("write<gate>", source))?;

    let mask_bytes = build_causal_mask_f16_bytes(t);
    mask.write_bytes_backend(&mask_bytes)
        .map_err(|source| E2eError::ggml("write<mask>", source))?;

    w_out
        .write_data_backend(output_weight)
        .map_err(|source| E2eError::ggml("write<W_out>", source))?;

    backend
        .compute(&mut graph)
        .map_err(|source| E2eError::ggml("compute(fused_attn)", source))?;

    output
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read<output>", source))
}
///
/// Processes one token using the KV cache accumulated during prefill (and
/// previous decode steps). The new K/V are appended to `state` BEFORE attention
/// so the token attends to itself.
pub(super) fn qwen35_full_attention_decode_step(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    rms_norm_eps: f32,
    state: &mut Qwen35FullAttentionState,
) -> Result<Vec<f32>, E2eError> {
    let PreparedAttention {
        mut q_values,
        mut k_values,
        v_proj,
        q_gate,
        hidden_features,
        query_features,
    } = project_and_prepare_qkv(attention, input, 1, rms_norm_eps, None)?;

    let hd = attention.head_dimension;

    // Position = number of tokens already in the cache.
    let position_offset = state.token_count();
    apply_neox_rope_in_place(
        &mut q_values,
        1,
        attention.head_count,
        attention.head_dimension,
        attention.rope_n_dims,
        attention.rope_freq_base,
        attention.rope_freq_scale,
        position_offset,
    )?;
    apply_neox_rope_in_place(
        &mut k_values,
        1,
        attention.kv_head_count,
        attention.head_dimension,
        attention.rope_n_dims,
        attention.rope_freq_base,
        attention.rope_freq_scale,
        position_offset,
    )?;

    // Append new K/V to cache BEFORE attention so the token attends to itself.
    state.append_batch(&k_values, &v_proj, 1)?;
    let total_tokens = state.token_count();

    let groups = attention.head_count / attention.kv_head_count;
    let mut head_outputs = vec![0.0_f32; query_features];
    for head in 0..attention.head_count {
        let kv_head = head / groups;
        let q = &q_values[head * hd..(head + 1) * hd];

        // Score against all cached K vectors.
        let mut scores = vec![f32::NEG_INFINITY; total_tokens];
        for (source, score) in scores.iter_mut().enumerate().take(total_tokens) {
            let k = state.k_head_at(source, kv_head, hd);
            *score = dot(q, k) * attention.attention_scale;
        }
        let weights = softmax_prefix(&scores, total_tokens);

        let dst = &mut head_outputs[head * hd..(head + 1) * hd];
        for (source, weight) in weights.iter().copied().enumerate() {
            let v = state.v_head_at(source, kv_head, hd);
            for index in 0..hd {
                dst[index] += v[index] * weight;
            }
        }

        let gate = &q_gate[head * hd..(head + 1) * hd];
        for index in 0..hd {
            dst[index] *= sigmoid_scalar(gate[index]);
        }
    }

    project_sequence(
        &head_outputs,
        1,
        query_features,
        hidden_features,
        &attention.output_weight_values,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35_full_attention_qgate_split_is_head_interleaved() {
        let q_full: Vec<f32> = vec![
            1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0,
        ];
        let (q_values, q_gate) = deinterleave_q_gate(&q_full, 1, 2, 3).unwrap();
        assert_eq!(q_values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(q_gate, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn qwen35_full_attention_qgate_split_multi_token() {
        let q_full: Vec<f32> = vec![
            1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0, 5.0, 6.0, 50.0, 60.0, 7.0, 8.0, 70.0, 80.0,
        ];
        let (q_values, q_gate) = deinterleave_q_gate(&q_full, 2, 2, 2).unwrap();
        assert_eq!(q_values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(q_gate, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn rope_identity_at_position_zero() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let original = values.clone();
        apply_neox_rope_in_place(&mut values, 1, 1, 4, 4, 10000.0, 1.0, 0).unwrap();
        for (a, b) in values.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
        }
    }

    #[test]
    fn rope_rotates_at_nonzero_position() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0];
        apply_neox_rope_in_place(&mut values, 2, 1, 4, 4, 1.0, 1.0, 0).unwrap();

        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[1] - 2.0).abs() < 1e-6);
        assert!((values[2] - 3.0).abs() < 1e-6);
        assert!((values[3] - 4.0).abs() < 1e-6);

        let cos1 = 1.0_f32.cos();
        let sin1 = 1.0_f32.sin();
        assert!(
            (values[4] - cos1).abs() < 1e-6,
            "expected {cos1}, got {}",
            values[4]
        );
        assert!((values[5]).abs() < 1e-6);
        assert!(
            (values[6] - sin1).abs() < 1e-6,
            "expected {sin1}, got {}",
            values[6]
        );
        assert!((values[7]).abs() < 1e-6);
    }

    #[test]
    fn rope_preserves_dims_beyond_n_rot() {
        let mut values = [
            1.0_f32, 2.0, 3.0, 4.0, 99.0, 88.0, 1.0, 2.0, 3.0, 4.0, 99.0, 88.0,
        ];
        apply_neox_rope_in_place(&mut values, 2, 1, 6, 4, 10000.0, 1.0, 0).unwrap();
        assert!((values[4] - 99.0).abs() < 1e-6);
        assert!((values[5] - 88.0).abs() < 1e-6);
        assert!((values[10] - 99.0).abs() < 1e-6);
        assert!((values[11] - 88.0).abs() < 1e-6);
    }

    #[test]
    fn rope_multi_head_applies_same_rotation_per_head() {
        let mut buf = [
            0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        apply_neox_rope_in_place(&mut buf, 2, 2, 4, 4, 1.0, 1.0, 0).unwrap();
        assert_eq!(&buf[8..12], &buf[12..16]);
    }

    #[test]
    fn rope_position_offset_matches_sequential() {
        // RoPE at offset=2 for 1 token should match position 2 from a 3-token batch.
        let mut batch = vec![0.0_f32; 3 * 4]; // 3 tokens, hd=4
        batch[2 * 4] = 1.0;
        batch[2 * 4 + 1] = 2.0;
        batch[2 * 4 + 2] = 3.0;
        batch[2 * 4 + 3] = 4.0;
        apply_neox_rope_in_place(&mut batch, 3, 1, 4, 4, 10000.0, 1.0, 0).unwrap();

        let mut single = vec![1.0, 2.0, 3.0, 4.0];
        apply_neox_rope_in_place(&mut single, 1, 1, 4, 4, 10000.0, 1.0, 2).unwrap();

        for (i, (a, b)) in single.iter().zip(&batch[8..12]).enumerate() {
            assert!((a - b).abs() < 1e-6, "dim {i}: offset={a} vs batch={b}");
        }
    }

    #[test]
    fn full_attention_prefill_then_decode_matches_full_reprocess() {
        // Build a small deterministic plan: 2 heads, 1 kv_head (GQA), hd=4.
        let head_count = 2;
        let kv_head_count = 1;
        let hd = 4;
        let query_features = head_count * hd; // 8
        let kv_features = kv_head_count * hd; // 4
        let hidden = 6;

        // Q weight: hidden → query_features*2 (Q+Gate interleaved)
        let q_weight: Vec<f32> = (0..hidden * query_features * 2)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let k_weight: Vec<f32> = (0..hidden * kv_features)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.08)
            .collect();
        let v_weight: Vec<f32> = (0..hidden * kv_features)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let output_weight: Vec<f32> = (0..query_features * hidden)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
            .collect();
        let q_norm = vec![1.0_f32; hd];
        let k_norm = vec![1.0_f32; hd];

        let plan = super::super::plan::Qwen35FullAttentionLayerPlan {
            norm_values: vec![1.0; hidden],
            q_norm_values: q_norm,
            k_norm_values: k_norm,
            q_weight_values: q_weight,
            k_weight_values: k_weight,
            v_weight_values: v_weight,
            output_weight_values: output_weight,
            head_count,
            kv_head_count,
            head_dimension: hd,
            attention_scale: 1.0 / (hd as f32).sqrt(),
            rope_n_dims: hd,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
        };

        // 3-token prompt + 1 decode token.
        let prompt: Vec<f32> = (0..3 * hidden).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let new_token: Vec<f32> = (0..hidden).map(|i| (i as f32 + 50.0) * 0.05).collect();

        crate::backend::ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        // Full reprocess: 4 tokens at once.
        let full_input: Vec<f32> = prompt.iter().chain(new_token.iter()).copied().collect();
        let full_output =
            qwen35_full_attention_inference(&plan, &full_input, 4, 1e-5, &backend).unwrap();
        let expected = &full_output[3 * hidden..4 * hidden];

        // Prefill 3 tokens, then decode 1.
        let mut state = Qwen35FullAttentionState::new(4, kv_head_count, hd).unwrap();
        let _prefill_out =
            qwen35_full_attention_prefill(&plan, &prompt, 3, 1e-5, &mut state, &backend).unwrap();
        let decode_out =
            qwen35_full_attention_decode_step(&plan, &new_token, 1e-5, &mut state).unwrap();

        for (i, (a, b)) in decode_out.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "feature {i}: decode={a} vs full={b}, diff={}",
                (a - b).abs()
            );
        }
    }
}
