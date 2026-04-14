use super::attention::QkvProjections;
use super::error::E2eError;
use super::numeric::checked_mul;
use ggml_rs::{Backend, Bytes, Context, Graph, Length, Shape2D, Tensor};

/// Slack constant added to memory estimates for ggml graph/tensor overhead.
pub(super) const PROJECTION_SLACK_BYTES: usize = 4 * 1024 * 1024;

/// Raw linear-attention projection outputs read back from the persistent
/// projection graph.
///
/// Unlike [`super::linear_attention::LinearProjections`], this carries only
/// the four GPU-readback buffers and omits derived dimension fields
/// (`conv_channels`, `hidden_features`), which the caller must supply
/// from the layer plan.
#[derive(Debug)]
pub(super) struct RawLinearProjections {
    pub(super) qkv: Vec<f32>,
    pub(super) z: Vec<f32>,
    pub(super) alpha: Vec<f32>,
    pub(super) beta: Vec<f32>,
}

pub(super) fn rms_norm_with_weight(
    input: &[f32],
    hidden_features: usize,
    sequence_length: usize,
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, E2eError> {
    if weight.len() != hidden_features {
        return Err(E2eError::BufferLengthMismatch {
            expected: hidden_features,
            actual: weight.len(),
        });
    }
    let expected_input_len = checked_mul(hidden_features, sequence_length)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }

    let mut output = vec![0.0_f32; input.len()];
    for (src, dst) in input
        .chunks_exact(hidden_features)
        .zip(output.chunks_exact_mut(hidden_features))
    {
        let mean_square = src
            .iter()
            .copied()
            .map(|value| f64::from(value) * f64::from(value))
            .sum::<f64>()
            / hidden_features as f64;
        let inv_rms = 1.0_f32 / ((mean_square as f32) + eps).sqrt();
        dst.iter_mut()
            .zip(src.iter().zip(weight.iter()))
            .for_each(|(d, (&s, &w))| *d = s * inv_rms * w);
    }
    Ok(output)
}

pub(super) fn rms_norm_single(
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, E2eError> {
    let mut output = vec![0.0_f32; input.len()];
    rms_norm_single_into(input, weight, eps, &mut output)?;
    Ok(output)
}

/// In-place variant of [`rms_norm_single`] that writes into a pre-allocated
/// destination buffer, avoiding a heap allocation per call.
pub(super) fn rms_norm_single_into(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    dst: &mut [f32],
) -> Result<(), E2eError> {
    if input.len() != weight.len() {
        return Err(E2eError::BufferLengthMismatch {
            expected: weight.len(),
            actual: input.len(),
        });
    }
    if dst.len() < input.len() {
        return Err(E2eError::BufferLengthMismatch {
            expected: input.len(),
            actual: dst.len(),
        });
    }
    let mean_square = input
        .iter()
        .copied()
        .map(|value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        / input.len() as f64;
    let inv_rms = 1.0_f32 / ((mean_square as f32) + eps).sqrt();
    dst.iter_mut()
        .zip(input.iter().copied().zip(weight.iter().copied()))
        .for_each(|(d, (value, scale))| *d = value * inv_rms * scale);
    Ok(())
}

/// In-place RMS normalization that reads and writes the same buffer.
fn rms_norm_single_in_place(data: &mut [f32], weight: &[f32], eps: f32) -> Result<(), E2eError> {
    if data.len() != weight.len() {
        return Err(E2eError::BufferLengthMismatch {
            expected: weight.len(),
            actual: data.len(),
        });
    }
    let mean_square = data
        .iter()
        .copied()
        .map(|value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        / data.len() as f64;
    let inv_rms = 1.0_f32 / ((mean_square as f32) + eps).sqrt();
    data.iter_mut()
        .zip(weight.iter().copied())
        .for_each(|(d, scale)| *d *= inv_rms * scale);
    Ok(())
}

pub(super) fn add_in_place(accumulator: &mut [f32], addend: &[f32]) -> Result<(), E2eError> {
    if accumulator.len() != addend.len() {
        return Err(E2eError::BufferLengthMismatch {
            expected: accumulator.len(),
            actual: addend.len(),
        });
    }
    for (lhs, rhs) in accumulator.iter_mut().zip(addend.iter().copied()) {
        *lhs += rhs;
    }
    Ok(())
}

pub(super) fn project_sequence(
    input: &[f32],
    sequence_length: usize,
    input_features: usize,
    output_features: usize,
    weight: &[f32],
) -> Result<Vec<f32>, E2eError> {
    let expected_input_len = checked_mul(sequence_length, input_features)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }
    let expected_weight_len = checked_mul(input_features, output_features)?;
    if weight.len() != expected_weight_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_weight_len,
            actual: weight.len(),
        });
    }

    let mut output = vec![0.0_f32; checked_mul(sequence_length, output_features)?];
    for (input_row, dst_row) in input
        .chunks_exact(input_features)
        .zip(output.chunks_exact_mut(output_features))
    {
        for (feature, weights_row) in weight.chunks_exact(input_features).enumerate() {
            dst_row[feature] = super::numeric::dot(input_row, weights_row);
        }
    }
    Ok(output)
}

pub(super) fn head_slice(
    values: &[f32],
    token: usize,
    head: usize,
    head_count: usize,
    head_dimension: usize,
) -> &[f32] {
    let token_offset = token * head_count * head_dimension;
    let head_offset = token_offset + head * head_dimension;
    &values[head_offset..head_offset + head_dimension]
}

pub(super) fn head_slice_mut(
    values: &mut [f32],
    token: usize,
    head: usize,
    head_count: usize,
    head_dimension: usize,
) -> &mut [f32] {
    let token_offset = token * head_count * head_dimension;
    let head_offset = token_offset + head * head_dimension;
    &mut values[head_offset..head_offset + head_dimension]
}

pub(super) fn per_head_rms_norm(
    input: &[f32],
    _sequence_length: usize,
    head_count: usize,
    head_dimension: usize,
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, E2eError> {
    if weight.len() != head_dimension {
        return Err(E2eError::BufferLengthMismatch {
            expected: head_dimension,
            actual: weight.len(),
        });
    }
    let token_features = checked_mul(head_count, head_dimension)?;
    let mut output = input.to_vec();
    for token_slice in output.chunks_exact_mut(token_features) {
        for head_slice in token_slice.chunks_exact_mut(head_dimension) {
            rms_norm_single_in_place(head_slice, weight, eps)?;
        }
    }
    Ok(output)
}

pub(super) fn per_head_l2_norm(
    input: &[f32],
    _sequence_length: usize,
    head_count: usize,
    head_dimension: usize,
    eps: f32,
) -> Result<Vec<f32>, E2eError> {
    let token_features = checked_mul(head_count, head_dimension)?;
    let mut output = input.to_vec();
    for token_slice in output.chunks_exact_mut(token_features) {
        for head_slice in token_slice.chunks_exact_mut(head_dimension) {
            let norm = head_slice.iter().map(|v| v * v).sum::<f32>();
            let inv = 1.0_f32 / norm.sqrt().max(eps);
            head_slice.iter_mut().for_each(|v| *v *= inv);
        }
    }
    Ok(output)
}

pub(super) fn gather_embeddings(
    embedding_values: &[f32],
    hidden_features: usize,
    vocab_size: usize,
    token_ids: &[i32],
) -> Result<Vec<f32>, E2eError> {
    let expected_embedding_len = checked_mul(hidden_features, vocab_size)?;
    if embedding_values.len() != expected_embedding_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_embedding_len,
            actual: embedding_values.len(),
        });
    }

    let mut output = vec![0.0_f32; checked_mul(hidden_features, token_ids.len())?];
    for (dst_slice, &token_id) in output
        .chunks_exact_mut(hidden_features)
        .zip(token_ids.iter())
    {
        let token_index = super::numeric::validate_token_id(token_id, vocab_size)?;
        let src_offset = token_index * hidden_features;
        dst_slice.copy_from_slice(&embedding_values[src_offset..src_offset + hidden_features]);
    }
    Ok(output)
}

/// Estimate the backend memory needed for a single matmul projection.
pub(super) fn recommended_single_projection_memory(
    input_features: usize,
    output_features: usize,
    sequence_length: usize,
) -> Result<Bytes, E2eError> {
    let mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(input_features, output_features),
        Shape2D::new(input_features, sequence_length),
    )
    .map_err(|source| E2eError::ggml("recommended_backend_matmul_memory(single)", source))?;
    let total = mem
        .get()
        .checked_add(PROJECTION_SLACK_BYTES)
        .ok_or(E2eError::MemorySizeOverflow)?;
    Ok(Bytes::new(total))
}

/// Compute a single matmul projection using a ggml compute graph.
///
/// General-purpose replacement for `project_sequence` when a backend is
/// available.  Used for output projections in both full and linear attention.
pub(super) fn project_sequence_graph(
    input: &[f32],
    sequence_length: usize,
    input_features: usize,
    output_features: usize,
    weight: &[f32],
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    let ctx_size =
        recommended_single_projection_memory(input_features, output_features, sequence_length)?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| E2eError::ggml("Context::new_no_alloc_bytes(proj)", source))?;

    let w = ctx
        .new_tensor_2d::<f32>(Shape2D::new(input_features, output_features))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W>", source))?;
    let x = ctx
        .new_tensor_2d::<f32>(Shape2D::new(input_features, sequence_length))
        .map_err(|source| E2eError::ggml("new_tensor_2d<X>", source))?;

    let y = ctx
        .mul_mat(&w, &x)
        .map_err(|source| E2eError::ggml("mul_mat(proj)", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(proj)", source))?;
    graph.build_forward_expand(&y);

    let _buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("allocate_tensors(proj)", source))?;

    w.write_data_backend(weight)
        .map_err(|source| E2eError::ggml("write_data_backend<W>", source))?;
    x.write_data_backend(input)
        .map_err(|source| E2eError::ggml("write_data_backend<X>", source))?;

    backend
        .compute(&mut graph)
        .map_err(|source| E2eError::ggml("compute(proj)", source))?;

    y.read_data_backend()
        .map_err(|source| E2eError::ggml("read_data_backend<Y>", source))
}

// ---------------------------------------------------------------------------
// LM head (output projection): rms_norm → weight → matmul → logits
// ---------------------------------------------------------------------------

/// Estimate backend memory for the LM head graph.
pub(super) fn recommended_lm_head_memory(
    hidden_features: usize,
    vocab_size: usize,
) -> Result<Bytes, E2eError> {
    let matmul_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, vocab_size),
        Shape2D::new(hidden_features, 1),
    )
    .map_err(|source| E2eError::ggml("recommended_backend_matmul_memory(lm_head)", source))?;

    // Extra for norm input tensor, norm weight, and intermediate tensors.
    let slack = hidden_features
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_add(PROJECTION_SLACK_BYTES))
        .ok_or(E2eError::MemorySizeOverflow)?;

    let total = matmul_mem
        .get()
        .checked_add(slack)
        .ok_or(E2eError::MemorySizeOverflow)?;
    Ok(Bytes::new(total))
}

/// One-shot LM head: rms_norm(hidden, eps) * norm_weight → matmul(output_weight).
///
/// Kept as `#[cfg(test)]` reference implementation for parity tests and benchmarks.
/// Production code uses the persistent `LmHeadResources` / `build_lm_head_graph` path.
#[cfg(test)]
pub(super) fn lm_head_graph(
    hidden_state: &[f32],
    norm_weight: &[f32],
    output_weight: &[f32],
    hidden_features: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    if hidden_state.len() != hidden_features {
        return Err(E2eError::BufferLengthMismatch {
            expected: hidden_features,
            actual: hidden_state.len(),
        });
    }
    if norm_weight.len() != hidden_features {
        return Err(E2eError::BufferLengthMismatch {
            expected: hidden_features,
            actual: norm_weight.len(),
        });
    }
    let expected_output_len = checked_mul(hidden_features, vocab_size)?;
    if output_weight.len() != expected_output_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_output_len,
            actual: output_weight.len(),
        });
    }

    let ctx_size = recommended_lm_head_memory(hidden_features, vocab_size)?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| E2eError::ggml("Context::new_no_alloc_bytes(lm_head)", source))?;

    let w_out = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, vocab_size))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_OUT>", source))?;
    let norm_w = ctx
        .new_tensor_1d::<f32>(Length::new(hidden_features))
        .map_err(|source| E2eError::ggml("new_tensor_1d<NORM_W>", source))?;
    let x_in = ctx
        .new_tensor_1d::<f32>(Length::new(hidden_features))
        .map_err(|source| E2eError::ggml("new_tensor_1d<X_IN>", source))?;

    let x_normed = ctx
        .rms_norm(&x_in, rms_norm_eps)
        .map_err(|source| E2eError::ggml("rms_norm(lm_head)", source))?;
    let x_scaled = ctx
        .mul(&x_normed, &norm_w)
        .map_err(|source| E2eError::ggml("mul(lm_head_norm)", source))?;
    let x_2d = ctx
        .reshape_2d(&x_scaled, hidden_features, 1)
        .map_err(|source| E2eError::ggml("reshape_2d(lm_head)", source))?;
    let logits = ctx
        .mul_mat(&w_out, &x_2d)
        .map_err(|source| E2eError::ggml("mul_mat(lm_head)", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(lm_head)", source))?;
    graph.build_forward_expand(&logits);

    let _buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("allocate_tensors(lm_head)", source))?;

    w_out
        .write_data_backend(output_weight)
        .map_err(|source| E2eError::ggml("write<W_OUT>", source))?;
    norm_w
        .write_data_backend(norm_weight)
        .map_err(|source| E2eError::ggml("write<NORM_W>", source))?;
    x_in.write_data_backend(hidden_state)
        .map_err(|source| E2eError::ggml("write<X_IN>", source))?;

    backend
        .compute(&mut graph)
        .map_err(|source| E2eError::ggml("compute(lm_head)", source))?;

    logits
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read<logits>", source))
}

/// Built parts of a persistent LM head graph.
///
/// Separates weight tensors (uploaded once) from I/O tensors (used per step).
/// Returned by [`build_lm_head_graph`].
pub(super) struct LmHeadGraphParts<'ctx> {
    /// Weight tensor for output projection `[hidden_features, vocab_size]`.
    pub w_out: Tensor<'ctx, f32>,
    /// Weight tensor for RMS norm `[hidden_features]`.
    pub norm_w: Tensor<'ctx, f32>,
    /// Input tensor (hidden state) — written per step.
    pub x_in: Tensor<'ctx, f32>,
    /// Output tensor (logits) — read per step.
    pub logits: Tensor<'ctx, f32>,
    /// Compute graph: rms_norm → mul → reshape → mul_mat.
    pub graph: Graph<'ctx>,
}

/// Build an LM head ggml graph (rms_norm → weight → matmul) in the given context.
///
/// Returns [`LmHeadGraphParts`] containing the weight, input, output tensors
/// and compute graph. The caller must:
/// 1. Call `ctx.allocate_tensors(backend)` and keep the buffer alive.
/// 2. Upload weights once via `w_out.write_data_backend` and `norm_w.write_data_backend`.
/// 3. Per step: upload hidden state to the returned x_input, compute, read logits.
///
/// This is the building block for persistent LM head contexts in generation loops.
pub(super) fn build_lm_head_graph<'ctx>(
    ctx: &'ctx Context,
    hidden_features: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
) -> Result<LmHeadGraphParts<'ctx>, E2eError> {
    let w_out = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, vocab_size))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_OUT>(plm)", source))?;
    let norm_w = ctx
        .new_tensor_1d::<f32>(Length::new(hidden_features))
        .map_err(|source| E2eError::ggml("new_tensor_1d<NORM_W>(plm)", source))?;
    let x_in = ctx
        .new_tensor_1d::<f32>(Length::new(hidden_features))
        .map_err(|source| E2eError::ggml("new_tensor_1d<X_IN>(plm)", source))?;

    let x_normed = ctx
        .rms_norm(&x_in, rms_norm_eps)
        .map_err(|source| E2eError::ggml("rms_norm(plm)", source))?;
    let x_scaled = ctx
        .mul(&x_normed, &norm_w)
        .map_err(|source| E2eError::ggml("mul(plm_norm)", source))?;
    let x_2d = ctx
        .reshape_2d(&x_scaled, hidden_features, 1)
        .map_err(|source| E2eError::ggml("reshape_2d(plm)", source))?;
    let logits = ctx
        .mul_mat(&w_out, &x_2d)
        .map_err(|source| E2eError::ggml("mul_mat(plm)", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(plm)", source))?;
    graph.build_forward_expand(&logits);

    Ok(LmHeadGraphParts {
        w_out,
        norm_w,
        x_in,
        logits,
        graph,
    })
}

/// Argmax over a logits vector — returns the token index with the highest value.
pub(super) fn argmax_token_id(logits: &[f32]) -> Result<i32, E2eError> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| i32::try_from(idx).map_err(|_| E2eError::MemorySizeOverflow))
        .ok_or(E2eError::BufferLengthMismatch {
            expected: 1,
            actual: 0,
        })?
}

/// Run one sampling step through a pre-built LM head graph.
///
/// Writes `hidden_state` (a single token's hidden vector) to the input tensor,
/// recomputes the graph, reads back the logits, and returns the argmax token ID.
pub(super) fn lm_head_sample_step(
    hidden_state: &[f32],
    x_in: &Tensor<'_, f32>,
    logits_t: &Tensor<'_, f32>,
    lm_graph: &mut Graph<'_>,
    backend: &Backend,
) -> Result<i32, E2eError> {
    x_in.write_data_backend(hidden_state)
        .map_err(|source| E2eError::ggml("write<X_IN>(step)", source))?;
    backend
        .compute(lm_graph)
        .map_err(|source| E2eError::ggml("compute(lm_head_step)", source))?;
    let logits_data: Vec<f32> = logits_t
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read<logits>(step)", source))?;
    argmax_token_id(&logits_data)
}

// ---------------------------------------------------------------------------
// Persistent decode projections: built once, reused every decode step.
// ---------------------------------------------------------------------------

use ggml_rs::BackendBuffer;

/// Pre-built projection graphs for a single attention layer's decode step.
///
/// Each variant holds an input graph (hidden → projections) and an output graph
/// (core result → hidden), along with all tensor handles needed for per-step
/// I/O. The `BackendBuffer` keeps allocated memory alive.
///
/// # Lifetime
///
/// The `'ctx` lifetime ties all tensor/graph handles to the `Context` that
/// created them. The caller must ensure the `Context` outlives this value.
pub(super) enum PersistentDecodeProjection<'ctx> {
    FullAttention {
        x_in: Tensor<'ctx, f32>,
        q_out: Tensor<'ctx, f32>,
        k_out: Tensor<'ctx, f32>,
        v_out: Tensor<'ctx, f32>,
        input_graph: Graph<'ctx>,
        output: OutputProjectionGraph<'ctx>,
        _buffer: BackendBuffer<'ctx>,
    },
    LinearAttention {
        x_in: Tensor<'ctx, f32>,
        qkv_out: Tensor<'ctx, f32>,
        z_out: Tensor<'ctx, f32>,
        alpha_out: Tensor<'ctx, f32>,
        beta_out: Tensor<'ctx, f32>,
        input_graph: Graph<'ctx>,
        output: OutputProjectionGraph<'ctx>,
        _buffer: BackendBuffer<'ctx>,
    },
}

/// Sum `recommended_backend_matmul_memory` for a batch of projections.
///
/// Each entry is `(weight_shape, input_shape, label)`.  The label is used
/// in the error message if the memory query fails.  Returns the total plus
/// `2 × PROJECTION_SLACK_BYTES` for ggml graph/tensor overhead.
fn sum_matmul_memories(
    projections: &[(Shape2D, Shape2D, &'static str)],
) -> Result<Bytes, E2eError> {
    let total = projections
        .iter()
        .try_fold(0usize, |acc, &(weight, input, label)| {
            let mem = Context::recommended_backend_matmul_memory::<f32>(weight, input)
                .map_err(|source| E2eError::ggml(label, source))?;
            acc.checked_add(mem.get())
                .ok_or(E2eError::MemorySizeOverflow)
        })?
        .checked_add(PROJECTION_SLACK_BYTES * 2)
        .ok_or(E2eError::MemorySizeOverflow)?;
    Ok(Bytes::new(total))
}

/// Estimate ggml context metadata bytes for a full attention persistent
/// projection (both input and output graphs in a single context).
pub(super) fn recommended_persistent_full_attention_memory(
    hidden_features: usize,
    query_features_x2: usize,
    kv_features: usize,
    query_features: usize,
) -> Result<Bytes, E2eError> {
    let h1 = Shape2D::new(hidden_features, 1);
    sum_matmul_memories(&[
        (
            Shape2D::new(hidden_features, query_features_x2),
            h1,
            "mem(pfa_q)",
        ),
        (Shape2D::new(hidden_features, kv_features), h1, "mem(pfa_k)"),
        (Shape2D::new(hidden_features, kv_features), h1, "mem(pfa_v)"),
        (
            Shape2D::new(query_features, hidden_features),
            Shape2D::new(query_features, 1),
            "mem(pfa_out)",
        ),
    ])
}

/// Estimate ggml context metadata bytes for a linear attention persistent
/// projection (both input and output graphs in a single context).
pub(super) fn recommended_persistent_linear_attention_memory(
    hidden_features: usize,
    conv_channels: usize,
    inner_size: usize,
    time_step_rank: usize,
) -> Result<Bytes, E2eError> {
    let h1 = Shape2D::new(hidden_features, 1);
    sum_matmul_memories(&[
        (
            Shape2D::new(hidden_features, conv_channels),
            h1,
            "mem(pla_qkv)",
        ),
        (Shape2D::new(hidden_features, inner_size), h1, "mem(pla_z)"),
        (
            Shape2D::new(hidden_features, time_step_rank),
            h1,
            "mem(pla_alpha)",
        ),
        (
            Shape2D::new(hidden_features, time_step_rank),
            h1,
            "mem(pla_beta)",
        ),
        (
            Shape2D::new(inner_size, hidden_features),
            Shape2D::new(inner_size, 1),
            "mem(pla_out)",
        ),
    ])
}

/// Shared output projection sub-graph returned by [`build_output_projection_graph`].
pub(super) struct OutputProjectionGraph<'ctx> {
    /// Weight tensor `[input_features, output_features]` — uploaded once.
    pub w: Tensor<'ctx, f32>,
    /// Input tensor `[input_features, 1]` — written per step.
    pub x: Tensor<'ctx, f32>,
    /// Output tensor `[output_features, 1]` — read per step.
    pub y: Tensor<'ctx, f32>,
    /// Compute graph: matmul(W, X).
    pub graph: Graph<'ctx>,
}

/// Build a single matmul output projection graph: `y = W × x`.
///
/// Reused by both full-attention and linear-attention persistent graph builders.
fn build_output_projection_graph<'ctx>(
    ctx: &'ctx Context,
    input_features: usize,
    output_features: usize,
    label: &'static str,
) -> Result<OutputProjectionGraph<'ctx>, E2eError> {
    let w = ctx
        .new_tensor_2d::<f32>(Shape2D::new(input_features, output_features))
        .map_err(|source| E2eError::ggml(label, source))?;
    let x = ctx
        .new_tensor_2d::<f32>(Shape2D::new(input_features, 1))
        .map_err(|source| E2eError::ggml(label, source))?;
    let y = ctx
        .mul_mat(&w, &x)
        .map_err(|source| E2eError::ggml(label, source))?;
    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml(label, source))?;
    graph.build_forward_expand(&y);
    Ok(OutputProjectionGraph { w, x, y, graph })
}

/// Built parts of a persistent full attention projection graph pair.
///
/// Returned by [`build_persistent_full_attention_graphs`].
pub(super) struct FullAttentionGraphParts<'ctx> {
    /// Input tensor (hidden state) — written per step.
    pub x_in: Tensor<'ctx, f32>,
    /// Weight tensors (uploaded once after allocation).
    pub w_q: Tensor<'ctx, f32>,
    pub w_k: Tensor<'ctx, f32>,
    pub w_v: Tensor<'ctx, f32>,
    /// Projection outputs — read per step.
    pub q_out: Tensor<'ctx, f32>,
    pub k_out: Tensor<'ctx, f32>,
    pub v_out: Tensor<'ctx, f32>,
    /// Input projection compute graph.
    pub input_graph: Graph<'ctx>,
    /// Output projection sub-graph (shared structure with linear attention).
    pub output: OutputProjectionGraph<'ctx>,
}

/// Build a persistent full-attention input + output projection graph.
///
/// Creates tensors and graph nodes in `ctx`. After calling this, the caller
/// must call `ctx.allocate_tensors(backend)` and upload weights once.
///
/// Returns the constructed `PersistentDecodeProjection::FullAttention` variant
/// (minus `_buffer` — the caller must attach it after allocation).
pub(super) fn build_persistent_full_attention_graphs<'ctx>(
    ctx: &'ctx Context,
    hidden_features: usize,
    query_features_x2: usize,
    kv_features: usize,
    query_features: usize,
) -> Result<FullAttentionGraphParts<'ctx>, E2eError> {
    // Input projection tensors
    let w_q = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, query_features_x2))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_Q>(pfa)", source))?;
    let w_k = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, kv_features))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_K>(pfa)", source))?;
    let w_v = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, kv_features))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_V>(pfa)", source))?;
    let x_in = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, 1))
        .map_err(|source| E2eError::ggml("new_tensor_2d<X_IN>(pfa)", source))?;

    let q_out = ctx
        .mul_mat(&w_q, &x_in)
        .map_err(|source| E2eError::ggml("mul_mat<Q>(pfa)", source))?;
    let k_out = ctx
        .mul_mat(&w_k, &x_in)
        .map_err(|source| E2eError::ggml("mul_mat<K>(pfa)", source))?;
    let v_out = ctx
        .mul_mat(&w_v, &x_in)
        .map_err(|source| E2eError::ggml("mul_mat<V>(pfa)", source))?;

    let mut input_graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(pfa_in)", source))?;
    input_graph.build_forward_expand(&q_out);
    input_graph.build_forward_expand(&k_out);
    input_graph.build_forward_expand(&v_out);

    // Output projection sub-graph
    let output =
        build_output_projection_graph(ctx, query_features, hidden_features, "output_proj(pfa)")?;

    Ok(FullAttentionGraphParts {
        x_in,
        w_q,
        w_k,
        w_v,
        q_out,
        k_out,
        v_out,
        input_graph,
        output,
    })
}

/// Built parts of a persistent linear attention projection graph pair.
///
/// Returned by [`build_persistent_linear_attention_graphs`].
pub(super) struct LinearAttentionGraphParts<'ctx> {
    /// Input tensor (hidden state) — written per step.
    pub x_in: Tensor<'ctx, f32>,
    /// Weight tensors (uploaded once after allocation).
    pub w_qkv: Tensor<'ctx, f32>,
    pub w_z: Tensor<'ctx, f32>,
    pub w_alpha: Tensor<'ctx, f32>,
    pub w_beta: Tensor<'ctx, f32>,
    /// Projection outputs — read per step.
    pub qkv_out: Tensor<'ctx, f32>,
    pub z_out: Tensor<'ctx, f32>,
    pub alpha_out: Tensor<'ctx, f32>,
    pub beta_out: Tensor<'ctx, f32>,
    /// Input projection compute graph.
    pub input_graph: Graph<'ctx>,
    /// Output projection sub-graph (shared structure with full attention).
    pub output: OutputProjectionGraph<'ctx>,
}

/// Build a persistent linear attention input + output projection graph.
pub(super) fn build_persistent_linear_attention_graphs<'ctx>(
    ctx: &'ctx Context,
    hidden_features: usize,
    conv_channels: usize,
    inner_size: usize,
    time_step_rank: usize,
) -> Result<LinearAttentionGraphParts<'ctx>, E2eError> {
    let w_qkv = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, conv_channels))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_QKV>(pla)", source))?;
    let w_z = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, inner_size))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_Z>(pla)", source))?;
    let w_alpha = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, time_step_rank))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_ALPHA>(pla)", source))?;
    let w_beta = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, time_step_rank))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_BETA>(pla)", source))?;
    let x_in = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, 1))
        .map_err(|source| E2eError::ggml("new_tensor_2d<X_IN>(pla)", source))?;

    let qkv_out = ctx
        .mul_mat(&w_qkv, &x_in)
        .map_err(|source| E2eError::ggml("mul_mat<QKV>(pla)", source))?;
    let z_out = ctx
        .mul_mat(&w_z, &x_in)
        .map_err(|source| E2eError::ggml("mul_mat<Z>(pla)", source))?;
    let alpha_out = ctx
        .mul_mat(&w_alpha, &x_in)
        .map_err(|source| E2eError::ggml("mul_mat<ALPHA>(pla)", source))?;
    let beta_out = ctx
        .mul_mat(&w_beta, &x_in)
        .map_err(|source| E2eError::ggml("mul_mat<BETA>(pla)", source))?;

    let mut input_graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(pla_in)", source))?;
    input_graph.build_forward_expand(&qkv_out);
    input_graph.build_forward_expand(&z_out);
    input_graph.build_forward_expand(&alpha_out);
    input_graph.build_forward_expand(&beta_out);

    // Output projection sub-graph
    let output =
        build_output_projection_graph(ctx, inner_size, hidden_features, "output_proj(pla)")?;

    Ok(LinearAttentionGraphParts {
        x_in,
        w_qkv,
        w_z,
        w_alpha,
        w_beta,
        qkv_out,
        z_out,
        alpha_out,
        beta_out,
        input_graph,
        output,
    })
}

impl<'ctx> PersistentDecodeProjection<'ctx> {
    /// Run the input projection step: upload hidden state, compute, read outputs.
    pub(super) fn project_input(
        &mut self,
        hidden_state: &[f32],
        backend: &Backend,
    ) -> Result<(), E2eError> {
        match self {
            Self::FullAttention {
                x_in, input_graph, ..
            }
            | Self::LinearAttention {
                x_in, input_graph, ..
            } => {
                x_in.write_data_backend(hidden_state)
                    .map_err(|source| E2eError::ggml("write<X_IN>(proj_step)", source))?;
                backend
                    .compute(input_graph)
                    .map_err(|source| E2eError::ggml("compute(proj_input_step)", source))?;
                Ok(())
            }
        }
    }

    /// Read raw QKV projection outputs for full attention.
    pub(super) fn read_full_attention_projections(&self) -> Result<QkvProjections, E2eError> {
        match self {
            Self::FullAttention {
                q_out,
                k_out,
                v_out,
                ..
            } => {
                let q_full: Vec<f32> = q_out
                    .read_data_backend()
                    .map_err(|source| E2eError::ggml("read<Q>(pfa_step)", source))?;
                let k_proj: Vec<f32> = k_out
                    .read_data_backend()
                    .map_err(|source| E2eError::ggml("read<K>(pfa_step)", source))?;
                let v_proj: Vec<f32> = v_out
                    .read_data_backend()
                    .map_err(|source| E2eError::ggml("read<V>(pfa_step)", source))?;
                Ok(QkvProjections {
                    q_full,
                    k_proj,
                    v_proj,
                })
            }
            Self::LinearAttention { .. } => Err(E2eError::BufferLengthMismatch {
                expected: 0,
                actual: 1,
            }),
        }
    }

    /// Read raw linear attention projection outputs.
    pub(super) fn read_linear_attention_projections(
        &self,
    ) -> Result<RawLinearProjections, E2eError> {
        match self {
            Self::LinearAttention {
                qkv_out,
                z_out,
                alpha_out,
                beta_out,
                ..
            } => {
                let qkv: Vec<f32> = qkv_out
                    .read_data_backend()
                    .map_err(|source| E2eError::ggml("read<QKV>(pla_step)", source))?;
                let z: Vec<f32> = z_out
                    .read_data_backend()
                    .map_err(|source| E2eError::ggml("read<Z>(pla_step)", source))?;
                let alpha: Vec<f32> = alpha_out
                    .read_data_backend()
                    .map_err(|source| E2eError::ggml("read<ALPHA>(pla_step)", source))?;
                let beta: Vec<f32> = beta_out
                    .read_data_backend()
                    .map_err(|source| E2eError::ggml("read<BETA>(pla_step)", source))?;
                Ok(RawLinearProjections {
                    qkv,
                    z,
                    alpha,
                    beta,
                })
            }
            Self::FullAttention { .. } => Err(E2eError::BufferLengthMismatch {
                expected: 0,
                actual: 1,
            }),
        }
    }

    /// Run the output projection step: upload core result, compute, read hidden.
    pub(super) fn project_output(
        &mut self,
        core_output: &[f32],
        backend: &Backend,
    ) -> Result<Vec<f32>, E2eError> {
        let output = match self {
            Self::FullAttention { output, .. } | Self::LinearAttention { output, .. } => output,
        };
        output
            .x
            .write_data_backend(core_output)
            .map_err(|source| E2eError::ggml("write<OUT_X>(proj_out_step)", source))?;
        backend
            .compute(&mut output.graph)
            .map_err(|source| E2eError::ggml("compute(proj_output_step)", source))?;
        output
            .y
            .read_data_backend()
            .map_err(|source| E2eError::ggml("read<OUT_Y>(proj_out_step)", source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_applies_weight_per_position() {
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32, 0.25];
        let output =
            rms_norm_with_weight(&input, 2, 2, &weight, 1e-5).expect("rms norm should succeed");
        assert_eq!(output.len(), input.len());
        assert!(output[0].is_finite());
        assert!(output[1].is_finite());
        assert!(output[2].is_finite());
        assert!(output[3].is_finite());
        assert!(output[0].abs() > output[1].abs());
    }

    #[test]
    fn rms_norm_eps_changes_scaled_output() {
        let input = vec![1.0_f32, 2.0];
        let weight = vec![1.0_f32, 1.0];
        let loose = rms_norm_with_weight(&input, 2, 1, &weight, 1e-5).expect("rms norm");
        let tight = rms_norm_with_weight(&input, 2, 1, &weight, 1e-6).expect("rms norm");
        assert_ne!(loose, tight);
    }

    #[test]
    fn project_sequence_graph_matches_host_projection() {
        use crate::backend::ensure_backends_loaded;
        use ggml_rs::BackendKind;

        let input_features = 8_usize;
        let output_features = 4_usize;
        let seq_len = 3_usize;

        let weight: Vec<f32> = (0..output_features * input_features)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let input: Vec<f32> = (0..seq_len * input_features)
            .map(|i| (i as f32 + 1.0) * 0.05)
            .collect();

        let host_result =
            project_sequence(&input, seq_len, input_features, output_features, &weight)
                .expect("host projection");

        ensure_backends_loaded();
        let backend = Backend::new(BackendKind::Cpu).expect("CPU backend");

        let graph_result = project_sequence_graph(
            &input,
            seq_len,
            input_features,
            output_features,
            &weight,
            &backend,
        )
        .expect("graph projection");

        assert_eq!(host_result.len(), graph_result.len());
        for (i, (h, g)) in host_result.iter().zip(graph_result.iter()).enumerate() {
            assert!(
                (h - g).abs() < 1e-5,
                "element {i}: host={h} vs graph={g}, diff={}",
                (h - g).abs()
            );
        }
    }

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax_token_id(&[1.0, 3.0, 2.0]).unwrap(), 1);
        assert_eq!(argmax_token_id(&[5.0, 1.0, 2.0]).unwrap(), 0);
        assert_eq!(argmax_token_id(&[-1.0, -2.0, -0.5]).unwrap(), 2);
    }

    #[test]
    fn argmax_empty_returns_error() {
        assert!(argmax_token_id(&[]).is_err());
    }

    #[test]
    fn lm_head_graph_matches_host_sampling() {
        use super::super::generation::greedy_next_token_id;
        use crate::backend::ensure_backends_loaded;
        use ggml_rs::BackendKind;

        let hidden_features = 8_usize;
        let vocab_size = 6_usize;
        let rms_norm_eps = 1e-5_f32;

        let norm_weight: Vec<f32> = (0..hidden_features)
            .map(|i| 0.8 + (i as f32) * 0.05)
            .collect();
        let output_weight: Vec<f32> = (0..hidden_features * vocab_size)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();
        let hidden_state: Vec<f32> = (0..hidden_features)
            .map(|i| (i as f32 + 1.0) * 0.2)
            .collect();

        // Host path: rms_norm_with_weight → greedy_next_token_id
        let normed = rms_norm_with_weight(
            &hidden_state,
            hidden_features,
            1,
            &norm_weight,
            rms_norm_eps,
        )
        .expect("host rms_norm");
        let host_token =
            greedy_next_token_id(&normed, 0, hidden_features, &output_weight, vocab_size)
                .expect("host sampling");

        // Graph path
        ensure_backends_loaded();
        let backend = Backend::new(BackendKind::Cpu).expect("CPU backend");
        let graph_logits = lm_head_graph(
            &hidden_state,
            &norm_weight,
            &output_weight,
            hidden_features,
            vocab_size,
            rms_norm_eps,
            &backend,
        )
        .expect("lm_head_graph");
        let graph_token = argmax_token_id(&graph_logits).expect("argmax");

        assert_eq!(
            host_token, graph_token,
            "host token {host_token} != graph token {graph_token}"
        );
    }

    #[test]
    fn lm_head_sample_step_matches_one_shot() {
        use crate::backend::ensure_backends_loaded;
        use ggml_rs::BackendKind;

        let hidden_features = 8_usize;
        let vocab_size = 6_usize;
        let rms_norm_eps = 1e-5_f32;

        let norm_weight: Vec<f32> = (0..hidden_features)
            .map(|i| 0.8 + (i as f32) * 0.05)
            .collect();
        let output_weight: Vec<f32> = (0..hidden_features * vocab_size)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();
        let hidden_state: Vec<f32> = (0..hidden_features)
            .map(|i| (i as f32 + 1.0) * 0.2)
            .collect();

        ensure_backends_loaded();
        let backend = Backend::new(BackendKind::Cpu).expect("CPU backend");

        // One-shot graph
        let one_shot_logits = lm_head_graph(
            &hidden_state,
            &norm_weight,
            &output_weight,
            hidden_features,
            vocab_size,
            rms_norm_eps,
            &backend,
        )
        .expect("one-shot");
        let one_shot_token = argmax_token_id(&one_shot_logits).expect("argmax");

        // Persistent graph via build_lm_head_graph
        let ctx_size = recommended_lm_head_memory(hidden_features, vocab_size).expect("mem");
        let ctx = Context::new_no_alloc_bytes(ctx_size).expect("ctx");
        let mut parts =
            build_lm_head_graph(&ctx, hidden_features, vocab_size, rms_norm_eps).expect("build");
        let _buf = ctx.allocate_tensors(&backend).expect("alloc");
        parts
            .w_out
            .write_data_backend(&output_weight)
            .expect("write w_out");
        parts
            .norm_w
            .write_data_backend(&norm_weight)
            .expect("write norm_w");

        let step_token = lm_head_sample_step(
            &hidden_state,
            &parts.x_in,
            &parts.logits,
            &mut parts.graph,
            &backend,
        )
        .expect("sample_step");

        assert_eq!(
            one_shot_token, step_token,
            "one-shot token {one_shot_token} != step token {step_token}"
        );
    }

    #[test]
    fn persistent_full_attention_projection_matches_one_shot() {
        use crate::backend::ensure_backends_loaded;
        use ggml_rs::BackendKind;

        let hidden = 8_usize;
        let qf_x2 = 12_usize; // query_features * 2 (Q+gate interleaved)
        let kv = 4_usize;
        let qf = 6_usize; // query_features (for output proj)

        let w_q: Vec<f32> = (0..hidden * qf_x2)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let w_k: Vec<f32> = (0..hidden * kv)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.08)
            .collect();
        let w_v: Vec<f32> = (0..hidden * kv)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let w_out: Vec<f32> = (0..qf * hidden)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
            .collect();
        let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.2).collect();

        ensure_backends_loaded();
        let backend = Backend::new(BackendKind::Cpu).expect("CPU backend");

        // One-shot host projection
        let q_host = project_sequence(&input, 1, hidden, qf_x2, &w_q).expect("host Q");
        let k_host = project_sequence(&input, 1, hidden, kv, &w_k).expect("host K");
        let v_host = project_sequence(&input, 1, hidden, kv, &w_v).expect("host V");

        // Persistent projection
        let ctx_size =
            recommended_persistent_full_attention_memory(hidden, qf_x2, kv, qf).expect("mem");
        let ctx = Context::new_no_alloc_bytes(ctx_size).expect("ctx");
        let g = build_persistent_full_attention_graphs(&ctx, hidden, qf_x2, kv, qf).expect("build");
        let _buf = ctx.allocate_tensors(&backend).expect("alloc");

        g.w_q.write_data_backend(&w_q).expect("write W_Q");
        g.w_k.write_data_backend(&w_k).expect("write W_K");
        g.w_v.write_data_backend(&w_v).expect("write W_V");
        g.output.w.write_data_backend(&w_out).expect("write W_OUT");

        let mut proj = PersistentDecodeProjection::FullAttention {
            x_in: g.x_in,
            q_out: g.q_out,
            k_out: g.k_out,
            v_out: g.v_out,
            input_graph: g.input_graph,
            output: g.output,
            _buffer: _buf,
        };
        proj.project_input(&input, &backend).expect("project_input");
        let qkv = proj.read_full_attention_projections().expect("read QKV");

        for (i, (h, p)) in q_host.iter().zip(qkv.q_full.iter()).enumerate() {
            assert!(
                (h - p).abs() < 1e-5,
                "Q[{i}]: host={h} vs persistent={p}, diff={}",
                (h - p).abs()
            );
        }
        for (i, (h, p)) in k_host.iter().zip(qkv.k_proj.iter()).enumerate() {
            assert!(
                (h - p).abs() < 1e-5,
                "K[{i}]: host={h} vs persistent={p}, diff={}",
                (h - p).abs()
            );
        }
        for (i, (h, p)) in v_host.iter().zip(qkv.v_proj.iter()).enumerate() {
            assert!(
                (h - p).abs() < 1e-5,
                "V[{i}]: host={h} vs persistent={p}, diff={}",
                (h - p).abs()
            );
        }

        // Output projection parity
        let core_out: Vec<f32> = (0..qf).map(|i| (i as f32 + 0.5) * 0.3).collect();
        let out_host = project_sequence(&core_out, 1, qf, hidden, &w_out).expect("host out");
        let out_pers = proj
            .project_output(&core_out, &backend)
            .expect("persistent out");

        for (i, (h, p)) in out_host.iter().zip(out_pers.iter()).enumerate() {
            assert!(
                (h - p).abs() < 1e-5,
                "OUT[{i}]: host={h} vs persistent={p}, diff={}",
                (h - p).abs()
            );
        }
    }

    #[test]
    fn persistent_linear_attention_projection_matches_one_shot() {
        use crate::backend::ensure_backends_loaded;
        use ggml_rs::BackendKind;

        let hidden = 8_usize;
        let conv_ch = 10_usize;
        let inner = 6_usize;
        let tsr = 4_usize; // time_step_rank

        let w_qkv: Vec<f32> = (0..hidden * conv_ch)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let w_z: Vec<f32> = (0..hidden * inner)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.08)
            .collect();
        let w_alpha: Vec<f32> = (0..hidden * tsr)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let w_beta: Vec<f32> = (0..hidden * tsr)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
            .collect();
        let w_out: Vec<f32> = (0..inner * hidden)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.04)
            .collect();
        let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.2).collect();

        ensure_backends_loaded();
        let backend = Backend::new(BackendKind::Cpu).expect("CPU backend");

        // One-shot host projections
        let qkv_host = project_sequence(&input, 1, hidden, conv_ch, &w_qkv).expect("host QKV");
        let z_host = project_sequence(&input, 1, hidden, inner, &w_z).expect("host Z");
        let alpha_host = project_sequence(&input, 1, hidden, tsr, &w_alpha).expect("host alpha");
        let beta_host = project_sequence(&input, 1, hidden, tsr, &w_beta).expect("host beta");

        // Persistent projection
        let ctx_size = recommended_persistent_linear_attention_memory(hidden, conv_ch, inner, tsr)
            .expect("mem");
        let ctx = Context::new_no_alloc_bytes(ctx_size).expect("ctx");
        let g = build_persistent_linear_attention_graphs(&ctx, hidden, conv_ch, inner, tsr)
            .expect("build");
        let _buf = ctx.allocate_tensors(&backend).expect("alloc");

        g.w_qkv.write_data_backend(&w_qkv).expect("write W_QKV");
        g.w_z.write_data_backend(&w_z).expect("write W_Z");
        g.w_alpha
            .write_data_backend(&w_alpha)
            .expect("write W_ALPHA");
        g.w_beta.write_data_backend(&w_beta).expect("write W_BETA");
        g.output.w.write_data_backend(&w_out).expect("write W_OUT");

        let mut proj = PersistentDecodeProjection::LinearAttention {
            x_in: g.x_in,
            qkv_out: g.qkv_out,
            z_out: g.z_out,
            alpha_out: g.alpha_out,
            beta_out: g.beta_out,
            input_graph: g.input_graph,
            output: g.output,
            _buffer: _buf,
        };
        proj.project_input(&input, &backend).expect("project_input");
        let raw = proj
            .read_linear_attention_projections()
            .expect("read linear");

        for (name, host, pers) in [
            ("QKV", &qkv_host, &raw.qkv),
            ("Z", &z_host, &raw.z),
            ("alpha", &alpha_host, &raw.alpha),
            ("beta", &beta_host, &raw.beta),
        ] {
            assert_eq!(host.len(), pers.len(), "{name} length mismatch");
            for (i, (h, p)) in host.iter().zip(pers.iter()).enumerate() {
                assert!(
                    (h - p).abs() < 1e-5,
                    "{name}[{i}]: host={h} vs persistent={p}, diff={}",
                    (h - p).abs()
                );
            }
        }

        // Output projection parity
        let core_out: Vec<f32> = (0..inner).map(|i| (i as f32 + 0.5) * 0.3).collect();
        let out_host = project_sequence(&core_out, 1, inner, hidden, &w_out).expect("host out");
        let out_pers = proj
            .project_output(&core_out, &backend)
            .expect("persistent out");

        for (i, (h, p)) in out_host.iter().zip(out_pers.iter()).enumerate() {
            assert!(
                (h - p).abs() < 1e-5,
                "OUT[{i}]: host={h} vs persistent={p}, diff={}",
                (h - p).abs()
            );
        }
    }
}
