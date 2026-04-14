use super::super::error::E2eError;
#[cfg(test)]
use super::super::numeric::checked_mul;
use super::projection::PROJECTION_SLACK_BYTES;
use ggml_rs::{Backend, Bytes, Context, Graph, Length, Shape2D, Tensor};

/// Estimate backend memory for the LM head graph.
pub(in crate::e2e) fn recommended_lm_head_memory(
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
pub(in crate::e2e) fn lm_head_graph(
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
pub(in crate::e2e) struct LmHeadGraphParts<'ctx> {
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
pub(in crate::e2e) fn build_lm_head_graph<'ctx>(
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
pub(in crate::e2e) fn argmax_token_id(logits: &[f32]) -> Result<i32, E2eError> {
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
pub(in crate::e2e) fn lm_head_sample_step(
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
