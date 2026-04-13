use super::error::E2eError;
use super::numeric::checked_mul;
use crate::inference::MlpWeights;
use ggml_rs::{Backend, Bytes, Context, Length, Shape2D};

const MLP_BACKEND_SLACK_BYTES: usize = 4 * 1024 * 1024;

/// MLP forward pass with in-graph layer pre-norm.
///
/// Accepts un-normed input and applies `rms_norm + weight` as the first graph
/// operation before gate/up/down projections. This eliminates the host↔device
/// round-trip for the layer norm.
pub(super) fn mlp_sequence_inference_with_weights(
    weights: &MlpWeights<f32>,
    input: &[f32],
    sequence_length: usize,
    norm_weight: &[f32],
    rms_norm_eps: f32,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    let hidden_features = weights.hidden_features;
    let ffn_features = weights.ffn_features;
    let expected_input_len = checked_mul(hidden_features, sequence_length)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }
    if norm_weight.len() != hidden_features {
        return Err(E2eError::BufferLengthMismatch {
            expected: hidden_features,
            actual: norm_weight.len(),
        });
    }

    let ctx_size =
        recommended_mlp_backend_memory_bytes(hidden_features, ffn_features, sequence_length)?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| E2eError::ggml("Context::new_no_alloc_bytes", source))?;

    let w_gate = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, ffn_features))
        .map_err(|source| E2eError::ggml("Context::new_tensor_2d<W_GATE>", source))?;
    let w_up = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, ffn_features))
        .map_err(|source| E2eError::ggml("Context::new_tensor_2d<W_UP>", source))?;
    let w_down = ctx
        .new_tensor_2d::<f32>(Shape2D::new(ffn_features, hidden_features))
        .map_err(|source| E2eError::ggml("Context::new_tensor_2d<W_DOWN>", source))?;
    let x_raw = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, sequence_length))
        .map_err(|source| E2eError::ggml("Context::new_tensor_2d<X>", source))?;
    let norm_w = ctx
        .new_tensor_1d::<f32>(Length::new(hidden_features))
        .map_err(|source| E2eError::ggml("new_tensor_1d<norm_w>", source))?;

    // In-graph layer pre-norm: rms_norm over ne[0]=hidden_features, then scale by weight.
    let x_normed = ctx
        .rms_norm(&x_raw, rms_norm_eps)
        .map_err(|source| E2eError::ggml("rms_norm(X)", source))?;
    let x = ctx
        .mul(&x_normed, &norm_w)
        .map_err(|source| E2eError::ggml("mul(norm)", source))?;

    let gate = ctx
        .mul_mat(&w_gate, &x)
        .map_err(|source| E2eError::ggml("Context::mul_mat(GATE)", source))?;
    let up = ctx
        .mul_mat(&w_up, &x)
        .map_err(|source| E2eError::ggml("Context::mul_mat(UP)", source))?;
    let activated = ctx
        .silu(&gate)
        .map_err(|source| E2eError::ggml("Context::silu", source))?;
    let fused = ctx
        .mul(&activated, &up)
        .map_err(|source| E2eError::ggml("Context::mul(GATE*UP)", source))?;
    let y = ctx
        .mul_mat(&w_down, &fused)
        .map_err(|source| E2eError::ggml("Context::mul_mat(DOWN)", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("Context::new_graph", source))?;
    graph.build_forward_expand(&y);
    let _buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("Context::allocate_tensors", source))?;

    w_gate
        .write_data_backend(weights.gate_values())
        .map_err(|source| E2eError::ggml("Tensor::write_data_backend<W_GATE>", source))?;
    w_up.write_data_backend(weights.up_values())
        .map_err(|source| E2eError::ggml("Tensor::write_data_backend<W_UP>", source))?;
    w_down
        .write_data_backend(weights.down_values())
        .map_err(|source| E2eError::ggml("Tensor::write_data_backend<W_DOWN>", source))?;
    x_raw
        .write_data_backend(input)
        .map_err(|source| E2eError::ggml("Tensor::write_data_backend<X>", source))?;
    norm_w
        .write_data_backend(norm_weight)
        .map_err(|source| E2eError::ggml("write<norm_w>", source))?;

    backend
        .compute(&mut graph)
        .map_err(|source| E2eError::ggml("Backend::compute", source))?;

    y.read_data_backend()
        .map_err(|source| E2eError::ggml("Tensor::read_data_backend<Y>", source))
}

pub(super) fn recommended_mlp_backend_memory_bytes(
    hidden_features: usize,
    ffn_features: usize,
    sequence_length: usize,
) -> Result<Bytes, E2eError> {
    let gate_projection = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, ffn_features),
        Shape2D::new(hidden_features, sequence_length),
    )
    .map_err(|source| E2eError::ggml("Context::recommended_backend_matmul_memory(gate)", source))?;
    let up_projection = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, ffn_features),
        Shape2D::new(hidden_features, sequence_length),
    )
    .map_err(|source| E2eError::ggml("Context::recommended_backend_matmul_memory(up)", source))?;
    let down_projection = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(ffn_features, hidden_features),
        Shape2D::new(ffn_features, sequence_length),
    )
    .map_err(|source| E2eError::ggml("Context::recommended_backend_matmul_memory(down)", source))?;

    let total = gate_projection
        .get()
        .checked_add(up_projection.get())
        .and_then(|value| value.checked_add(down_projection.get()))
        .and_then(|value| value.checked_add(MLP_BACKEND_SLACK_BYTES))
        .ok_or(E2eError::MemorySizeOverflow)?;
    Ok(Bytes::new(total))
}
