use super::error::E2eError;
use super::numeric::checked_mul;
use crate::inference::MlpWeights;
use ggml_rs::{Backend, BackendBuffer, Bytes, Context, Graph, Length, Shape2D, Tensor};

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

// ---------------------------------------------------------------------------
// Persistent MLP: build graph once, upload weights once, step per-token
// ---------------------------------------------------------------------------

/// Pre-built MLP compute graph with weights resident on the backend.
///
/// Built once per layer, the graph is fixed at `seq_len=1`. Per-step cost is
/// only the hidden vector I/O (~6 KB write + ~6 KB read at hidden=1536),
/// eliminating the ~165 MB weight upload per layer per token.
pub(super) struct PersistentMlp<'ctx> {
    x_in: Tensor<'ctx, f32>,
    y_out: Tensor<'ctx, f32>,
    graph: Graph<'ctx>,
    _buffer: BackendBuffer<'ctx>,
    hidden_features: usize,
}

impl<'ctx> PersistentMlp<'ctx> {
    /// Execute one decode step: write hidden input → compute → read output.
    pub(super) fn step(&mut self, hidden: &[f32], backend: &Backend) -> Result<Vec<f32>, E2eError> {
        if hidden.len() != self.hidden_features {
            return Err(E2eError::BufferLengthMismatch {
                expected: self.hidden_features,
                actual: hidden.len(),
            });
        }
        self.x_in
            .write_data_backend(hidden)
            .map_err(|source| E2eError::ggml("PersistentMlp::write(x_in)", source))?;
        backend
            .compute(&mut self.graph)
            .map_err(|source| E2eError::ggml("PersistentMlp::compute", source))?;
        self.y_out
            .read_data_backend()
            .map_err(|source| E2eError::ggml("PersistentMlp::read(y_out)", source))
    }
}

/// Build a persistent MLP graph for single-token decode (`seq_len=1`).
///
/// Creates a dedicated ggml context, builds the graph, uploads all weights
/// (gate, up, down, norm) once. Returns `(Context, PersistentMlp<'static>)`
/// where the context owns the tensors and must outlive the handle.
///
/// # Safety (transmute)
///
/// Same pattern as `build_one_persistent_full`: the returned `PersistentMlp`
/// carries `'static` via transmute. `Tensor`/`Graph`/`BackendBuffer` only hold
/// `PhantomData<&'ctx Context>` (raw pointers, not real references). Callers
/// must ensure LIFO drop order: handle drops before context.
pub(super) fn build_persistent_mlp(
    weights: &MlpWeights<f32>,
    norm_weight: &[f32],
    rms_norm_eps: f32,
    backend: &Backend,
) -> Result<(Context, PersistentMlp<'static>), E2eError> {
    let hidden_features = weights.hidden_features;
    let ffn_features = weights.ffn_features;

    let ctx_size = recommended_persistent_mlp_memory(hidden_features, ffn_features)?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| E2eError::ggml("Context(pmlp)", source))?;

    let w_gate = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, ffn_features))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_GATE>(pmlp)", source))?;
    let w_up = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, ffn_features))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_UP>(pmlp)", source))?;
    let w_down = ctx
        .new_tensor_2d::<f32>(Shape2D::new(ffn_features, hidden_features))
        .map_err(|source| E2eError::ggml("new_tensor_2d<W_DOWN>(pmlp)", source))?;
    let x_in = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, 1))
        .map_err(|source| E2eError::ggml("new_tensor_2d<X_IN>(pmlp)", source))?;
    let norm_w = ctx
        .new_tensor_1d::<f32>(Length::new(hidden_features))
        .map_err(|source| E2eError::ggml("new_tensor_1d<NORM_W>(pmlp)", source))?;

    // In-graph layer pre-norm: rms_norm then scale by weight.
    let x_normed = ctx
        .rms_norm(&x_in, rms_norm_eps)
        .map_err(|source| E2eError::ggml("rms_norm(pmlp)", source))?;
    let x = ctx
        .mul(&x_normed, &norm_w)
        .map_err(|source| E2eError::ggml("mul(norm)(pmlp)", source))?;

    let gate = ctx
        .mul_mat(&w_gate, &x)
        .map_err(|source| E2eError::ggml("mul_mat(GATE)(pmlp)", source))?;
    let up = ctx
        .mul_mat(&w_up, &x)
        .map_err(|source| E2eError::ggml("mul_mat(UP)(pmlp)", source))?;
    let activated = ctx
        .silu(&gate)
        .map_err(|source| E2eError::ggml("silu(pmlp)", source))?;
    let fused = ctx
        .mul(&activated, &up)
        .map_err(|source| E2eError::ggml("mul(GATE*UP)(pmlp)", source))?;
    let y_out = ctx
        .mul_mat(&w_down, &fused)
        .map_err(|source| E2eError::ggml("mul_mat(DOWN)(pmlp)", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(pmlp)", source))?;
    graph.build_forward_expand(&y_out);

    let buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("allocate_tensors(pmlp)", source))?;

    // Upload weights once.
    w_gate
        .write_data_backend(weights.gate_values())
        .map_err(|source| E2eError::ggml("write<W_GATE>(pmlp)", source))?;
    w_up.write_data_backend(weights.up_values())
        .map_err(|source| E2eError::ggml("write<W_UP>(pmlp)", source))?;
    w_down
        .write_data_backend(weights.down_values())
        .map_err(|source| E2eError::ggml("write<W_DOWN>(pmlp)", source))?;
    norm_w
        .write_data_backend(norm_weight)
        .map_err(|source| E2eError::ggml("write<NORM_W>(pmlp)", source))?;

    // SAFETY: see doc comment above.
    let mlp = unsafe {
        std::mem::transmute::<PersistentMlp<'_>, PersistentMlp<'static>>(PersistentMlp {
            x_in,
            y_out,
            graph,
            _buffer: buffer,
            hidden_features,
        })
    };
    Ok((ctx, mlp))
}

/// Recommended context size for a persistent MLP graph (seq_len=1).
pub(super) fn recommended_persistent_mlp_memory(
    hidden_features: usize,
    ffn_features: usize,
) -> Result<Bytes, E2eError> {
    recommended_mlp_backend_memory_bytes(hidden_features, ffn_features, 1)
}
