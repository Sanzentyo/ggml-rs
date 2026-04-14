use super::error::{E2eError, GgmlResultExt};
use super::numeric::checked_mul;
use super::tensor_ops::{MATMUL_GRAPH_SLACK_BYTES, upload_weight};
use crate::inference::MlpWeights;
use ggml_rs::{Backend, BackendBuffer, Bytes, Context, Graph, Length, Shape2D, Tensor};

// ---------------------------------------------------------------------------
// Shared MLP graph topology builder
// ---------------------------------------------------------------------------

/// Tensor handles for a built MLP graph: norm → gate → up → silu → mul → down.
struct MlpGraphParts<'ctx> {
    w_gate: Tensor<'ctx, f32>,
    w_up: Tensor<'ctx, f32>,
    w_down: Tensor<'ctx, f32>,
    x_in: Tensor<'ctx, f32>,
    norm_w: Tensor<'ctx, f32>,
    y_out: Tensor<'ctx, f32>,
    graph: Graph<'ctx>,
}

/// Build the MLP compute graph topology in the given context.
///
/// Creates weight tensors, input tensor, and the operation chain:
/// `rms_norm(x) * norm_w → gate matmul → silu → mul(up matmul) → down matmul`.
///
/// Does **not** allocate backend memory or upload data — the caller manages
/// those steps (one-shot: allocate + upload all + compute + read; persistent:
/// allocate + upload weights + return struct).
fn build_mlp_graph<'ctx>(
    ctx: &'ctx Context,
    hidden_features: usize,
    ffn_features: usize,
    sequence_length: usize,
    rms_norm_eps: f32,
) -> Result<MlpGraphParts<'ctx>, E2eError> {
    let w_gate = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, ffn_features))
        .ggml_ctx("new<W_GATE>(mlp)")?;
    let w_up = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, ffn_features))
        .ggml_ctx("new<W_UP>(mlp)")?;
    let w_down = ctx
        .new_tensor_2d::<f32>(Shape2D::new(ffn_features, hidden_features))
        .ggml_ctx("new<W_DOWN>(mlp)")?;
    let x_in = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, sequence_length))
        .ggml_ctx("new<X>(mlp)")?;
    let norm_w = ctx
        .new_tensor_1d::<f32>(Length::new(hidden_features))
        .ggml_ctx("new<norm_w>(mlp)")?;

    let x_normed = ctx
        .rms_norm(&x_in, rms_norm_eps)
        .ggml_ctx("rms_norm(mlp)")?;
    let x = ctx.mul(&x_normed, &norm_w).ggml_ctx("mul(norm)(mlp)")?;

    let gate = ctx.mul_mat(&w_gate, &x).ggml_ctx("mul_mat(GATE)(mlp)")?;
    let up = ctx.mul_mat(&w_up, &x).ggml_ctx("mul_mat(UP)(mlp)")?;
    let activated = ctx.silu(&gate).ggml_ctx("silu(mlp)")?;
    let fused = ctx.mul(&activated, &up).ggml_ctx("mul(GATE*UP)(mlp)")?;
    let y_out = ctx
        .mul_mat(&w_down, &fused)
        .ggml_ctx("mul_mat(DOWN)(mlp)")?;

    let mut graph = ctx.new_graph().ggml_ctx("new_graph(mlp)")?;
    graph.build_forward_expand(&y_out);

    Ok(MlpGraphParts {
        w_gate,
        w_up,
        w_down,
        x_in,
        norm_w,
        y_out,
        graph,
    })
}

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
    let ctx = Context::new_no_alloc_bytes(ctx_size).ggml_ctx("Context::new_no_alloc_bytes")?;

    let MlpGraphParts {
        w_gate,
        w_up,
        w_down,
        x_in,
        norm_w,
        y_out,
        mut graph,
    } = build_mlp_graph(
        &ctx,
        hidden_features,
        ffn_features,
        sequence_length,
        rms_norm_eps,
    )?;

    let _buffer = ctx
        .allocate_tensors(backend)
        .ggml_ctx("allocate_tensors(mlp)")?;

    upload_weight(&w_gate, weights.gate_values(), "write<W_GATE>")?;
    upload_weight(&w_up, weights.up_values(), "write<W_UP>")?;
    upload_weight(&w_down, weights.down_values(), "write<W_DOWN>")?;
    upload_weight(&x_in, input, "write<X>")?;
    upload_weight(&norm_w, norm_weight, "write<norm_w>")?;

    backend.compute(&mut graph).ggml_ctx("Backend::compute")?;

    y_out.read_data_backend().ggml_ctx("read<Y>(mlp)")
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
    .ggml_ctx("Context::recommended_backend_matmul_memory(gate)")?;
    let up_projection = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, ffn_features),
        Shape2D::new(hidden_features, sequence_length),
    )
    .ggml_ctx("Context::recommended_backend_matmul_memory(up)")?;
    let down_projection = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(ffn_features, hidden_features),
        Shape2D::new(ffn_features, sequence_length),
    )
    .ggml_ctx("Context::recommended_backend_matmul_memory(down)")?;

    let total = gate_projection
        .get()
        .checked_add(up_projection.get())
        .and_then(|value| value.checked_add(down_projection.get()))
        .and_then(|value| value.checked_add(MATMUL_GRAPH_SLACK_BYTES))
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
        upload_weight(&self.x_in, hidden, "PersistentMlp::write(x_in)")?;
        backend
            .compute(&mut self.graph)
            .ggml_ctx("PersistentMlp::compute")?;
        self.y_out
            .read_data_backend()
            .ggml_ctx("PersistentMlp::read(y_out)")
    }
}

/// Build a persistent MLP graph for single-token decode (`seq_len=1`).
///
/// Creates a dedicated ggml context, builds the graph, uploads all weights
/// (gate, up, down, norm) once. Returns `(PersistentMlp<'static>, Context)`
/// where the context owns the tensors and must outlive the handle.
/// The handle is first in the tuple so that even if stored as a single
/// value, it drops before the context (Rust drops tuple fields in order).
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
) -> Result<(PersistentMlp<'static>, Context), E2eError> {
    let hidden_features = weights.hidden_features;
    let ffn_features = weights.ffn_features;

    let ctx_size = recommended_persistent_mlp_memory(hidden_features, ffn_features)?;
    let ctx = Context::new_no_alloc_bytes(ctx_size).ggml_ctx("Context(pmlp)")?;

    let parts = build_mlp_graph(&ctx, hidden_features, ffn_features, 1, rms_norm_eps)?;

    let buffer = ctx
        .allocate_tensors(backend)
        .ggml_ctx("allocate_tensors(pmlp)")?;

    // Upload weights once.
    upload_weight(&parts.w_gate, weights.gate_values(), "write<W_GATE>(pmlp)")?;
    upload_weight(&parts.w_up, weights.up_values(), "write<W_UP>(pmlp)")?;
    upload_weight(&parts.w_down, weights.down_values(), "write<W_DOWN>(pmlp)")?;
    upload_weight(&parts.norm_w, norm_weight, "write<NORM_W>(pmlp)")?;

    // SAFETY: see doc comment above.
    let mlp = unsafe {
        std::mem::transmute::<PersistentMlp<'_>, PersistentMlp<'static>>(PersistentMlp {
            x_in: parts.x_in,
            y_out: parts.y_out,
            graph: parts.graph,
            _buffer: buffer,
            hidden_features,
        })
    };
    Ok((mlp, ctx))
}

/// Recommended context size for a persistent MLP graph (seq_len=1).
pub(super) fn recommended_persistent_mlp_memory(
    hidden_features: usize,
    ffn_features: usize,
) -> Result<Bytes, E2eError> {
    recommended_mlp_backend_memory_bytes(hidden_features, ffn_features, 1)
}
