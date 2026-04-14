use super::super::error::E2eError;
use ggml_rs::{Backend, Bytes, Context, Graph, Shape2D, Tensor};

/// Slack constant added to memory estimates for ggml graph/tensor overhead.
pub(in crate::e2e) const PROJECTION_SLACK_BYTES: usize = 4 * 1024 * 1024;

/// Estimate the backend memory needed for a single matmul projection.
fn recommended_single_projection_memory(
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
pub(in crate::e2e) fn project_sequence_graph(
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

/// Upload a weight buffer to a backend tensor with a descriptive error label.
pub(in crate::e2e) fn upload_weight(
    tensor: &Tensor<'_, f32>,
    data: &[f32],
    label: &'static str,
) -> Result<(), E2eError> {
    tensor
        .write_data_backend(data)
        .map_err(|source| E2eError::ggml(label, source))
}

/// Specification for a single projection in a batch.
pub(in crate::e2e) struct ProjectionSpec {
    /// Error label for the weight tensor creation (`new_tensor_2d`).
    pub weight_label: &'static str,
    /// Error label for the matmul operation.
    pub matmul_label: &'static str,
    /// Output dimension of this projection (columns of the weight matrix).
    pub out_features: usize,
}

/// A projection built by [`build_batch_projections`]: weight tensor + output.
pub(in crate::e2e) struct BuiltProjection<'ctx> {
    /// Weight tensor `[input_features, out_features]` — upload once.
    pub w: Tensor<'ctx, f32>,
    /// Matmul output `[out_features, sequence_length]` — read after compute.
    pub y: Tensor<'ctx, f32>,
}

/// Build N parallel `mul_mat` projections from a shared input tensor.
///
/// Creates one weight tensor + one `mul_mat` output per spec. Does **not**
/// create a graph or call `build_forward_expand` — the caller manages graph
/// topology (some callers conditionally expand only a subset of outputs).
pub(in crate::e2e) fn build_batch_projections<'ctx>(
    ctx: &'ctx Context,
    x_in: &Tensor<'ctx, f32>,
    input_features: usize,
    specs: &[ProjectionSpec],
) -> Result<Vec<BuiltProjection<'ctx>>, E2eError> {
    let mut projs = Vec::with_capacity(specs.len());
    for spec in specs {
        let w = ctx
            .new_tensor_2d::<f32>(Shape2D::new(input_features, spec.out_features))
            .map_err(|source| E2eError::ggml(spec.weight_label, source))?;
        let y = ctx
            .mul_mat(&w, x_in)
            .map_err(|source| E2eError::ggml(spec.matmul_label, source))?;
        projs.push(BuiltProjection { w, y });
    }
    Ok(projs)
}

/// Shared output projection sub-graph returned by [`build_output_projection_graph`].
pub(in crate::e2e) struct OutputProjectionGraph<'ctx> {
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
pub(super) fn build_output_projection_graph<'ctx>(
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

/// Sum `recommended_backend_matmul_memory` for a batch of projections.
///
/// Each entry is `(weight_shape, input_shape, label)`.  The label is used
/// in the error message if the memory query fails.  Returns the total plus
/// `2 × PROJECTION_SLACK_BYTES` for ggml graph/tensor overhead.
pub(super) fn sum_matmul_memories(
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
