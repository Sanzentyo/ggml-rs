//! Linear attention input projections: QKV, gate(Z), alpha, beta.
//!
//! Handles dimension validation, memory estimation, and both GPU-graph and
//! host-fallback projection paths.

use crate::e2e::error::{E2eError, GgmlResultExt};
use crate::e2e::numeric::checked_mul;
use crate::e2e::plan::Qwen35LinearAttentionLayerPlan;
use crate::e2e::tensor_ops::{
    PROJECTION_SLACK_BYTES, ProjectionSpec, execute_batch_projections, project_sequence,
};
use ggml_rs::{Backend, Bytes, Context, Shape2D};

/// Shared input projections for both core and decode paths.
pub(in crate::e2e) struct LinearProjections {
    pub(in crate::e2e) qkv: Vec<f32>,
    pub(in crate::e2e) z: Vec<f32>,
    pub(in crate::e2e) alpha: Vec<f32>,
    pub(in crate::e2e) beta: Vec<f32>,
    pub(in crate::e2e) conv_channels: usize,
    pub(in crate::e2e) hidden_features: usize,
}

/// Output of the fused projection + conv graph, which computes all four linear
/// projections AND the causal depthwise convolution + SiLU in a single ggml
/// compute graph, eliminating the host↔device round-trip between them.
pub(super) struct FusedLinearOutputs {
    /// Post-conv, post-SiLU activation `[seq_len × conv_channels]`.
    pub(super) conv: Vec<f32>,
    /// Pre-conv QKV tail `[conv_tail_rows × conv_channels]` — only the last
    /// `kernel_size - 1` rows needed by `capture_conv_buffer` for decode state
    /// continuity. `None` when no decode state is needed.
    pub(super) qkv_pre_conv_tail: Option<Vec<f32>>,
    pub(super) z: Vec<f32>,
    pub(super) alpha: Vec<f32>,
    pub(super) beta: Vec<f32>,
    pub(super) conv_channels: usize,
}

/// Validated dimension bundle for linear attention operations.
///
/// Consolidates dimension derivation (`hidden` from output weight matrix,
/// `conv_channels` from group/state dims) and provides memory estimation
/// for the fused projection + conv graph.
#[derive(Debug, Clone, Copy)]
pub(super) struct LinearAttentionDims {
    /// Hidden features (H) — derived from output weight matrix.
    pub(super) hidden: usize,
    /// Inner size (IS) — from plan.
    pub(super) inner_size: usize,
    /// Total conv channel width: IS + 2 * G * S.
    pub(super) conv_channels: usize,
    /// Timestep rank (R) — from plan.
    pub(super) time_step_rank: usize,
    /// Conv kernel size (K) — from plan.
    pub(super) kernel_size: usize,
}

impl LinearAttentionDims {
    /// Derive and validate all linear attention dimensions from a layer plan.
    pub(super) fn new(attention: &Qwen35LinearAttentionLayerPlan) -> Result<Self, E2eError> {
        let hidden = super::linear_attention_hidden_features(attention)?;
        let conv_channels = super::linear_attention_conv_channels(attention)?;
        Ok(Self {
            hidden,
            inner_size: attention.inner_size,
            conv_channels,
            time_step_rank: attention.time_step_rank,
            kernel_size: attention.conv_kernel,
        })
    }

    /// Conservative memory estimate for the fused projection + conv graph.
    pub(super) fn estimate_fused_memory(&self, seq_len: usize) -> Result<usize, E2eError> {
        let input_shape = Shape2D::new(self.hidden, seq_len);

        let proj_mem = [
            Context::recommended_backend_matmul_memory::<f32>(
                Shape2D::new(self.hidden, self.conv_channels),
                input_shape,
            )
            .ggml_ctx("matmul_mem(QKV)")?,
            Context::recommended_backend_matmul_memory::<f32>(
                Shape2D::new(self.hidden, self.inner_size),
                input_shape,
            )
            .ggml_ctx("matmul_mem(Z)")?,
            Context::recommended_backend_matmul_memory::<f32>(
                Shape2D::new(self.hidden, self.time_step_rank),
                input_shape,
            )
            .ggml_ctx("matmul_mem(alpha)")?,
            Context::recommended_backend_matmul_memory::<f32>(
                Shape2D::new(self.hidden, self.time_step_rank),
                input_shape,
            )
            .ggml_ctx("matmul_mem(beta)")?,
        ];
        let proj_total: usize = proj_mem.iter().map(|b| b.get()).sum();

        let pad = self.kernel_size.saturating_sub(1);
        let padded_len = pad + seq_len;
        let conv_tensor_bytes = std::mem::size_of::<f32>()
            * (seq_len * self.conv_channels
                + pad * self.conv_channels
                + padded_len * self.conv_channels
                + self.kernel_size * self.conv_channels
                + seq_len * self.conv_channels
                + seq_len * self.conv_channels
                + self.hidden
                + self.hidden * seq_len);

        proj_total
            .checked_add(conv_tensor_bytes)
            .and_then(|v| v.checked_add(PROJECTION_SLACK_BYTES * 2))
            .ok_or(E2eError::MemorySizeOverflow)
    }
}

/// Estimate the backend memory needed for 4 linear-attention input projections.
///
/// Sums estimates for QKV, gate(Z), alpha, and beta matmuls sharing the same
/// input tensor, plus slack for graph overhead.
fn recommended_linear_projection_memory(
    hidden_features: usize,
    conv_channels: usize,
    inner_size: usize,
    time_step_rank: usize,
    sequence_length: usize,
) -> Result<Bytes, E2eError> {
    let input_shape = Shape2D::new(hidden_features, sequence_length);
    let qkv_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, conv_channels),
        input_shape,
    )
    .ggml_ctx("recommended_backend_matmul_memory(QKV)")?;
    let z_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, inner_size),
        input_shape,
    )
    .ggml_ctx("recommended_backend_matmul_memory(Z)")?;
    let alpha_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, time_step_rank),
        input_shape,
    )
    .ggml_ctx("recommended_backend_matmul_memory(alpha)")?;
    let beta_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, time_step_rank),
        input_shape,
    )
    .ggml_ctx("recommended_backend_matmul_memory(beta)")?;
    let total = qkv_mem
        .get()
        .checked_add(z_mem.get())
        .and_then(|v| v.checked_add(alpha_mem.get()))
        .and_then(|v| v.checked_add(beta_mem.get()))
        .and_then(|v| v.checked_add(PROJECTION_SLACK_BYTES))
        .ok_or(E2eError::MemorySizeOverflow)?;
    Ok(Bytes::new(total))
}

/// Build the 4 linear-attention projection specs (QKV, Z, alpha, beta).
pub(super) fn linear_projection_specs(dims: &LinearAttentionDims) -> [ProjectionSpec; 4] {
    [
        ProjectionSpec {
            weight_label: "new<W_QKV>",
            matmul_label: "mul_mat(QKV)",
            out_features: dims.conv_channels,
        },
        ProjectionSpec {
            weight_label: "new<W_Z>",
            matmul_label: "mul_mat(Z)",
            out_features: dims.inner_size,
        },
        ProjectionSpec {
            weight_label: "new<W_alpha>",
            matmul_label: "mul_mat(alpha)",
            out_features: dims.time_step_rank,
        },
        ProjectionSpec {
            weight_label: "new<W_beta>",
            matmul_label: "mul_mat(beta)",
            out_features: dims.time_step_rank,
        },
    ]
}

/// Compute QKV, gate, alpha, and beta projections in a single ggml graph.
///
/// Batches four `mul_mat` operations sharing the same input tensor.
fn project_linear_inputs_graph(
    attention: &Qwen35LinearAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    hidden_features: usize,
    conv_channels: usize,
    backend: &Backend,
) -> Result<LinearProjections, E2eError> {
    let dims = LinearAttentionDims::new(attention)?;

    let ctx_size = recommended_linear_projection_memory(
        hidden_features,
        conv_channels,
        dims.inner_size,
        dims.time_step_rank,
        sequence_length,
    )?;

    let results = execute_batch_projections(
        ctx_size,
        hidden_features,
        sequence_length,
        &linear_projection_specs(&dims),
        input,
        &[
            (&attention.qkv_weight_values, "write<W_QKV>"),
            (&attention.gate_weight_values, "write<W_Z>"),
            (&attention.alpha_weight_values, "write<W_alpha>"),
            (&attention.beta_weight_values, "write<W_beta>"),
        ],
        backend,
    )?;

    let mut iter = results.into_iter();
    Ok(LinearProjections {
        qkv: iter.next().expect("4 specs → 4 results"),
        z: iter.next().expect("4 specs → 4 results"),
        alpha: iter.next().expect("4 specs → 4 results"),
        beta: iter.next().expect("4 specs → 4 results"),
        conv_channels,
        hidden_features,
    })
}

/// Project input through QKV, gate, alpha, and beta weights. Validates
/// input dimensions via checked arithmetic (catches malformed weights early).
///
/// When `backend` is `Some`, uses a single ggml compute graph for all four
/// projections. When `None`, falls back to host-side scalar dot products.
pub(in crate::e2e) fn project_linear_inputs(
    attention: &Qwen35LinearAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    backend: Option<&Backend>,
) -> Result<LinearProjections, E2eError> {
    let dims = LinearAttentionDims::new(attention)?;
    let hidden_features = dims.hidden;
    let conv_channels = dims.conv_channels;

    let expected_input_len = checked_mul(hidden_features, sequence_length)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }

    if let Some(backend) = backend {
        return project_linear_inputs_graph(
            attention,
            input,
            sequence_length,
            hidden_features,
            conv_channels,
            backend,
        );
    }

    let qkv = project_sequence(
        input,
        sequence_length,
        hidden_features,
        conv_channels,
        &attention.qkv_weight_values,
    )?;
    let z = project_sequence(
        input,
        sequence_length,
        hidden_features,
        attention.inner_size,
        &attention.gate_weight_values,
    )?;
    let alpha = project_sequence(
        input,
        sequence_length,
        hidden_features,
        attention.time_step_rank,
        &attention.alpha_weight_values,
    )?;
    let beta = project_sequence(
        input,
        sequence_length,
        hidden_features,
        attention.time_step_rank,
        &attention.beta_weight_values,
    )?;

    Ok(LinearProjections {
        qkv,
        z,
        alpha,
        beta,
        conv_channels,
        hidden_features,
    })
}
