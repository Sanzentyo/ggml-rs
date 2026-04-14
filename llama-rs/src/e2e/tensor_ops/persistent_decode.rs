use super::super::attention::QkvProjections;
use super::super::error::E2eError;
use super::projection::{
    BuiltProjection, OutputProjectionGraph, ProjectionSpec, build_batch_projections,
    build_output_projection_graph, sum_matmul_memories,
};
use ggml_rs::{Backend, BackendBuffer, Bytes, Context, Graph, Shape2D, Tensor};

/// Raw linear-attention projection outputs read back from the persistent
/// projection graph.
///
/// Unlike [`super::super::linear_attention::LinearProjections`], this carries only
/// the four GPU-readback buffers and omits derived dimension fields
/// (`conv_channels`, `hidden_features`), which the caller must supply
/// from the layer plan.
#[derive(Debug)]
pub(in crate::e2e) struct RawLinearProjections {
    pub(in crate::e2e) qkv: Vec<f32>,
    pub(in crate::e2e) z: Vec<f32>,
    pub(in crate::e2e) alpha: Vec<f32>,
    pub(in crate::e2e) beta: Vec<f32>,
}

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
pub(in crate::e2e) enum PersistentDecodeProjection<'ctx> {
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

/// Estimate ggml context metadata bytes for a full attention persistent
/// projection (both input and output graphs in a single context).
pub(in crate::e2e) fn recommended_persistent_full_attention_memory(
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
pub(in crate::e2e) fn recommended_persistent_linear_attention_memory(
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

/// Built parts of a persistent full attention projection graph pair.
///
/// Returned by [`build_persistent_full_attention_graphs`].
pub(in crate::e2e) struct FullAttentionGraphParts<'ctx> {
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
pub(in crate::e2e) fn build_persistent_full_attention_graphs<'ctx>(
    ctx: &'ctx Context,
    hidden_features: usize,
    query_features_x2: usize,
    kv_features: usize,
    query_features: usize,
) -> Result<FullAttentionGraphParts<'ctx>, E2eError> {
    let x_in = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, 1))
        .map_err(|source| E2eError::ggml("new_tensor_2d<X_IN>(pfa)", source))?;

    let projs = build_batch_projections(
        ctx,
        &x_in,
        hidden_features,
        &[
            ProjectionSpec {
                weight_label: "new_tensor_2d<W_Q>(pfa)",
                matmul_label: "mul_mat<Q>(pfa)",
                out_features: query_features_x2,
            },
            ProjectionSpec {
                weight_label: "new_tensor_2d<W_K>(pfa)",
                matmul_label: "mul_mat<K>(pfa)",
                out_features: kv_features,
            },
            ProjectionSpec {
                weight_label: "new_tensor_2d<W_V>(pfa)",
                matmul_label: "mul_mat<V>(pfa)",
                out_features: kv_features,
            },
        ],
    )?;
    let [q, k, v]: [BuiltProjection<'_>; 3] = projs
        .try_into()
        .ok()
        .expect("internal spec mismatch: expected 3 projections");

    let mut input_graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(pfa_in)", source))?;
    input_graph.build_forward_expand(&q.y);
    input_graph.build_forward_expand(&k.y);
    input_graph.build_forward_expand(&v.y);

    // Output projection sub-graph
    let output =
        build_output_projection_graph(ctx, query_features, hidden_features, "output_proj(pfa)")?;

    Ok(FullAttentionGraphParts {
        x_in,
        w_q: q.w,
        w_k: k.w,
        w_v: v.w,
        q_out: q.y,
        k_out: k.y,
        v_out: v.y,
        input_graph,
        output,
    })
}

/// Built parts of a persistent linear attention projection graph pair.
///
/// Returned by [`build_persistent_linear_attention_graphs`].
pub(in crate::e2e) struct LinearAttentionGraphParts<'ctx> {
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
pub(in crate::e2e) fn build_persistent_linear_attention_graphs<'ctx>(
    ctx: &'ctx Context,
    hidden_features: usize,
    conv_channels: usize,
    inner_size: usize,
    time_step_rank: usize,
) -> Result<LinearAttentionGraphParts<'ctx>, E2eError> {
    let x_in = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, 1))
        .map_err(|source| E2eError::ggml("new_tensor_2d<X_IN>(pla)", source))?;

    let projs = build_batch_projections(
        ctx,
        &x_in,
        hidden_features,
        &[
            ProjectionSpec {
                weight_label: "new_tensor_2d<W_QKV>(pla)",
                matmul_label: "mul_mat<QKV>(pla)",
                out_features: conv_channels,
            },
            ProjectionSpec {
                weight_label: "new_tensor_2d<W_Z>(pla)",
                matmul_label: "mul_mat<Z>(pla)",
                out_features: inner_size,
            },
            ProjectionSpec {
                weight_label: "new_tensor_2d<W_ALPHA>(pla)",
                matmul_label: "mul_mat<ALPHA>(pla)",
                out_features: time_step_rank,
            },
            ProjectionSpec {
                weight_label: "new_tensor_2d<W_BETA>(pla)",
                matmul_label: "mul_mat<BETA>(pla)",
                out_features: time_step_rank,
            },
        ],
    )?;
    let [qkv, z, alpha, beta]: [BuiltProjection<'_>; 4] = projs
        .try_into()
        .ok()
        .expect("internal spec mismatch: expected 4 projections");

    let mut input_graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(pla_in)", source))?;
    input_graph.build_forward_expand(&qkv.y);
    input_graph.build_forward_expand(&z.y);
    input_graph.build_forward_expand(&alpha.y);
    input_graph.build_forward_expand(&beta.y);

    // Output projection sub-graph
    let output =
        build_output_projection_graph(ctx, inner_size, hidden_features, "output_proj(pla)")?;

    Ok(LinearAttentionGraphParts {
        x_in,
        w_qkv: qkv.w,
        w_z: z.w,
        w_alpha: alpha.w,
        w_beta: beta.w,
        qkv_out: qkv.y,
        z_out: z.y,
        alpha_out: alpha.y,
        beta_out: beta.y,
        input_graph,
        output,
    })
}

impl<'ctx> PersistentDecodeProjection<'ctx> {
    /// Run the input projection step: upload hidden state, compute, read outputs.
    pub(in crate::e2e) fn project_input(
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
    pub(in crate::e2e) fn read_full_attention_projections(
        &self,
    ) -> Result<QkvProjections, E2eError> {
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
    pub(in crate::e2e) fn read_linear_attention_projections(
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
    pub(in crate::e2e) fn project_output(
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
