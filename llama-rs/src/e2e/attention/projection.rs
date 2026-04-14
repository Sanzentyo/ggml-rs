//! QKV projection, deinterleaving, and preparation for full attention.

use super::shared::validate_gqa_heads;
use crate::e2e::error::{E2eError, GgmlResultExt};
use crate::e2e::numeric::checked_mul;
use crate::e2e::plan::Qwen35FullAttentionLayerPlan;
use crate::e2e::tensor_ops::{
    MATMUL_GRAPH_SLACK_BYTES, ProjectionSpec, execute_batch_projections, per_head_rms_norm,
    project_sequence,
};
use ggml_rs::{Backend, Bytes, Context, Shape2D, Tensor};

/// Base slack for the full-attention (Qwen3.5) memory estimate.
///
/// This accounts for ggml context overhead, intermediate tensor descriptors,
/// and alignment padding in the fused QKV + deinterleave + scoring graph.
const FULL_ATTENTION_BASE_SLACK_BYTES: usize = 1_048_576;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// Projected and normalized Q, K, V + gate vectors (pre-RoPE).
pub(in crate::e2e) struct PreparedAttention {
    pub(in crate::e2e) q_values: Vec<f32>,
    pub(in crate::e2e) k_values: Vec<f32>,
    pub(in crate::e2e) v_proj: Vec<f32>,
    pub(in crate::e2e) q_gate: Vec<f32>,
    pub(in crate::e2e) hidden_features: usize,
    pub(in crate::e2e) query_features: usize,
}

/// Host-side QKV projection outputs for full attention.
///
/// Used by both the fused graph builder (`project_qkv_graph`) and the
/// persistent projection reader (`read_full_attention_projections`).
#[derive(Debug)]
pub(in crate::e2e) struct QkvProjections {
    pub(in crate::e2e) q_full: Vec<f32>,
    pub(in crate::e2e) k_proj: Vec<f32>,
    pub(in crate::e2e) v_proj: Vec<f32>,
}

/// Validated derived dimensions for full (gated) attention.
///
/// Constructed once from `Qwen35FullAttentionLayerPlan`, validates all
/// dimension invariants upfront: GQA divisibility, hidden size consistency,
/// and weight buffer lengths.
#[derive(Debug, Clone, Copy)]
pub(super) struct FullAttentionDims {
    /// Per-head feature dimension (D).
    pub(super) d: usize,
    /// Number of query/gate heads (H).
    pub(super) h: usize,
    /// Number of KV heads (Hkv ≤ H, for GQA).
    pub(super) hkv: usize,
    /// Model hidden size (`H * D`, derived from output weight matrix).
    pub(super) hidden: usize,
    /// Total query features (`H * D`).
    pub(super) qf: usize,
    /// Total Q+gate interleaved features (`H * D * 2`).
    pub(super) qf2: usize,
    /// Total KV features per tensor (`Hkv * D`).
    pub(super) kvf: usize,
}

impl FullAttentionDims {
    pub(super) fn new(attention: &Qwen35FullAttentionLayerPlan) -> Result<Self, E2eError> {
        let d = attention.head_dimension;
        let h = attention.head_count;
        let hkv = attention.kv_head_count;

        let hidden = full_attention_hidden_features(attention)?;

        validate_gqa_heads(h, hkv)?;

        let qf = h * d;
        let qf2 = qf * 2;
        let kvf = hkv * d;

        Ok(Self {
            d,
            h,
            hkv,
            hidden,
            qf,
            qf2,
            kvf,
        })
    }

    /// Conservative memory estimate for the fully-fused attention graph.
    ///
    /// Returns `Err` if the estimate overflows `usize`.
    pub(super) fn estimate_memory(&self, t: usize) -> Result<Bytes, E2eError> {
        let Self {
            d,
            h: _,
            hkv: _,
            hidden,
            qf,
            qf2,
            kvf,
        } = *self;

        let weight_bytes = checked_mul(hidden, qf2)?
            .checked_add(checked_mul(hidden, checked_mul(kvf, 2)?)?)
            .and_then(|v| v.checked_add(checked_mul(qf, hidden).ok()?))
            .and_then(|v| v.checked_add(checked_mul(d, 2).ok()?))
            .and_then(|v| v.checked_add(hidden))
            .and_then(|v| v.checked_mul(4))
            .ok_or(E2eError::MemorySizeOverflow)?;

        let data_bytes = checked_mul(hidden, checked_mul(t, 2)?)?
            .checked_add(checked_mul(qf2, t)?)
            .and_then(|v| v.checked_add(checked_mul(kvf, checked_mul(t, 2).ok()?).ok()?))
            .and_then(|v| v.checked_add(checked_mul(qf, checked_mul(t, 4).ok()?).ok()?))
            .and_then(|v| v.checked_add(checked_mul(kvf, checked_mul(t, 2).ok()?).ok()?))
            .and_then(|v| v.checked_add(checked_mul(qf, checked_mul(t, 3).ok()?).ok()?))
            .and_then(|v| v.checked_add(checked_mul(t, t).ok()?))
            .and_then(|v| v.checked_add(checked_mul(hidden, t).ok()?))
            .and_then(|v| v.checked_mul(4))
            .ok_or(E2eError::MemorySizeOverflow)?;

        let mask_bytes_estimate = checked_mul(checked_mul(t, t)?, 2)?; // f16

        let total = weight_bytes
            .checked_add(data_bytes)
            .and_then(|v| v.checked_add(mask_bytes_estimate))
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_add(FULL_ATTENTION_BASE_SLACK_BYTES))
            .ok_or(E2eError::MemorySizeOverflow)?;

        Ok(Bytes::new(total))
    }
}

// ---------------------------------------------------------------------------
// QKV deinterleaving
// ---------------------------------------------------------------------------

/// De-interleave ggml's `[Q_h0, G_h0, Q_h1, G_h1, ...]` layout into
/// separate Q and gate buffers.
///
/// Both call sites (prefill multi-token and decode single-token) go through
/// this function so validation is unified.
pub(super) fn deinterleave_q_gate(
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

/// Graph-side deinterleave of the fused Q+gate projection output.
///
/// Given `q_full` shaped `[qf2, T]` where `qf2 = 2 × H × D` (interleaved
/// `[Q_h0(D), G_h0(D), Q_h1(D), G_h1(D), ...]` per token), builds strided
/// views that extract contiguous Q `[D, H, T]` and gate `[D, H, T]` tensors.
///
/// Both output tensors are made contiguous via `ctx.cont()` so downstream
/// operations (RoPE, RMS norm) see standard strides.
pub(super) fn graph_deinterleave_q_gate<'ctx>(
    ctx: &'ctx Context,
    q_full: &Tensor<'ctx, f32>,
    head_dimension: usize,
    head_count: usize,
    sequence_length: usize,
) -> Result<(Tensor<'ctx, f32>, Tensor<'ctx, f32>), E2eError> {
    let d = head_dimension;
    let h = head_count;
    let t = sequence_length;

    let q_full_3d = ctx
        .reshape_3d(q_full, d, 2 * h, t)
        .ggml_ctx("reshape_3d(Q_full)")?;

    let elem = std::mem::size_of::<f32>();
    let stride_h = 2 * d * elem; // skip one Q + one gate head
    let stride_t = 2 * h * d * elem; // full token

    // Q: even-indexed heads (offset 0)
    let q_view = ctx
        .view_3d(&q_full_3d, d, h, t, stride_h, stride_t, 0)
        .ggml_ctx("view_3d(Q)")?;
    let q_cont = ctx.cont(&q_view).ggml_ctx("cont(Q_deinterleave)")?;

    // Gate: odd-indexed heads (offset D × sizeof(f32))
    let gate_view = ctx
        .view_3d(&q_full_3d, d, h, t, stride_h, stride_t, d * elem)
        .ggml_ctx("view_3d(Gate)")?;
    let gate_cont = ctx.cont(&gate_view).ggml_ctx("cont(Gate_deinterleave)")?;

    Ok((q_cont, gate_cont))
}

// ---------------------------------------------------------------------------
// QKV projection functions
// ---------------------------------------------------------------------------

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
    .ggml_ctx("recommended_backend_matmul_memory(Q)")?;
    let k_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, kv_features),
        Shape2D::new(hidden_features, sequence_length),
    )
    .ggml_ctx("recommended_backend_matmul_memory(K)")?;
    let v_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, kv_features),
        Shape2D::new(hidden_features, sequence_length),
    )
    .ggml_ctx("recommended_backend_matmul_memory(V)")?;
    let total = q_mem
        .get()
        .checked_add(k_mem.get())
        .and_then(|v| v.checked_add(v_mem.get()))
        .and_then(|v| v.checked_add(MATMUL_GRAPH_SLACK_BYTES))
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
    attention: &Qwen35FullAttentionLayerPlan,
    backend: &Backend,
) -> Result<QkvProjections, E2eError> {
    let ctx_size = recommended_qkv_projection_memory(
        hidden_features,
        query_features_x2,
        kv_features,
        sequence_length,
    )?;

    let specs = [
        ProjectionSpec {
            weight_label: "new_tensor_2d<W_Q>",
            matmul_label: "mul_mat(Q)",
            out_features: query_features_x2,
        },
        ProjectionSpec {
            weight_label: "new_tensor_2d<W_K>",
            matmul_label: "mul_mat(K)",
            out_features: kv_features,
        },
        ProjectionSpec {
            weight_label: "new_tensor_2d<W_V>",
            matmul_label: "mul_mat(V)",
            out_features: kv_features,
        },
    ];

    let results = execute_batch_projections(
        ctx_size,
        hidden_features,
        sequence_length,
        &specs,
        input,
        &[
            (&attention.q_weight_values, "write<W_Q>"),
            (&attention.k_weight_values, "write<W_K>"),
            (&attention.v_weight_values, "write<W_V>"),
        ],
        backend,
    )?;

    let mut iter = results.into_iter();
    Ok(QkvProjections {
        q_full: iter.next().expect("3 specs → 3 results"),
        k_proj: iter.next().expect("3 specs → 3 results"),
        v_proj: iter.next().expect("3 specs → 3 results"),
    })
}

/// Derive `hidden_features` from the output weight matrix dimensions.
pub(in crate::e2e) fn full_attention_hidden_features(
    attention: &Qwen35FullAttentionLayerPlan,
) -> Result<usize, E2eError> {
    let query_features = attention
        .head_count
        .checked_mul(attention.head_dimension)
        .ok_or(E2eError::MemorySizeOverflow)?;
    attention
        .output_weight_values
        .len()
        .checked_div(query_features)
        .filter(|&h| h > 0 && h * query_features == attention.output_weight_values.len())
        .ok_or(E2eError::BufferLengthMismatch {
            expected: 1,
            actual: 0,
        })
}

/// Post-process raw QKV projection outputs: deinterleave Q/gate and apply
/// per-head RMS norm. Usable from both the one-shot and persistent projection
/// paths.
pub(in crate::e2e) fn prepare_qkv_from_raw(
    attention: &Qwen35FullAttentionLayerPlan,
    q_full: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    sequence_length: usize,
    hidden_features: usize,
    rms_norm_eps: f32,
) -> Result<PreparedAttention, E2eError> {
    let query_features = checked_mul(attention.head_count, attention.head_dimension)?;

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

/// Shared projection + deinterleave + per-head RMS norm for both core and
/// decode paths. The caller applies RoPE with its own position_offset.
///
/// When `backend` is `Some`, uses ggml compute graphs for projections
/// (prefill/inference path). When `None`, falls back to host-side scalar
/// dot products (decode path, where graph overhead exceeds benefit).
pub(super) fn project_and_prepare_qkv(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    backend: Option<&Backend>,
) -> Result<PreparedAttention, E2eError> {
    let hidden_features = full_attention_hidden_features(attention)?;
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

    let qkv = if let Some(backend) = backend {
        project_qkv_graph(
            input,
            sequence_length,
            hidden_features,
            query_features_x2,
            kv_features,
            attention,
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
        QkvProjections {
            q_full,
            k_proj,
            v_proj,
        }
    };

    prepare_qkv_from_raw(
        attention,
        qkv.q_full,
        qkv.k_proj,
        qkv.v_proj,
        sequence_length,
        hidden_features,
        rms_norm_eps,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal `Qwen35FullAttentionLayerPlan` stub for dimension tests.
    fn stub_full_plan(
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        hd: usize,
    ) -> Qwen35FullAttentionLayerPlan {
        let qf = heads * hd;
        let kvf = kv_heads * hd;
        Qwen35FullAttentionLayerPlan {
            norm_values: vec![1.0; hidden],
            q_norm_values: vec![1.0; hd],
            k_norm_values: vec![1.0; hd],
            q_weight_values: vec![0.0; hidden * qf * 2],
            k_weight_values: vec![0.0; hidden * kvf],
            v_weight_values: vec![0.0; hidden * kvf],
            output_weight_values: vec![0.0; qf * hidden],
            head_count: heads,
            kv_head_count: kv_heads,
            head_dimension: hd,
            attention_scale: 1.0 / (hd as f32).sqrt(),
            rope_n_dims: hd,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
        }
    }

    #[test]
    fn full_hidden_features_basic() {
        let plan = stub_full_plan(128, 4, 2, 32);
        assert_eq!(full_attention_hidden_features(&plan).unwrap(), 128);
    }

    #[test]
    fn full_hidden_features_qwen35_dimensions() {
        // Qwen3.5 0.6B: hidden=1536, heads=8, kv=4, hd=128
        let plan = stub_full_plan(1536, 8, 4, 128);
        assert_eq!(full_attention_hidden_features(&plan).unwrap(), 1536);
    }

    #[test]
    fn full_hidden_features_zero_heads() {
        // head_count=0 → query_features overflow/zero
        let mut plan = stub_full_plan(64, 2, 1, 32);
        plan.head_count = 0;
        assert!(full_attention_hidden_features(&plan).is_err());
    }

    #[test]
    fn full_hidden_features_empty_weights() {
        let mut plan = stub_full_plan(64, 2, 1, 32);
        plan.output_weight_values = vec![];
        assert!(full_attention_hidden_features(&plan).is_err());
    }

    #[test]
    fn full_attention_dims_delegates_to_hidden_features() {
        let plan = stub_full_plan(128, 4, 2, 32);
        let dims = FullAttentionDims::new(&plan).unwrap();
        assert_eq!(dims.hidden, 128);
        assert_eq!(dims.d, 32);
        assert_eq!(dims.h, 4);
        assert_eq!(dims.hkv, 2);
        assert_eq!(dims.qf, 128);
        assert_eq!(dims.qf2, 256);
        assert_eq!(dims.kvf, 64);
    }

    #[test]
    fn full_attention_dims_rejects_bad_gqa() {
        // kv_heads > heads → GQA validation fails
        let plan = stub_full_plan(128, 2, 4, 32);
        assert!(FullAttentionDims::new(&plan).is_err());
    }
}
