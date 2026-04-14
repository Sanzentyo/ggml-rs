//! Qwen3.5 linear attention with causal depthwise convolution and delta-net
//! recurrence.
//!
//! Provides full-sequence inference, prefill (capturing conv buffer and SSM
//! states), and single-token decode step.

use super::error::E2eError;
use super::numeric::{checked_mul, sigmoid_scalar, silu_scalar, softplus_scalar};
use super::plan::Qwen35LinearAttentionLayerPlan;
use super::state::LinearAttentionState;
use super::tensor_ops::upload_weight;
use super::tensor_ops::{
    PROJECTION_SLACK_BYTES, ProjectionSpec, build_batch_projections, head_slice, head_slice_mut,
    per_head_l2_norm, project_sequence, project_sequence_graph, rms_norm_single,
    rms_norm_single_into,
};
use ggml_rs::{Backend, Bytes, Context, Length, Shape2D};

pub(super) fn qwen35_linear_attention_inference(
    attention: &Qwen35LinearAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    qwen35_linear_attention_core(
        attention,
        input,
        sequence_length,
        rms_norm_eps,
        attn_norm_weight,
        None,
        backend,
    )
}

/// Prefill variant: runs the full sequence AND captures conv buffer + SSM states.
pub(super) fn qwen35_linear_attention_prefill(
    attention: &Qwen35LinearAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    state: &mut LinearAttentionState,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    qwen35_linear_attention_core(
        attention,
        input,
        sequence_length,
        rms_norm_eps,
        attn_norm_weight,
        Some(state),
        backend,
    )
}

/// Shared input projections for both core and decode paths.
pub(super) struct LinearProjections {
    pub(super) qkv: Vec<f32>,
    pub(super) z: Vec<f32>,
    pub(super) alpha: Vec<f32>,
    pub(super) beta: Vec<f32>,
    pub(super) conv_channels: usize,
    pub(super) hidden_features: usize,
}

/// Output of the fused projection + conv graph, which computes all four linear
/// projections AND the causal depthwise convolution + SiLU in a single ggml
/// compute graph, eliminating the host↔device round-trip between them.
struct FusedLinearOutputs {
    /// Post-conv, post-SiLU activation `[seq_len × conv_channels]`.
    conv: Vec<f32>,
    /// Pre-conv QKV tail `[conv_tail_rows × conv_channels]` — only the last
    /// `kernel_size - 1` rows needed by `capture_conv_buffer` for decode state
    /// continuity. `None` when no decode state is needed.
    qkv_pre_conv_tail: Option<Vec<f32>>,
    z: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    conv_channels: usize,
}

/// Validated dimension bundle for linear attention operations.
///
/// Consolidates dimension derivation (`hidden` from output weight matrix,
/// `conv_channels` from group/state dims) and provides memory estimation
/// for the fused projection + conv graph.
#[derive(Debug, Clone, Copy)]
struct LinearAttentionDims {
    /// Hidden features (H) — derived from output weight matrix.
    hidden: usize,
    /// Inner size (IS) — from plan.
    inner_size: usize,
    /// Total conv channel width: IS + 2 * G * S.
    conv_channels: usize,
    /// Timestep rank (R) — from plan.
    time_step_rank: usize,
    /// Conv kernel size (K) — from plan.
    kernel_size: usize,
}

impl LinearAttentionDims {
    /// Derive and validate all linear attention dimensions from a layer plan.
    fn new(attention: &Qwen35LinearAttentionLayerPlan) -> Result<Self, E2eError> {
        let hidden = linear_attention_hidden_features(attention)?;
        let conv_channels = linear_attention_conv_channels(attention)?;
        Ok(Self {
            hidden,
            inner_size: attention.inner_size,
            conv_channels,
            time_step_rank: attention.time_step_rank,
            kernel_size: attention.conv_kernel,
        })
    }

    /// Conservative memory estimate for the fused projection + conv graph.
    fn estimate_fused_memory(&self, seq_len: usize) -> Result<usize, E2eError> {
        let input_shape = Shape2D::new(self.hidden, seq_len);

        let proj_mem = [
            Context::recommended_backend_matmul_memory::<f32>(
                Shape2D::new(self.hidden, self.conv_channels),
                input_shape,
            )
            .map_err(|source| E2eError::ggml("matmul_mem(QKV)", source))?,
            Context::recommended_backend_matmul_memory::<f32>(
                Shape2D::new(self.hidden, self.inner_size),
                input_shape,
            )
            .map_err(|source| E2eError::ggml("matmul_mem(Z)", source))?,
            Context::recommended_backend_matmul_memory::<f32>(
                Shape2D::new(self.hidden, self.time_step_rank),
                input_shape,
            )
            .map_err(|source| E2eError::ggml("matmul_mem(alpha)", source))?,
            Context::recommended_backend_matmul_memory::<f32>(
                Shape2D::new(self.hidden, self.time_step_rank),
                input_shape,
            )
            .map_err(|source| E2eError::ggml("matmul_mem(beta)", source))?,
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
    .map_err(|source| E2eError::ggml("recommended_backend_matmul_memory(QKV)", source))?;
    let z_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, inner_size),
        input_shape,
    )
    .map_err(|source| E2eError::ggml("recommended_backend_matmul_memory(Z)", source))?;
    let alpha_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, time_step_rank),
        input_shape,
    )
    .map_err(|source| E2eError::ggml("recommended_backend_matmul_memory(alpha)", source))?;
    let beta_mem = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, time_step_rank),
        input_shape,
    )
    .map_err(|source| E2eError::ggml("recommended_backend_matmul_memory(beta)", source))?;
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
fn linear_projection_specs(dims: &LinearAttentionDims) -> [ProjectionSpec; 4] {
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
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| E2eError::ggml("Context::new_no_alloc_bytes(linear_proj)", source))?;

    let x = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, sequence_length))
        .map_err(|source| E2eError::ggml("new<X>", source))?;

    let projs =
        build_batch_projections(&ctx, &x, hidden_features, &linear_projection_specs(&dims))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(linear_proj)", source))?;
    for p in &projs {
        graph.build_forward_expand(&p.y);
    }

    let _buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("allocate_tensors(linear_proj)", source))?;

    upload_weight(&projs[0].w, &attention.qkv_weight_values, "write<W_QKV>")?;
    upload_weight(&projs[1].w, &attention.gate_weight_values, "write<W_Z>")?;
    upload_weight(
        &projs[2].w,
        &attention.alpha_weight_values,
        "write<W_alpha>",
    )?;
    upload_weight(&projs[3].w, &attention.beta_weight_values, "write<W_beta>")?;
    upload_weight(&x, input, "write<X>")?;

    backend
        .compute(&mut graph)
        .map_err(|source| E2eError::ggml("compute(linear_proj)", source))?;

    let qkv = projs[0]
        .y
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read<QKV>", source))?;
    let z = projs[1]
        .y
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read<Z>", source))?;
    let alpha = projs[2]
        .y
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read<alpha>", source))?;
    let beta = projs[3]
        .y
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read<beta>", source))?;

    Ok(LinearProjections {
        qkv,
        z,
        alpha,
        beta,
        conv_channels,
        hidden_features,
    })
}

/// Derive `hidden_features` from the output weight matrix dimensions.
pub(super) fn linear_attention_hidden_features(
    attention: &Qwen35LinearAttentionLayerPlan,
) -> Result<usize, E2eError> {
    let is = attention.inner_size;
    if is == 0 {
        return Err(E2eError::BufferLengthMismatch {
            expected: 1,
            actual: 0,
        });
    }
    let total = attention.ssm_out_weight_values.len();
    let h = total / is;
    if h > 0 && h * is == total {
        Ok(h)
    } else {
        Err(E2eError::BufferLengthMismatch {
            expected: 1,
            actual: 0,
        })
    }
}

/// Compute `conv_channels` from plan dimensions.
pub(super) fn linear_attention_conv_channels(
    attention: &Qwen35LinearAttentionLayerPlan,
) -> Result<usize, E2eError> {
    Ok(attention.inner_size
        + checked_mul(checked_mul(attention.group_count, attention.state_size)?, 2)?)
}

/// Core linear attention decode logic: conv → split/norm → SSM recurrence → z-gating.
///
/// Takes raw projections and returns the SSM output (before output projection).
/// The caller is responsible for projecting the output.
pub(super) fn linear_attention_decode_core(
    projections: LinearProjections,
    attention: &Qwen35LinearAttentionLayerPlan,
    rms_norm_eps: f32,
    state: &mut LinearAttentionState,
    decode_scratch: Option<&mut LinearDecodeScratch>,
) -> Result<Vec<f32>, E2eError> {
    let LinearProjections {
        qkv,
        z,
        alpha,
        beta,
        conv_channels,
        hidden_features: _,
    } = projections;

    if attention.group_count == 0 || attention.state_size == 0 || attention.time_step_rank == 0 {
        return Err(E2eError::BufferLengthMismatch {
            expected: 1,
            actual: 0,
        });
    }

    let conv = causal_depthwise_conv_decode_step(&qkv, state, &attention.conv_weight_values)?;

    let qk_features = checked_mul(attention.group_count, attention.state_size)?;
    let (q_heads, k_heads) = split_and_norm_qk(&conv, 1, attention, conv_channels, rms_norm_eps)?;

    let v_raw = &conv[qk_features * 2..conv_channels];

    let scale = 1.0_f32 / (attention.state_size as f32).sqrt();

    // Use pre-allocated scratch when available, falling back to fresh allocation.
    let mut owned_scratch;
    let scratch = match decode_scratch {
        Some(s) => {
            s.output.fill(0.0);
            s
        }
        None => {
            owned_scratch = LinearDecodeScratch::new(attention.state_size, attention.inner_size);
            &mut owned_scratch
        }
    };

    for head in 0..attention.time_step_rank {
        let src_group = head % attention.group_count;
        let q = head_slice(
            &q_heads,
            0,
            src_group,
            attention.group_count,
            attention.state_size,
        );
        let k = head_slice(
            &k_heads,
            0,
            src_group,
            attention.group_count,
            attention.state_size,
        );
        let v = head_slice(
            v_raw,
            0,
            head,
            attention.time_step_rank,
            attention.state_size,
        );
        let z_head = head_slice(&z, 0, head, attention.time_step_rank, attention.state_size);

        let gate = softplus_scalar(alpha[head] + attention.dt_bias_values[head])
            * attention.ssm_a_values[head];
        let beta_value = sigmoid_scalar(beta[head]);

        let state_size_sq = checked_mul(attention.state_size, attention.state_size)?;
        let state_offset = checked_mul(head, state_size_sq)?;
        let ssm_state = &mut state.ssm_states[state_offset..state_offset + state_size_sq];

        ssm_recurrence_step(
            ssm_state,
            q,
            k,
            v,
            z_head,
            gate.exp(),
            beta_value,
            attention.state_size,
            scale,
            &mut scratch.ssm,
        );
        rms_norm_single_into(
            &scratch.ssm.out,
            &attention.ssm_norm_values,
            rms_norm_eps,
            &mut scratch.norm_buf,
        )?;
        let dst = head_slice_mut(
            &mut scratch.output,
            0,
            head,
            attention.time_step_rank,
            attention.state_size,
        );
        dst.iter_mut()
            .zip(scratch.norm_buf.iter().zip(z_head.iter()))
            .for_each(|(d, (&n, &z))| *d = n * silu_scalar(z));
    }

    Ok(scratch.output.clone())
}

/// Project input through QKV, gate, alpha, and beta weights. Validates
/// input dimensions via checked arithmetic (catches malformed weights early).
///
/// When `backend` is `Some`, uses a single ggml compute graph for all four
/// projections. When `None`, falls back to host-side scalar dot products.
fn project_linear_inputs(
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

/// Split conv output into Q, K, V regions and L2-normalize Q and K heads.
///
/// Returns `(q_heads, k_heads)` — the caller retains ownership of `conv` for
/// V slicing (avoids an extra copy).
fn split_and_norm_qk(
    conv: &[f32],
    sequence_length: usize,
    attention: &Qwen35LinearAttentionLayerPlan,
    conv_channels: usize,
    rms_norm_eps: f32,
) -> Result<(Vec<f32>, Vec<f32>), E2eError> {
    let qk_features = checked_mul(attention.group_count, attention.state_size)?;
    let mut q_raw = vec![0.0_f32; checked_mul(sequence_length, qk_features)?];
    let mut k_raw = vec![0.0_f32; q_raw.len()];
    debug_assert_eq!(conv.len(), checked_mul(sequence_length, conv_channels)?);
    for ((conv_row, q_dst), k_dst) in conv
        .chunks_exact(conv_channels)
        .zip(q_raw.chunks_exact_mut(qk_features))
        .zip(k_raw.chunks_exact_mut(qk_features))
    {
        q_dst.copy_from_slice(&conv_row[..qk_features]);
        k_dst.copy_from_slice(&conv_row[qk_features..qk_features * 2]);
    }

    let q_heads = per_head_l2_norm(
        &q_raw,
        sequence_length,
        attention.group_count,
        attention.state_size,
        rms_norm_eps,
    )?;
    let k_heads = per_head_l2_norm(
        &k_raw,
        sequence_length,
        attention.group_count,
        attention.state_size,
        rms_norm_eps,
    )?;

    Ok((q_heads, k_heads))
}

fn qwen35_linear_attention_core(
    attention: &Qwen35LinearAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    mut state: Option<&mut LinearAttentionState>,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    let dims = LinearAttentionDims::new(attention)?;
    let hidden_features = dims.hidden;

    let expected_input_len = checked_mul(hidden_features, sequence_length)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }

    // Fused projection + conv: single graph, no host round-trip.
    // Only read back the tail rows of qkv_pre_conv needed for conv buffer seeding.
    let conv_tail_rows = if state.is_some() {
        dims.kernel_size.saturating_sub(1).min(sequence_length)
    } else {
        0
    };
    let FusedLinearOutputs {
        conv,
        qkv_pre_conv_tail,
        z,
        alpha,
        beta,
        conv_channels,
    } = project_and_conv_fused_graph(
        attention,
        &dims,
        input,
        sequence_length,
        attn_norm_weight,
        rms_norm_eps,
        conv_tail_rows,
        backend,
    )?;

    // Capture pre-conv QKV tail into conv buffer for future decode steps.
    if let Some(ref mut state) = state {
        if let Some(ref tail) = qkv_pre_conv_tail {
            state.capture_conv_buffer(tail, conv_tail_rows)?;
        } else {
            // kernel_size == 1: no conv history needed, but reset valid count.
            state.conv_valid = 0;
        }
    }

    let qk_features = checked_mul(attention.group_count, attention.state_size)?;
    let (q_heads, k_heads) = split_and_norm_qk(
        &conv,
        sequence_length,
        attention,
        conv_channels,
        rms_norm_eps,
    )?;

    // Extract V from conv output (the region after Q and K).
    let mut v_heads = vec![0.0_f32; checked_mul(sequence_length, attention.inner_size)?];
    for (conv_row, v_dst) in conv
        .chunks_exact(conv_channels)
        .zip(v_heads.chunks_exact_mut(attention.inner_size))
    {
        v_dst.copy_from_slice(&conv_row[qk_features * 2..]);
    }

    if !attention
        .time_step_rank
        .is_multiple_of(attention.group_count)
    {
        return Err(E2eError::BufferLengthMismatch {
            expected: attention.group_count,
            actual: attention.time_step_rank,
        });
    }
    if attention.inner_size != checked_mul(attention.time_step_rank, attention.state_size)? {
        return Err(E2eError::BufferLengthMismatch {
            expected: checked_mul(attention.time_step_rank, attention.state_size)?,
            actual: attention.inner_size,
        });
    }
    let mut output = vec![0.0_f32; checked_mul(sequence_length, attention.inner_size)?];
    let state_size = attention.state_size;
    let time_step_rank = attention.time_step_rank;
    let group_count = attention.group_count;
    let state_size_sq = checked_mul(state_size, state_size)?;
    let states_len = checked_mul(time_step_rank, state_size_sq)?;
    // When persistent state exists, write recurrence directly into it to avoid
    // an extra allocation + memcpy. Otherwise use a temporary buffer.
    let mut owned_states;
    let states: &mut [f32] = match state {
        Some(ref mut s) => {
            debug_assert_eq!(
                s.ssm_states.len(),
                states_len,
                "SSM state size mismatch: expected {states_len}, got {}",
                s.ssm_states.len()
            );
            s.ssm_states[..states_len].fill(0.0);
            &mut s.ssm_states[..states_len]
        }
        None => {
            owned_states = vec![0.0_f32; states_len];
            &mut owned_states
        }
    };
    let scale = 1.0_f32 / (state_size as f32).sqrt();
    let mut scratch = SsmScratch::new(state_size);

    for token in 0..sequence_length {
        let token_rank_base = token * time_step_rank;
        for (head, state_chunk) in states.chunks_exact_mut(state_size_sq).enumerate() {
            let src_group = head % group_count;
            let q = head_slice(&q_heads, token, src_group, group_count, state_size);
            let k = head_slice(&k_heads, token, src_group, group_count, state_size);
            let v = head_slice(&v_heads, token, head, time_step_rank, state_size);
            let z_head = head_slice(&z, token, head, time_step_rank, state_size);
            let gate =
                softplus_scalar(alpha[token_rank_base + head] + attention.dt_bias_values[head])
                    * attention.ssm_a_values[head];
            let beta_value = sigmoid_scalar(beta[token_rank_base + head]);
            ssm_recurrence_step(
                state_chunk,
                q,
                k,
                v,
                z_head,
                gate.exp(),
                beta_value,
                state_size,
                scale,
                &mut scratch,
            );
            let normalized =
                rms_norm_single(&scratch.out, &attention.ssm_norm_values, rms_norm_eps)?;
            let dst = head_slice_mut(&mut output, token, head, time_step_rank, state_size);
            dst.iter_mut()
                .zip(normalized.iter().zip(z_head.iter()))
                .for_each(|(d, (&n, &z))| *d = n * silu_scalar(z));
        }
    }

    // SSM states are already in persistent state (written directly) or in a
    // temporary buffer (discarded). No capture step needed.

    project_sequence_graph(
        &output,
        sequence_length,
        attention.inner_size,
        hidden_features,
        &attention.ssm_out_weight_values,
        backend,
    )
}

/// Reusable scratch buffers for SSM recurrence, avoiding per-head allocation.
struct SsmScratch {
    sk: Vec<f32>,
    delta: Vec<f32>,
    out: Vec<f32>,
}

impl SsmScratch {
    fn new(state_size: usize) -> Self {
        Self {
            sk: vec![0.0_f32; state_size],
            delta: vec![0.0_f32; state_size],
            out: vec![0.0_f32; state_size],
        }
    }

    fn clear(&mut self) {
        self.sk.fill(0.0);
        self.delta.fill(0.0);
        self.out.fill(0.0);
    }
}

/// Reusable scratch buffers for [`linear_attention_decode_core`], avoiding
/// per-call and per-head heap allocations during autoregressive decode.
///
/// Created once per decode session and passed into every decode step.
/// Bundles the SSM recurrence scratch, the per-call output buffer, and a
/// temporary buffer for RMS normalization.
pub(super) struct LinearDecodeScratch {
    ssm: SsmScratch,
    /// Per-call output accumulator, sized to `inner_size`.
    output: Vec<f32>,
    /// Temporary for RMS norm result, sized to `state_size`.
    norm_buf: Vec<f32>,
}

impl LinearDecodeScratch {
    /// Create scratch buffers for a linear attention layer with the given
    /// `state_size` and `inner_size`.
    pub(super) fn new(state_size: usize, inner_size: usize) -> Self {
        Self {
            ssm: SsmScratch::new(state_size),
            output: vec![0.0_f32; inner_size],
            norm_buf: vec![0.0_f32; state_size],
        }
    }
}

/// Single SSM recurrence step: decay → sk → delta → update → read.
///
/// Operates on one head's `state_size × state_size` state matrix. Results are
/// written into `scratch.out`; the caller is responsible for post-processing
/// (RMS norm, SiLU gating, etc.).
///
/// The loop order is preserved exactly to match the original implementation and
/// maintain floating-point parity.
#[allow(clippy::too_many_arguments)]
fn ssm_recurrence_step(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    _z: &[f32],
    decay: f32,
    beta_value: f32,
    state_size: usize,
    scale: f32,
    scratch: &mut SsmScratch,
) {
    debug_assert_eq!(state.len(), state_size * state_size);
    debug_assert_eq!(q.len(), state_size);
    debug_assert_eq!(k.len(), state_size);
    debug_assert_eq!(v.len(), state_size);

    scratch.clear();

    // Decay state and accumulate sk (row-major order preserved).
    for (row, row_slice) in state.chunks_exact_mut(state_size).enumerate() {
        let k_row = k[row];
        for (col, s) in row_slice.iter_mut().enumerate() {
            *s *= decay;
            scratch.sk[col] += *s * k_row;
        }
    }

    // Delta = (v - sk) * beta
    scratch
        .delta
        .iter_mut()
        .zip(v.iter().zip(scratch.sk.iter()))
        .for_each(|(d, (&vi, &ski))| *d = (vi - ski) * beta_value);

    // State update: state[row][col] += k[row] * delta[col]
    for (row_slice, &k_row) in state.chunks_exact_mut(state_size).zip(k.iter()) {
        row_slice
            .iter_mut()
            .zip(scratch.delta.iter())
            .for_each(|(s, &d)| *s += k_row * d);
    }

    // Read output: row-major traversal for contiguous access + auto-vectorization.
    for row in 0..state_size {
        let qr_scaled = q[row] * scale;
        let row_slice = &state[row * state_size..(row + 1) * state_size];
        scratch
            .out
            .iter_mut()
            .zip(row_slice.iter())
            .for_each(|(o, &s)| *o += s * qr_scaled);
    }
}

#[cfg(test)]
pub(super) fn causal_depthwise_conv(
    input: &[f32],
    sequence_length: usize,
    channels: usize,
    kernel_size: usize,
    weight: &[f32],
) -> Result<Vec<f32>, E2eError> {
    let expected_input_len = checked_mul(sequence_length, channels)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }
    let expected_weight_len = checked_mul(kernel_size, channels)?;
    if weight.len() != expected_weight_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_weight_len,
            actual: weight.len(),
        });
    }
    let mut output = vec![0.0_f32; input.len()];
    for token in 0..sequence_length {
        let start_tap = kernel_size.saturating_sub(token + 1);
        for channel in 0..channels {
            let weight_base = channel * kernel_size;
            let mut sum = 0.0_f32;
            for tap in start_tap..kernel_size {
                let src_token = token + tap + 1 - kernel_size;
                sum += input[src_token * channels + channel] * weight[weight_base + tap];
            }
            output[token * channels + channel] = silu_scalar(sum);
        }
    }
    Ok(output)
}

/// Graph-accelerated causal depthwise convolution via `ggml_ssm_conv` + SiLU.
///
/// Standalone version used by tests and benchmarks for parity checking.
/// Production code uses the fused `project_and_conv_fused_graph` which
/// combines projection + conv.
#[cfg(test)]
pub(super) fn causal_depthwise_conv_graph(
    input: &[f32],
    sequence_length: usize,
    channels: usize,
    kernel_size: usize,
    weight: &[f32],
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    if kernel_size == 0 {
        return Err(E2eError::BufferLengthMismatch {
            expected: 1,
            actual: 0,
        });
    }
    let expected_input_len = checked_mul(sequence_length, channels)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }
    let expected_weight_len = checked_mul(kernel_size, channels)?;
    if weight.len() != expected_weight_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_weight_len,
            actual: weight.len(),
        });
    }

    let pad = kernel_size - 1;
    let padded_len = pad + sequence_length;

    // Transpose input [seq_len × channels] (channels-fast) →
    // [channels][padded_len] (time-fast) with kernel_size-1 zero-padding on
    // the left of each channel row.
    let mut sx_data = vec![0.0_f32; padded_len * channels];
    for token in 0..sequence_length {
        for ch in 0..channels {
            sx_data[ch * padded_len + pad + token] = input[token * channels + ch];
        }
    }

    // Context metadata: 3 tensors (sx, c, conv_out) + graph + silu output.
    // ggml_ssm_conv is a direct loop — no large intermediates.
    let tensor_overhead = 3
        * std::mem::size_of::<f32>()
        * (padded_len * channels + kernel_size * channels + sequence_length * channels);
    let ctx_size = Bytes::new(tensor_overhead + PROJECTION_SLACK_BYTES);

    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| E2eError::ggml("Context::new_no_alloc_bytes(conv)", source))?;

    // sx: [padded_len, channels, 1] — ggml ne[0]=padded_len, ne[1]=channels, ne[2]=1
    let sx = ctx
        .new_tensor_3d::<f32>(ggml_rs::Shape3D::new(padded_len, channels, 1))
        .map_err(|source| E2eError::ggml("new_tensor_3d<sx>", source))?;

    // c: [kernel_size, channels] — ggml ne[0]=kernel_size, ne[1]=channels
    let c = ctx
        .new_tensor_2d::<f32>(Shape2D::new(kernel_size, channels))
        .map_err(|source| E2eError::ggml("new_tensor_2d<c>", source))?;

    // ssm_conv(sx, c) → [channels, seq_len, 1]
    let conv_out = ctx
        .ssm_conv(&sx, &c)
        .map_err(|source| E2eError::ggml("ssm_conv", source))?;

    // SiLU activation
    let result = ctx
        .silu(&conv_out)
        .map_err(|source| E2eError::ggml("silu(conv)", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(conv)", source))?;
    graph.build_forward_expand(&result);

    let _buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("allocate_tensors(conv)", source))?;

    upload_weight(&sx, &sx_data, "write<sx>")?;
    upload_weight(&c, weight, "write<c>")?;

    backend
        .compute(&mut graph)
        .map_err(|source| E2eError::ggml("compute(conv)", source))?;

    result
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read_data_backend(conv)", source))
}

/// Fused projection + convolution graph for linear attention prefill.
///
/// Combines four input projections (`mul_mat`), in-graph transpose + left-pad,
/// `ggml_ssm_conv`, and SiLU activation into a single compute graph. This
/// eliminates the host↔device round-trip between projection and convolution.
///
/// When `conv_tail_rows > 0`, reads back only the last `conv_tail_rows` rows of
/// the pre-conv QKV tensor (for decode state continuity). When 0, skips the
/// readback entirely — the allocator can reuse the memory.
#[allow(clippy::too_many_arguments)] // GPU graph builder — 8 params, all needed
fn project_and_conv_fused_graph(
    attention: &Qwen35LinearAttentionLayerPlan,
    dims: &LinearAttentionDims,
    input: &[f32],
    sequence_length: usize,
    attn_norm_weight: &[f32],
    rms_norm_eps: f32,
    conv_tail_rows: usize,
    backend: &Backend,
) -> Result<FusedLinearOutputs, E2eError> {
    let LinearAttentionDims {
        hidden,
        conv_channels,
        kernel_size,
        ..
    } = *dims;

    if kernel_size == 0 {
        return Err(E2eError::BufferLengthMismatch {
            expected: 1,
            actual: 0,
        });
    }
    if sequence_length == 0 {
        return Err(E2eError::BufferLengthMismatch {
            expected: 1,
            actual: 0,
        });
    }

    let pad = kernel_size - 1;
    let padded_len = pad + sequence_length;

    let total_bytes = dims.estimate_fused_memory(sequence_length)?;
    let ctx = Context::new_no_alloc_bytes(Bytes::new(total_bytes))
        .map_err(|source| E2eError::ggml("Context::new_no_alloc_bytes(fused)", source))?;

    // --- Projection tensors via shared builder ---
    let x_raw = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden, sequence_length))
        .map_err(|source| E2eError::ggml("new<X>", source))?;

    // Layer pre-norm weight: [hidden]
    let norm_w = ctx
        .new_tensor_1d::<f32>(Length::new(hidden))
        .map_err(|source| E2eError::ggml("new<norm_w>", source))?;

    // In-graph layer pre-norm: rms_norm(X, eps) * norm_weight
    let x_normed = ctx
        .rms_norm(&x_raw, rms_norm_eps)
        .map_err(|source| E2eError::ggml("rms_norm(X_layer)", source))?;
    let x = ctx
        .mul(&x_normed, &norm_w)
        .map_err(|source| E2eError::ggml("mul(X_layer_norm)", source))?;

    // --- Projection: 4 matmuls sharing normed input X ---
    let projs = build_batch_projections(&ctx, &x, hidden, &linear_projection_specs(dims))?;
    let (w_qkv, qkv_out) = (&projs[0].w, &projs[0].y);
    let (w_z, z_out) = (&projs[1].w, &projs[1].y);
    let (w_alpha, alpha_out) = (&projs[2].w, &projs[2].y);
    let (w_beta, beta_out) = (&projs[3].w, &projs[3].y);

    // --- In-graph conv: transpose → cont → pad → ssm_conv → silu ---
    // qkv_out shape: ne[0]=conv_channels, ne[1]=seq_len (channels-fast)
    // ssm_conv wants: ne[0]=padded_len, ne[1]=conv_channels (time-fast)

    // Step 1: Transpose to [seq_len, conv_channels] (time-fast)
    let qkv_t = ctx
        .transpose(qkv_out)
        .map_err(|source| E2eError::ggml("transpose(QKV)", source))?;
    // Step 2: Make contiguous (transpose is just a stride swap)
    let qkv_cont = ctx
        .cont(&qkv_t)
        .map_err(|source| E2eError::ggml("cont(QKV_t)", source))?;

    // Step 3: Left-pad with zeros for causal boundary.
    // Build conv sub-graph; the `zeros` tensor (if any) and conv kernel `c`
    // must be kept alive for data upload after allocation.
    let (silu_out, zeros_tensor, c_tensor) = if pad > 0 {
        let zeros = ctx
            .new_tensor_2d::<f32>(Shape2D::new(pad, conv_channels))
            .map_err(|source| E2eError::ggml("new_tensor_2d<zeros>", source))?;
        let padded = ctx
            .concat(&zeros, &qkv_cont, 0)
            .map_err(|source| E2eError::ggml("concat(pad)", source))?;
        let padded_3d = ctx
            .reshape_3d(&padded, padded_len, conv_channels, 1)
            .map_err(|source| E2eError::ggml("reshape_3d(padded)", source))?;
        let c = ctx
            .new_tensor_2d::<f32>(Shape2D::new(kernel_size, conv_channels))
            .map_err(|source| E2eError::ggml("new_tensor_2d<c>", source))?;
        let conv_out = ctx
            .ssm_conv(&padded_3d, &c)
            .map_err(|source| E2eError::ggml("ssm_conv", source))?;
        let silu_out = ctx
            .silu(&conv_out)
            .map_err(|source| E2eError::ggml("silu(conv)", source))?;
        (silu_out, Some(zeros), c)
    } else {
        // kernel_size == 1: no padding, but still transpose → cont for layout.
        let qkv_3d = ctx
            .reshape_3d(&qkv_cont, sequence_length, conv_channels, 1)
            .map_err(|source| E2eError::ggml("reshape_3d(no_pad)", source))?;
        let c = ctx
            .new_tensor_2d::<f32>(Shape2D::new(kernel_size, conv_channels))
            .map_err(|source| E2eError::ggml("new_tensor_2d<c>(no_pad)", source))?;
        let conv_out = ctx
            .ssm_conv(&qkv_3d, &c)
            .map_err(|source| E2eError::ggml("ssm_conv(no_pad)", source))?;
        let silu_out = ctx
            .silu(&conv_out)
            .map_err(|source| E2eError::ggml("silu(no_pad)", source))?;
        (silu_out, None, c)
    };

    // --- Build graph, allocate, upload, compute, read — all in one scope so
    //     the backend buffer stays alive through the read-back. ---
    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("new_graph(fused)", source))?;
    graph.build_forward_expand(&silu_out);
    // Only keep qkv_out alive when we need to read tail rows for conv state.
    if conv_tail_rows > 0 {
        graph.build_forward_expand(qkv_out);
    }
    graph.build_forward_expand(z_out);
    graph.build_forward_expand(alpha_out);
    graph.build_forward_expand(beta_out);

    let _buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("allocate_tensors(fused)", source))?;

    // Upload projection weights and input.
    upload_weight(w_qkv, &attention.qkv_weight_values, "write<W_QKV>")?;
    upload_weight(w_z, &attention.gate_weight_values, "write<W_Z>")?;
    upload_weight(w_alpha, &attention.alpha_weight_values, "write<W_alpha>")?;
    upload_weight(w_beta, &attention.beta_weight_values, "write<W_beta>")?;
    upload_weight(&x_raw, input, "write<X>")?;
    upload_weight(&norm_w, attn_norm_weight, "write<norm_w>")?;

    // Upload zero padding (only when kernel_size > 1).
    if let Some(ref zeros) = zeros_tensor {
        let zero_data = vec![0.0_f32; pad * conv_channels];
        upload_weight(zeros, &zero_data, "write<zeros>")?;
    }

    // Upload conv kernel weights.
    upload_weight(&c_tensor, &attention.conv_weight_values, "write<c>")?;

    backend
        .compute(&mut graph)
        .map_err(|source| E2eError::ggml("compute(fused)", source))?;

    // --- Read back results (buffer is still alive) ---
    let conv = silu_out
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read<conv_silu>", source))?;
    // Read only the tail rows of pre-conv QKV needed for decode state seeding.
    let qkv_pre_conv_tail = if conv_tail_rows > 0 {
        let tail_offset = checked_mul(
            sequence_length.saturating_sub(conv_tail_rows),
            conv_channels,
        )?;
        let tail_len = checked_mul(conv_tail_rows, conv_channels)?;
        Some(
            qkv_out
                .read_data_backend_at(tail_offset, tail_len)
                .map_err(|source| E2eError::ggml("read<qkv_tail>", source))?,
        )
    } else {
        None
    };
    let z = z_out
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read<Z>", source))?;
    let alpha = alpha_out
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read<alpha>", source))?;
    let beta = beta_out
        .read_data_backend()
        .map_err(|source| E2eError::ggml("read<beta>", source))?;

    Ok(FusedLinearOutputs {
        conv,
        qkv_pre_conv_tail,
        z,
        alpha,
        beta,
        conv_channels,
    })
}

/// Convolve a single new QKV row using the conv buffer from previous tokens.
///
/// The buffer holds the last `kernel_size - 1` pre-conv QKV rows. The new row
/// is the "current" sample. After computing the output, the new row is pushed
/// into the buffer (shifting out the oldest if full).
pub(super) fn causal_depthwise_conv_decode_step(
    new_row: &[f32],
    state: &mut LinearAttentionState,
    weight: &[f32],
) -> Result<Vec<f32>, E2eError> {
    let channels = state.conv_channels;
    let kernel_size = state.conv_kernel;
    if new_row.len() != channels {
        return Err(E2eError::BufferLengthMismatch {
            expected: channels,
            actual: new_row.len(),
        });
    }
    let expected_weight_len = checked_mul(kernel_size, channels)?;
    if weight.len() != expected_weight_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_weight_len,
            actual: weight.len(),
        });
    }

    let mut output = vec![0.0_f32; channels];
    for channel in 0..channels {
        let mut sum = 0.0_f32;
        for tap in 0..kernel_size {
            let lookback = kernel_size - 1 - tap;
            let value = if lookback == 0 {
                new_row[channel]
            } else {
                let buffer_idx = state.conv_valid as isize - lookback as isize;
                if buffer_idx < 0 {
                    0.0 // zero padding for positions before start
                } else {
                    state.conv_buffer[buffer_idx as usize * channels + channel]
                }
            };
            sum += value * weight[channel * kernel_size + tap];
        }
        output[channel] = silu_scalar(sum);
    }

    // Push new row into buffer for the next decode step.
    state.push_conv_row(new_row)?;
    Ok(output)
}

/// Single-token decode step for Qwen3.5 linear attention.
///
/// Uses the conv buffer for convolution and the SSM states for the recurrence.
/// Both are updated in-place.
pub(super) fn qwen35_linear_attention_decode_step(
    attention: &Qwen35LinearAttentionLayerPlan,
    input: &[f32],
    rms_norm_eps: f32,
    state: &mut LinearAttentionState,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    let projections = project_linear_inputs(attention, input, 1, Some(backend))?;
    let hidden_features = projections.hidden_features;

    let output = linear_attention_decode_core(projections, attention, rms_norm_eps, state, None)?;

    project_sequence_graph(
        &output,
        1,
        attention.inner_size,
        hidden_features,
        &attention.ssm_out_weight_values,
        backend,
    )
}

/// Phase timings for a single linear attention prefill invocation.
///
/// Used by `bench_graphs` to compare the relative costs of QKV projection +
/// causal depthwise conv (GPU graph), QK split/norm + V extraction (CPU),
/// SSM recurrence (CPU), and output projection (GPU graph).
#[cfg(test)]
pub(super) struct LinearAttentionPhaseTimings {
    /// Fused projection + causal depthwise conv (GPU graph).
    pub(super) proj_conv_ms: f64,
    /// QK split, per-head L2 norm, V extraction (CPU).
    pub(super) qk_split_norm_ms: f64,
    /// SSM recurrence loop (CPU).
    pub(super) ssm_recurrence_ms: f64,
    /// Output projection (GPU graph).
    pub(super) output_proj_ms: f64,
}

/// Run a single linear attention prefill and return per-phase wall-clock timings.
///
/// This decomposes `qwen35_linear_attention_core` into its four major phases
/// with explicit timing barriers, allowing isolated cost comparison between:
/// - GPU graph work (projections + conv)
/// - CPU scalar work (QK norm, SSM recurrence)
#[cfg(test)]
pub(super) fn bench_linear_attention_phases(
    attention: &Qwen35LinearAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    attn_norm_weight: &[f32],
    state: &mut LinearAttentionState,
    backend: &Backend,
) -> Result<LinearAttentionPhaseTimings, E2eError> {
    use std::time::Instant;

    let dims = LinearAttentionDims::new(attention)?;

    // ── Phase 1: fused projection + causal depthwise conv (GPU graph) ──
    let conv_tail_rows = dims.kernel_size.saturating_sub(1).min(sequence_length);
    let t0 = Instant::now();
    let FusedLinearOutputs {
        conv,
        qkv_pre_conv_tail,
        z,
        alpha,
        beta,
        conv_channels,
    } = project_and_conv_fused_graph(
        attention,
        &dims,
        input,
        sequence_length,
        attn_norm_weight,
        rms_norm_eps,
        conv_tail_rows,
        backend,
    )?;
    let proj_conv_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Capture pre-conv QKV tail into conv buffer.
    if let Some(ref tail) = qkv_pre_conv_tail {
        state.capture_conv_buffer(tail, conv_tail_rows)?;
    } else {
        state.conv_valid = 0;
    }

    // ── Phase 2: QK split + per-head L2 norm + V extraction (CPU) ──
    let t1 = Instant::now();
    let qk_features = checked_mul(attention.group_count, attention.state_size)?;
    let (_q_heads, _k_heads) = split_and_norm_qk(
        &conv,
        sequence_length,
        attention,
        conv_channels,
        rms_norm_eps,
    )?;
    let mut v_heads = vec![0.0_f32; checked_mul(sequence_length, attention.inner_size)?];
    for (conv_row, v_dst) in conv
        .chunks_exact(conv_channels)
        .zip(v_heads.chunks_exact_mut(attention.inner_size))
    {
        v_dst.copy_from_slice(&conv_row[qk_features * 2..]);
    }
    let qk_split_norm_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // ── Phase 3: SSM recurrence (CPU) ──
    let t2 = Instant::now();
    let state_size = attention.state_size;
    let time_step_rank = attention.time_step_rank;
    let group_count = attention.group_count;
    let state_size_sq = checked_mul(state_size, state_size)?;
    let states_len = checked_mul(time_step_rank, state_size_sq)?;
    debug_assert_eq!(
        state.ssm_states.len(),
        states_len,
        "SSM state size mismatch"
    );
    state.ssm_states[..states_len].fill(0.0);
    let states = &mut state.ssm_states[..states_len];
    let mut output = vec![0.0_f32; checked_mul(sequence_length, attention.inner_size)?];
    let scale = 1.0_f32 / (state_size as f32).sqrt();
    let mut scratch = SsmScratch::new(state_size);

    for token in 0..sequence_length {
        let token_rank_base = token * time_step_rank;
        for (head, state_chunk) in states.chunks_exact_mut(state_size_sq).enumerate() {
            let src_group = head % group_count;
            let q = head_slice(&_q_heads, token, src_group, group_count, state_size);
            let k = head_slice(&_k_heads, token, src_group, group_count, state_size);
            let v = head_slice(&v_heads, token, head, time_step_rank, state_size);
            let z_head = head_slice(&z, token, head, time_step_rank, state_size);
            let gate =
                softplus_scalar(alpha[token_rank_base + head] + attention.dt_bias_values[head])
                    * attention.ssm_a_values[head];
            let beta_value = sigmoid_scalar(beta[token_rank_base + head]);
            ssm_recurrence_step(
                state_chunk,
                q,
                k,
                v,
                z_head,
                gate.exp(),
                beta_value,
                state_size,
                scale,
                &mut scratch,
            );
            let normalized =
                rms_norm_single(&scratch.out, &attention.ssm_norm_values, rms_norm_eps)?;
            let dst = head_slice_mut(&mut output, token, head, time_step_rank, state_size);
            dst.iter_mut()
                .zip(normalized.iter().zip(z_head.iter()))
                .for_each(|(d, (&n, &z_val))| *d = n * silu_scalar(z_val));
        }
    }
    let ssm_recurrence_ms = t2.elapsed().as_secs_f64() * 1000.0;

    // ── Phase 4: Output projection (GPU graph) ──
    let t3 = Instant::now();
    let _result = project_sequence_graph(
        &output,
        sequence_length,
        attention.inner_size,
        dims.hidden,
        &attention.ssm_out_weight_values,
        backend,
    )?;
    let output_proj_ms = t3.elapsed().as_secs_f64() * 1000.0;

    Ok(LinearAttentionPhaseTimings {
        proj_conv_ms,
        qk_split_norm_ms,
        ssm_recurrence_ms,
        output_proj_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::super::plan::Qwen35LinearAttentionLayerPlan;
    use super::*;

    #[test]
    fn qwen35_linear_head_group_mapping_is_tiled() {
        let group_count = 2_usize;
        let time_step_rank = 4_usize;
        let state_size = 2_usize;
        let inner_size = time_step_rank * state_size;
        let hidden = inner_size;
        let conv_channels = inner_size + 2 * group_count * state_size;
        let conv_kernel = 2_usize;

        let mut qkv_weight = vec![0.0_f32; hidden * conv_channels];
        for i in 0..hidden.min(conv_channels) {
            qkv_weight[i * conv_channels + i] = 1.0;
        }
        let gate_weight = vec![0.0_f32; hidden * inner_size];
        let alpha_weight = vec![0.0_f32; hidden * time_step_rank];
        let beta_weight = vec![0.0_f32; hidden * time_step_rank];
        let mut conv_weight = vec![0.0_f32; conv_channels * conv_kernel];
        for ch in 0..conv_channels {
            conv_weight[ch * conv_kernel + (conv_kernel - 1)] = 1.0;
        }
        let dt_bias = vec![0.0_f32; time_step_rank];
        let ssm_a = vec![-1.0_f32; time_step_rank];
        let ssm_norm = vec![1.0_f32; state_size];
        let mut ssm_out_weight = vec![0.0_f32; inner_size * hidden];
        for i in 0..inner_size.min(hidden) {
            ssm_out_weight[i * hidden + i] = 1.0;
        }

        let plan = Qwen35LinearAttentionLayerPlan {
            norm_values: vec![1.0_f32; hidden],
            qkv_weight_values: qkv_weight,
            gate_weight_values: gate_weight,
            alpha_weight_values: alpha_weight,
            beta_weight_values: beta_weight,
            conv_weight_values: conv_weight,
            dt_bias_values: dt_bias,
            ssm_a_values: ssm_a,
            ssm_norm_values: ssm_norm,
            ssm_out_weight_values: ssm_out_weight,
            state_size,
            group_count,
            time_step_rank,
            inner_size,
            conv_kernel,
        };

        let input: Vec<f32> = (0..inner_size).map(|i| (i + 1) as f32 * 0.1).collect();
        let sequence_length = 1;

        crate::backend::ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        let result = qwen35_linear_attention_inference(
            &plan,
            &input,
            sequence_length,
            1e-5,
            &plan.norm_values,
            &backend,
        );
        assert!(
            result.is_ok(),
            "inference should succeed: {:?}",
            result.err()
        );
        let output_tiled = result.unwrap();
        assert_eq!(output_tiled.len(), inner_size);

        let plan_no_repeat = Qwen35LinearAttentionLayerPlan {
            group_count: time_step_rank,
            ..plan.clone()
        };

        let plan_bad = Qwen35LinearAttentionLayerPlan {
            group_count: 3,
            ..plan.clone()
        };
        let bad_result = qwen35_linear_attention_inference(
            &plan_bad,
            &input,
            sequence_length,
            1e-5,
            &plan_bad.norm_values,
            &backend,
        );
        assert!(
            bad_result.is_err(),
            "should fail with indivisible group_count"
        );
        let _ = plan_no_repeat;
    }

    #[test]
    fn conv_decode_step_matches_full_conv_last_token() {
        let channels = 4;
        let kernel_size = 3;
        let weight: Vec<f32> = (0..channels * kernel_size)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        // 4-token sequence through full conv.
        let input: Vec<f32> = (0..4 * channels).map(|i| (i as f32 + 1.0) * 0.05).collect();
        let full = causal_depthwise_conv(&input, 4, channels, kernel_size, &weight).unwrap();

        // Decode: feed tokens one at a time through the conv decode step.
        let mut state =
            super::super::state::LinearAttentionState::new(kernel_size, channels, 1, 1).unwrap();
        let mut last_output = vec![];
        for token in 0..4 {
            let row = &input[token * channels..(token + 1) * channels];
            last_output = causal_depthwise_conv_decode_step(row, &mut state, &weight).unwrap();
        }

        // The last decode output should match the last token from full conv.
        let expected = &full[3 * channels..4 * channels];
        for (i, (a, b)) in last_output.iter().zip(expected).enumerate() {
            assert!((a - b).abs() < 1e-6, "channel {i}: decode={a} vs full={b}");
        }
    }

    #[test]
    fn linear_attention_prefill_then_decode_matches_full_reprocess() {
        let group_count = 2_usize;
        let time_step_rank = 4_usize;
        let state_size = 2_usize;
        let inner_size = time_step_rank * state_size;
        let hidden = inner_size;
        let conv_channels = inner_size + 2 * group_count * state_size;
        let conv_kernel = 2_usize;

        // Build a deterministic plan with non-trivial weights.
        let mut qkv_weight = vec![0.0_f32; hidden * conv_channels];
        for i in 0..hidden.min(conv_channels) {
            qkv_weight[i * conv_channels + i] = 1.0;
        }
        let mut gate_weight = vec![0.0_f32; hidden * inner_size];
        for i in 0..hidden.min(inner_size) {
            gate_weight[i * inner_size + i] = 0.5;
        }
        let alpha_weight = vec![0.01_f32; hidden * time_step_rank];
        let beta_weight = vec![0.01_f32; hidden * time_step_rank];
        let mut conv_weight = vec![0.0_f32; conv_channels * conv_kernel];
        for ch in 0..conv_channels {
            conv_weight[ch * conv_kernel + (conv_kernel - 1)] = 1.0;
        }
        let dt_bias = vec![0.0_f32; time_step_rank];
        let ssm_a = vec![-1.0_f32; time_step_rank];
        let ssm_norm = vec![1.0_f32; state_size];
        let mut ssm_out_weight = vec![0.0_f32; inner_size * hidden];
        for i in 0..inner_size.min(hidden) {
            ssm_out_weight[i * hidden + i] = 1.0;
        }

        let plan = Qwen35LinearAttentionLayerPlan {
            norm_values: vec![1.0_f32; hidden],
            qkv_weight_values: qkv_weight,
            gate_weight_values: gate_weight,
            alpha_weight_values: alpha_weight,
            beta_weight_values: beta_weight,
            conv_weight_values: conv_weight,
            dt_bias_values: dt_bias,
            ssm_a_values: ssm_a,
            ssm_norm_values: ssm_norm,
            ssm_out_weight_values: ssm_out_weight,
            state_size,
            group_count,
            time_step_rank,
            inner_size,
            conv_kernel,
        };

        // 3-token prompt + 1 decode token.
        let prompt: Vec<f32> = (0..3 * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect();
        let new_token: Vec<f32> = (0..hidden).map(|i| (i as f32 + 100.0) * 0.01).collect();

        crate::backend::ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        // Full reprocess: all 4 tokens at once.
        let full_input: Vec<f32> = prompt.iter().chain(new_token.iter()).copied().collect();
        let norm_weight = &plan.norm_values;
        let full_output =
            qwen35_linear_attention_inference(&plan, &full_input, 4, 1e-5, norm_weight, &backend)
                .unwrap();
        let expected = &full_output[3 * hidden..4 * hidden];

        // Prefill 3 tokens, then decode 1 token.
        let mut state = super::super::state::LinearAttentionState::new(
            conv_kernel,
            conv_channels,
            time_step_rank,
            state_size,
        )
        .unwrap();
        let _prefill_output = qwen35_linear_attention_prefill(
            &plan,
            &prompt,
            3,
            1e-5,
            norm_weight,
            &mut state,
            &backend,
        )
        .unwrap();

        // Decode path: apply host-side rms_norm + weight to match the in-graph
        // norm that inference/prefill now perform.
        let normalized_token = super::super::tensor_ops::rms_norm_with_weight(
            &new_token,
            hidden,
            1,
            norm_weight,
            1e-5,
        )
        .unwrap();
        let decode_output = qwen35_linear_attention_decode_step(
            &plan,
            &normalized_token,
            1e-5,
            &mut state,
            &backend,
        )
        .unwrap();

        for (i, (a, b)) in decode_output.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "feature {i}: decode={a} vs full={b}, diff={}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn conv_graph_matches_host_basic() {
        let channels = 6;
        let kernel_size = 3;
        let seq_len = 5;

        let weight: Vec<f32> = (0..channels * kernel_size)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let input: Vec<f32> = (0..seq_len * channels)
            .map(|i| (i as f32 + 1.0) * 0.05)
            .collect();

        let host_out = causal_depthwise_conv(&input, seq_len, channels, kernel_size, &weight)
            .expect("host conv");

        crate::backend::ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        let graph_out =
            causal_depthwise_conv_graph(&input, seq_len, channels, kernel_size, &weight, &backend)
                .expect("graph conv");

        assert_eq!(host_out.len(), graph_out.len());
        for (i, (h, g)) in host_out.iter().zip(graph_out.iter()).enumerate() {
            assert!(
                (h - g).abs() < 1e-5,
                "index {i}: host={h} vs graph={g}, diff={}",
                (h - g).abs()
            );
        }
    }

    #[test]
    fn conv_graph_matches_host_single_token() {
        let channels = 4;
        let kernel_size = 3;
        let seq_len = 1;

        let weight: Vec<f32> = (0..channels * kernel_size)
            .map(|i| (i as f32 + 0.5) * 0.2)
            .collect();
        let input: Vec<f32> = (0..channels).map(|i| i as f32 + 1.0).collect();

        let host_out =
            causal_depthwise_conv(&input, seq_len, channels, kernel_size, &weight).unwrap();

        crate::backend::ensure_backends_loaded();
        let backend = Backend::new(ggml_rs::BackendKind::Cpu).unwrap();

        let graph_out =
            causal_depthwise_conv_graph(&input, seq_len, channels, kernel_size, &weight, &backend)
                .unwrap();

        for (i, (h, g)) in host_out.iter().zip(graph_out.iter()).enumerate() {
            assert!(
                (h - g).abs() < 1e-5,
                "seq=1 index {i}: host={h} vs graph={g}",
            );
        }
    }

    #[test]
    fn conv_graph_matches_host_seq_less_than_kernel() {
        let channels = 4;
        let kernel_size = 4;
        let seq_len = 2;

        let weight: Vec<f32> = (0..channels * kernel_size)
            .map(|i| (i as f32 + 1.0) * 0.05)
            .collect();
        let input: Vec<f32> = (0..seq_len * channels)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        let host_out =
            causal_depthwise_conv(&input, seq_len, channels, kernel_size, &weight).unwrap();

        crate::backend::ensure_backends_loaded();
        let backend = Backend::new(ggml_rs::BackendKind::Cpu).unwrap();

        let graph_out =
            causal_depthwise_conv_graph(&input, seq_len, channels, kernel_size, &weight, &backend)
                .unwrap();

        for (i, (h, g)) in host_out.iter().zip(graph_out.iter()).enumerate() {
            assert!(
                (h - g).abs() < 1e-5,
                "seq<kernel index {i}: host={h} vs graph={g}",
            );
        }
    }

    #[test]
    fn conv_graph_rejects_kernel_size_zero() {
        crate::backend::ensure_backends_loaded();
        let backend = Backend::new(ggml_rs::BackendKind::Cpu).unwrap();
        let result = causal_depthwise_conv_graph(&[1.0], 1, 1, 0, &[], &backend);
        assert!(result.is_err());
    }
}
