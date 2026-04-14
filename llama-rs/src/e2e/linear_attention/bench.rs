//! Benchmark instrumentation for linear attention phase timing.
//!
//! Decomposes the full-sequence linear attention into its four major phases
//! with explicit timing barriers, allowing isolated cost comparison.

use super::super::error::E2eError;
use super::super::numeric::{checked_mul, sigmoid_scalar, silu_scalar, softplus_scalar};
use super::super::plan::Qwen35LinearAttentionLayerPlan;
use super::super::state::LinearAttentionState;
use super::super::tensor_ops::{
    head_slice, head_slice_mut, project_sequence_graph, rms_norm_single,
};
use super::conv::project_and_conv_fused_graph;
use super::projection::{FusedLinearOutputs, LinearAttentionDims};
use super::ssm::{SsmScratch, SsmStepScalars, split_and_norm_qk, ssm_recurrence_step};
use ggml_rs::Backend;

/// Phase timings for a single linear attention prefill invocation.
///
/// Used by `bench_graphs` to compare the relative costs of QKV projection +
/// causal depthwise conv (GPU graph), QK split/norm + V extraction (CPU),
/// SSM recurrence (CPU), and output projection (GPU graph).
pub(in crate::e2e) struct LinearAttentionPhaseTimings {
    /// Fused projection + causal depthwise conv (GPU graph).
    pub(in crate::e2e) proj_conv_ms: f64,
    /// QK split, per-head L2 norm, V extraction (CPU).
    pub(in crate::e2e) qk_split_norm_ms: f64,
    /// SSM recurrence loop (CPU).
    pub(in crate::e2e) ssm_recurrence_ms: f64,
    /// Output projection (GPU graph).
    pub(in crate::e2e) output_proj_ms: f64,
}

/// Run a single linear attention prefill and return per-phase wall-clock timings.
///
/// This decomposes `qwen35_linear_attention_core` into its four major phases
/// with explicit timing barriers, allowing isolated cost comparison between:
/// - GPU graph work (projections + conv)
/// - CPU scalar work (QK norm, SSM recurrence)
pub(in crate::e2e) fn bench_linear_attention_phases(
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
                SsmStepScalars {
                    decay: gate.exp(),
                    beta_value,
                    scale,
                },
                state_size,
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
