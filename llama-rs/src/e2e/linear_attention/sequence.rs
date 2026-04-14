//! Full-sequence linear attention core: processes all tokens at once.
//!
//! Contains the main `qwen35_linear_attention_core` function which handles
//! both inference (stateless) and prefill (captures conv buffer + SSM states).

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

pub(super) fn qwen35_linear_attention_core(
    attention: &Qwen35LinearAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
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
