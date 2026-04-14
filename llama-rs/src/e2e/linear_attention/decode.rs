//! Single-token decode path for linear attention.
//!
//! Contains `linear_attention_decode_core` (the core recurrence for one token)
//! and `qwen35_linear_attention_decode_step` (the full decode wrapper including
//! projection and output).

use super::super::error::E2eError;
use super::super::numeric::{checked_mul, sigmoid_scalar, silu_scalar, softplus_scalar};
use super::super::plan::Qwen35LinearAttentionLayerPlan;
use super::super::state::LinearAttentionState;
use super::super::tensor_ops::{
    head_slice, head_slice_mut, project_sequence_graph, rms_norm_single_into,
};
use super::conv::causal_depthwise_conv_decode_step;
use super::projection::{LinearProjections, project_linear_inputs};
use super::ssm::{LinearDecodeScratch, split_and_norm_qk, ssm_recurrence_step};
use ggml_rs::Backend;

/// Core linear attention decode logic: conv → split/norm → SSM recurrence → z-gating.
///
/// Takes raw projections and returns the SSM output (before output projection).
/// The caller is responsible for projecting the output.
pub(in crate::e2e) fn linear_attention_decode_core(
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

/// Single-token decode step for Qwen3.5 linear attention.
///
/// Uses the conv buffer for convolution and the SSM states for the recurrence.
/// Both are updated in-place.
pub(in crate::e2e) fn qwen35_linear_attention_decode_step(
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
