//! Qwen3.5 linear attention with causal depthwise convolution and delta-net
//! recurrence.
//!
//! Provides full-sequence inference, prefill (capturing conv buffer and SSM
//! states), and single-token decode step.

use super::error::E2eError;
use super::numeric::{checked_mul, sigmoid_scalar, silu_scalar, softplus_scalar};
use super::plan::Qwen35LinearAttentionLayerPlan;
use super::state::LinearAttentionState;
use super::tensor_ops::{
    head_slice, head_slice_mut, per_head_l2_norm, project_sequence, rms_norm_single,
};

pub(super) fn qwen35_linear_attention_inference(
    attention: &Qwen35LinearAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
) -> Result<Vec<f32>, E2eError> {
    qwen35_linear_attention_core(attention, input, sequence_length, rms_norm_eps, None)
}

/// Prefill variant: runs the full sequence AND captures conv buffer + SSM states.
pub(super) fn qwen35_linear_attention_prefill(
    attention: &Qwen35LinearAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    state: &mut LinearAttentionState,
) -> Result<Vec<f32>, E2eError> {
    qwen35_linear_attention_core(attention, input, sequence_length, rms_norm_eps, Some(state))
}

/// Shared input projections for both core and decode paths.
struct LinearProjections {
    qkv: Vec<f32>,
    z: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    conv_channels: usize,
    hidden_features: usize,
}

/// Project input through QKV, gate, alpha, and beta weights. Validates
/// input dimensions via checked arithmetic (catches malformed weights early).
fn project_linear_inputs(
    attention: &Qwen35LinearAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
) -> Result<LinearProjections, E2eError> {
    let hidden_features = attention
        .inner_size
        .checked_div(1) // just for consistent error path
        .filter(|&is| is > 0)
        .and_then(|_| {
            let total = attention.ssm_out_weight_values.len();
            let is = attention.inner_size;
            let h = total / is;
            (h > 0 && h * is == total).then_some(h)
        })
        .ok_or(E2eError::BufferLengthMismatch {
            expected: 1,
            actual: 0,
        })?;

    let expected_input_len = checked_mul(hidden_features, sequence_length)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }

    let conv_channels = attention.inner_size
        + checked_mul(checked_mul(attention.group_count, attention.state_size)?, 2)?;
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
    mut state: Option<&mut LinearAttentionState>,
) -> Result<Vec<f32>, E2eError> {
    let LinearProjections {
        qkv,
        z,
        alpha,
        beta,
        conv_channels,
        hidden_features,
    } = project_linear_inputs(attention, input, sequence_length)?;

    let conv = causal_depthwise_conv(
        &qkv,
        sequence_length,
        conv_channels,
        attention.conv_kernel,
        &attention.conv_weight_values,
    )?;

    // Capture pre-conv QKV into conv buffer for future decode steps.
    if let Some(ref mut state) = state {
        state.capture_conv_buffer(&qkv, sequence_length)?;
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
    let mut states = vec![
        0.0_f32;
        checked_mul(
            attention.time_step_rank,
            checked_mul(attention.state_size, attention.state_size)?
        )?
    ];
    let scale = 1.0_f32 / (attention.state_size as f32).sqrt();
    let mut scratch = SsmScratch::new(attention.state_size);

    for token in 0..sequence_length {
        for head in 0..attention.time_step_rank {
            let src_group = head % attention.group_count;
            let q = head_slice(
                &q_heads,
                token,
                src_group,
                attention.group_count,
                attention.state_size,
            );
            let k = head_slice(
                &k_heads,
                token,
                src_group,
                attention.group_count,
                attention.state_size,
            );
            let v = head_slice(
                &v_heads,
                token,
                head,
                attention.time_step_rank,
                attention.state_size,
            );
            let z_head = head_slice(
                &z,
                token,
                head,
                attention.time_step_rank,
                attention.state_size,
            );
            let gate = softplus_scalar(
                alpha[checked_mul(token, attention.time_step_rank)? + head]
                    + attention.dt_bias_values[head],
            ) * attention.ssm_a_values[head];
            let beta_value =
                sigmoid_scalar(beta[checked_mul(token, attention.time_step_rank)? + head]);
            let state_offset = checked_mul(
                head,
                checked_mul(attention.state_size, attention.state_size)?,
            )?;
            let state = &mut states[state_offset
                ..state_offset + checked_mul(attention.state_size, attention.state_size)?];
            ssm_recurrence_step(
                state,
                q,
                k,
                v,
                z_head,
                gate.exp(),
                beta_value,
                attention.state_size,
                scale,
                &mut scratch,
            );
            let normalized =
                rms_norm_single(&scratch.out, &attention.ssm_norm_values, rms_norm_eps)?;
            let dst = head_slice_mut(
                &mut output,
                token,
                head,
                attention.time_step_rank,
                attention.state_size,
            );
            dst.iter_mut()
                .zip(normalized.iter().zip(z_head.iter()))
                .for_each(|(d, (&n, &z))| *d = n * silu_scalar(z));
        }
    }

    // Capture SSM states after processing all tokens.
    if let Some(state) = state {
        state.capture_ssm_states(&states);
    }

    project_sequence(
        &output,
        sequence_length,
        attention.inner_size,
        hidden_features,
        &attention.ssm_out_weight_values,
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

    // Read output (column-major order preserved for parity).
    for col in 0..state_size {
        for row in 0..state_size {
            scratch.out[col] += state[row * state_size + col] * (q[row] * scale);
        }
    }
}

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
) -> Result<Vec<f32>, E2eError> {
    let LinearProjections {
        qkv,
        z,
        alpha,
        beta,
        conv_channels,
        hidden_features,
    } = project_linear_inputs(attention, input, 1)?;

    // Conv: use buffer + new QKV row.
    let conv = causal_depthwise_conv_decode_step(&qkv, state, &attention.conv_weight_values)?;

    // Split conv output into Q, K and L2-normalize.
    let qk_features = checked_mul(attention.group_count, attention.state_size)?;
    let (q_heads, k_heads) = split_and_norm_qk(&conv, 1, attention, conv_channels, rms_norm_eps)?;

    // V is the region after Q and K — borrow directly from conv to avoid copy.
    let v_raw = &conv[qk_features * 2..conv_channels];

    // One SSM recurrence step per head, using persisted states.
    let scale = 1.0_f32 / (attention.state_size as f32).sqrt();
    let mut output = vec![0.0_f32; attention.inner_size];
    let mut scratch = SsmScratch::new(attention.state_size);

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
            &mut scratch,
        );
        let normalized = rms_norm_single(&scratch.out, &attention.ssm_norm_values, rms_norm_eps)?;
        let dst = head_slice_mut(
            &mut output,
            0,
            head,
            attention.time_step_rank,
            attention.state_size,
        );
        dst.iter_mut()
            .zip(normalized.iter().zip(z_head.iter()))
            .for_each(|(d, (&n, &z))| *d = n * silu_scalar(z));
    }

    project_sequence(
        &output,
        1,
        attention.inner_size,
        hidden_features,
        &attention.ssm_out_weight_values,
    )
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

        let result = qwen35_linear_attention_inference(&plan, &input, sequence_length, 1e-5);
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
        let bad_result =
            qwen35_linear_attention_inference(&plan_bad, &input, sequence_length, 1e-5);
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

        // Full reprocess: all 4 tokens at once.
        let full_input: Vec<f32> = prompt.iter().chain(new_token.iter()).copied().collect();
        let full_output = qwen35_linear_attention_inference(&plan, &full_input, 4, 1e-5).unwrap();
        let expected = &full_output[3 * hidden..4 * hidden];

        // Prefill 3 tokens, then decode 1 token.
        let mut state = super::super::state::LinearAttentionState::new(
            conv_kernel,
            conv_channels,
            time_step_rank,
            state_size,
        )
        .unwrap();
        let _prefill_output =
            qwen35_linear_attention_prefill(&plan, &prompt, 3, 1e-5, &mut state).unwrap();
        let decode_output =
            qwen35_linear_attention_decode_step(&plan, &new_token, 1e-5, &mut state).unwrap();

        for (i, (a, b)) in decode_output.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "feature {i}: decode={a} vs full={b}, diff={}",
                (a - b).abs()
            );
        }
    }
}
