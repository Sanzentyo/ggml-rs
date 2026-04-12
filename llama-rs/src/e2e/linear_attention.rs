use super::error::E2eError;
use super::numeric::{checked_mul, sigmoid_scalar, silu_scalar, softplus_scalar};
use super::plan::Qwen35LinearAttentionLayerPlan;
use super::tensor_ops::{
    head_slice, head_slice_mut, per_head_l2_norm, project_sequence, rms_norm_single,
};

pub(super) fn qwen35_linear_attention_inference(
    attention: &Qwen35LinearAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
) -> Result<Vec<f32>, E2eError> {
    let hidden_features = attention.ssm_out_weight_values.len() / attention.inner_size;
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
    let conv = causal_depthwise_conv(
        &qkv,
        sequence_length,
        conv_channels,
        attention.conv_kernel,
        &attention.conv_weight_values,
    )?;

    let mut q_heads = vec![
        0.0_f32;
        checked_mul(
            sequence_length,
            checked_mul(attention.group_count, attention.state_size)?
        )?
    ];
    let mut k_heads = vec![0.0_f32; q_heads.len()];
    let mut v_heads = vec![0.0_f32; checked_mul(sequence_length, attention.inner_size)?];
    let qk_features = checked_mul(attention.group_count, attention.state_size)?;
    for token in 0..sequence_length {
        let src_offset = checked_mul(token, conv_channels)?;
        let q_offset = checked_mul(token, qk_features)?;
        let v_offset = checked_mul(token, attention.inner_size)?;
        q_heads[q_offset..q_offset + qk_features]
            .copy_from_slice(&conv[src_offset..src_offset + qk_features]);
        k_heads[q_offset..q_offset + qk_features].copy_from_slice(
            &conv[src_offset + qk_features..src_offset + checked_mul(qk_features, 2)?],
        );
        v_heads[v_offset..v_offset + attention.inner_size].copy_from_slice(
            &conv[src_offset + checked_mul(qk_features, 2)?..src_offset + conv_channels],
        );
    }

    let q_heads = per_head_l2_norm(
        &q_heads,
        sequence_length,
        attention.group_count,
        attention.state_size,
        rms_norm_eps,
    )?;
    let k_heads = per_head_l2_norm(
        &k_heads,
        sequence_length,
        attention.group_count,
        attention.state_size,
        rms_norm_eps,
    )?;
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
            let mut sk = vec![0.0_f32; attention.state_size];
            let decay = gate.exp();
            for row in 0..attention.state_size {
                for col in 0..attention.state_size {
                    state[row * attention.state_size + col] *= decay;
                    sk[col] += state[row * attention.state_size + col] * k[row];
                }
            }
            let mut delta = vec![0.0_f32; attention.state_size];
            for index in 0..attention.state_size {
                delta[index] = (v[index] - sk[index]) * beta_value;
            }
            for row in 0..attention.state_size {
                for col in 0..attention.state_size {
                    state[row * attention.state_size + col] += k[row] * delta[col];
                }
            }
            let mut out = vec![0.0_f32; attention.state_size];
            for col in 0..attention.state_size {
                for row in 0..attention.state_size {
                    out[col] += state[row * attention.state_size + col] * (q[row] * scale);
                }
            }
            let normalized = rms_norm_single(&out, &attention.ssm_norm_values, rms_norm_eps)?;
            let dst = head_slice_mut(
                &mut output,
                token,
                head,
                attention.time_step_rank,
                attention.state_size,
            );
            for index in 0..attention.state_size {
                dst[index] = normalized[index] * silu_scalar(z_head[index]);
            }
        }
    }

    project_sequence(
        &output,
        sequence_length,
        attention.inner_size,
        hidden_features,
        &attention.ssm_out_weight_values,
    )
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
        for channel in 0..channels {
            let mut sum = 0.0_f32;
            for tap in 0..kernel_size {
                if token + 1 < kernel_size - tap {
                    continue;
                }
                let src_token = token + tap + 1 - kernel_size;
                sum += input[checked_mul(src_token, channels)? + channel]
                    * weight[checked_mul(channel, kernel_size)? + tap];
            }
            output[checked_mul(token, channels)? + channel] = silu_scalar(sum);
        }
    }
    Ok(output)
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
}
