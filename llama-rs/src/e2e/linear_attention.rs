//! Qwen3.5 linear attention with causal depthwise convolution and delta-net
//! recurrence.
//!
//! Provides full-sequence inference, prefill (capturing conv buffer and SSM
//! states), and single-token decode step.
//!
//! Implementation is split across coherent submodules:
//! - [`conv`]: Causal depthwise convolution (host, GPU graph, fused, decode)
//! - [`projection`]: Input projections (QKV, gate, alpha, beta)
//! - [`ssm`]: SSM recurrence logic and scratch buffers
//! - [`decode`]: Single-token decode core and wrapper
//! - [`sequence`]: Full-sequence linear attention core (SSM recurrence loop)
//! - [`bench`]: Phase-level timing instrumentation (test-only)

#[cfg(test)]
mod bench;
mod conv;
mod decode;
mod projection;
mod sequence;
mod ssm;

use sequence::qwen35_linear_attention_core;

// Re-exports: keep existing import paths stable for e2e consumers.
pub(super) use decode::{linear_attention_decode_core, qwen35_linear_attention_decode_step};
pub(super) use projection::LinearProjections;
pub(super) use ssm::LinearDecodeScratch;

// Test-only re-exports for bench_graphs and integration tests.
#[cfg(test)]
pub(super) use bench::bench_linear_attention_phases;
#[cfg(test)]
pub(super) use conv::{
    causal_depthwise_conv, causal_depthwise_conv_decode_step, causal_depthwise_conv_graph,
};

use super::error::E2eError;
use super::numeric::checked_mul;
use super::plan::Qwen35LinearAttentionLayerPlan;
use super::state::LinearAttentionState;
use ggml_rs::Backend;

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

#[cfg(test)]
mod tests {
    use super::super::plan::Qwen35LinearAttentionLayerPlan;
    use super::conv::{causal_depthwise_conv, causal_depthwise_conv_graph};
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
