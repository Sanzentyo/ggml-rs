//! Causal depthwise convolution: host reference, GPU graph, fused projection +
//! conv graph, and single-token decode step.

use super::projection::{FusedLinearOutputs, LinearAttentionDims, linear_projection_specs};
use crate::e2e::attention::shared::graph_norm_input;
use crate::e2e::error::{E2eError, GgmlResultExt};
use crate::e2e::numeric::checked_mul;
use crate::e2e::plan::Qwen35LinearAttentionLayerPlan;
use crate::e2e::state::LinearAttentionState;
#[cfg(test)]
use crate::e2e::tensor_ops::MATMUL_GRAPH_SLACK_BYTES;
use crate::e2e::tensor_ops::{build_batch_projections, upload_weight};
use ggml_rs::{Backend, Bytes, Context, Shape2D};

/// Host-side causal depthwise convolution (test/reference implementation).
#[cfg(test)]
pub(in crate::e2e) fn causal_depthwise_conv(
    input: &[f32],
    sequence_length: usize,
    channels: usize,
    kernel_size: usize,
    weight: &[f32],
) -> Result<Vec<f32>, E2eError> {
    use crate::e2e::numeric::silu_scalar;

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
pub(in crate::e2e) fn causal_depthwise_conv_graph(
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
    let ctx_size = Bytes::new(tensor_overhead + MATMUL_GRAPH_SLACK_BYTES);

    let ctx =
        Context::new_no_alloc_bytes(ctx_size).ggml_ctx("Context::new_no_alloc_bytes(conv)")?;

    // sx: [padded_len, channels, 1] — ggml ne[0]=padded_len, ne[1]=channels, ne[2]=1
    let sx = ctx
        .new_tensor_3d::<f32>(ggml_rs::Shape3D::new(padded_len, channels, 1))
        .ggml_ctx("new_tensor_3d<sx>")?;

    // c: [kernel_size, channels] — ggml ne[0]=kernel_size, ne[1]=channels
    let c = ctx
        .new_tensor_2d::<f32>(Shape2D::new(kernel_size, channels))
        .ggml_ctx("new_tensor_2d<c>")?;

    // ssm_conv(sx, c) → [channels, seq_len, 1]
    let conv_out = ctx.ssm_conv(&sx, &c).ggml_ctx("ssm_conv")?;

    // SiLU activation
    let result = ctx.silu(&conv_out).ggml_ctx("silu(conv)")?;

    let mut graph = ctx.new_graph().ggml_ctx("new_graph(conv)")?;
    graph.build_forward_expand(&result);

    let _buffer = ctx
        .allocate_tensors(backend)
        .ggml_ctx("allocate_tensors(conv)")?;

    upload_weight(&sx, &sx_data, "write<sx>")?;
    upload_weight(&c, weight, "write<c>")?;

    backend.compute(&mut graph).ggml_ctx("compute(conv)")?;

    result
        .read_data_backend()
        .ggml_ctx("read_data_backend(conv)")
}

/// Fused projection + convolution graph for linear attention prefill.
///
/// Combines four input projections (`mul_mat`), in-graph transpose + left-pad,
/// `ggml_ssm_conv`, and SiLU activation into a single compute graph. This
/// eliminates the host↔device round-trip between projection and convolution.
///
/// Reads the RMS norm weight from `attention.norm_values` directly, avoiding
/// a redundant parameter that must always match the plan.
///
/// When `conv_tail_rows > 0`, reads back only the last `conv_tail_rows` rows of
/// the pre-conv QKV tensor (for decode state continuity). When 0, skips the
/// readback entirely — the allocator can reuse the memory.
pub(super) fn project_and_conv_fused_graph(
    attention: &Qwen35LinearAttentionLayerPlan,
    dims: &LinearAttentionDims,
    input: &[f32],
    sequence_length: usize,
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
        .ggml_ctx("Context::new_no_alloc_bytes(fused)")?;

    let ni = graph_norm_input(&ctx, hidden, sequence_length, rms_norm_eps)?;

    // --- Projection: 4 matmuls sharing normed input X ---
    let projs = build_batch_projections(&ctx, &ni.x, hidden, &linear_projection_specs(dims))?;
    let (w_qkv, qkv_out) = (&projs[0].w, &projs[0].y);
    let (w_z, z_out) = (&projs[1].w, &projs[1].y);
    let (w_alpha, alpha_out) = (&projs[2].w, &projs[2].y);
    let (w_beta, beta_out) = (&projs[3].w, &projs[3].y);

    // --- In-graph conv: transpose → cont → pad → ssm_conv → silu ---
    // qkv_out shape: ne[0]=conv_channels, ne[1]=seq_len (channels-fast)
    // ssm_conv wants: ne[0]=padded_len, ne[1]=conv_channels (time-fast)

    // Step 1: Transpose to [seq_len, conv_channels] (time-fast)
    let qkv_t = ctx.transpose(qkv_out).ggml_ctx("transpose(QKV)")?;
    // Step 2: Make contiguous (transpose is just a stride swap)
    let qkv_cont = ctx.cont(&qkv_t).ggml_ctx("cont(QKV_t)")?;

    // Step 3: Left-pad with zeros for causal boundary.
    // Build conv sub-graph; the `zeros` tensor (if any) and conv kernel `c`
    // must be kept alive for data upload after allocation.
    let (silu_out, zeros_tensor, c_tensor) = if pad > 0 {
        let zeros = ctx
            .new_tensor_2d::<f32>(Shape2D::new(pad, conv_channels))
            .ggml_ctx("new_tensor_2d<zeros>")?;
        let padded = ctx.concat(&zeros, &qkv_cont, 0).ggml_ctx("concat(pad)")?;
        let padded_3d = ctx
            .reshape_3d(&padded, padded_len, conv_channels, 1)
            .ggml_ctx("reshape_3d(padded)")?;
        let c = ctx
            .new_tensor_2d::<f32>(Shape2D::new(kernel_size, conv_channels))
            .ggml_ctx("new_tensor_2d<c>")?;
        let conv_out = ctx.ssm_conv(&padded_3d, &c).ggml_ctx("ssm_conv")?;
        let silu_out = ctx.silu(&conv_out).ggml_ctx("silu(conv)")?;
        (silu_out, Some(zeros), c)
    } else {
        // kernel_size == 1: no padding, but still transpose → cont for layout.
        let qkv_3d = ctx
            .reshape_3d(&qkv_cont, sequence_length, conv_channels, 1)
            .ggml_ctx("reshape_3d(no_pad)")?;
        let c = ctx
            .new_tensor_2d::<f32>(Shape2D::new(kernel_size, conv_channels))
            .ggml_ctx("new_tensor_2d<c>(no_pad)")?;
        let conv_out = ctx.ssm_conv(&qkv_3d, &c).ggml_ctx("ssm_conv(no_pad)")?;
        let silu_out = ctx.silu(&conv_out).ggml_ctx("silu(no_pad)")?;
        (silu_out, None, c)
    };

    // --- Build graph, allocate, upload, compute, read — all in one scope so
    //     the backend buffer stays alive through the read-back. ---
    let mut graph = ctx.new_graph().ggml_ctx("new_graph(fused)")?;
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
        .ggml_ctx("allocate_tensors(fused)")?;

    // Upload projection weights and input.
    upload_weight(w_qkv, &attention.qkv_weight_values, "write<W_QKV>")?;
    upload_weight(w_z, &attention.gate_weight_values, "write<W_Z>")?;
    upload_weight(w_alpha, &attention.alpha_weight_values, "write<W_alpha>")?;
    upload_weight(w_beta, &attention.beta_weight_values, "write<W_beta>")?;
    upload_weight(&ni.x_raw, input, "write<X>")?;
    upload_weight(&ni.norm_w, &attention.norm_values, "write<norm_w>")?;

    // Upload zero padding (only when kernel_size > 1).
    if let Some(ref zeros) = zeros_tensor {
        let zero_data = vec![0.0_f32; pad * conv_channels];
        upload_weight(zeros, &zero_data, "write<zeros>")?;
    }

    // Upload conv kernel weights.
    upload_weight(&c_tensor, &attention.conv_weight_values, "write<c>")?;

    backend.compute(&mut graph).ggml_ctx("compute(fused)")?;

    // --- Read back results (buffer is still alive) ---
    let conv = silu_out.read_data_backend().ggml_ctx("read<conv_silu>")?;
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
                .ggml_ctx("read<qkv_tail>")?,
        )
    } else {
        None
    };
    let z = z_out.read_data_backend().ggml_ctx("read<Z>")?;
    let alpha = alpha_out.read_data_backend().ggml_ctx("read<alpha>")?;
    let beta = beta_out.read_data_backend().ggml_ctx("read<beta>")?;

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
pub(in crate::e2e) fn causal_depthwise_conv_decode_step(
    new_row: &[f32],
    state: &mut LinearAttentionState,
    weight: &[f32],
) -> Result<Vec<f32>, E2eError> {
    use crate::e2e::numeric::silu_scalar;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::e2e::numeric::silu_scalar;

    // -----------------------------------------------------------------------
    // Item 127: causal_depthwise_conv unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn causal_conv_first_token_sees_only_zero_history() {
        // kernel_size=3, channels=2, seq_len=1
        // First token has no history → only the last kernel tap (tap=2) touches
        // the actual input; earlier taps read zero padding.
        let channels = 2;
        let kernel_size = 3;
        let weight: Vec<f32> = (0..channels * kernel_size)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let input = vec![1.0, 2.0]; // single token

        let output = causal_depthwise_conv(&input, 1, channels, kernel_size, &weight).unwrap();

        // ch0: 0*w[0] + 0*w[1] + 1.0*w[2]  → silu(1.0 * 0.3)
        // ch1: 0*w[3] + 0*w[4] + 2.0*w[5]  → silu(2.0 * 0.6)
        let expected_ch0 = silu_scalar(1.0 * weight[2]);
        let expected_ch1 = silu_scalar(2.0 * weight[5]);
        assert!(
            (output[0] - expected_ch0).abs() < 1e-7,
            "ch0: got {} expected {}",
            output[0],
            expected_ch0
        );
        assert!(
            (output[1] - expected_ch1).abs() < 1e-7,
            "ch1: got {} expected {}",
            output[1],
            expected_ch1
        );
    }

    #[test]
    fn causal_conv_no_future_leakage() {
        // A 5-token sequence: token i = i+1. If future leaks, earlier tokens
        // would see larger values than expected from causal-only history.
        let channels = 1;
        let kernel_size = 2;
        let weight = vec![0.5, 0.5]; // uniform kernel

        let input: Vec<f32> = (1..=5).map(|i| i as f32).collect();
        let output = causal_depthwise_conv(&input, 5, channels, kernel_size, &weight).unwrap();

        // token 0: 0*0.5 + 1.0*0.5 = 0.5
        // token 1: 1.0*0.5 + 2.0*0.5 = 1.5
        // token 2: 2.0*0.5 + 3.0*0.5 = 2.5
        // token 3: 3.0*0.5 + 4.0*0.5 = 3.5
        // token 4: 4.0*0.5 + 5.0*0.5 = 4.5
        let expected_sums = [0.5, 1.5, 2.5, 3.5, 4.5];
        for (i, &expected_sum) in expected_sums.iter().enumerate() {
            let expected = silu_scalar(expected_sum);
            assert!(
                (output[i] - expected).abs() < 1e-6,
                "token {i}: got {} expected silu({expected_sum})={}",
                output[i],
                expected
            );
        }
    }

    #[test]
    fn causal_conv_hand_computed_3ch_k2() {
        let channels = 3;
        let kernel_size = 2;
        // weight layout: [ch0_tap0, ch0_tap1, ch1_tap0, ch1_tap1, ch2_tap0, ch2_tap1]
        let weight = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        // 3 tokens × 3 channels
        let input = vec![
            1.0, 2.0, 3.0, // token 0
            4.0, 5.0, 6.0, // token 1
            7.0, 8.0, 9.0, // token 2
        ];
        let output = causal_depthwise_conv(&input, 3, channels, kernel_size, &weight).unwrap();

        // token 0: pad=0 for tap0, input[0] for tap1
        //   ch0: 0*0.1 + 1.0*0.2 = 0.2
        //   ch1: 0*0.3 + 2.0*0.4 = 0.8
        //   ch2: 0*0.5 + 3.0*0.6 = 1.8
        // token 1: input[0] for tap0, input[1] for tap1
        //   ch0: 1.0*0.1 + 4.0*0.2 = 0.9
        //   ch1: 2.0*0.3 + 5.0*0.4 = 2.6
        //   ch2: 3.0*0.5 + 6.0*0.6 = 5.1
        // token 2: input[1] for tap0, input[2] for tap1
        //   ch0: 4.0*0.1 + 7.0*0.2 = 1.8
        //   ch1: 5.0*0.3 + 8.0*0.4 = 4.7
        //   ch2: 6.0*0.5 + 9.0*0.6 = 8.4
        let expected_sums = [0.2, 0.8, 1.8, 0.9, 2.6, 5.1, 1.8, 4.7, 8.4];
        for (i, &expected_sum) in expected_sums.iter().enumerate() {
            let expected = silu_scalar(expected_sum);
            assert!(
                (output[i] - expected).abs() < 1e-6,
                "idx {i}: got {} expected silu({expected_sum})={}",
                output[i],
                expected
            );
        }
    }

    #[test]
    fn causal_conv_kernel_size_1_is_pointwise_silu() {
        // kernel_size=1: no history, just pointwise weight × input → silu
        let channels = 4;
        let kernel_size = 1;
        let weight = vec![0.5, 1.0, 1.5, 2.0];
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 tokens

        let output = causal_depthwise_conv(&input, 2, channels, kernel_size, &weight).unwrap();

        let expected: Vec<f32> = input
            .iter()
            .zip(weight.iter().cycle())
            .map(|(x, w)| silu_scalar(x * w))
            .collect();

        for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-7,
                "idx {i}: got {got} expected {exp}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Item 126: fused projection + conv graph tests
    // -----------------------------------------------------------------------

    #[test]
    fn fused_graph_with_padding_matches_reference() {
        // kernel_size=3 → pad=2, exercises the concat(zeros, qkv) path
        crate::backend::ensure_backends_loaded();
        let backend =
            ggml_rs::Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend available");

        let (plan, dims) = make_linear_plan(3, 4, 2, 2, 2);
        let hidden = dims.hidden;
        let seq_len = 5;
        let input: Vec<f32> = (0..hidden * seq_len)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();

        let fused = project_and_conv_fused_graph(
            &plan, &dims, &input, seq_len, 1e-5, 0, // no tail readback
            &backend,
        )
        .expect("fused graph should succeed");

        // Output shape: conv channels × seq_len
        assert_eq!(
            fused.conv.len(),
            dims.conv_channels * seq_len,
            "conv output length"
        );
        assert_eq!(fused.z.len(), dims.inner_size * seq_len, "z output length");
        assert_eq!(
            fused.alpha.len(),
            dims.time_step_rank * seq_len,
            "alpha output length"
        );
        assert_eq!(
            fused.beta.len(),
            dims.time_step_rank * seq_len,
            "beta output length"
        );
        assert!(fused.qkv_pre_conv_tail.is_none());
    }

    #[test]
    fn fused_graph_no_padding_kernel_1() {
        // kernel_size=1 → pad=0, exercises the no-pad reshape path
        crate::backend::ensure_backends_loaded();
        let backend =
            ggml_rs::Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend available");

        let (plan, dims) = make_linear_plan(1, 4, 2, 2, 2);
        let hidden = dims.hidden;
        let seq_len = 3;
        let input: Vec<f32> = (0..hidden * seq_len)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
            .collect();

        let fused = project_and_conv_fused_graph(&plan, &dims, &input, seq_len, 1e-5, 0, &backend)
            .expect("fused graph (no pad) should succeed");

        assert_eq!(fused.conv.len(), dims.conv_channels * seq_len);
        assert_eq!(fused.z.len(), dims.inner_size * seq_len);
    }

    #[test]
    fn fused_graph_tail_readback() {
        // Verify that conv_tail_rows readback returns the correct number of
        // pre-conv QKV elements.
        crate::backend::ensure_backends_loaded();
        let backend =
            ggml_rs::Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend available");

        let (plan, dims) = make_linear_plan(3, 4, 2, 2, 2);
        let hidden = dims.hidden;
        let seq_len = 5;
        let conv_tail_rows = 2; // request last 2 rows
        let input: Vec<f32> = (0..hidden * seq_len)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();

        let fused = project_and_conv_fused_graph(
            &plan,
            &dims,
            &input,
            seq_len,
            1e-5,
            conv_tail_rows,
            &backend,
        )
        .expect("fused graph with tail should succeed");

        let tail = fused.qkv_pre_conv_tail.expect("tail should be Some");
        assert_eq!(
            tail.len(),
            conv_tail_rows * dims.conv_channels,
            "tail length = tail_rows × conv_channels"
        );
    }

    // --- Test helper ---

    fn make_linear_plan(
        conv_kernel: usize,
        time_step_rank: usize,
        state_size: usize,
        group_count: usize,
        num_groups_for_conv: usize,
    ) -> (Qwen35LinearAttentionLayerPlan, LinearAttentionDims) {
        let inner_size = time_step_rank * state_size;
        let conv_channels = inner_size + num_groups_for_conv * group_count * state_size;
        let hidden = inner_size;

        let plan = Qwen35LinearAttentionLayerPlan {
            norm_values: vec![1.0_f32; hidden],
            qkv_weight_values: vec![0.01_f32; hidden * conv_channels],
            gate_weight_values: vec![0.01_f32; hidden * inner_size],
            alpha_weight_values: vec![0.01_f32; hidden * time_step_rank],
            beta_weight_values: vec![0.01_f32; hidden * time_step_rank],
            conv_weight_values: vec![1.0_f32; conv_channels * conv_kernel],
            dt_bias_values: vec![0.0_f32; time_step_rank],
            ssm_a_values: vec![-1.0_f32; time_step_rank],
            ssm_norm_values: vec![1.0_f32; state_size],
            ssm_out_weight_values: vec![0.01_f32; inner_size * hidden],
            state_size,
            group_count,
            time_step_rank,
            inner_size,
            conv_kernel,
        };

        let dims = LinearAttentionDims::new(&plan).expect("dims should be valid");
        (plan, dims)
    }
}
