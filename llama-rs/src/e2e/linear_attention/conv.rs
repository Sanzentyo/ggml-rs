//! Causal depthwise convolution: host reference, GPU graph, fused projection +
//! conv graph, and single-token decode step.

use super::projection::{FusedLinearOutputs, LinearAttentionDims, linear_projection_specs};
use crate::e2e::attention::shared::graph_norm_input;
use crate::e2e::error::{E2eError, GgmlResultExt};
use crate::e2e::numeric::checked_mul;
use crate::e2e::plan::Qwen35LinearAttentionLayerPlan;
use crate::e2e::state::LinearAttentionState;
#[cfg(test)]
use crate::e2e::tensor_ops::PROJECTION_SLACK_BYTES;
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
    let ctx_size = Bytes::new(tensor_overhead + PROJECTION_SLACK_BYTES);

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
/// When `conv_tail_rows > 0`, reads back only the last `conv_tail_rows` rows of
/// the pre-conv QKV tensor (for decode state continuity). When 0, skips the
/// readback entirely — the allocator can reuse the memory.
#[allow(clippy::too_many_arguments)] // GPU graph builder — 8 params, all needed
pub(super) fn project_and_conv_fused_graph(
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
    upload_weight(&ni.norm_w, attn_norm_weight, "write<norm_w>")?;

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
