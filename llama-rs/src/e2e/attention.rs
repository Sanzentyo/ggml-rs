//! Qwen3.5 full (standard-style) attention with gated Q and NeoX RoPE.
//!
//! Provides full-sequence inference, prefill (capturing KV cache), and
//! single-token decode step using cached state.

use super::error::E2eError;
use super::numeric::{checked_mul, dot, sigmoid_scalar, softmax_prefix};
use super::plan::Qwen35FullAttentionLayerPlan;
use super::state::Qwen35FullAttentionState;
use super::tensor_ops::{head_slice, head_slice_mut, per_head_rms_norm, project_sequence};

pub(super) fn qwen35_full_attention_inference(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
) -> Result<Vec<f32>, E2eError> {
    qwen35_full_attention_core(attention, input, sequence_length, rms_norm_eps, None)
}

/// Prefill variant: computes full attention AND stores post-RoPE K + raw V in `state`.
pub(super) fn qwen35_full_attention_prefill(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    state: &mut Qwen35FullAttentionState,
) -> Result<Vec<f32>, E2eError> {
    qwen35_full_attention_core(attention, input, sequence_length, rms_norm_eps, Some(state))
}

fn qwen35_full_attention_core(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
    state: Option<&mut Qwen35FullAttentionState>,
) -> Result<Vec<f32>, E2eError> {
    let hidden_features =
        attention.output_weight_values.len() / attention.head_count / attention.head_dimension;
    let expected_input_len = checked_mul(hidden_features, sequence_length)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }

    let query_features = checked_mul(attention.head_count, attention.head_dimension)?;
    let kv_features = checked_mul(attention.kv_head_count, attention.head_dimension)?;
    let q_full = project_sequence(
        input,
        sequence_length,
        hidden_features,
        checked_mul(query_features, 2)?,
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

    let hd = attention.head_dimension;
    let (q_values, q_gate) =
        deinterleave_q_gate(&q_full, sequence_length, attention.head_count, hd)?;

    let mut q_values = per_head_rms_norm(
        &q_values,
        sequence_length,
        attention.head_count,
        attention.head_dimension,
        &attention.q_norm_values,
        rms_norm_eps,
    )?;
    let mut k_values = per_head_rms_norm(
        &k_proj,
        sequence_length,
        attention.kv_head_count,
        attention.head_dimension,
        &attention.k_norm_values,
        rms_norm_eps,
    )?;

    // Apply NeoX-style RoPE to Q and K after normalization.
    apply_neox_rope_in_place(
        &mut q_values,
        sequence_length,
        attention.head_count,
        attention.head_dimension,
        attention.rope_n_dims,
        attention.rope_freq_base,
        attention.rope_freq_scale,
        0,
    )?;
    apply_neox_rope_in_place(
        &mut k_values,
        sequence_length,
        attention.kv_head_count,
        attention.head_dimension,
        attention.rope_n_dims,
        attention.rope_freq_base,
        attention.rope_freq_scale,
        0,
    )?;

    // Capture post-RoPE K and raw V into the KV cache if we are in prefill mode.
    if let Some(state) = state {
        state.append_batch(&k_values, &v_proj, sequence_length)?;
    }

    let groups = attention.head_count / attention.kv_head_count;
    let mut head_outputs = vec![0.0_f32; checked_mul(sequence_length, query_features)?];
    for token in 0..sequence_length {
        for head in 0..attention.head_count {
            let kv_head = head / groups;
            let mut scores = vec![f32::NEG_INFINITY; sequence_length];
            for (source, score) in scores.iter_mut().enumerate().take(token + 1) {
                let q = head_slice(
                    &q_values,
                    token,
                    head,
                    attention.head_count,
                    attention.head_dimension,
                );
                let k = head_slice(
                    &k_values,
                    source,
                    kv_head,
                    attention.kv_head_count,
                    attention.head_dimension,
                );
                *score = dot(q, k) * attention.attention_scale;
            }
            let weights = softmax_prefix(&scores, token + 1);
            let dst = head_slice_mut(
                &mut head_outputs,
                token,
                head,
                attention.head_count,
                attention.head_dimension,
            );
            for (source, weight) in weights.iter().copied().enumerate() {
                let v = head_slice(
                    &v_proj,
                    source,
                    kv_head,
                    attention.kv_head_count,
                    attention.head_dimension,
                );
                for index in 0..attention.head_dimension {
                    dst[index] += v[index] * weight;
                }
            }
            let gate = head_slice(
                &q_gate,
                token,
                head,
                attention.head_count,
                attention.head_dimension,
            );
            for index in 0..attention.head_dimension {
                dst[index] *= sigmoid_scalar(gate[index]);
            }
        }
    }

    project_sequence(
        &head_outputs,
        sequence_length,
        query_features,
        hidden_features,
        &attention.output_weight_values,
    )
}

/// De-interleave ggml's `[Q_h0, G_h0, Q_h1, G_h1, ...]` layout into
/// separate Q and gate buffers.
///
/// Both call sites (prefill multi-token and decode single-token) go through
/// this function so validation is unified.
fn deinterleave_q_gate(
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

/// Apply NeoX-style rotary position embedding in-place.
///
/// For each token at position `position_offset + pos`, rotates dimension pairs
/// `(x[k], x[k + n_rot/2])` for `k` in `0..n_rot/2` using angle
/// `theta_k = pos * freq_base^(-2k / n_rot)`.
/// Dimensions beyond `n_rot` are left unchanged.
///
/// `position_offset` shifts the starting position (0 for prefill, prompt_len for decode).
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_neox_rope_in_place(
    values: &mut [f32],
    sequence_length: usize,
    head_count: usize,
    head_dimension: usize,
    n_rot: usize,
    freq_base: f32,
    freq_scale: f32,
    position_offset: usize,
) -> Result<(), E2eError> {
    let total_features = checked_mul(head_count, head_dimension)?;
    let expected_len = checked_mul(sequence_length, total_features)?;
    if values.len() != expected_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_len,
            actual: values.len(),
        });
    }
    debug_assert!(n_rot <= head_dimension && n_rot.is_multiple_of(2));

    let half_rot = n_rot / 2;
    let theta_scale = freq_base.powf(-2.0 / n_rot as f32);

    let cache_size = checked_mul(sequence_length, half_rot)?;
    let mut cos_cache = vec![0.0_f32; cache_size];
    let mut sin_cache = vec![0.0_f32; cache_size];
    for pos in 0..sequence_length {
        let mut theta = (position_offset + pos) as f32;
        for k in 0..half_rot {
            let cache_idx = pos * half_rot + k;
            let angle = theta * freq_scale;
            cos_cache[cache_idx] = angle.cos();
            sin_cache[cache_idx] = angle.sin();
            theta *= theta_scale;
        }
    }

    for pos in 0..sequence_length {
        let token_base = pos * total_features;
        for head in 0..head_count {
            let head_base = token_base + head * head_dimension;
            for k in 0..half_rot {
                let cache_idx = pos * half_rot + k;
                let cos_t = cos_cache[cache_idx];
                let sin_t = sin_cache[cache_idx];
                let idx0 = head_base + k;
                let idx1 = head_base + k + half_rot;
                let x0 = values[idx0];
                let x1 = values[idx1];
                values[idx0] = x0 * cos_t - x1 * sin_t;
                values[idx1] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    Ok(())
}

/// Single-token decode step for Qwen3.5 full attention.
///
/// Processes one token using the KV cache accumulated during prefill (and
/// previous decode steps). The new K/V are appended to `state` BEFORE attention
/// so the token attends to itself.
pub(super) fn qwen35_full_attention_decode_step(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    rms_norm_eps: f32,
    state: &mut Qwen35FullAttentionState,
) -> Result<Vec<f32>, E2eError> {
    let hidden_features =
        attention.output_weight_values.len() / attention.head_count / attention.head_dimension;
    if input.len() != hidden_features {
        return Err(E2eError::BufferLengthMismatch {
            expected: hidden_features,
            actual: input.len(),
        });
    }

    let query_features = checked_mul(attention.head_count, attention.head_dimension)?;
    let kv_features = checked_mul(attention.kv_head_count, attention.head_dimension)?;
    let q_full = project_sequence(
        input,
        1,
        hidden_features,
        checked_mul(query_features, 2)?,
        &attention.q_weight_values,
    )?;
    let k_proj = project_sequence(
        input,
        1,
        hidden_features,
        kv_features,
        &attention.k_weight_values,
    )?;
    let v_proj = project_sequence(
        input,
        1,
        hidden_features,
        kv_features,
        &attention.v_weight_values,
    )?;

    // De-interleave Q/Gate for single token.
    let hd = attention.head_dimension;
    let (q_values, q_gate) = deinterleave_q_gate(&q_full, 1, attention.head_count, hd)?;

    let mut q_values = per_head_rms_norm(
        &q_values,
        1,
        attention.head_count,
        attention.head_dimension,
        &attention.q_norm_values,
        rms_norm_eps,
    )?;
    let mut k_values = per_head_rms_norm(
        &k_proj,
        1,
        attention.kv_head_count,
        attention.head_dimension,
        &attention.k_norm_values,
        rms_norm_eps,
    )?;

    // Position = number of tokens already in the cache.
    let position_offset = state.token_count();
    apply_neox_rope_in_place(
        &mut q_values,
        1,
        attention.head_count,
        attention.head_dimension,
        attention.rope_n_dims,
        attention.rope_freq_base,
        attention.rope_freq_scale,
        position_offset,
    )?;
    apply_neox_rope_in_place(
        &mut k_values,
        1,
        attention.kv_head_count,
        attention.head_dimension,
        attention.rope_n_dims,
        attention.rope_freq_base,
        attention.rope_freq_scale,
        position_offset,
    )?;

    // Append new K/V to cache BEFORE attention so the token attends to itself.
    state.append_batch(&k_values, &v_proj, 1)?;
    let total_tokens = state.token_count();

    let groups = attention.head_count / attention.kv_head_count;
    let mut head_outputs = vec![0.0_f32; query_features];
    for head in 0..attention.head_count {
        let kv_head = head / groups;
        let q = &q_values[head * hd..(head + 1) * hd];

        // Score against all cached K vectors.
        let mut scores = vec![f32::NEG_INFINITY; total_tokens];
        for (source, score) in scores.iter_mut().enumerate().take(total_tokens) {
            let k = state.k_head_at(source, kv_head, hd);
            *score = dot(q, k) * attention.attention_scale;
        }
        let weights = softmax_prefix(&scores, total_tokens);

        let dst = &mut head_outputs[head * hd..(head + 1) * hd];
        for (source, weight) in weights.iter().copied().enumerate() {
            let v = state.v_head_at(source, kv_head, hd);
            for index in 0..hd {
                dst[index] += v[index] * weight;
            }
        }

        let gate = &q_gate[head * hd..(head + 1) * hd];
        for index in 0..hd {
            dst[index] *= sigmoid_scalar(gate[index]);
        }
    }

    project_sequence(
        &head_outputs,
        1,
        query_features,
        hidden_features,
        &attention.output_weight_values,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35_full_attention_qgate_split_is_head_interleaved() {
        let q_full: Vec<f32> = vec![
            1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0,
        ];
        let (q_values, q_gate) = deinterleave_q_gate(&q_full, 1, 2, 3).unwrap();
        assert_eq!(q_values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(q_gate, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn qwen35_full_attention_qgate_split_multi_token() {
        let q_full: Vec<f32> = vec![
            1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0, 5.0, 6.0, 50.0, 60.0, 7.0, 8.0, 70.0, 80.0,
        ];
        let (q_values, q_gate) = deinterleave_q_gate(&q_full, 2, 2, 2).unwrap();
        assert_eq!(q_values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(q_gate, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn rope_identity_at_position_zero() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let original = values.clone();
        apply_neox_rope_in_place(&mut values, 1, 1, 4, 4, 10000.0, 1.0, 0).unwrap();
        for (a, b) in values.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
        }
    }

    #[test]
    fn rope_rotates_at_nonzero_position() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0];
        apply_neox_rope_in_place(&mut values, 2, 1, 4, 4, 1.0, 1.0, 0).unwrap();

        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[1] - 2.0).abs() < 1e-6);
        assert!((values[2] - 3.0).abs() < 1e-6);
        assert!((values[3] - 4.0).abs() < 1e-6);

        let cos1 = 1.0_f32.cos();
        let sin1 = 1.0_f32.sin();
        assert!(
            (values[4] - cos1).abs() < 1e-6,
            "expected {cos1}, got {}",
            values[4]
        );
        assert!((values[5]).abs() < 1e-6);
        assert!(
            (values[6] - sin1).abs() < 1e-6,
            "expected {sin1}, got {}",
            values[6]
        );
        assert!((values[7]).abs() < 1e-6);
    }

    #[test]
    fn rope_preserves_dims_beyond_n_rot() {
        let mut values = [
            1.0_f32, 2.0, 3.0, 4.0, 99.0, 88.0, 1.0, 2.0, 3.0, 4.0, 99.0, 88.0,
        ];
        apply_neox_rope_in_place(&mut values, 2, 1, 6, 4, 10000.0, 1.0, 0).unwrap();
        assert!((values[4] - 99.0).abs() < 1e-6);
        assert!((values[5] - 88.0).abs() < 1e-6);
        assert!((values[10] - 99.0).abs() < 1e-6);
        assert!((values[11] - 88.0).abs() < 1e-6);
    }

    #[test]
    fn rope_multi_head_applies_same_rotation_per_head() {
        let mut buf = [
            0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        apply_neox_rope_in_place(&mut buf, 2, 2, 4, 4, 1.0, 1.0, 0).unwrap();
        assert_eq!(&buf[8..12], &buf[12..16]);
    }

    #[test]
    fn rope_position_offset_matches_sequential() {
        // RoPE at offset=2 for 1 token should match position 2 from a 3-token batch.
        let mut batch = vec![0.0_f32; 3 * 4]; // 3 tokens, hd=4
        batch[2 * 4] = 1.0;
        batch[2 * 4 + 1] = 2.0;
        batch[2 * 4 + 2] = 3.0;
        batch[2 * 4 + 3] = 4.0;
        apply_neox_rope_in_place(&mut batch, 3, 1, 4, 4, 10000.0, 1.0, 0).unwrap();

        let mut single = vec![1.0, 2.0, 3.0, 4.0];
        apply_neox_rope_in_place(&mut single, 1, 1, 4, 4, 10000.0, 1.0, 2).unwrap();

        for (i, (a, b)) in single.iter().zip(&batch[8..12]).enumerate() {
            assert!((a - b).abs() < 1e-6, "dim {i}: offset={a} vs batch={b}");
        }
    }

    #[test]
    fn full_attention_prefill_then_decode_matches_full_reprocess() {
        // Build a small deterministic plan: 2 heads, 1 kv_head (GQA), hd=4.
        let head_count = 2;
        let kv_head_count = 1;
        let hd = 4;
        let query_features = head_count * hd; // 8
        let kv_features = kv_head_count * hd; // 4
        let hidden = 6;

        // Q weight: hidden → query_features*2 (Q+Gate interleaved)
        let q_weight: Vec<f32> = (0..hidden * query_features * 2)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let k_weight: Vec<f32> = (0..hidden * kv_features)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.08)
            .collect();
        let v_weight: Vec<f32> = (0..hidden * kv_features)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let output_weight: Vec<f32> = (0..query_features * hidden)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
            .collect();
        let q_norm = vec![1.0_f32; hd];
        let k_norm = vec![1.0_f32; hd];

        let plan = super::super::plan::Qwen35FullAttentionLayerPlan {
            norm_values: vec![1.0; hidden],
            q_norm_values: q_norm,
            k_norm_values: k_norm,
            q_weight_values: q_weight,
            k_weight_values: k_weight,
            v_weight_values: v_weight,
            output_weight_values: output_weight,
            head_count,
            kv_head_count,
            head_dimension: hd,
            attention_scale: 1.0 / (hd as f32).sqrt(),
            rope_n_dims: hd,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
        };

        // 3-token prompt + 1 decode token.
        let prompt: Vec<f32> = (0..3 * hidden).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let new_token: Vec<f32> = (0..hidden).map(|i| (i as f32 + 50.0) * 0.05).collect();

        // Full reprocess: 4 tokens at once.
        let full_input: Vec<f32> = prompt.iter().chain(new_token.iter()).copied().collect();
        let full_output = qwen35_full_attention_inference(&plan, &full_input, 4, 1e-5).unwrap();
        let expected = &full_output[3 * hidden..4 * hidden];

        // Prefill 3 tokens, then decode 1.
        let mut state = Qwen35FullAttentionState::new(4, kv_head_count, hd).unwrap();
        let _prefill_out =
            qwen35_full_attention_prefill(&plan, &prompt, 3, 1e-5, &mut state).unwrap();
        let decode_out =
            qwen35_full_attention_decode_step(&plan, &new_token, 1e-5, &mut state).unwrap();

        for (i, (a, b)) in decode_out.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "feature {i}: decode={a} vs full={b}, diff={}",
                (a - b).abs()
            );
        }
    }
}
