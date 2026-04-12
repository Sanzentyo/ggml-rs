use super::error::E2eError;
use super::numeric::{checked_mul, dot, sigmoid_scalar, softmax_prefix};
use super::plan::Qwen35FullAttentionLayerPlan;
use super::tensor_ops::{head_slice, head_slice_mut, per_head_rms_norm, project_sequence};

pub(super) fn qwen35_full_attention_inference(
    attention: &Qwen35FullAttentionLayerPlan,
    input: &[f32],
    sequence_length: usize,
    rms_norm_eps: f32,
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

    let mut q_values = vec![0.0_f32; checked_mul(sequence_length, query_features)?];
    let mut q_gate = vec![0.0_f32; checked_mul(sequence_length, query_features)?];
    let hd = attention.head_dimension;
    // ggml layout per token: [Q_h0(D), G_h0(D), Q_h1(D), G_h1(D), ...]
    let per_token_qg_features = checked_mul(checked_mul(attention.head_count, hd)?, 2)?;
    let expected_qg_len = checked_mul(sequence_length, per_token_qg_features)?;
    if q_full.len() != expected_qg_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_qg_len,
            actual: q_full.len(),
        });
    }
    for token in 0..sequence_length {
        let token_base = token * per_token_qg_features;
        let dst_token_base = token * query_features;
        for head in 0..attention.head_count {
            for dim in 0..hd {
                let src_q = token_base + head * 2 * hd + dim;
                let src_g = token_base + head * 2 * hd + hd + dim;
                let dst = dst_token_base + head * hd + dim;
                q_values[dst] = q_full[src_q];
                q_gate[dst] = q_full[src_g];
            }
        }
    }

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
    )?;
    apply_neox_rope_in_place(
        &mut k_values,
        sequence_length,
        attention.kv_head_count,
        attention.head_dimension,
        attention.rope_n_dims,
        attention.rope_freq_base,
        attention.rope_freq_scale,
    )?;

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

/// Apply NeoX-style rotary position embedding in-place.
///
/// For each token at position `pos`, rotates dimension pairs `(x[k], x[k + n_rot/2])`
/// for `k` in `0..n_rot/2` using angle `theta_k = pos * freq_base^(-2k / n_rot)`.
/// Dimensions beyond `n_rot` are left unchanged.
///
/// The RoPE cache (cos/sin per position per dimension pair) is computed once and
/// reused across all heads.
pub(super) fn apply_neox_rope_in_place(
    values: &mut [f32],
    sequence_length: usize,
    head_count: usize,
    head_dimension: usize,
    n_rot: usize,
    freq_base: f32,
    freq_scale: f32,
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
        let mut theta = pos as f32;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35_full_attention_qgate_split_is_head_interleaved() {
        let head_count = 2;
        let hd = 3;
        let query_features = head_count * hd;
        let per_token_qg = query_features * 2;
        let sequence_length = 1;

        let q_full: Vec<f32> = vec![
            1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0,
        ];
        assert_eq!(q_full.len(), per_token_qg);

        let mut q_values = vec![0.0_f32; query_features];
        let mut q_gate = vec![0.0_f32; query_features];
        for token in 0..sequence_length {
            let token_base = token * per_token_qg;
            let dst_token_base = token * query_features;
            for head in 0..head_count {
                for dim in 0..hd {
                    let src_q = token_base + head * 2 * hd + dim;
                    let src_g = token_base + head * 2 * hd + hd + dim;
                    let dst = dst_token_base + head * hd + dim;
                    q_values[dst] = q_full[src_q];
                    q_gate[dst] = q_full[src_g];
                }
            }
        }

        assert_eq!(q_values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(q_gate, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn qwen35_full_attention_qgate_split_multi_token() {
        let head_count = 2;
        let hd = 2;
        let query_features = head_count * hd;
        let per_token_qg = query_features * 2;
        let sequence_length = 2;

        let q_full: Vec<f32> = vec![
            1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0, 5.0, 6.0, 50.0, 60.0, 7.0, 8.0, 70.0, 80.0,
        ];
        assert_eq!(q_full.len(), sequence_length * per_token_qg);

        let mut q_values = vec![0.0_f32; sequence_length * query_features];
        let mut q_gate = vec![0.0_f32; sequence_length * query_features];
        for token in 0..sequence_length {
            let token_base = token * per_token_qg;
            let dst_token_base = token * query_features;
            for head in 0..head_count {
                for dim in 0..hd {
                    let src_q = token_base + head * 2 * hd + dim;
                    let src_g = token_base + head * 2 * hd + hd + dim;
                    let dst = dst_token_base + head * hd + dim;
                    q_values[dst] = q_full[src_q];
                    q_gate[dst] = q_full[src_g];
                }
            }
        }

        assert_eq!(q_values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(q_gate, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn rope_identity_at_position_zero() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let original = values.clone();
        apply_neox_rope_in_place(&mut values, 1, 1, 4, 4, 10000.0, 1.0).unwrap();
        for (a, b) in values.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
        }
    }

    #[test]
    fn rope_rotates_at_nonzero_position() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0];
        apply_neox_rope_in_place(&mut values, 2, 1, 4, 4, 1.0, 1.0).unwrap();

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
        apply_neox_rope_in_place(&mut values, 2, 1, 6, 4, 10000.0, 1.0).unwrap();
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
        apply_neox_rope_in_place(&mut buf, 2, 2, 4, 4, 1.0, 1.0).unwrap();
        assert_eq!(&buf[8..12], &buf[12..16]);
    }
}
