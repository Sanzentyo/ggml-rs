use super::error::E2eError;
use super::numeric::checked_mul;

pub(super) fn rms_norm_with_weight(
    input: &[f32],
    hidden_features: usize,
    sequence_length: usize,
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, E2eError> {
    if weight.len() != hidden_features {
        return Err(E2eError::BufferLengthMismatch {
            expected: hidden_features,
            actual: weight.len(),
        });
    }
    let expected_input_len = checked_mul(hidden_features, sequence_length)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }

    let mut output = vec![0.0_f32; input.len()];
    for (src, dst) in input
        .chunks_exact(hidden_features)
        .zip(output.chunks_exact_mut(hidden_features))
    {
        let mean_square = src
            .iter()
            .copied()
            .map(|value| f64::from(value) * f64::from(value))
            .sum::<f64>()
            / hidden_features as f64;
        let inv_rms = 1.0_f32 / ((mean_square as f32) + eps).sqrt();
        dst.iter_mut()
            .zip(src.iter().zip(weight.iter()))
            .for_each(|(d, (&s, &w))| *d = s * inv_rms * w);
    }
    Ok(output)
}

pub(super) fn rms_norm_single(
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, E2eError> {
    if input.len() != weight.len() {
        return Err(E2eError::BufferLengthMismatch {
            expected: weight.len(),
            actual: input.len(),
        });
    }
    let mean_square = input
        .iter()
        .copied()
        .map(|value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        / input.len() as f64;
    let inv_rms = 1.0_f32 / ((mean_square as f32) + eps).sqrt();
    Ok(input
        .iter()
        .copied()
        .zip(weight.iter().copied())
        .map(|(value, scale)| value * inv_rms * scale)
        .collect())
}

pub(super) fn add_in_place(accumulator: &mut [f32], addend: &[f32]) -> Result<(), E2eError> {
    if accumulator.len() != addend.len() {
        return Err(E2eError::BufferLengthMismatch {
            expected: accumulator.len(),
            actual: addend.len(),
        });
    }
    for (lhs, rhs) in accumulator.iter_mut().zip(addend.iter().copied()) {
        *lhs += rhs;
    }
    Ok(())
}

pub(super) fn project_sequence(
    input: &[f32],
    sequence_length: usize,
    input_features: usize,
    output_features: usize,
    weight: &[f32],
) -> Result<Vec<f32>, E2eError> {
    let expected_input_len = checked_mul(sequence_length, input_features)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }
    let expected_weight_len = checked_mul(input_features, output_features)?;
    if weight.len() != expected_weight_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_weight_len,
            actual: weight.len(),
        });
    }

    let mut output = vec![0.0_f32; checked_mul(sequence_length, output_features)?];
    for (input_row, dst_row) in input
        .chunks_exact(input_features)
        .zip(output.chunks_exact_mut(output_features))
    {
        for (feature, weights_row) in weight.chunks_exact(input_features).enumerate() {
            dst_row[feature] = super::numeric::dot(input_row, weights_row);
        }
    }
    Ok(output)
}

pub(super) fn head_slice(
    values: &[f32],
    token: usize,
    head: usize,
    head_count: usize,
    head_dimension: usize,
) -> &[f32] {
    let token_offset = token * head_count * head_dimension;
    let head_offset = token_offset + head * head_dimension;
    &values[head_offset..head_offset + head_dimension]
}

pub(super) fn head_slice_mut(
    values: &mut [f32],
    token: usize,
    head: usize,
    head_count: usize,
    head_dimension: usize,
) -> &mut [f32] {
    let token_offset = token * head_count * head_dimension;
    let head_offset = token_offset + head * head_dimension;
    &mut values[head_offset..head_offset + head_dimension]
}

pub(super) fn per_head_rms_norm(
    input: &[f32],
    _sequence_length: usize,
    head_count: usize,
    head_dimension: usize,
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, E2eError> {
    if weight.len() != head_dimension {
        return Err(E2eError::BufferLengthMismatch {
            expected: head_dimension,
            actual: weight.len(),
        });
    }
    let token_features = checked_mul(head_count, head_dimension)?;
    let mut output = input.to_vec();
    for token_slice in output.chunks_exact_mut(token_features) {
        for head_slice in token_slice.chunks_exact_mut(head_dimension) {
            let normalized = rms_norm_single(head_slice, weight, eps)?;
            head_slice.copy_from_slice(&normalized);
        }
    }
    Ok(output)
}

pub(super) fn per_head_l2_norm(
    input: &[f32],
    _sequence_length: usize,
    head_count: usize,
    head_dimension: usize,
    eps: f32,
) -> Result<Vec<f32>, E2eError> {
    let token_features = checked_mul(head_count, head_dimension)?;
    let mut output = input.to_vec();
    for token_slice in output.chunks_exact_mut(token_features) {
        for head_slice in token_slice.chunks_exact_mut(head_dimension) {
            let norm = head_slice.iter().map(|v| v * v).sum::<f32>();
            let inv = 1.0_f32 / norm.sqrt().max(eps);
            head_slice.iter_mut().for_each(|v| *v *= inv);
        }
    }
    Ok(output)
}

pub(super) fn gather_embeddings(
    embedding_values: &[f32],
    hidden_features: usize,
    vocab_size: usize,
    token_ids: &[i32],
) -> Result<Vec<f32>, E2eError> {
    let expected_embedding_len = checked_mul(hidden_features, vocab_size)?;
    if embedding_values.len() != expected_embedding_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_embedding_len,
            actual: embedding_values.len(),
        });
    }

    let mut output = vec![0.0_f32; checked_mul(hidden_features, token_ids.len())?];
    for (dst_slice, &token_id) in output
        .chunks_exact_mut(hidden_features)
        .zip(token_ids.iter())
    {
        let token_index = super::numeric::validate_token_id(token_id, vocab_size)?;
        let src_offset = token_index * hidden_features;
        dst_slice.copy_from_slice(&embedding_values[src_offset..src_offset + hidden_features]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_applies_weight_per_position() {
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32, 0.25];
        let output =
            rms_norm_with_weight(&input, 2, 2, &weight, 1e-5).expect("rms norm should succeed");
        assert_eq!(output.len(), input.len());
        assert!(output[0].is_finite());
        assert!(output[1].is_finite());
        assert!(output[2].is_finite());
        assert!(output[3].is_finite());
        assert!(output[0].abs() > output[1].abs());
    }

    #[test]
    fn rms_norm_eps_changes_scaled_output() {
        let input = vec![1.0_f32, 2.0];
        let weight = vec![1.0_f32, 1.0];
        let loose = rms_norm_with_weight(&input, 2, 1, &weight, 1e-5).expect("rms norm");
        let tight = rms_norm_with_weight(&input, 2, 1, &weight, 1e-6).expect("rms norm");
        assert_ne!(loose, tight);
    }
}
