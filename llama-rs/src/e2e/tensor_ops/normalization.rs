use super::super::error::E2eError;
use super::super::numeric::checked_mul;

pub(in crate::e2e) fn rms_norm_with_weight(
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

pub(in crate::e2e) fn rms_norm_single(
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, E2eError> {
    let mut output = vec![0.0_f32; input.len()];
    rms_norm_single_into(input, weight, eps, &mut output)?;
    Ok(output)
}

/// In-place variant of [`rms_norm_single`] that writes into a pre-allocated
/// destination buffer, avoiding a heap allocation per call.
pub(in crate::e2e) fn rms_norm_single_into(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    dst: &mut [f32],
) -> Result<(), E2eError> {
    if input.len() != weight.len() {
        return Err(E2eError::BufferLengthMismatch {
            expected: weight.len(),
            actual: input.len(),
        });
    }
    if dst.len() < input.len() {
        return Err(E2eError::BufferLengthMismatch {
            expected: input.len(),
            actual: dst.len(),
        });
    }
    let mean_square = input
        .iter()
        .copied()
        .map(|value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        / input.len() as f64;
    let inv_rms = 1.0_f32 / ((mean_square as f32) + eps).sqrt();
    dst.iter_mut()
        .zip(input.iter().copied().zip(weight.iter().copied()))
        .for_each(|(d, (value, scale))| *d = value * inv_rms * scale);
    Ok(())
}

/// In-place RMS normalization that reads and writes the same buffer.
fn rms_norm_single_in_place(data: &mut [f32], weight: &[f32], eps: f32) -> Result<(), E2eError> {
    if data.len() != weight.len() {
        return Err(E2eError::BufferLengthMismatch {
            expected: weight.len(),
            actual: data.len(),
        });
    }
    let mean_square = data
        .iter()
        .copied()
        .map(|value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        / data.len() as f64;
    let inv_rms = 1.0_f32 / ((mean_square as f32) + eps).sqrt();
    data.iter_mut()
        .zip(weight.iter().copied())
        .for_each(|(d, scale)| *d *= inv_rms * scale);
    Ok(())
}

pub(in crate::e2e) fn per_head_rms_norm(
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
            rms_norm_single_in_place(head_slice, weight, eps)?;
        }
    }
    Ok(output)
}

pub(in crate::e2e) fn per_head_l2_norm(
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
