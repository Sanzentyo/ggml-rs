use super::super::error::E2eError;
use super::super::numeric::checked_mul;

pub(in crate::e2e) fn add_in_place(
    accumulator: &mut [f32],
    addend: &[f32],
) -> Result<(), E2eError> {
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

pub(in crate::e2e) fn project_sequence(
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
            dst_row[feature] = super::super::numeric::dot(input_row, weights_row);
        }
    }
    Ok(output)
}

pub(in crate::e2e) fn head_slice(
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

pub(in crate::e2e) fn head_slice_mut(
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

pub(in crate::e2e) fn gather_embeddings(
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
        let token_index = super::super::numeric::validate_token_id(token_id, vocab_size)?;
        let src_offset = token_index * hidden_features;
        dst_slice.copy_from_slice(&embedding_values[src_offset..src_offset + hidden_features]);
    }
    Ok(output)
}
