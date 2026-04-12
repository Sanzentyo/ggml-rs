use super::error::E2eError;
use ggml_rs::GgufValue;

pub(super) fn checked_mul(lhs: usize, rhs: usize) -> Result<usize, E2eError> {
    lhs.checked_mul(rhs).ok_or(E2eError::MemorySizeOverflow)
}

pub(super) fn validate_token_id(token_id: i32, vocab_size: usize) -> Result<usize, E2eError> {
    if token_id < 0 {
        return Err(E2eError::InvalidTokenId {
            token_id,
            vocab_size,
        });
    }
    let token_index = usize::try_from(token_id).map_err(|_| E2eError::InvalidTokenId {
        token_id,
        vocab_size,
    })?;
    if token_index >= vocab_size {
        return Err(E2eError::InvalidTokenId {
            token_id,
            vocab_size,
        });
    }
    Ok(token_index)
}

pub(super) fn value_to_i32(value: &GgufValue) -> Option<i32> {
    match value {
        GgufValue::U8(value) => Some(i32::from(*value)),
        GgufValue::I8(value) => Some(i32::from(*value)),
        GgufValue::U16(value) => Some(i32::from(*value)),
        GgufValue::I16(value) => Some(i32::from(*value)),
        GgufValue::U32(value) => i32::try_from(*value).ok(),
        GgufValue::I32(value) => Some(*value),
        GgufValue::U64(value) => i32::try_from(*value).ok(),
        GgufValue::I64(value) => i32::try_from(*value).ok(),
        GgufValue::F32(value) if value.fract() == 0.0 => i32::try_from(*value as i64).ok(),
        GgufValue::F64(value) if value.fract() == 0.0 => i32::try_from(*value as i64).ok(),
        _ => None,
    }
}

pub(super) fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .copied()
        .zip(rhs.iter().copied())
        .fold(0.0_f32, |acc, (lhs, rhs)| acc + lhs * rhs)
}

pub(super) fn softmax_prefix(scores: &[f32], len: usize) -> Vec<f32> {
    let max_score = scores[..len]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exp_scores = Vec::with_capacity(len);
    let mut denom = 0.0_f32;
    for value in scores[..len].iter().copied() {
        let exp_value = (value - max_score).exp();
        denom += exp_value;
        exp_scores.push(exp_value);
    }
    exp_scores.into_iter().map(|value| value / denom).collect()
}

pub(super) fn sigmoid_scalar(value: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-value).exp())
}

pub(super) fn silu_scalar(value: f32) -> f32 {
    value * sigmoid_scalar(value)
}

pub(super) fn softplus_scalar(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0_f32 + value.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ggml_rs::GgufValue;

    #[test]
    fn converts_numeric_gguf_values_to_i32() {
        assert_eq!(value_to_i32(&GgufValue::U32(12)), Some(12));
        assert_eq!(value_to_i32(&GgufValue::I32(-5)), Some(-5));
        assert_eq!(value_to_i32(&GgufValue::F32(42.0)), Some(42));
        assert_eq!(value_to_i32(&GgufValue::String("x".to_string())), None);
    }
}
