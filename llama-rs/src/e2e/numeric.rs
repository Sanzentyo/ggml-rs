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

/// Convert a single f32 to IEEE 754 half-precision (f16) bits.
pub(super) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;

    if exponent == 0xFF {
        return if mantissa == 0 {
            sign | 0x7C00 // Inf
        } else {
            sign | 0x7E00 // NaN (quiet)
        };
    }

    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7C00; // overflow → Inf
    }
    if unbiased < -24 {
        return sign; // too small → zero
    }
    if unbiased < -14 {
        let shift = (-14 - unbiased) as u32;
        let m = (mantissa | 0x80_0000) >> (shift + 13);
        return sign | m as u16;
    }
    let exp16 = ((unbiased + 15) as u16) << 10;
    let man16 = (mantissa >> 13) as u16;
    sign | exp16 | man16
}

/// Build a causal attention mask as f16 little-endian bytes.
///
/// Layout: `[Tkv, T]` (row = query token, col = key token).
/// Allowed positions → 0.0 (f16), blocked positions → -Inf (f16 0xFC00).
///
/// Returns an error if `seq_len² × 2` overflows `usize`.
pub(super) fn build_causal_mask_f16_bytes(seq_len: usize) -> Result<Vec<u8>, super::E2eError> {
    let zero_f16 = f32_to_f16_bits(0.0_f32).to_le_bytes();
    let neg_inf_f16 = f32_to_f16_bits(f32::NEG_INFINITY).to_le_bytes();
    let total_bytes = seq_len
        .checked_mul(seq_len)
        .and_then(|n| n.checked_mul(2))
        .ok_or(super::E2eError::MemorySizeOverflow)?;
    let mut buf = vec![0u8; total_bytes];
    for row in 0..seq_len {
        for col in 0..seq_len {
            let bytes = if col > row { &neg_inf_f16 } else { &zero_f16 };
            let offset = (col + row * seq_len) * 2;
            buf[offset..offset + 2].copy_from_slice(bytes);
        }
    }
    Ok(buf)
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
