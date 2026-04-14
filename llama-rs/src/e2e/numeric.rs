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

    // ── checked_mul ──────────────────────────────────────────────

    #[test]
    fn checked_mul_success() {
        assert_eq!(checked_mul(3, 4).unwrap(), 12);
        assert_eq!(checked_mul(0, 999).unwrap(), 0);
    }

    #[test]
    fn checked_mul_overflow() {
        let err = checked_mul(usize::MAX, 2).unwrap_err();
        assert!(
            matches!(err, E2eError::MemorySizeOverflow),
            "expected MemorySizeOverflow, got {err:?}"
        );
    }

    // ── validate_token_id ────────────────────────────────────────

    #[test]
    fn validate_token_id_valid() {
        assert_eq!(validate_token_id(0, 10).unwrap(), 0);
        assert_eq!(validate_token_id(5, 10).unwrap(), 5);
        assert_eq!(validate_token_id(9, 10).unwrap(), 9);
    }

    #[test]
    fn validate_token_id_negative() {
        let err = validate_token_id(-1, 10).unwrap_err();
        assert!(matches!(
            err,
            E2eError::InvalidTokenId {
                token_id: -1,
                vocab_size: 10
            }
        ));
    }

    #[test]
    fn validate_token_id_out_of_range() {
        assert!(validate_token_id(10, 10).is_err());
        assert!(validate_token_id(100, 10).is_err());
    }

    // ── value_to_i32 ─────────────────────────────────────────────

    #[test]
    fn converts_numeric_gguf_values_to_i32() {
        assert_eq!(value_to_i32(&GgufValue::U8(255)), Some(255));
        assert_eq!(value_to_i32(&GgufValue::I8(-128)), Some(-128));
        assert_eq!(value_to_i32(&GgufValue::U16(60000)), Some(60000));
        assert_eq!(value_to_i32(&GgufValue::I16(-30000)), Some(-30000));
        assert_eq!(value_to_i32(&GgufValue::U32(12)), Some(12));
        assert_eq!(value_to_i32(&GgufValue::I32(-5)), Some(-5));
        assert_eq!(value_to_i32(&GgufValue::F32(42.0)), Some(42));
        assert_eq!(value_to_i32(&GgufValue::String("x".to_string())), None);
    }

    #[test]
    fn value_to_i32_overflow_u32() {
        assert_eq!(value_to_i32(&GgufValue::U32(u32::MAX)), None);
    }

    #[test]
    fn value_to_i32_fractional_f32_returns_none() {
        assert_eq!(value_to_i32(&GgufValue::F32(1.5)), None);
        assert_eq!(value_to_i32(&GgufValue::F64(0.1)), None);
    }

    #[test]
    fn value_to_i32_integral_float_out_of_range() {
        assert_eq!(value_to_i32(&GgufValue::F32(3_000_000_000.0)), None);
        assert_eq!(value_to_i32(&GgufValue::F64(3_000_000_000.0)), None);
    }

    // ── dot ──────────────────────────────────────────────────────

    #[test]
    fn dot_product_basic() {
        let result = dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn dot_product_empty() {
        assert_eq!(dot(&[], &[]), 0.0);
    }

    // ── softmax_prefix ───────────────────────────────────────────

    #[test]
    fn softmax_prefix_uniform_scores() {
        let result = softmax_prefix(&[1.0, 1.0, 1.0], 3);
        for p in &result {
            assert!((p - 1.0 / 3.0).abs() < 1e-6, "expected ~0.333, got {p}");
        }
    }

    #[test]
    fn softmax_prefix_dominated_score() {
        let result = softmax_prefix(&[0.0, 0.0, 100.0], 3);
        assert!(result[2] > 0.999, "dominant score should get ~1.0");
        assert!(result[0] < 0.001);
    }

    #[test]
    fn softmax_prefix_uses_only_prefix() {
        // Tail element (1000.0) must be ignored when len=2
        let result = softmax_prefix(&[0.0, 0.0, 1000.0], 2);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn softmax_prefix_len_zero_returns_empty() {
        let result = softmax_prefix(&[1.0, 2.0], 0);
        assert!(result.is_empty());
    }

    // ── sigmoid / silu / softplus ────────────────────────────────

    #[test]
    fn sigmoid_known_values() {
        assert!((sigmoid_scalar(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid_scalar(100.0) > 0.999);
        assert!(sigmoid_scalar(-100.0) < 0.001);
    }

    #[test]
    fn silu_known_values() {
        assert!((silu_scalar(0.0)).abs() < 1e-6, "silu(0) = 0");
        // For large x, sigmoid(x) → 1, so silu(x) ≈ x
        let large = 10.0_f32;
        assert!((silu_scalar(large) - large).abs() < 0.01);
    }

    #[test]
    fn softplus_below_threshold() {
        let expected = (1.0_f32 + 1.0_f32.exp()).ln(); // ln(1+e) ≈ 1.3133
        assert!((softplus_scalar(1.0) - expected).abs() < 1e-6);
    }

    #[test]
    fn softplus_above_threshold_returns_identity() {
        assert_eq!(softplus_scalar(25.0), 25.0);
        assert_eq!(softplus_scalar(100.0), 100.0);
    }

    #[test]
    fn softplus_at_boundary() {
        // 20.0 is NOT above threshold (> 20.0 check), so it uses ln path
        let expected = (1.0_f32 + 20.0_f32.exp()).ln();
        assert!((softplus_scalar(20.0) - expected).abs() < 1e-4);
        // 20.001 should take the identity branch
        assert_eq!(softplus_scalar(20.001), 20.001);
    }

    // ── f32_to_f16_bits ──────────────────────────────────────────

    #[test]
    fn f16_converts_zero() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
    }

    #[test]
    fn f16_preserves_negative_zero() {
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
    }

    #[test]
    fn f16_converts_one() {
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
    }

    #[test]
    fn f16_converts_negative_one() {
        assert_eq!(f32_to_f16_bits(-1.0), 0xBC00);
    }

    #[test]
    fn f16_converts_infinity() {
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7C00);
    }

    #[test]
    fn f16_converts_negative_infinity() {
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xFC00);
    }

    #[test]
    fn f16_converts_nan() {
        let bits = f32_to_f16_bits(f32::NAN);
        // NaN: exponent all 1s, mantissa nonzero — quiet NaN → 0x7E00
        assert_eq!(bits & 0x7FFF, 0x7E00);
    }

    #[test]
    fn f16_overflow_to_inf() {
        // 100_000.0 exceeds f16 max (~65504), should clamp to infinity
        assert_eq!(f32_to_f16_bits(100_000.0), 0x7C00);
    }

    #[test]
    fn f16_subnormal_smallest_positive() {
        // 2^-24 is the smallest positive f16 subnormal (0x0001)
        let val = 2.0_f32.powi(-24);
        assert_eq!(f32_to_f16_bits(val), 0x0001);
    }

    #[test]
    fn f16_below_subnormal_flushes_to_zero() {
        // Below 2^-25 should flush to zero
        let val = 2.0_f32.powi(-25);
        assert_eq!(f32_to_f16_bits(val), 0x0000);
    }

    // ── build_causal_mask_f16_bytes ──────────────────────────────

    #[test]
    fn causal_mask_seq_1() {
        let mask = build_causal_mask_f16_bytes(1).unwrap();
        // 1×1 mask: only position (0,0), which is allowed → 0.0 in f16
        assert_eq!(mask, vec![0x00, 0x00]);
    }

    #[test]
    fn causal_mask_seq_3_lower_triangular() {
        let mask = build_causal_mask_f16_bytes(3).unwrap();
        assert_eq!(mask.len(), 3 * 3 * 2);
        let zero = f32_to_f16_bits(0.0).to_le_bytes();
        let neg_inf = f32_to_f16_bits(f32::NEG_INFINITY).to_le_bytes();

        for row in 0..3_usize {
            for col in 0..3_usize {
                let offset = (col + row * 3) * 2;
                let cell = &mask[offset..offset + 2];
                let expected = if col > row { &neg_inf[..] } else { &zero[..] };
                assert_eq!(
                    cell, expected,
                    "mask[{row},{col}]: expected {:?}, got {:?}",
                    expected, cell
                );
            }
        }
    }

    #[test]
    fn causal_mask_overflow() {
        let err = build_causal_mask_f16_bytes(usize::MAX).unwrap_err();
        assert!(matches!(err, E2eError::MemorySizeOverflow));
    }
}
