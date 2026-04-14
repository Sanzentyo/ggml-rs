use super::error::E2eError;
use super::numeric::checked_mul;
use crate::model::GgufModel;

pub(super) fn decode_norm_tensor(
    model: &GgufModel,
    tensor_name: &str,
    hidden_features: usize,
    role: &'static str,
) -> Result<Vec<f32>, E2eError> {
    let values = model
        .tensor_values::<f32>(tensor_name)
        .map_err(|source| E2eError::model("GgufModel::tensor_values(norm)", source))?;
    if values.len() != hidden_features {
        return Err(E2eError::NormWeightLengthMismatch {
            tensor_name: format!("{role}:{tensor_name}"),
            expected: hidden_features,
            actual: values.len(),
        });
    }
    Ok(values)
}

pub(super) fn decode_exact_tensor(
    model: &GgufModel,
    tensor_name: &str,
    expected_len: usize,
    role: &'static str,
) -> Result<Vec<f32>, E2eError> {
    let values = model
        .tensor_values::<f32>(tensor_name)
        .map_err(|source| E2eError::model("GgufModel::tensor_values(exact)", source))?;
    if values.len() != expected_len {
        return Err(E2eError::NormWeightLengthMismatch {
            tensor_name: format!("{role}:{tensor_name}"),
            expected: expected_len,
            actual: values.len(),
        });
    }
    Ok(values)
}

pub(super) fn decode_matrix_tensor(
    model: &GgufModel,
    tensor_name: &str,
    input_features: usize,
    output_features: usize,
    role: &'static str,
) -> Result<Vec<f32>, E2eError> {
    let values = model
        .tensor_values::<f32>(tensor_name)
        .map_err(|source| E2eError::model("GgufModel::tensor_values(matrix)", source))?;
    let expected = checked_mul(input_features, output_features)?;
    if values.len() != expected {
        return Err(E2eError::OutputWeightLengthMismatch {
            tensor_name: format!("{role}:{tensor_name}"),
            expected,
            actual: values.len(),
        });
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a stub model with one tensor of the given f32 element count.
    fn stub_model(name: &str, f32_count: usize) -> GgufModel {
        GgufModel::stub(&[(name, f32_count * 4)], vec![])
    }

    // ── decode_norm_tensor ──────────────────────────────────────
    #[test]
    fn norm_tensor_valid_length() {
        let model = stub_model("norm.weight", 4);
        let v = decode_norm_tensor(&model, "norm.weight", 4, "test").unwrap();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn norm_tensor_length_mismatch() {
        let model = stub_model("norm.weight", 4);
        let err = decode_norm_tensor(&model, "norm.weight", 8, "test").unwrap_err();
        assert!(
            matches!(
                err,
                E2eError::NormWeightLengthMismatch {
                    expected: 8,
                    actual: 4,
                    ..
                }
            ),
            "expected NormWeightLengthMismatch, got {err:?}"
        );
    }

    #[test]
    fn norm_tensor_missing() {
        let model = stub_model("other", 4);
        let err = decode_norm_tensor(&model, "norm.weight", 4, "test").unwrap_err();
        assert!(
            matches!(err, E2eError::Model { .. }),
            "expected Model error, got {err:?}"
        );
    }

    // ── decode_exact_tensor ─────────────────────────────────────
    #[test]
    fn exact_tensor_valid() {
        let model = stub_model("bias", 6);
        let v = decode_exact_tensor(&model, "bias", 6, "test").unwrap();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn exact_tensor_length_mismatch() {
        let model = stub_model("bias", 6);
        let err = decode_exact_tensor(&model, "bias", 3, "test").unwrap_err();
        assert!(
            matches!(
                err,
                E2eError::NormWeightLengthMismatch {
                    expected: 3,
                    actual: 6,
                    ..
                }
            ),
            "expected NormWeightLengthMismatch, got {err:?}"
        );
    }

    // ── decode_matrix_tensor ────────────────────────────────────
    #[test]
    fn matrix_tensor_valid() {
        // 3×4 = 12 elements
        let model = stub_model("weight", 12);
        let v = decode_matrix_tensor(&model, "weight", 3, 4, "test").unwrap();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn matrix_tensor_dimension_mismatch() {
        // Tensor has 12 elements but we expect 3×5 = 15
        let model = stub_model("weight", 12);
        let err = decode_matrix_tensor(&model, "weight", 3, 5, "test").unwrap_err();
        assert!(
            matches!(
                err,
                E2eError::OutputWeightLengthMismatch {
                    expected: 15,
                    actual: 12,
                    ..
                }
            ),
            "expected OutputWeightLengthMismatch, got {err:?}"
        );
    }

    #[test]
    fn matrix_tensor_overflow_dimensions() {
        let model = stub_model("weight", 4);
        let err = decode_matrix_tensor(&model, "weight", usize::MAX, 2, "test").unwrap_err();
        assert!(
            matches!(err, E2eError::MemorySizeOverflow),
            "expected MemorySizeOverflow, got {err:?}"
        );
    }
}
