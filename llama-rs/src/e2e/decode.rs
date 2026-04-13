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
