//! Embedding-oriented tensor summary helpers.

use crate::model::{GgufModel, ModelError};
use std::error::Error as StdError;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Basic descriptive stats computed from an embedding tensor.
pub struct EmbeddingStats {
    pub len: usize,
    pub mean: f32,
    pub l2_norm: f32,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug)]
/// Errors surfaced by embedding-summary routines.
pub enum EmbeddingError {
    Model {
        context: &'static str,
        source: ModelError,
    },
    EmptyTensor {
        tensor_name: String,
    },
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model { context, source } => write!(f, "{context}: {source}"),
            Self::EmptyTensor { tensor_name } => write!(f, "tensor is empty: {tensor_name}"),
        }
    }
}

impl StdError for EmbeddingError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Model { source, .. } => Some(source),
            Self::EmptyTensor { .. } => None,
        }
    }
}

impl EmbeddingError {
    fn model(context: &'static str, source: ModelError) -> Self {
        Self::Model { context, source }
    }
}

pub fn summarize_embedding_tensor(
    model: &GgufModel,
    tensor_name: &str,
) -> Result<EmbeddingStats, EmbeddingError> {
    let mut values = Vec::new();
    model
        .decode_tensor_f32_into(tensor_name, &mut values)
        .map_err(|source| EmbeddingError::model("GgufModel::decode_tensor_f32_into", source))?;
    if values.is_empty() {
        return Err(EmbeddingError::EmptyTensor {
            tensor_name: tensor_name.to_string(),
        });
    }

    let len = values.len();
    // Use f64 accumulation for numerically stable summary stats.
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &value in &values {
        sum += f64::from(value);
        sum_sq += f64::from(value) * f64::from(value);
        min = min.min(value);
        max = max.max(value);
    }

    let mean = (sum / len as f64) as f32;
    let l2_norm = (sum_sq as f32).sqrt();

    Ok(EmbeddingStats {
        len,
        mean,
        l2_norm,
        min,
        max,
    })
}
