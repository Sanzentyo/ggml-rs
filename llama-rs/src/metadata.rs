//! GGUF metadata parsing for architecture-aware inference configuration.

use crate::model::GgufModel;
use ggml_rs::GgufValue;
use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt;
use std::num::NonZeroUsize;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Model architecture declared in GGUF metadata.
pub enum ModelArchitecture {
    Llama,
    Other(String),
}

impl ModelArchitecture {
    fn from_string(value: &str) -> Self {
        match value {
            "llama" => Self::Llama,
            other => Self::Other(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Llama => "llama",
            Self::Other(value) => value,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Parsed metadata for transformer-like architectures that use
/// `{architecture}.embedding_length` style keys.
pub struct TransformerMetadata {
    architecture: ModelArchitecture,
    block_count: NonZeroUsize,
    embedding_length: NonZeroUsize,
    attention_head_count: NonZeroUsize,
    attention_head_count_kv: NonZeroUsize,
    feed_forward_length: Option<NonZeroUsize>,
    context_length: Option<NonZeroUsize>,
    rope_dimension_count: Option<NonZeroUsize>,
    rope_freq_base: f32,
    rope_freq_scale: f32,
}

impl TransformerMetadata {
    pub fn architecture(&self) -> &ModelArchitecture {
        &self.architecture
    }

    pub fn block_count(&self) -> usize {
        self.block_count.get()
    }

    pub fn embedding_length(&self) -> usize {
        self.embedding_length.get()
    }

    pub fn attention_head_count(&self) -> usize {
        self.attention_head_count.get()
    }

    pub fn attention_head_count_kv(&self) -> usize {
        self.attention_head_count_kv.get()
    }

    pub fn feed_forward_length(&self) -> Option<usize> {
        self.feed_forward_length.map(NonZeroUsize::get)
    }

    pub fn context_length(&self) -> Option<usize> {
        self.context_length.map(NonZeroUsize::get)
    }

    pub fn rope_dimension_count(&self) -> Option<usize> {
        self.rope_dimension_count.map(NonZeroUsize::get)
    }

    pub fn rope_freq_base(&self) -> f32 {
        self.rope_freq_base
    }

    pub fn rope_freq_scale(&self) -> f32 {
        self.rope_freq_scale
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Parsed metadata required by the current LLaMA inference helpers.
pub struct LlamaModelMetadata {
    block_count: NonZeroUsize,
    embedding_length: NonZeroUsize,
    attention_head_count: NonZeroUsize,
    attention_head_count_kv: NonZeroUsize,
    feed_forward_length: Option<NonZeroUsize>,
    context_length: Option<NonZeroUsize>,
    rope_dimension_count: Option<NonZeroUsize>,
    rope_freq_base: f32,
    rope_freq_scale: f32,
}

impl LlamaModelMetadata {
    pub fn block_count(self) -> usize {
        self.block_count.get()
    }

    pub fn embedding_length(self) -> usize {
        self.embedding_length.get()
    }

    pub fn attention_head_count(self) -> usize {
        self.attention_head_count.get()
    }

    pub fn attention_head_count_kv(self) -> usize {
        self.attention_head_count_kv.get()
    }

    pub fn feed_forward_length(self) -> Option<usize> {
        self.feed_forward_length.map(NonZeroUsize::get)
    }

    pub fn context_length(self) -> Option<usize> {
        self.context_length.map(NonZeroUsize::get)
    }

    pub fn rope_dimension_count(self) -> Option<usize> {
        self.rope_dimension_count.map(NonZeroUsize::get)
    }

    pub fn rope_freq_base(self) -> f32 {
        self.rope_freq_base
    }

    pub fn rope_freq_scale(self) -> f32 {
        self.rope_freq_scale
    }
}

impl From<TransformerMetadata> for LlamaModelMetadata {
    fn from(value: TransformerMetadata) -> Self {
        Self {
            block_count: value.block_count,
            embedding_length: value.embedding_length,
            attention_head_count: value.attention_head_count,
            attention_head_count_kv: value.attention_head_count_kv,
            feed_forward_length: value.feed_forward_length,
            context_length: value.context_length,
            rope_dimension_count: value.rope_dimension_count,
            rope_freq_base: value.rope_freq_base,
            rope_freq_scale: value.rope_freq_scale,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Architecture-specific metadata currently recognized by `llama-rs`.
pub enum ModelMetadata {
    Llama(LlamaModelMetadata),
    Unsupported { architecture: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors surfaced while parsing GGUF metadata.
pub enum MetadataError {
    MissingRequiredKey {
        key: String,
    },
    InvalidValueType {
        key: String,
        expected: &'static str,
        actual: &'static str,
    },
    InvalidPositiveValue {
        key: String,
        value: String,
    },
    InvalidFloatValue {
        key: String,
        value: String,
    },
    MissingArchitecture,
    UnsupportedArchitecture {
        architecture: String,
    },
    InvalidAttentionLayout {
        embedding_length: usize,
        attention_head_count: usize,
        attention_head_count_kv: usize,
    },
}

impl fmt::Display for MetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequiredKey { key } => {
                write!(f, "required GGUF metadata key is missing: {key}")
            }
            Self::InvalidValueType {
                key,
                expected,
                actual,
            } => write!(
                f,
                "invalid GGUF metadata type for key `{key}`: expected {expected}, got {actual}"
            ),
            Self::InvalidPositiveValue { key, value } => write!(
                f,
                "invalid non-positive GGUF metadata value for key `{key}`: {value}"
            ),
            Self::InvalidFloatValue { key, value } => {
                write!(
                    f,
                    "invalid GGUF float metadata value for key `{key}`: {value}"
                )
            }
            Self::MissingArchitecture => {
                write!(f, "GGUF metadata key `general.architecture` is missing")
            }
            Self::UnsupportedArchitecture { architecture } => {
                write!(f, "unsupported GGUF architecture: {architecture}")
            }
            Self::InvalidAttentionLayout {
                embedding_length,
                attention_head_count,
                attention_head_count_kv,
            } => write!(
                f,
                "invalid attention layout from metadata: embedding_length={embedding_length}, head_count={attention_head_count}, head_count_kv={attention_head_count_kv}"
            ),
        }
    }
}

impl StdError for MetadataError {}

/// Resolves transformer-style metadata using `general.architecture` as key prefix.
pub fn resolve_transformer_metadata(
    model: &GgufModel,
) -> Result<TransformerMetadata, MetadataError> {
    let entries = model
        .report()
        .kv_entries
        .iter()
        .map(|entry| (entry.key.as_str(), &entry.value));
    resolve_transformer_metadata_from_kv(entries)
}

/// Resolves transformer-style metadata from arbitrary GGUF KV entries.
pub fn resolve_transformer_metadata_from_kv<'a>(
    entries: impl IntoIterator<Item = (&'a str, &'a GgufValue)>,
) -> Result<TransformerMetadata, MetadataError> {
    let kv: HashMap<&str, &GgufValue> = entries.into_iter().collect();
    let architecture = parse_architecture(&kv)?;
    parse_transformer_metadata(&kv, architecture)
}

/// Resolves architecture metadata from a loaded GGUF model.
pub fn resolve_model_metadata(model: &GgufModel) -> Result<ModelMetadata, MetadataError> {
    let entries = model
        .report()
        .kv_entries
        .iter()
        .map(|entry| (entry.key.as_str(), &entry.value));
    resolve_model_metadata_from_kv(entries)
}

/// Resolves architecture metadata from arbitrary GGUF KV entries.
pub fn resolve_model_metadata_from_kv<'a>(
    entries: impl IntoIterator<Item = (&'a str, &'a GgufValue)>,
) -> Result<ModelMetadata, MetadataError> {
    let transformer = resolve_transformer_metadata_from_kv(entries)?;
    match transformer.architecture() {
        ModelArchitecture::Llama => Ok(ModelMetadata::Llama(transformer.into())),
        ModelArchitecture::Other(architecture) => Ok(ModelMetadata::Unsupported {
            architecture: architecture.clone(),
        }),
    }
}

/// Resolves and validates LLaMA metadata from a loaded GGUF model.
pub fn resolve_llama_metadata(model: &GgufModel) -> Result<LlamaModelMetadata, MetadataError> {
    let entries = model
        .report()
        .kv_entries
        .iter()
        .map(|entry| (entry.key.as_str(), &entry.value));
    resolve_llama_metadata_from_kv(entries)
}

/// Resolves and validates LLaMA metadata from arbitrary GGUF KV entries.
pub fn resolve_llama_metadata_from_kv<'a>(
    entries: impl IntoIterator<Item = (&'a str, &'a GgufValue)>,
) -> Result<LlamaModelMetadata, MetadataError> {
    let transformer = resolve_transformer_metadata_from_kv(entries)?;
    match transformer.architecture() {
        ModelArchitecture::Llama => Ok(transformer.into()),
        ModelArchitecture::Other(architecture) => Err(MetadataError::UnsupportedArchitecture {
            architecture: architecture.clone(),
        }),
    }
}

fn parse_architecture(kv: &HashMap<&str, &GgufValue>) -> Result<ModelArchitecture, MetadataError> {
    let value = kv
        .get("general.architecture")
        .copied()
        .ok_or(MetadataError::MissingArchitecture)?;
    let architecture = match value {
        GgufValue::String(value) => value.as_str(),
        other => {
            return Err(MetadataError::InvalidValueType {
                key: "general.architecture".to_string(),
                expected: "string",
                actual: other.type_name(),
            });
        }
    };
    Ok(ModelArchitecture::from_string(architecture))
}

fn parse_transformer_metadata(
    kv: &HashMap<&str, &GgufValue>,
    architecture: ModelArchitecture,
) -> Result<TransformerMetadata, MetadataError> {
    let prefix = architecture.as_str();
    let block_count_key = format!("{prefix}.block_count");
    let embedding_length_key = format!("{prefix}.embedding_length");
    let head_count_key = format!("{prefix}.attention.head_count");
    let head_count_kv_key = format!("{prefix}.attention.head_count_kv");
    let feed_forward_key = format!("{prefix}.feed_forward_length");
    let context_length_key = format!("{prefix}.context_length");
    let rope_dim_key = format!("{prefix}.rope.dimension_count");
    let rope_base_key = format!("{prefix}.rope.freq_base");
    let rope_scale_key = format!("{prefix}.rope.freq_scale");

    let block_count = required_non_zero_usize(kv, &block_count_key)?;
    let embedding_length = required_non_zero_usize(kv, &embedding_length_key)?;
    let attention_head_count = required_non_zero_usize(kv, &head_count_key)?;
    let attention_head_count_kv =
        optional_non_zero_usize(kv, &head_count_kv_key)?.unwrap_or(attention_head_count);
    let feed_forward_length = optional_non_zero_usize(kv, &feed_forward_key)?;
    let context_length = optional_non_zero_usize(kv, &context_length_key)?;
    let rope_dimension_count = optional_non_zero_usize(kv, &rope_dim_key)?;
    let rope_freq_base = optional_f32(kv, &rope_base_key)?.unwrap_or(10_000.0);
    let rope_freq_scale = optional_f32(kv, &rope_scale_key)?.unwrap_or(1.0);

    if embedding_length.get() % attention_head_count.get() != 0
        || attention_head_count.get() % attention_head_count_kv.get() != 0
    {
        return Err(MetadataError::InvalidAttentionLayout {
            embedding_length: embedding_length.get(),
            attention_head_count: attention_head_count.get(),
            attention_head_count_kv: attention_head_count_kv.get(),
        });
    }

    Ok(TransformerMetadata {
        architecture,
        block_count,
        embedding_length,
        attention_head_count,
        attention_head_count_kv,
        feed_forward_length,
        context_length,
        rope_dimension_count,
        rope_freq_base,
        rope_freq_scale,
    })
}

fn required_non_zero_usize(
    kv: &HashMap<&str, &GgufValue>,
    key: &str,
) -> Result<NonZeroUsize, MetadataError> {
    let value = kv
        .get(key)
        .copied()
        .ok_or_else(|| MetadataError::MissingRequiredKey {
            key: key.to_string(),
        })?;
    to_non_zero_usize(key, value)
}

fn optional_non_zero_usize(
    kv: &HashMap<&str, &GgufValue>,
    key: &str,
) -> Result<Option<NonZeroUsize>, MetadataError> {
    kv.get(key)
        .copied()
        .map(|value| to_non_zero_usize(key, value))
        .transpose()
}

fn optional_f32(kv: &HashMap<&str, &GgufValue>, key: &str) -> Result<Option<f32>, MetadataError> {
    kv.get(key)
        .copied()
        .map(|value| to_f32(key, value))
        .transpose()
}

fn to_non_zero_usize(key: &str, value: &GgufValue) -> Result<NonZeroUsize, MetadataError> {
    let usize_value = to_usize(key, value)?;
    NonZeroUsize::new(usize_value).ok_or_else(|| MetadataError::InvalidPositiveValue {
        key: key.to_string(),
        value: usize_value.to_string(),
    })
}

fn to_usize(key: &str, value: &GgufValue) -> Result<usize, MetadataError> {
    let number = match value {
        GgufValue::U8(value) => *value as u128,
        GgufValue::I8(value) if *value >= 0 => *value as u128,
        GgufValue::U16(value) => *value as u128,
        GgufValue::I16(value) if *value >= 0 => *value as u128,
        GgufValue::U32(value) => *value as u128,
        GgufValue::I32(value) if *value >= 0 => *value as u128,
        GgufValue::U64(value) => *value as u128,
        GgufValue::I64(value) if *value >= 0 => *value as u128,
        GgufValue::F32(value) if *value >= 0.0 && value.fract() == 0.0 => *value as u128,
        GgufValue::F64(value) if *value >= 0.0 && value.fract() == 0.0 => *value as u128,
        GgufValue::I8(value) => {
            return Err(MetadataError::InvalidPositiveValue {
                key: key.to_string(),
                value: value.to_string(),
            });
        }
        GgufValue::I16(value) => {
            return Err(MetadataError::InvalidPositiveValue {
                key: key.to_string(),
                value: value.to_string(),
            });
        }
        GgufValue::I32(value) => {
            return Err(MetadataError::InvalidPositiveValue {
                key: key.to_string(),
                value: value.to_string(),
            });
        }
        GgufValue::I64(value) => {
            return Err(MetadataError::InvalidPositiveValue {
                key: key.to_string(),
                value: value.to_string(),
            });
        }
        GgufValue::F32(value) => {
            return Err(MetadataError::InvalidPositiveValue {
                key: key.to_string(),
                value: value.to_string(),
            });
        }
        GgufValue::F64(value) => {
            return Err(MetadataError::InvalidPositiveValue {
                key: key.to_string(),
                value: value.to_string(),
            });
        }
        other => {
            return Err(MetadataError::InvalidValueType {
                key: key.to_string(),
                expected: "integer-like",
                actual: other.type_name(),
            });
        }
    };

    usize::try_from(number).map_err(|_| MetadataError::InvalidPositiveValue {
        key: key.to_string(),
        value: number.to_string(),
    })
}

fn to_f32(key: &str, value: &GgufValue) -> Result<f32, MetadataError> {
    let out = match value {
        GgufValue::F32(value) => *value,
        GgufValue::F64(value) => *value as f32,
        GgufValue::U8(value) => *value as f32,
        GgufValue::I8(value) => *value as f32,
        GgufValue::U16(value) => *value as f32,
        GgufValue::I16(value) => *value as f32,
        GgufValue::U32(value) => *value as f32,
        GgufValue::I32(value) => *value as f32,
        GgufValue::U64(value) => *value as f32,
        GgufValue::I64(value) => *value as f32,
        other => {
            return Err(MetadataError::InvalidValueType {
                key: key.to_string(),
                expected: "numeric",
                actual: other.type_name(),
            });
        }
    };
    if out.is_finite() {
        Ok(out)
    } else {
        Err(MetadataError::InvalidFloatValue {
            key: key.to_string(),
            value: out.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MetadataError, resolve_llama_metadata_from_kv, resolve_transformer_metadata_from_kv,
    };
    use ggml_rs::GgufValue;

    #[test]
    fn parses_llama_metadata() {
        let kv = [
            (
                "general.architecture",
                GgufValue::String("llama".to_string()),
            ),
            ("llama.block_count", GgufValue::U32(24)),
            ("llama.embedding_length", GgufValue::U32(4096)),
            ("llama.feed_forward_length", GgufValue::U32(11008)),
            ("llama.attention.head_count", GgufValue::U32(32)),
            ("llama.attention.head_count_kv", GgufValue::U32(8)),
            ("llama.rope.dimension_count", GgufValue::U32(128)),
            ("llama.rope.freq_base", GgufValue::F32(10_000.0)),
        ];
        let resolved = resolve_llama_metadata_from_kv(kv.iter().map(|(k, v)| (*k, v)))
            .expect("metadata should parse");
        assert_eq!(resolved.block_count(), 24);
        assert_eq!(resolved.embedding_length(), 4096);
        assert_eq!(resolved.feed_forward_length(), Some(11008));
        assert_eq!(resolved.attention_head_count(), 32);
        assert_eq!(resolved.attention_head_count_kv(), 8);
        assert_eq!(resolved.rope_dimension_count(), Some(128));
    }

    #[test]
    fn reports_missing_required_metadata() {
        let kv = [
            (
                "general.architecture",
                GgufValue::String("llama".to_string()),
            ),
            ("llama.block_count", GgufValue::U32(24)),
        ];
        let error = resolve_llama_metadata_from_kv(kv.iter().map(|(k, v)| (*k, v)))
            .expect_err("missing keys should error");
        assert!(matches!(error, MetadataError::MissingRequiredKey { .. }));
    }

    #[test]
    fn parses_transformer_metadata_for_non_llama_prefix() {
        let kv = [
            (
                "general.architecture",
                GgufValue::String("mistral".to_string()),
            ),
            ("mistral.block_count", GgufValue::U32(32)),
            ("mistral.embedding_length", GgufValue::U32(4096)),
            ("mistral.attention.head_count", GgufValue::U32(32)),
            ("mistral.attention.head_count_kv", GgufValue::U32(8)),
        ];
        let resolved = resolve_transformer_metadata_from_kv(kv.iter().map(|(k, v)| (*k, v)))
            .expect("transformer metadata should parse");
        assert_eq!(resolved.embedding_length(), 4096);
        assert_eq!(resolved.attention_head_count(), 32);
        assert_eq!(resolved.attention_head_count_kv(), 8);
    }
}
