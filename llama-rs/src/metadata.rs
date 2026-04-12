//! GGUF metadata parsing for architecture-aware inference configuration.

use crate::model::GgufModel;
use ggml_rs::{GgufArrayValue, GgufValue};
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
    attention_key_length: Option<NonZeroUsize>,
    attention_value_length: Option<NonZeroUsize>,
    feed_forward_length: Option<NonZeroUsize>,
    context_length: Option<NonZeroUsize>,
    attention_scale: Option<f32>,
    attention_layer_norm_rms_epsilon: f32,
    rope_dimension_count: Option<NonZeroUsize>,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    rope_original_context_length: Option<NonZeroUsize>,
    rope_dimension_sections: Option<[i32; 4]>,
    full_attention_interval: Option<NonZeroUsize>,
    ssm_conv_kernel: Option<NonZeroUsize>,
    ssm_state_size: Option<NonZeroUsize>,
    ssm_group_count: Option<NonZeroUsize>,
    ssm_time_step_rank: Option<NonZeroUsize>,
    ssm_inner_size: Option<NonZeroUsize>,
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

    pub fn attention_key_length(&self) -> Option<usize> {
        self.attention_key_length.map(NonZeroUsize::get)
    }

    pub fn attention_value_length(&self) -> Option<usize> {
        self.attention_value_length.map(NonZeroUsize::get)
    }

    pub fn feed_forward_length(&self) -> Option<usize> {
        self.feed_forward_length.map(NonZeroUsize::get)
    }

    pub fn context_length(&self) -> Option<usize> {
        self.context_length.map(NonZeroUsize::get)
    }

    pub fn attention_scale(&self) -> Option<f32> {
        self.attention_scale
    }

    pub fn attention_layer_norm_rms_epsilon(&self) -> f32 {
        self.attention_layer_norm_rms_epsilon
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

    pub fn rope_original_context_length(&self) -> Option<usize> {
        self.rope_original_context_length.map(NonZeroUsize::get)
    }

    pub fn rope_dimension_sections(&self) -> Option<[i32; 4]> {
        self.rope_dimension_sections
    }

    pub fn full_attention_interval(&self) -> Option<usize> {
        self.full_attention_interval.map(NonZeroUsize::get)
    }

    pub fn ssm_conv_kernel(&self) -> Option<usize> {
        self.ssm_conv_kernel.map(NonZeroUsize::get)
    }

    pub fn ssm_state_size(&self) -> Option<usize> {
        self.ssm_state_size.map(NonZeroUsize::get)
    }

    pub fn ssm_group_count(&self) -> Option<usize> {
        self.ssm_group_count.map(NonZeroUsize::get)
    }

    pub fn ssm_time_step_rank(&self) -> Option<usize> {
        self.ssm_time_step_rank.map(NonZeroUsize::get)
    }

    pub fn ssm_inner_size(&self) -> Option<usize> {
        self.ssm_inner_size.map(NonZeroUsize::get)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Parsed metadata required by the current LLaMA inference helpers.
pub struct LlamaModelMetadata {
    block_count: NonZeroUsize,
    embedding_length: NonZeroUsize,
    attention_head_count: NonZeroUsize,
    attention_head_count_kv: NonZeroUsize,
    attention_key_length: Option<NonZeroUsize>,
    attention_value_length: Option<NonZeroUsize>,
    feed_forward_length: Option<NonZeroUsize>,
    context_length: Option<NonZeroUsize>,
    attention_scale: Option<f32>,
    attention_layer_norm_rms_epsilon: f32,
    rope_dimension_count: Option<NonZeroUsize>,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    rope_original_context_length: Option<NonZeroUsize>,
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

    pub fn attention_key_length(self) -> Option<usize> {
        self.attention_key_length.map(NonZeroUsize::get)
    }

    pub fn attention_value_length(self) -> Option<usize> {
        self.attention_value_length.map(NonZeroUsize::get)
    }

    pub fn feed_forward_length(self) -> Option<usize> {
        self.feed_forward_length.map(NonZeroUsize::get)
    }

    pub fn context_length(self) -> Option<usize> {
        self.context_length.map(NonZeroUsize::get)
    }

    pub fn attention_scale(self) -> Option<f32> {
        self.attention_scale
    }

    pub fn attention_layer_norm_rms_epsilon(self) -> f32 {
        self.attention_layer_norm_rms_epsilon
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

    pub fn rope_original_context_length(self) -> Option<usize> {
        self.rope_original_context_length.map(NonZeroUsize::get)
    }
}

impl From<TransformerMetadata> for LlamaModelMetadata {
    fn from(value: TransformerMetadata) -> Self {
        Self {
            block_count: value.block_count,
            embedding_length: value.embedding_length,
            attention_head_count: value.attention_head_count,
            attention_head_count_kv: value.attention_head_count_kv,
            attention_key_length: value.attention_key_length,
            attention_value_length: value.attention_value_length,
            feed_forward_length: value.feed_forward_length,
            context_length: value.context_length,
            attention_scale: value.attention_scale,
            attention_layer_norm_rms_epsilon: value.attention_layer_norm_rms_epsilon,
            rope_dimension_count: value.rope_dimension_count,
            rope_freq_base: value.rope_freq_base,
            rope_freq_scale: value.rope_freq_scale,
            rope_original_context_length: value.rope_original_context_length,
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
    let key_length_key = format!("{prefix}.attention.key_length");
    let value_length_key = format!("{prefix}.attention.value_length");
    let feed_forward_key = format!("{prefix}.feed_forward_length");
    let context_length_key = format!("{prefix}.context_length");
    let attention_scale_key = format!("{prefix}.attention.scale");
    let norm_rms_eps_key = format!("{prefix}.attention.layer_norm_rms_epsilon");
    let rope_dim_key = format!("{prefix}.rope.dimension_count");
    let rope_base_key = format!("{prefix}.rope.freq_base");
    let rope_scale_key = format!("{prefix}.rope.freq_scale");
    let rope_original_context_length_key = format!("{prefix}.rope.scaling.original_context_length");
    let rope_dimension_sections_key = format!("{prefix}.rope.dimension_sections");
    let full_attention_interval_key = format!("{prefix}.full_attention_interval");
    let ssm_conv_kernel_key = format!("{prefix}.ssm.conv_kernel");
    let ssm_state_size_key = format!("{prefix}.ssm.state_size");
    let ssm_group_count_key = format!("{prefix}.ssm.group_count");
    let ssm_time_step_rank_key = format!("{prefix}.ssm.time_step_rank");
    let ssm_inner_size_key = format!("{prefix}.ssm.inner_size");

    let block_count = required_non_zero_usize(kv, &block_count_key)?;
    let embedding_length = required_non_zero_usize(kv, &embedding_length_key)?;
    let attention_head_count = required_non_zero_usize(kv, &head_count_key)?;
    let attention_head_count_kv =
        optional_non_zero_usize(kv, &head_count_kv_key)?.unwrap_or(attention_head_count);
    let attention_key_length = optional_non_zero_usize(kv, &key_length_key)?;
    let attention_value_length = optional_non_zero_usize(kv, &value_length_key)?;
    let feed_forward_length = optional_non_zero_usize(kv, &feed_forward_key)?;
    let context_length = optional_non_zero_usize(kv, &context_length_key)?;
    let attention_scale = optional_f32(kv, &attention_scale_key)?;
    let attention_layer_norm_rms_epsilon = optional_f32(kv, &norm_rms_eps_key)?.unwrap_or(1e-5);
    let rope_dimension_count = optional_non_zero_usize(kv, &rope_dim_key)?;
    let rope_freq_base = optional_f32(kv, &rope_base_key)?.unwrap_or(10_000.0);
    let rope_freq_scale = optional_f32(kv, &rope_scale_key)?.unwrap_or(1.0);
    let rope_original_context_length =
        optional_non_zero_usize(kv, &rope_original_context_length_key)?;
    let rope_dimension_sections = optional_i32_array_4(kv, &rope_dimension_sections_key)?;
    let full_attention_interval = optional_non_zero_usize(kv, &full_attention_interval_key)?;
    let ssm_conv_kernel = optional_non_zero_usize(kv, &ssm_conv_kernel_key)?;
    let ssm_state_size = optional_non_zero_usize(kv, &ssm_state_size_key)?;
    let ssm_group_count = optional_non_zero_usize(kv, &ssm_group_count_key)?;
    let ssm_time_step_rank = optional_non_zero_usize(kv, &ssm_time_step_rank_key)?;
    let ssm_inner_size = optional_non_zero_usize(kv, &ssm_inner_size_key)?;

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
        attention_key_length,
        attention_value_length,
        feed_forward_length,
        context_length,
        attention_scale,
        attention_layer_norm_rms_epsilon,
        rope_dimension_count,
        rope_freq_base,
        rope_freq_scale,
        rope_original_context_length,
        rope_dimension_sections,
        full_attention_interval,
        ssm_conv_kernel,
        ssm_state_size,
        ssm_group_count,
        ssm_time_step_rank,
        ssm_inner_size,
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

fn optional_i32_array_4(
    kv: &HashMap<&str, &GgufValue>,
    key: &str,
) -> Result<Option<[i32; 4]>, MetadataError> {
    let Some(value) = kv.get(key).copied() else {
        return Ok(None);
    };
    match value {
        GgufValue::Array(GgufArrayValue::I32(values)) if values.len() == 4 => {
            Ok(Some([values[0], values[1], values[2], values[3]]))
        }
        GgufValue::Array(GgufArrayValue::I32(_)) => Err(MetadataError::InvalidValueType {
            key: key.to_string(),
            expected: "i32 array of length 4",
            actual: "i32 array of wrong length",
        }),
        other => Err(MetadataError::InvalidValueType {
            key: key.to_string(),
            expected: "i32 array of length 4",
            actual: other.type_name(),
        }),
    }
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
            ("llama.attention.key_length", GgufValue::U32(128)),
            ("llama.attention.value_length", GgufValue::U32(128)),
            ("llama.attention.scale", GgufValue::F32(0.125)),
            (
                "llama.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-6),
            ),
            ("llama.rope.dimension_count", GgufValue::U32(128)),
            ("llama.rope.freq_base", GgufValue::F32(10_000.0)),
            (
                "llama.rope.scaling.original_context_length",
                GgufValue::U32(8192),
            ),
        ];
        let resolved = resolve_llama_metadata_from_kv(kv.iter().map(|(k, v)| (*k, v)))
            .expect("metadata should parse");
        assert_eq!(resolved.block_count(), 24);
        assert_eq!(resolved.embedding_length(), 4096);
        assert_eq!(resolved.feed_forward_length(), Some(11008));
        assert_eq!(resolved.attention_head_count(), 32);
        assert_eq!(resolved.attention_head_count_kv(), 8);
        assert_eq!(resolved.attention_key_length(), Some(128));
        assert_eq!(resolved.attention_value_length(), Some(128));
        assert_eq!(resolved.attention_scale(), Some(0.125));
        assert_eq!(resolved.attention_layer_norm_rms_epsilon(), 1e-6);
        assert_eq!(resolved.rope_dimension_count(), Some(128));
        assert_eq!(resolved.rope_original_context_length(), Some(8192));
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
            ("mistral.attention.key_length", GgufValue::U32(128)),
            ("mistral.attention.value_length", GgufValue::U32(128)),
            ("mistral.attention.scale", GgufValue::F32(0.25)),
            (
                "mistral.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-6),
            ),
        ];
        let resolved = resolve_transformer_metadata_from_kv(kv.iter().map(|(k, v)| (*k, v)))
            .expect("transformer metadata should parse");
        assert_eq!(resolved.embedding_length(), 4096);
        assert_eq!(resolved.attention_head_count(), 32);
        assert_eq!(resolved.attention_head_count_kv(), 8);
        assert_eq!(resolved.attention_key_length(), Some(128));
        assert_eq!(resolved.attention_value_length(), Some(128));
        assert_eq!(resolved.attention_scale(), Some(0.25));
        assert_eq!(resolved.attention_layer_norm_rms_epsilon(), 1e-6);
    }

    #[test]
    fn parses_qwen35_ssm_metadata() {
        let kv = [
            (
                "general.architecture",
                GgufValue::String("qwen35".to_string()),
            ),
            ("qwen35.block_count", GgufValue::U32(32)),
            ("qwen35.embedding_length", GgufValue::U32(2560)),
            ("qwen35.attention.head_count", GgufValue::U32(16)),
            ("qwen35.attention.head_count_kv", GgufValue::U32(4)),
            ("qwen35.attention.key_length", GgufValue::U32(256)),
            (
                "qwen35.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-6),
            ),
            ("qwen35.full_attention_interval", GgufValue::U32(4)),
            ("qwen35.ssm.conv_kernel", GgufValue::U32(4)),
            ("qwen35.ssm.state_size", GgufValue::U32(128)),
            ("qwen35.ssm.group_count", GgufValue::U32(16)),
            ("qwen35.ssm.time_step_rank", GgufValue::U32(32)),
            ("qwen35.ssm.inner_size", GgufValue::U32(4096)),
        ];
        let resolved = resolve_transformer_metadata_from_kv(kv.iter().map(|(k, v)| (*k, v)))
            .expect("transformer metadata should parse");
        assert_eq!(resolved.architecture().as_str(), "qwen35");
        assert_eq!(resolved.full_attention_interval(), Some(4));
        assert_eq!(resolved.ssm_conv_kernel(), Some(4));
        assert_eq!(resolved.ssm_state_size(), Some(128));
        assert_eq!(resolved.ssm_group_count(), Some(16));
        assert_eq!(resolved.ssm_time_step_rank(), Some(32));
        assert_eq!(resolved.ssm_inner_size(), Some(4096));
    }

    #[test]
    fn parses_rope_dimension_sections() {
        let kv = [
            (
                "general.architecture",
                GgufValue::String("qwen35".to_string()),
            ),
            ("qwen35.block_count", GgufValue::U32(32)),
            ("qwen35.embedding_length", GgufValue::U32(2560)),
            ("qwen35.attention.head_count", GgufValue::U32(16)),
            ("qwen35.attention.head_count_kv", GgufValue::U32(4)),
            (
                "qwen35.rope.dimension_sections",
                GgufValue::Array(ggml_rs::GgufArrayValue::I32(vec![11, 11, 10, 0])),
            ),
        ];
        let resolved = resolve_transformer_metadata_from_kv(kv.iter().map(|(k, v)| (*k, v)))
            .expect("transformer metadata should parse");
        assert_eq!(resolved.rope_dimension_sections(), Some([11, 11, 10, 0]));
    }

    #[test]
    fn rope_dimension_sections_absent_returns_none() {
        let kv = [
            (
                "general.architecture",
                GgufValue::String("llama".to_string()),
            ),
            ("llama.block_count", GgufValue::U32(24)),
            ("llama.embedding_length", GgufValue::U32(4096)),
            ("llama.attention.head_count", GgufValue::U32(32)),
        ];
        let resolved = resolve_transformer_metadata_from_kv(kv.iter().map(|(k, v)| (*k, v)))
            .expect("transformer metadata should parse");
        assert_eq!(resolved.rope_dimension_sections(), None);
    }
}
