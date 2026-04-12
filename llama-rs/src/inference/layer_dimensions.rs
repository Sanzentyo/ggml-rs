use super::InferenceError;
use crate::metadata::{LlamaModelMetadata, MetadataError, resolve_llama_metadata};
use crate::model::GgufModel;
use crate::naming::{LlamaLayerTensorNames, resolve_llama_layer_tensor_names};

#[derive(Debug, Clone, Copy, PartialEq)]
/// Indicates whether layer dimensions were resolved from full metadata or heuristics.
pub enum MetadataResolutionMode {
    FullMetadata,
    TensorHeuristic,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Auto-resolved per-layer dimensions derived from GGUF metadata + tensor sizes.
pub struct LlamaLayerDimensions {
    pub resolution_mode: MetadataResolutionMode,
    pub hidden_features: usize,
    pub ffn_features: usize,
    pub query_head_count: usize,
    pub kv_head_count: usize,
    pub head_dimension: usize,
    pub attention_scale: Option<f32>,
    pub attention_layer_norm_rms_epsilon: f32,
    pub rope_dimension_count: Option<usize>,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub rope_original_context_length: Option<usize>,
}

/// Resolves per-layer dimensions from GGUF metadata and tensor lengths.
pub fn resolve_llama_layer_dimensions(
    model: &GgufModel,
    layer: usize,
) -> Result<LlamaLayerDimensions, InferenceError> {
    let layer_names = resolve_llama_layer_tensor_names(model, layer)
        .map_err(|source| InferenceError::naming("resolve_llama_layer_tensor_names", source))?;
    resolve_llama_layer_dimensions_from_names(model, &layer_names)
}

fn resolve_llama_layer_dimensions_from_names(
    model: &GgufModel,
    layer_names: &LlamaLayerTensorNames,
) -> Result<LlamaLayerDimensions, InferenceError> {
    let metadata = match resolve_llama_metadata(model) {
        Ok(metadata) => Some(metadata),
        Err(MetadataError::MissingRequiredKey { .. })
        | Err(MetadataError::MissingArchitecture)
        | Err(MetadataError::UnsupportedArchitecture { .. }) => None,
        Err(source) => {
            return Err(InferenceError::metadata("resolve_llama_metadata", source));
        }
    };
    let resolution_mode = if metadata.is_some() {
        MetadataResolutionMode::FullMetadata
    } else {
        MetadataResolutionMode::TensorHeuristic
    };

    let hidden_features = metadata.map_or_else(
        || infer_hidden_features_from_query_weight(model, layer_names),
        |metadata| Ok(metadata.embedding_length()),
    )?;
    let inferred_kv_features =
        infer_projection_output_features(model, &layer_names.attn_k, hidden_features)?;
    let inferred_layout =
        infer_attention_layout_from_features(hidden_features, inferred_kv_features)?;
    let query_head_count = metadata.map_or(inferred_layout.query_head_count, |metadata| {
        metadata.attention_head_count()
    });
    let kv_head_count = metadata.map_or(inferred_layout.kv_head_count, |metadata| {
        metadata.attention_head_count_kv()
    });
    let head_dimension = metadata
        .and_then(LlamaModelMetadata::attention_key_length)
        .unwrap_or_else(|| hidden_features / query_head_count);
    if hidden_features == 0
        || query_head_count == 0
        || kv_head_count == 0
        || !hidden_features.is_multiple_of(query_head_count)
        || !query_head_count.is_multiple_of(kv_head_count)
    {
        return Err(InferenceError::InvalidAttentionLayout {
            hidden_features,
            query_head_count,
            kv_head_count,
        });
    }
    let rope_dimension_count = metadata.and_then(LlamaModelMetadata::rope_dimension_count);
    if let Some(rope_dimensions) = rope_dimension_count
        && rope_dimensions > head_dimension
    {
        return Err(InferenceError::InvalidRopeDimensions {
            rope_dimensions,
            head_dimension,
        });
    }

    let ffn_features = infer_ffn_features(
        model,
        layer_names,
        hidden_features,
        metadata.and_then(LlamaModelMetadata::feed_forward_length),
    )?;
    validate_attention_projection_lengths(
        model,
        layer_names,
        hidden_features,
        query_head_count,
        kv_head_count,
        head_dimension,
    )?;

    Ok(LlamaLayerDimensions {
        resolution_mode,
        hidden_features,
        ffn_features,
        query_head_count,
        kv_head_count,
        head_dimension,
        attention_scale: metadata.and_then(LlamaModelMetadata::attention_scale),
        attention_layer_norm_rms_epsilon: metadata
            .map_or(1e-5, LlamaModelMetadata::attention_layer_norm_rms_epsilon),
        rope_dimension_count,
        rope_freq_base: metadata.map_or(10_000.0, LlamaModelMetadata::rope_freq_base),
        rope_freq_scale: metadata.map_or(1.0, LlamaModelMetadata::rope_freq_scale),
        rope_original_context_length: metadata
            .and_then(LlamaModelMetadata::rope_original_context_length),
    })
}

fn infer_hidden_features_from_query_weight(
    model: &GgufModel,
    layer_names: &LlamaLayerTensorNames,
) -> Result<usize, InferenceError> {
    let q_len = model
        .tensor_len(&layer_names.attn_q)
        .map_err(|source| InferenceError::model("GgufModel::tensor_len(attn_q)", source))?;
    let hidden = (q_len as f64).sqrt() as usize;
    if hidden == 0 || hidden * hidden != q_len {
        return Err(InferenceError::InvalidAttentionWeightShape {
            tensor_name: layer_names.attn_q.clone(),
            expected: hidden * hidden,
            actual: q_len,
        });
    }
    Ok(hidden)
}

fn infer_projection_output_features(
    model: &GgufModel,
    tensor_name: &str,
    hidden_features: usize,
) -> Result<usize, InferenceError> {
    let weight_len = model
        .tensor_len(tensor_name)
        .map_err(|source| InferenceError::model("GgufModel::tensor_len(attn_proj)", source))?;
    if hidden_features == 0 || !weight_len.is_multiple_of(hidden_features) {
        return Err(InferenceError::InvalidAttentionWeightShape {
            tensor_name: tensor_name.to_string(),
            expected: hidden_features.saturating_mul(weight_len / hidden_features.max(1)),
            actual: weight_len,
        });
    }
    Ok(weight_len / hidden_features)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InferredAttentionLayout {
    pub(crate) query_head_count: usize,
    pub(crate) kv_head_count: usize,
}

pub(crate) trait HeadLayoutStrategy {
    fn resolve(
        &self,
        hidden_features: usize,
        kv_features: usize,
    ) -> Result<InferredAttentionLayout, InferenceError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PreferredHeadLayoutStrategy;

impl HeadLayoutStrategy for PreferredHeadLayoutStrategy {
    fn resolve(
        &self,
        hidden_features: usize,
        kv_features: usize,
    ) -> Result<InferredAttentionLayout, InferenceError> {
        const PREFERRED_HEAD_DIMENSIONS: &[usize] =
            &[128, 96, 80, 64, 48, 40, 32, 24, 16, 8, 4, 2, 1];

        for &head_dimension in PREFERRED_HEAD_DIMENSIONS {
            if !hidden_features.is_multiple_of(head_dimension)
                || !kv_features.is_multiple_of(head_dimension)
            {
                continue;
            }
            let query_head_count = hidden_features / head_dimension;
            let kv_head_count = kv_features / head_dimension;
            if query_head_count >= kv_head_count && query_head_count.is_multiple_of(kv_head_count) {
                return Ok(InferredAttentionLayout {
                    query_head_count,
                    kv_head_count,
                });
            }
        }

        let head_dimension = gcd(hidden_features, kv_features);
        if head_dimension == 0 {
            return Err(InferenceError::InvalidAttentionLayout {
                hidden_features,
                query_head_count: 0,
                kv_head_count: 0,
            });
        }
        let query_head_count = hidden_features / head_dimension;
        let kv_head_count = kv_features / head_dimension;
        if query_head_count >= kv_head_count && query_head_count.is_multiple_of(kv_head_count) {
            Ok(InferredAttentionLayout {
                query_head_count,
                kv_head_count,
            })
        } else {
            Err(InferenceError::InvalidAttentionLayout {
                hidden_features,
                query_head_count,
                kv_head_count,
            })
        }
    }
}

pub(crate) fn infer_attention_layout_from_features(
    hidden_features: usize,
    kv_features: usize,
) -> Result<InferredAttentionLayout, InferenceError> {
    PreferredHeadLayoutStrategy.resolve(hidden_features, kv_features)
}

fn gcd(mut lhs: usize, mut rhs: usize) -> usize {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}

fn infer_ffn_features(
    model: &GgufModel,
    layer_names: &LlamaLayerTensorNames,
    hidden_features: usize,
    metadata_ffn_features: Option<usize>,
) -> Result<usize, InferenceError> {
    let gate_len = model
        .tensor_len(&layer_names.ffn_gate)
        .map_err(|source| InferenceError::model("GgufModel::tensor_len(ffn_gate)", source))?;
    if gate_len % hidden_features != 0 {
        return Err(InferenceError::InvalidMlpWeightShape {
            tensor_name: layer_names.ffn_gate.clone(),
            hidden_features,
            weight_len: gate_len,
        });
    }
    let inferred_ffn = gate_len / hidden_features;
    let ffn_features = metadata_ffn_features.unwrap_or(inferred_ffn);
    let expected = hidden_features * ffn_features;
    if gate_len != expected {
        return Err(InferenceError::InvalidWeightLength {
            tensor_name: layer_names.ffn_gate.clone(),
            expected,
            actual: gate_len,
        });
    }
    for tensor_name in [&layer_names.ffn_up, &layer_names.ffn_down] {
        let weight_len = model
            .tensor_len(tensor_name)
            .map_err(|source| InferenceError::model("GgufModel::tensor_len(ffn)", source))?;
        if weight_len != expected {
            return Err(InferenceError::InvalidWeightLength {
                tensor_name: tensor_name.clone(),
                expected,
                actual: weight_len,
            });
        }
    }
    Ok(ffn_features)
}

fn validate_attention_projection_lengths(
    model: &GgufModel,
    layer_names: &LlamaLayerTensorNames,
    hidden_features: usize,
    query_head_count: usize,
    kv_head_count: usize,
    head_dimension: usize,
) -> Result<(), InferenceError> {
    let query_features = query_head_count
        .checked_mul(head_dimension)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let kv_features = kv_head_count
        .checked_mul(head_dimension)
        .ok_or(InferenceError::MemorySizeOverflow)?;

    let expected_q = hidden_features
        .checked_mul(query_features)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let expected_kv = hidden_features
        .checked_mul(kv_features)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let expected_o = query_features
        .checked_mul(hidden_features)
        .ok_or(InferenceError::MemorySizeOverflow)?;

    let q_len = model
        .tensor_len(&layer_names.attn_q)
        .map_err(|source| InferenceError::model("GgufModel::tensor_len(attn_q)", source))?;
    if q_len != expected_q {
        return Err(InferenceError::InvalidAttentionWeightShape {
            tensor_name: layer_names.attn_q.clone(),
            expected: expected_q,
            actual: q_len,
        });
    }
    let k_len = model
        .tensor_len(&layer_names.attn_k)
        .map_err(|source| InferenceError::model("GgufModel::tensor_len(attn_k)", source))?;
    if k_len != expected_kv {
        return Err(InferenceError::InvalidAttentionWeightShape {
            tensor_name: layer_names.attn_k.clone(),
            expected: expected_kv,
            actual: k_len,
        });
    }
    let v_len = model
        .tensor_len(&layer_names.attn_v)
        .map_err(|source| InferenceError::model("GgufModel::tensor_len(attn_v)", source))?;
    if v_len != expected_kv {
        return Err(InferenceError::InvalidAttentionWeightShape {
            tensor_name: layer_names.attn_v.clone(),
            expected: expected_kv,
            actual: v_len,
        });
    }
    let o_len = model
        .tensor_len(&layer_names.attn_output)
        .map_err(|source| InferenceError::model("GgufModel::tensor_len(attn_output)", source))?;
    if o_len != expected_o {
        return Err(InferenceError::InvalidAttentionWeightShape {
            tensor_name: layer_names.attn_output.clone(),
            expected: expected_o,
            actual: o_len,
        });
    }
    Ok(())
}
