//! Token-id based end-to-end generation helpers for transformer-style GGUF models.
//!
//! This module supports both direct token-id prompts and GGUF tokenizer-backed
//! prompt text (currently for `tokenizer.ggml.model=gpt2`).

use crate::backend::{LlamaBackend, ensure_backends_loaded};
use crate::inference::{
    AttentionInferenceConfig, AttentionMaskPolicy, AttentionWeights, InferenceError,
    MlpInferenceConfig, MlpWeights,
    attention_inference_with_weights_on_backend_repeats_with_length,
    resolve_llama_layer_dimensions,
};
use crate::metadata::{MetadataError, resolve_transformer_metadata};
use crate::model::{GgufModel, ModelError};
use crate::naming::{LlamaLayerTensorNames, NamingError, resolve_llama_layer_tensor_names};
use crate::tokenizer::{TokenizerError, tokenize_text_prompt};
use ggml_rs::{Backend, Bytes, Context, GgufValue, Shape2D};
use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;

const RMS_NORM_EPS: f32 = 1e-5;
const MLP_BACKEND_SLACK_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum E2eError {
    #[error("{context}: {source}")]
    Model {
        context: &'static str,
        #[source]
        source: ModelError,
    },
    #[error("{context}: {source}")]
    Metadata {
        context: &'static str,
        #[source]
        source: MetadataError,
    },
    #[error("{context}: {source}")]
    Naming {
        context: &'static str,
        #[source]
        source: NamingError,
    },
    #[error("{context}: {source}")]
    Inference {
        context: &'static str,
        #[source]
        source: InferenceError,
    },
    #[error("{context}: {source}")]
    Tokenizer {
        context: &'static str,
        #[source]
        source: TokenizerError,
    },
    #[error("{context}: {source}")]
    Ggml {
        context: &'static str,
        #[source]
        source: ggml_rs::Error,
    },
    #[error("prompt_token_ids must not be empty")]
    EmptyPrompt,
    #[error("invalid token id {token_id}; valid range is [0, {vocab_size})")]
    InvalidTokenId { token_id: i32, vocab_size: usize },
    #[error(
        "token embedding tensor `{tensor_name}` has incompatible shape: hidden_features={hidden_features}, tensor_len={tensor_len}"
    )]
    InvalidTokenEmbeddingShape {
        tensor_name: String,
        hidden_features: usize,
        tensor_len: usize,
    },
    #[error(
        "output projection tensor `{tensor_name}` length mismatch: expected {expected}, got {actual}"
    )]
    OutputWeightLengthMismatch {
        tensor_name: String,
        expected: usize,
        actual: usize,
    },
    #[error("norm tensor `{tensor_name}` length mismatch: expected {expected}, got {actual}")]
    NormWeightLengthMismatch {
        tensor_name: String,
        expected: usize,
        actual: usize,
    },
    #[error("hidden feature mismatch at layer {layer}: expected {expected}, got {actual}")]
    HiddenFeatureMismatch {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "requested total sequence length {requested} exceeds model context length {context_length}"
    )]
    SequenceTooLong {
        requested: usize,
        context_length: usize,
    },
    #[error(
        "MLP gate tensor `{tensor_name}` has incompatible shape for hidden_features={hidden_features}: tensor_len={tensor_len}"
    )]
    InvalidMlpGateShape {
        tensor_name: String,
        hidden_features: usize,
        tensor_len: usize,
    },
    #[error("buffer length mismatch: expected {expected}, got {actual}")]
    BufferLengthMismatch { expected: usize, actual: usize },
    #[error("memory size overflow while building generation graph")]
    MemorySizeOverflow,
}

impl E2eError {
    fn model(context: &'static str, source: ModelError) -> Self {
        Self::Model { context, source }
    }

    fn metadata(context: &'static str, source: MetadataError) -> Self {
        Self::Metadata { context, source }
    }

    fn naming(context: &'static str, source: NamingError) -> Self {
        Self::Naming { context, source }
    }

    fn inference(context: &'static str, source: InferenceError) -> Self {
        Self::Inference { context, source }
    }

    fn tokenizer(context: &'static str, source: TokenizerError) -> Self {
        Self::Tokenizer { context, source }
    }

    fn ggml(context: &'static str, source: ggml_rs::Error) -> Self {
        Self::Ggml { context, source }
    }
}

#[derive(Debug, Clone)]
pub struct E2eGenerationConfig {
    pub backend: LlamaBackend,
    pub prompt_token_ids: Vec<i32>,
    pub max_new_tokens: usize,
    pub pad_token_id: i32,
    pub eos_token_id: Option<i32>,
    pub mixed_layer_policy: MixedLayerPolicy,
}

impl E2eGenerationConfig {
    pub fn new(
        backend: LlamaBackend,
        prompt_token_ids: Vec<i32>,
        max_new_tokens: usize,
    ) -> Result<Self, E2eError> {
        if prompt_token_ids.is_empty() {
            return Err(E2eError::EmptyPrompt);
        }
        Ok(Self {
            backend,
            prompt_token_ids,
            max_new_tokens,
            pad_token_id: 0,
            eos_token_id: None,
            mixed_layer_policy: MixedLayerPolicy::Strict,
        })
    }

    pub const fn with_pad_token_id(mut self, pad_token_id: i32) -> Self {
        self.pad_token_id = pad_token_id;
        self
    }

    pub const fn with_eos_token_id(mut self, eos_token_id: Option<i32>) -> Self {
        self.eos_token_id = eos_token_id;
        self
    }

    pub const fn with_mixed_layer_policy(mut self, mixed_layer_policy: MixedLayerPolicy) -> Self {
        self.mixed_layer_policy = mixed_layer_policy;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedLayerPolicy {
    Strict,
    SkipUnsupportedAttention,
}

#[derive(Debug, Clone)]
pub struct E2eGenerationReport {
    pub backend_name: String,
    pub prompt_token_count: usize,
    pub generated_token_ids: Vec<i32>,
    pub all_token_ids: Vec<i32>,
    pub attention_layer_count: usize,
    pub mlp_only_layer_count: usize,
    pub elapsed: Duration,
}

impl E2eGenerationReport {
    pub fn avg_generated_token_ms(&self) -> f64 {
        if self.generated_token_ids.is_empty() {
            0.0
        } else {
            self.elapsed.as_secs_f64() * 1000.0 / self.generated_token_ids.len() as f64
        }
    }
}

#[derive(Debug, Clone)]
struct GlobalTensorNames {
    token_embedding: String,
    output_norm: String,
    output: Option<String>,
}

#[derive(Debug, Clone)]
struct LayerPlan {
    attention: Option<AttentionLayerPlan>,
    mlp: MlpLayerPlan,
}

#[derive(Debug, Clone)]
struct AttentionLayerPlan {
    weights: AttentionWeights<f32>,
    norm_values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct MlpLayerPlan {
    weights: MlpWeights<f32>,
    norm_values: Vec<f32>,
}

pub fn resolve_eos_token_id(model: &GgufModel) -> Option<i32> {
    model
        .kv_value("tokenizer.ggml.eos_token_id")
        .and_then(value_to_i32)
}

pub fn tokenize_prompt_text(model: &GgufModel, prompt_text: &str) -> Result<Vec<i32>, E2eError> {
    tokenize_text_prompt(model, prompt_text)
        .map_err(|source| E2eError::tokenizer("tokenize_text_prompt", source))
}

pub fn generate_token_ids_from_path(
    model_path: impl AsRef<Path>,
    config: &E2eGenerationConfig,
) -> Result<E2eGenerationReport, E2eError> {
    let model =
        GgufModel::open(model_path).map_err(|source| E2eError::model("GgufModel::open", source))?;
    generate_token_ids_from_model(&model, config)
}

pub fn generate_token_ids_from_model(
    model: &GgufModel,
    config: &E2eGenerationConfig,
) -> Result<E2eGenerationReport, E2eError> {
    let prompt_token_count = config.prompt_token_ids.len();
    if prompt_token_count == 0 {
        return Err(E2eError::EmptyPrompt);
    }

    let total_sequence_length = prompt_token_count
        .checked_add(config.max_new_tokens)
        .ok_or(E2eError::MemorySizeOverflow)?;
    let metadata = resolve_transformer_metadata(model)
        .map_err(|source| E2eError::metadata("resolve_transformer_metadata", source))?;
    if let Some(context_length) = metadata.context_length()
        && total_sequence_length > context_length
    {
        return Err(E2eError::SequenceTooLong {
            requested: total_sequence_length,
            context_length,
        });
    }

    let hidden_features = metadata.embedding_length();
    let global_names = resolve_global_tensor_names(model)?;

    let token_embedding_values = model
        .tensor_values::<f32>(&global_names.token_embedding)
        .map_err(|source| E2eError::model("GgufModel::tensor_values(token_embedding)", source))?;
    if hidden_features == 0 || !token_embedding_values.len().is_multiple_of(hidden_features) {
        return Err(E2eError::InvalidTokenEmbeddingShape {
            tensor_name: global_names.token_embedding.clone(),
            hidden_features,
            tensor_len: token_embedding_values.len(),
        });
    }
    let vocab_size = token_embedding_values.len() / hidden_features;

    let output_weight_values = if let Some(output_name) = global_names.output.as_deref() {
        let values = model.tensor_values::<f32>(output_name).map_err(|source| {
            E2eError::model("GgufModel::tensor_values(output_projection)", source)
        })?;
        let expected = checked_mul(hidden_features, vocab_size)?;
        if values.len() != expected {
            return Err(E2eError::OutputWeightLengthMismatch {
                tensor_name: output_name.to_string(),
                expected,
                actual: values.len(),
            });
        }
        Some(values)
    } else {
        None
    };
    let output_weight_values = output_weight_values
        .as_deref()
        .unwrap_or(token_embedding_values.as_slice());

    let output_norm_values = decode_norm_tensor(
        model,
        &global_names.output_norm,
        hidden_features,
        "output_norm",
    )?;
    let layer_plans = build_layer_plans(
        model,
        metadata.block_count(),
        hidden_features,
        total_sequence_length,
        config.mixed_layer_policy,
    )?;
    let attention_layer_count = layer_plans
        .iter()
        .filter(|layer_plan| layer_plan.attention.is_some())
        .count();
    let mlp_only_layer_count = layer_plans.len() - attention_layer_count;

    let _ = validate_token_id(config.pad_token_id, vocab_size)?;
    if let Some(eos_token_id) = config.eos_token_id {
        let _ = validate_token_id(eos_token_id, vocab_size)?;
    }
    for &token_id in &config.prompt_token_ids {
        let _ = validate_token_id(token_id, vocab_size)?;
    }

    let mut all_token_ids = vec![config.pad_token_id; total_sequence_length];
    all_token_ids[..prompt_token_count].copy_from_slice(&config.prompt_token_ids);
    let mut generated_token_ids = Vec::with_capacity(config.max_new_tokens);
    let mut current_token_count = prompt_token_count;
    ensure_backends_loaded();
    let backend = Backend::new(config.backend.into())
        .map_err(|source| E2eError::ggml("Backend::new", source))?;
    let backend_name = backend
        .name()
        .map(|name| name.to_string())
        .map_err(|source| E2eError::ggml("Backend::name", source))?;

    let start = Instant::now();
    for _step in 0..config.max_new_tokens {
        let active_token_ids = &all_token_ids[..current_token_count];
        let mut hidden = gather_embeddings(
            &token_embedding_values,
            hidden_features,
            vocab_size,
            active_token_ids,
        )?;

        for layer_plan in &layer_plans {
            if let Some(attention) = &layer_plan.attention {
                let normalized_attn = rms_norm_with_weight(
                    &hidden,
                    hidden_features,
                    current_token_count,
                    &attention.norm_values,
                )?;
                let attention_output =
                    attention_inference_with_weights_on_backend_repeats_with_length(
                        &attention.weights,
                        &normalized_attn,
                        current_token_count,
                        &backend,
                        1,
                    )
                    .map_err(|source| {
                        E2eError::inference(
                            "attention_inference_with_weights_on_backend_repeats_with_length",
                            source,
                        )
                    })?;
                add_in_place(&mut hidden, &attention_output)?;
            }

            let normalized_ffn = rms_norm_with_weight(
                &hidden,
                hidden_features,
                current_token_count,
                &layer_plan.mlp.norm_values,
            )?;
            let mlp_output = mlp_sequence_inference_with_weights(
                &layer_plan.mlp.weights,
                &normalized_ffn,
                current_token_count,
                &backend,
            )?;
            add_in_place(&mut hidden, &mlp_output)?;
        }

        let normalized_output = rms_norm_with_weight(
            &hidden,
            hidden_features,
            current_token_count,
            &output_norm_values,
        )?;
        let last_index = current_token_count
            .checked_sub(1)
            .ok_or(E2eError::EmptyPrompt)?;
        let next_token_id = greedy_next_token_id(
            &normalized_output,
            last_index,
            hidden_features,
            output_weight_values,
            vocab_size,
        )?;

        generated_token_ids.push(next_token_id);
        if current_token_count < total_sequence_length {
            all_token_ids[current_token_count] = next_token_id;
            current_token_count += 1;
        }

        if config.eos_token_id.is_some_and(|eos| eos == next_token_id) {
            break;
        }
    }
    let elapsed = start.elapsed();

    Ok(E2eGenerationReport {
        backend_name,
        prompt_token_count,
        generated_token_ids,
        all_token_ids: all_token_ids[..current_token_count].to_vec(),
        attention_layer_count,
        mlp_only_layer_count,
        elapsed,
    })
}

fn build_layer_plans(
    model: &GgufModel,
    block_count: usize,
    hidden_features: usize,
    total_sequence_length: usize,
    mixed_layer_policy: MixedLayerPolicy,
) -> Result<Vec<LayerPlan>, E2eError> {
    let mut layer_plans = Vec::with_capacity(block_count);
    for layer in 0..block_count {
        match resolve_llama_layer_tensor_names(model, layer) {
            Ok(names) => {
                match build_full_layer_plan(model, names, hidden_features, total_sequence_length) {
                    Ok(full_plan) => layer_plans.push(full_plan),
                    Err(source) => match mixed_layer_policy {
                        MixedLayerPolicy::Strict => return Err(source),
                        MixedLayerPolicy::SkipUnsupportedAttention => {
                            layer_plans.push(build_mlp_only_layer_plan(
                                model,
                                layer,
                                hidden_features,
                            )?);
                        }
                    },
                }
            }
            Err(source) => match mixed_layer_policy {
                MixedLayerPolicy::Strict => {
                    return Err(E2eError::naming("resolve_llama_layer_tensor_names", source));
                }
                MixedLayerPolicy::SkipUnsupportedAttention => {
                    if matches!(
                        source,
                        NamingError::NoLayersDetected | NamingError::LayerNotFound { .. }
                    ) {
                        return Err(E2eError::naming("resolve_llama_layer_tensor_names", source));
                    }
                    layer_plans.push(build_mlp_only_layer_plan(model, layer, hidden_features)?);
                }
            },
        }
    }
    Ok(layer_plans)
}

fn build_full_layer_plan(
    model: &GgufModel,
    names: LlamaLayerTensorNames,
    hidden_features: usize,
    total_sequence_length: usize,
) -> Result<LayerPlan, E2eError> {
    let layer = names.layer;
    let dimensions = resolve_llama_layer_dimensions(model, layer)
        .map_err(|source| E2eError::inference("resolve_llama_layer_dimensions", source))?;
    if dimensions.hidden_features != hidden_features {
        return Err(E2eError::HiddenFeatureMismatch {
            layer,
            expected: hidden_features,
            actual: dimensions.hidden_features,
        });
    }
    let attention_config =
        AttentionInferenceConfig::from_layer_dimensions(dimensions, total_sequence_length)
            .map_err(|source| {
                E2eError::inference("AttentionInferenceConfig::from_layer_dimensions", source)
            })?
            .with_mask(AttentionMaskPolicy::Causal { past_tokens: 0 });
    let attention_weights = AttentionWeights::from_model_layer(model, &names, attention_config)
        .map_err(|source| E2eError::inference("AttentionWeights::from_model_layer", source))?;
    let mlp_weights = MlpWeights::from_model_layer(model, &names, hidden_features)
        .map_err(|source| E2eError::inference("MlpWeights::from_model_layer", source))?;
    let attn_norm_values =
        decode_norm_tensor(model, &names.attn_norm, hidden_features, "attn_norm")?;
    let ffn_norm_values = decode_norm_tensor(model, &names.ffn_norm, hidden_features, "ffn_norm")?;

    Ok(LayerPlan {
        attention: Some(AttentionLayerPlan {
            weights: attention_weights,
            norm_values: attn_norm_values,
        }),
        mlp: MlpLayerPlan {
            weights: mlp_weights,
            norm_values: ffn_norm_values,
        },
    })
}

fn build_mlp_only_layer_plan(
    model: &GgufModel,
    layer: usize,
    hidden_features: usize,
) -> Result<LayerPlan, E2eError> {
    let ffn_norm = resolve_required_layer_tensor_name(
        model,
        layer,
        "ffn_norm",
        layer_ffn_norm_candidates(layer),
    )?;
    let ffn_gate = resolve_required_layer_tensor_name(
        model,
        layer,
        "ffn_gate",
        layer_ffn_gate_candidates(layer),
    )?;
    let ffn_up =
        resolve_required_layer_tensor_name(model, layer, "ffn_up", layer_ffn_up_candidates(layer))?;
    let ffn_down = resolve_required_layer_tensor_name(
        model,
        layer,
        "ffn_down",
        layer_ffn_down_candidates(layer),
    )?;

    let gate_len = model
        .tensor_len(&ffn_gate)
        .map_err(|source| E2eError::model("GgufModel::tensor_len(ffn_gate)", source))?;
    if hidden_features == 0 || !gate_len.is_multiple_of(hidden_features) {
        return Err(E2eError::InvalidMlpGateShape {
            tensor_name: ffn_gate,
            hidden_features,
            tensor_len: gate_len,
        });
    }
    let ffn_features = gate_len / hidden_features;
    let mlp_config = MlpInferenceConfig::new(hidden_features, ffn_features)
        .map_err(|source| E2eError::inference("MlpInferenceConfig::new", source))?;
    let mlp_weights = MlpWeights::from_model(model, &ffn_gate, &ffn_up, &ffn_down, mlp_config)
        .map_err(|source| E2eError::inference("MlpWeights::from_model", source))?;
    let ffn_norm_values = decode_norm_tensor(model, &ffn_norm, hidden_features, "ffn_norm")?;

    Ok(LayerPlan {
        attention: None,
        mlp: MlpLayerPlan {
            weights: mlp_weights,
            norm_values: ffn_norm_values,
        },
    })
}

fn resolve_global_tensor_names(model: &GgufModel) -> Result<GlobalTensorNames, E2eError> {
    let token_embedding = resolve_required_global_tensor_name(
        model,
        "token_embedding",
        global_token_embedding_candidates(),
    )?;
    let output_norm =
        resolve_required_global_tensor_name(model, "output_norm", global_output_norm_candidates())?;
    let output = resolve_optional_global_tensor_name(model, global_output_projection_candidates());

    Ok(GlobalTensorNames {
        token_embedding,
        output_norm,
        output,
    })
}

fn resolve_required_global_tensor_name(
    model: &GgufModel,
    role: &'static str,
    candidates: Vec<String>,
) -> Result<String, E2eError> {
    if let Some(name) = resolve_first_tensor_name(model, &candidates) {
        return Ok(name);
    }
    Err(E2eError::naming(
        "resolve_required_global_tensor_name",
        NamingError::MissingGlobalTensor {
            role,
            tried: candidates,
        },
    ))
}

fn resolve_optional_global_tensor_name(
    model: &GgufModel,
    candidates: Vec<String>,
) -> Option<String> {
    resolve_first_tensor_name(model, &candidates)
}

fn resolve_required_layer_tensor_name(
    model: &GgufModel,
    layer: usize,
    role: &'static str,
    candidates: Vec<String>,
) -> Result<String, E2eError> {
    if let Some(name) = resolve_first_tensor_name(model, &candidates) {
        return Ok(name);
    }
    Err(E2eError::naming(
        "resolve_required_layer_tensor_name",
        NamingError::MissingLayerTensor {
            layer,
            role,
            tried: candidates,
        },
    ))
}

fn resolve_first_tensor_name(model: &GgufModel, candidates: &[String]) -> Option<String> {
    candidates
        .iter()
        .find(|name| model.find_tensor(name.as_str()).is_some())
        .cloned()
}

fn global_token_embedding_candidates() -> Vec<String> {
    vec![
        "token_embd.weight".to_string(),
        "tok_embeddings.weight".to_string(),
        "model.embed_tokens.weight".to_string(),
    ]
}

fn global_output_norm_candidates() -> Vec<String> {
    vec![
        "output_norm.weight".to_string(),
        "norm.weight".to_string(),
        "model.norm.weight".to_string(),
    ]
}

fn global_output_projection_candidates() -> Vec<String> {
    vec![
        "output.weight".to_string(),
        "lm_head.weight".to_string(),
        "model.lm_head.weight".to_string(),
    ]
}

fn layer_ffn_norm_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.ffn_norm.weight"),
        format!("blk.{layer}.post_attention_norm.weight"),
        format!("layers.{layer}.ffn_norm.weight"),
        format!("model.layers.{layer}.post_attention_layernorm.weight"),
        format!("model.layers.{layer}.post_attention_norm.weight"),
    ]
}

fn layer_ffn_gate_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.ffn_gate.weight"),
        format!("layers.{layer}.feed_forward.w1.weight"),
        format!("model.layers.{layer}.mlp.gate_proj.weight"),
    ]
}

fn layer_ffn_up_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.ffn_up.weight"),
        format!("layers.{layer}.feed_forward.w3.weight"),
        format!("model.layers.{layer}.mlp.up_proj.weight"),
    ]
}

fn layer_ffn_down_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.ffn_down.weight"),
        format!("layers.{layer}.feed_forward.w2.weight"),
        format!("model.layers.{layer}.mlp.down_proj.weight"),
    ]
}

fn decode_norm_tensor(
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

fn gather_embeddings(
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
    for (position, &token_id) in token_ids.iter().enumerate() {
        let token_index = validate_token_id(token_id, vocab_size)?;
        let src_offset = checked_mul(token_index, hidden_features)?;
        let dst_offset = checked_mul(position, hidden_features)?;
        output[dst_offset..dst_offset + hidden_features]
            .copy_from_slice(&embedding_values[src_offset..src_offset + hidden_features]);
    }
    Ok(output)
}

fn rms_norm_with_weight(
    input: &[f32],
    hidden_features: usize,
    sequence_length: usize,
    weight: &[f32],
) -> Result<Vec<f32>, E2eError> {
    if weight.len() != hidden_features {
        return Err(E2eError::BufferLengthMismatch {
            expected: hidden_features,
            actual: weight.len(),
        });
    }
    let expected_input_len = checked_mul(hidden_features, sequence_length)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }

    let mut output = vec![0.0_f32; input.len()];
    for token in 0..sequence_length {
        let offset = checked_mul(token, hidden_features)?;
        let slice = &input[offset..offset + hidden_features];
        let mean_square = slice
            .iter()
            .copied()
            .map(|value| f64::from(value) * f64::from(value))
            .sum::<f64>()
            / hidden_features as f64;
        let inv_rms = 1.0_f32 / ((mean_square as f32) + RMS_NORM_EPS).sqrt();
        for (index, value) in slice.iter().copied().enumerate() {
            output[offset + index] = value * inv_rms * weight[index];
        }
    }
    Ok(output)
}

fn add_in_place(accumulator: &mut [f32], addend: &[f32]) -> Result<(), E2eError> {
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

fn mlp_sequence_inference_with_weights(
    weights: &MlpWeights<f32>,
    input: &[f32],
    sequence_length: usize,
    backend: &Backend,
) -> Result<Vec<f32>, E2eError> {
    let hidden_features = weights.hidden_features;
    let ffn_features = weights.ffn_features;
    let expected_input_len = checked_mul(hidden_features, sequence_length)?;
    if input.len() != expected_input_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_input_len,
            actual: input.len(),
        });
    }

    let ctx_size =
        recommended_mlp_backend_memory_bytes(hidden_features, ffn_features, sequence_length)?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| E2eError::ggml("Context::new_no_alloc_bytes", source))?;

    let w_gate = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, ffn_features))
        .map_err(|source| E2eError::ggml("Context::new_tensor_2d<W_GATE>", source))?;
    let w_up = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, ffn_features))
        .map_err(|source| E2eError::ggml("Context::new_tensor_2d<W_UP>", source))?;
    let w_down = ctx
        .new_tensor_2d::<f32>(Shape2D::new(ffn_features, hidden_features))
        .map_err(|source| E2eError::ggml("Context::new_tensor_2d<W_DOWN>", source))?;
    let x = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden_features, sequence_length))
        .map_err(|source| E2eError::ggml("Context::new_tensor_2d<X>", source))?;

    let gate = ctx
        .mul_mat(&w_gate, &x)
        .map_err(|source| E2eError::ggml("Context::mul_mat(GATE)", source))?;
    let up = ctx
        .mul_mat(&w_up, &x)
        .map_err(|source| E2eError::ggml("Context::mul_mat(UP)", source))?;
    let activated = ctx
        .silu(&gate)
        .map_err(|source| E2eError::ggml("Context::silu", source))?;
    let fused = ctx
        .mul(&activated, &up)
        .map_err(|source| E2eError::ggml("Context::mul(GATE*UP)", source))?;
    let y = ctx
        .mul_mat(&w_down, &fused)
        .map_err(|source| E2eError::ggml("Context::mul_mat(DOWN)", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| E2eError::ggml("Context::new_graph", source))?;
    graph.build_forward_expand(&y);
    let _buffer = ctx
        .allocate_tensors(backend)
        .map_err(|source| E2eError::ggml("Context::allocate_tensors", source))?;

    w_gate
        .write_data_backend(weights.gate_values())
        .map_err(|source| E2eError::ggml("Tensor::write_data_backend<W_GATE>", source))?;
    w_up.write_data_backend(weights.up_values())
        .map_err(|source| E2eError::ggml("Tensor::write_data_backend<W_UP>", source))?;
    w_down
        .write_data_backend(weights.down_values())
        .map_err(|source| E2eError::ggml("Tensor::write_data_backend<W_DOWN>", source))?;
    x.write_data_backend(input)
        .map_err(|source| E2eError::ggml("Tensor::write_data_backend<X>", source))?;

    backend
        .compute(&mut graph)
        .map_err(|source| E2eError::ggml("Backend::compute", source))?;

    y.read_data_backend::<f32>()
        .map_err(|source| E2eError::ggml("Tensor::read_data_backend<Y>", source))
}

fn recommended_mlp_backend_memory_bytes(
    hidden_features: usize,
    ffn_features: usize,
    sequence_length: usize,
) -> Result<Bytes, E2eError> {
    let gate_projection = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, ffn_features),
        Shape2D::new(hidden_features, sequence_length),
    )
    .map_err(|source| E2eError::ggml("Context::recommended_backend_matmul_memory(gate)", source))?;
    let up_projection = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(hidden_features, ffn_features),
        Shape2D::new(hidden_features, sequence_length),
    )
    .map_err(|source| E2eError::ggml("Context::recommended_backend_matmul_memory(up)", source))?;
    let down_projection = Context::recommended_backend_matmul_memory::<f32>(
        Shape2D::new(ffn_features, hidden_features),
        Shape2D::new(ffn_features, sequence_length),
    )
    .map_err(|source| E2eError::ggml("Context::recommended_backend_matmul_memory(down)", source))?;

    let total = gate_projection
        .get()
        .checked_add(up_projection.get())
        .and_then(|value| value.checked_add(down_projection.get()))
        .and_then(|value| value.checked_add(MLP_BACKEND_SLACK_BYTES))
        .ok_or(E2eError::MemorySizeOverflow)?;
    Ok(Bytes::new(total))
}

fn greedy_next_token_id(
    hidden_states: &[f32],
    token_index: usize,
    hidden_features: usize,
    output_weight: &[f32],
    vocab_size: usize,
) -> Result<i32, E2eError> {
    let expected_hidden = checked_mul(hidden_features, token_index + 1)?;
    if hidden_states.len() < expected_hidden {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_hidden,
            actual: hidden_states.len(),
        });
    }
    let expected_output_len = checked_mul(hidden_features, vocab_size)?;
    if output_weight.len() != expected_output_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_output_len,
            actual: output_weight.len(),
        });
    }

    let offset = checked_mul(token_index, hidden_features)?;
    let last_hidden = &hidden_states[offset..offset + hidden_features];
    let mut best_token = 0usize;
    let mut best_logit = f32::NEG_INFINITY;
    for (token, row) in output_weight.chunks_exact(hidden_features).enumerate() {
        let logit = row
            .iter()
            .copied()
            .zip(last_hidden.iter().copied())
            .fold(0.0_f32, |acc, (weight, value)| acc + weight * value);
        if logit > best_logit {
            best_logit = logit;
            best_token = token;
        }
    }
    i32::try_from(best_token).map_err(|_| E2eError::MemorySizeOverflow)
}

fn validate_token_id(token_id: i32, vocab_size: usize) -> Result<usize, E2eError> {
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

fn checked_mul(lhs: usize, rhs: usize) -> Result<usize, E2eError> {
    lhs.checked_mul(rhs).ok_or(E2eError::MemorySizeOverflow)
}

fn value_to_i32(value: &GgufValue) -> Option<i32> {
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

#[cfg(test)]
mod tests {
    use super::{greedy_next_token_id, rms_norm_with_weight, value_to_i32};
    use ggml_rs::GgufValue;

    #[test]
    fn converts_numeric_gguf_values_to_i32() {
        assert_eq!(value_to_i32(&GgufValue::U32(12)), Some(12));
        assert_eq!(value_to_i32(&GgufValue::I32(-5)), Some(-5));
        assert_eq!(value_to_i32(&GgufValue::F32(42.0)), Some(42));
        assert_eq!(value_to_i32(&GgufValue::String("x".to_string())), None);
    }

    #[test]
    fn rms_norm_applies_weight_per_position() {
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32, 0.25];
        let output = rms_norm_with_weight(&input, 2, 2, &weight).expect("rms norm should succeed");
        assert_eq!(output.len(), input.len());
        // Ensure both positions are normalized independently and weighted.
        assert!(output[0].is_finite());
        assert!(output[1].is_finite());
        assert!(output[2].is_finite());
        assert!(output[3].is_finite());
        assert!(output[0].abs() > output[1].abs());
    }

    #[test]
    fn greedy_sampler_picks_largest_logit_row() {
        let hidden_states = vec![
            // token 0
            0.0_f32, 0.0, // token 1 (target)
            1.0, 2.0,
        ];
        // vocab=3, hidden=2 (rows: token logits weights)
        let output_weight = vec![
            0.0_f32, 0.0, // dot = 0
            1.0, 0.0, // dot = 1
            0.0, 3.0, // dot = 6 (best)
        ];
        let token = greedy_next_token_id(&hidden_states, 1, 2, &output_weight, 3)
            .expect("sampler should succeed");
        assert_eq!(token, 2);
    }
}
