//! Public entry points for token generation.
//!
//! [`generate_token_ids_from_path`] and [`generate_token_ids_from_model`] are
//! the main API, plus helpers for EOS resolution and tokenization.

use super::super::config::{E2eGenerationConfig, E2eGenerationReport};
use super::super::decode::decode_norm_tensor;
use super::super::error::{E2eError, GgmlResultExt};
use super::super::numeric::{checked_mul, validate_token_id};
use super::super::planner::build_layer_plans;
use super::super::resolve::resolve_global_tensor_names;
use super::{GenerationInputs, GenerationMode};
use crate::backend::ensure_backends_loaded;
use crate::metadata::resolve_transformer_metadata;
use crate::model::GgufModel;
use crate::tokenizer::tokenize_text_prompt;
use ggml_rs::Backend;
use std::path::Path;
use std::time::Instant;

pub fn resolve_eos_token_id(model: &GgufModel) -> Option<i32> {
    use super::super::numeric::value_to_i32;
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
    let rms_norm_eps = metadata.attention_layer_norm_rms_epsilon();
    let global_names = resolve_global_tensor_names(model)?;

    let (token_embedding_values, vocab_size) =
        load_token_embeddings(model, &global_names.token_embedding, hidden_features)?;
    let output_weight_values = load_output_weights(
        model,
        global_names.output.as_deref(),
        hidden_features,
        vocab_size,
    )?;
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
        &metadata,
        hidden_features,
        total_sequence_length,
        config.mixed_layer_policy,
    )?;
    let attention_layer_count = layer_plans
        .iter()
        .filter(|layer_plan| layer_plan.attention.is_some())
        .count();
    let mlp_only_layer_count = layer_plans.len() - attention_layer_count;

    validate_config_token_ids(config, vocab_size)?;

    ensure_backends_loaded();
    let backend = Backend::new(config.backend.into()).ggml_ctx("Backend::new")?;
    let backend_name = backend
        .name()
        .map(|name: &str| name.to_string())
        .ggml_ctx("Backend::name")?;

    let start = Instant::now();
    let inputs = GenerationInputs {
        layer_plans: &layer_plans,
        token_embedding_values: &token_embedding_values,
        output_weight_values,
        output_norm_values: &output_norm_values,
        hidden_features,
        vocab_size,
        rms_norm_eps,
        prompt_token_ids: &config.prompt_token_ids,
        max_new_tokens: config.max_new_tokens,
        pad_token_id: config.pad_token_id,
        eos_token_id: config.eos_token_id,
        backend: &backend,
        total_sequence_length,
    };
    let output = super::loops::generate_from_plans(&inputs, GenerationMode::Auto)?;
    let elapsed = start.elapsed();

    Ok(E2eGenerationReport {
        backend_name,
        prompt_token_count,
        generated_token_ids: output.generated_token_ids,
        all_token_ids: output.all_token_ids,
        attention_layer_count,
        mlp_only_layer_count,
        elapsed,
    })
}

/// Load and validate token embedding tensor, returning values and derived vocab size.
fn load_token_embeddings(
    model: &GgufModel,
    tensor_name: &str,
    hidden_features: usize,
) -> Result<(Vec<f32>, usize), E2eError> {
    let values = model
        .tensor_values::<f32>(tensor_name)
        .map_err(|source| E2eError::model("GgufModel::tensor_values(token_embedding)", source))?;
    if hidden_features == 0 || !values.len().is_multiple_of(hidden_features) {
        return Err(E2eError::InvalidTokenEmbeddingShape {
            tensor_name: tensor_name.to_string(),
            hidden_features,
            tensor_len: values.len(),
        });
    }
    let vocab_size = values.len() / hidden_features;
    Ok((values, vocab_size))
}

/// Load optional output projection weights, validating shape against vocab × hidden.
fn load_output_weights(
    model: &GgufModel,
    output_name: Option<&str>,
    hidden_features: usize,
    vocab_size: usize,
) -> Result<Option<Vec<f32>>, E2eError> {
    let Some(output_name) = output_name else {
        return Ok(None);
    };
    let values = model
        .tensor_values::<f32>(output_name)
        .map_err(|source| E2eError::model("GgufModel::tensor_values(output_projection)", source))?;
    let expected = checked_mul(hidden_features, vocab_size)?;
    if values.len() != expected {
        return Err(E2eError::OutputWeightLengthMismatch {
            tensor_name: output_name.to_string(),
            expected,
            actual: values.len(),
        });
    }
    Ok(Some(values))
}

/// Validate that all token IDs in the config (pad, eos, prompt) are within vocab range.
fn validate_config_token_ids(
    config: &E2eGenerationConfig,
    vocab_size: usize,
) -> Result<(), E2eError> {
    validate_token_id(config.pad_token_id, vocab_size)?;
    if let Some(eos_token_id) = config.eos_token_id {
        validate_token_id(eos_token_id, vocab_size)?;
    }
    for &token_id in &config.prompt_token_ids {
        validate_token_id(token_id, vocab_size)?;
    }
    Ok(())
}
