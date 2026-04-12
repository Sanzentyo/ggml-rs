use super::attention::qwen35_full_attention_inference;
use super::config::{E2eGenerationConfig, E2eGenerationReport};
use super::decode::decode_norm_tensor;
use super::error::E2eError;
use super::linear_attention::qwen35_linear_attention_inference;
use super::mlp::mlp_sequence_inference_with_weights;
use super::numeric::{checked_mul, validate_token_id, value_to_i32};
use super::plan::AttentionLayerPlan;
use super::planner::build_layer_plans;
use super::resolve::resolve_global_tensor_names;
use super::tensor_ops::{add_in_place, gather_embeddings, rms_norm_with_weight};
use crate::backend::ensure_backends_loaded;
use crate::inference::attention_inference_with_weights_on_backend_repeats_with_length;
use crate::metadata::resolve_transformer_metadata;
use crate::model::GgufModel;
use crate::tokenizer::tokenize_text_prompt;
use ggml_rs::Backend;
use std::path::Path;
use std::time::Instant;

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
    let rms_norm_eps = metadata.attention_layer_norm_rms_epsilon();
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
                let attention_output = match attention {
                    AttentionLayerPlan::Standard(attention) => {
                        let normalized_attn = rms_norm_with_weight(
                            &hidden,
                            hidden_features,
                            current_token_count,
                            &attention.norm_values,
                            rms_norm_eps,
                        )?;
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
                        })?
                    }
                    AttentionLayerPlan::Qwen35Full(attention) => {
                        let normalized_attn = rms_norm_with_weight(
                            &hidden,
                            hidden_features,
                            current_token_count,
                            &attention.norm_values,
                            rms_norm_eps,
                        )?;
                        qwen35_full_attention_inference(
                            attention,
                            &normalized_attn,
                            current_token_count,
                            rms_norm_eps,
                        )?
                    }
                    AttentionLayerPlan::Qwen35Linear(attention) => {
                        let normalized_attn = rms_norm_with_weight(
                            &hidden,
                            hidden_features,
                            current_token_count,
                            &attention.norm_values,
                            rms_norm_eps,
                        )?;
                        qwen35_linear_attention_inference(
                            attention,
                            &normalized_attn,
                            current_token_count,
                            rms_norm_eps,
                        )?
                    }
                };
                add_in_place(&mut hidden, &attention_output)?;
            }

            let normalized_ffn = rms_norm_with_weight(
                &hidden,
                hidden_features,
                current_token_count,
                &layer_plan.mlp.norm_values,
                rms_norm_eps,
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
            rms_norm_eps,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampler_picks_largest_logit_row() {
        let hidden_states = vec![0.0_f32, 0.0, 1.0, 2.0];
        let output_weight = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 3.0];
        let token = greedy_next_token_id(&hidden_states, 1, 2, &output_weight, 3)
            .expect("sampler should succeed");
        assert_eq!(token, 2);
    }
}
