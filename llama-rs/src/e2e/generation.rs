use super::attention::{
    qwen35_full_attention_decode_step, qwen35_full_attention_inference,
    qwen35_full_attention_prefill,
};
use super::config::{E2eGenerationConfig, E2eGenerationReport};
use super::decode::decode_norm_tensor;
use super::error::E2eError;
use super::linear_attention::{
    qwen35_linear_attention_decode_step, qwen35_linear_attention_inference,
    qwen35_linear_attention_prefill,
};
use super::mlp::mlp_sequence_inference_with_weights;
use super::numeric::{checked_mul, validate_token_id, value_to_i32};
use super::plan::{AttentionLayerPlan, LayerPlan};
use super::planner::build_layer_plans;
use super::resolve::resolve_global_tensor_names;
use super::state::{GenerationState, LayerAttentionState};
use super::tensor_ops::{add_in_place, gather_embeddings, rms_norm_with_weight};
use crate::backend::ensure_backends_loaded;
use crate::inference::attention_inference_with_weights_on_backend_repeats_with_length;
use crate::metadata::resolve_transformer_metadata;
use crate::model::GgufModel;
use crate::tokenizer::tokenize_text_prompt;
use ggml_rs::Backend;
use std::path::Path;
use std::time::Instant;

/// Controls which execution strategy the generation loop uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GenerationMode {
    /// Automatically select based on layer types:
    /// - Standard attention present → FullReprocess
    /// - All Qwen3.5 → TwoPhase
    Auto,
    /// Always reprocess all tokens from scratch each step.
    FullReprocess,
    /// Prefill all prompt tokens, then decode one token at a time using cached state.
    TwoPhase,
}

/// Bundles all pre-resolved inputs needed by the core generation loop.
pub(super) struct GenerationInputs<'a> {
    pub layer_plans: &'a [LayerPlan],
    pub token_embedding_values: &'a [f32],
    pub output_weight_values: &'a [f32],
    pub output_norm_values: &'a [f32],
    pub hidden_features: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub prompt_token_ids: &'a [i32],
    pub max_new_tokens: usize,
    pub pad_token_id: i32,
    pub eos_token_id: Option<i32>,
    pub backend: &'a Backend,
    pub total_sequence_length: usize,
}

/// Result of the core generation loop (before wrapping in the public report).
pub(super) struct GenerationOutput {
    pub generated_token_ids: Vec<i32>,
    pub all_token_ids: Vec<i32>,
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
    ensure_backends_loaded();
    let backend = Backend::new(config.backend.into())
        .map_err(|source| E2eError::ggml("Backend::new", source))?;
    let backend_name = backend
        .name()
        .map(|name| name.to_string())
        .map_err(|source| E2eError::ggml("Backend::name", source))?;

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
    let output = generate_from_plans(&inputs, GenerationMode::Auto)?;
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

/// Core generation loop, decoupled from model resolution.
///
/// When `mode` is `Auto`, selects `FullReprocess` if Standard attention is present
/// or `max_new_tokens == 0`, otherwise `TwoPhase`.
pub(super) fn generate_from_plans(
    inputs: &GenerationInputs<'_>,
    mode: GenerationMode,
) -> Result<GenerationOutput, E2eError> {
    let prompt_token_count = inputs.prompt_token_ids.len();
    if prompt_token_count == 0 {
        return Err(E2eError::EmptyPrompt);
    }

    let mut all_token_ids = vec![inputs.pad_token_id; inputs.total_sequence_length];
    all_token_ids[..prompt_token_count].copy_from_slice(inputs.prompt_token_ids);
    let mut generated_token_ids = Vec::with_capacity(inputs.max_new_tokens);
    let mut current_token_count = prompt_token_count;

    let effective_mode = match mode {
        GenerationMode::Auto => {
            let has_standard = inputs
                .layer_plans
                .iter()
                .any(|p| matches!(p.attention, Some(AttentionLayerPlan::Standard(_))));
            if has_standard || inputs.max_new_tokens == 0 {
                GenerationMode::FullReprocess
            } else {
                GenerationMode::TwoPhase
            }
        }
        other => other,
    };

    match effective_mode {
        GenerationMode::FullReprocess | GenerationMode::Auto => {
            full_reprocess_loop(
                inputs,
                &mut all_token_ids,
                &mut generated_token_ids,
                &mut current_token_count,
            )?;
        }
        GenerationMode::TwoPhase => {
            two_phase_loop(
                inputs,
                &mut all_token_ids,
                &mut generated_token_ids,
                &mut current_token_count,
            )?;
        }
    }

    Ok(GenerationOutput {
        generated_token_ids,
        all_token_ids: all_token_ids[..current_token_count].to_vec(),
    })
}

fn full_reprocess_loop(
    inputs: &GenerationInputs<'_>,
    all_token_ids: &mut [i32],
    generated_token_ids: &mut Vec<i32>,
    current_token_count: &mut usize,
) -> Result<(), E2eError> {
    for _step in 0..inputs.max_new_tokens {
        let active_token_ids = &all_token_ids[..*current_token_count];
        let mut hidden = gather_embeddings(
            inputs.token_embedding_values,
            inputs.hidden_features,
            inputs.vocab_size,
            active_token_ids,
        )?;

        for layer_plan in inputs.layer_plans {
            if let Some(attention) = &layer_plan.attention {
                let attention_output = match attention {
                    AttentionLayerPlan::Standard(attention) => {
                        let normalized_attn = rms_norm_with_weight(
                            &hidden,
                            inputs.hidden_features,
                            *current_token_count,
                            &attention.norm_values,
                            inputs.rms_norm_eps,
                        )?;
                        attention_inference_with_weights_on_backend_repeats_with_length(
                            &attention.weights,
                            &normalized_attn,
                            *current_token_count,
                            inputs.backend,
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
                            inputs.hidden_features,
                            *current_token_count,
                            &attention.norm_values,
                            inputs.rms_norm_eps,
                        )?;
                        qwen35_full_attention_inference(
                            attention,
                            &normalized_attn,
                            *current_token_count,
                            inputs.rms_norm_eps,
                        )?
                    }
                    AttentionLayerPlan::Qwen35Linear(attention) => {
                        let normalized_attn = rms_norm_with_weight(
                            &hidden,
                            inputs.hidden_features,
                            *current_token_count,
                            &attention.norm_values,
                            inputs.rms_norm_eps,
                        )?;
                        qwen35_linear_attention_inference(
                            attention,
                            &normalized_attn,
                            *current_token_count,
                            inputs.rms_norm_eps,
                        )?
                    }
                };
                add_in_place(&mut hidden, &attention_output)?;
            }

            let normalized_ffn = rms_norm_with_weight(
                &hidden,
                inputs.hidden_features,
                *current_token_count,
                &layer_plan.mlp.norm_values,
                inputs.rms_norm_eps,
            )?;
            let mlp_output = mlp_sequence_inference_with_weights(
                &layer_plan.mlp.weights,
                &normalized_ffn,
                *current_token_count,
                inputs.backend,
            )?;
            add_in_place(&mut hidden, &mlp_output)?;
        }

        let normalized_output = rms_norm_with_weight(
            &hidden,
            inputs.hidden_features,
            *current_token_count,
            inputs.output_norm_values,
            inputs.rms_norm_eps,
        )?;
        let last_index = current_token_count
            .checked_sub(1)
            .ok_or(E2eError::EmptyPrompt)?;
        let next_token_id = greedy_next_token_id(
            &normalized_output,
            last_index,
            inputs.hidden_features,
            inputs.output_weight_values,
            inputs.vocab_size,
        )?;

        generated_token_ids.push(next_token_id);
        if *current_token_count < inputs.total_sequence_length {
            all_token_ids[*current_token_count] = next_token_id;
            *current_token_count += 1;
        }

        if inputs.eos_token_id.is_some_and(|eos| eos == next_token_id) {
            break;
        }
    }
    Ok(())
}

fn two_phase_loop(
    inputs: &GenerationInputs<'_>,
    all_token_ids: &mut [i32],
    generated_token_ids: &mut Vec<i32>,
    current_token_count: &mut usize,
) -> Result<(), E2eError> {
    let prompt_token_count = inputs.prompt_token_ids.len();
    let mut state = GenerationState::new(inputs.layer_plans, inputs.total_sequence_length)?;

    // Phase 1: Prefill — process all prompt tokens at once, capturing state.
    let prompt_ids = &all_token_ids[..prompt_token_count];
    let mut hidden = gather_embeddings(
        inputs.token_embedding_values,
        inputs.hidden_features,
        inputs.vocab_size,
        prompt_ids,
    )?;

    for (layer_idx, layer_plan) in inputs.layer_plans.iter().enumerate() {
        if let Some(attention) = &layer_plan.attention {
            let attention_output = match (attention, &mut state.layers[layer_idx]) {
                (AttentionLayerPlan::Qwen35Full(attn), LayerAttentionState::Qwen35Full(s)) => {
                    let normalized_attn = rms_norm_with_weight(
                        &hidden,
                        inputs.hidden_features,
                        prompt_token_count,
                        &attn.norm_values,
                        inputs.rms_norm_eps,
                    )?;
                    qwen35_full_attention_prefill(
                        attn,
                        &normalized_attn,
                        prompt_token_count,
                        inputs.rms_norm_eps,
                        s,
                    )?
                }
                (AttentionLayerPlan::Qwen35Linear(attn), LayerAttentionState::Qwen35Linear(s)) => {
                    let normalized_attn = rms_norm_with_weight(
                        &hidden,
                        inputs.hidden_features,
                        prompt_token_count,
                        &attn.norm_values,
                        inputs.rms_norm_eps,
                    )?;
                    qwen35_linear_attention_prefill(
                        attn,
                        &normalized_attn,
                        prompt_token_count,
                        inputs.rms_norm_eps,
                        s,
                    )?
                }
                _ => {
                    unreachable!("two-phase loop should not encounter Standard or mismatched state")
                }
            };
            add_in_place(&mut hidden, &attention_output)?;
        }

        let normalized_ffn = rms_norm_with_weight(
            &hidden,
            inputs.hidden_features,
            prompt_token_count,
            &layer_plan.mlp.norm_values,
            inputs.rms_norm_eps,
        )?;
        let mlp_output = mlp_sequence_inference_with_weights(
            &layer_plan.mlp.weights,
            &normalized_ffn,
            prompt_token_count,
            inputs.backend,
        )?;
        add_in_place(&mut hidden, &mlp_output)?;
    }

    // Sample first token from the last prompt position.
    let normalized_output = rms_norm_with_weight(
        &hidden,
        inputs.hidden_features,
        prompt_token_count,
        inputs.output_norm_values,
        inputs.rms_norm_eps,
    )?;
    let last_index = prompt_token_count
        .checked_sub(1)
        .ok_or(E2eError::EmptyPrompt)?;
    let first_token_id = greedy_next_token_id(
        &normalized_output,
        last_index,
        inputs.hidden_features,
        inputs.output_weight_values,
        inputs.vocab_size,
    )?;

    generated_token_ids.push(first_token_id);
    all_token_ids[prompt_token_count] = first_token_id;
    *current_token_count = prompt_token_count + 1;

    let eos_hit = inputs.eos_token_id.is_some_and(|eos| eos == first_token_id);

    // Phase 2: Decode — one token at a time using cached state.
    if !eos_hit {
        for _step in 1..inputs.max_new_tokens {
            let new_token_id = all_token_ids[*current_token_count - 1];
            let mut hidden = gather_embeddings(
                inputs.token_embedding_values,
                inputs.hidden_features,
                inputs.vocab_size,
                &[new_token_id],
            )?;

            for (layer_idx, layer_plan) in inputs.layer_plans.iter().enumerate() {
                if let Some(attention) = &layer_plan.attention {
                    let attention_output = match (attention, &mut state.layers[layer_idx]) {
                        (
                            AttentionLayerPlan::Qwen35Full(attn),
                            LayerAttentionState::Qwen35Full(s),
                        ) => {
                            let normalized_attn = rms_norm_with_weight(
                                &hidden,
                                inputs.hidden_features,
                                1,
                                &attn.norm_values,
                                inputs.rms_norm_eps,
                            )?;
                            qwen35_full_attention_decode_step(
                                attn,
                                &normalized_attn,
                                inputs.rms_norm_eps,
                                s,
                            )?
                        }
                        (
                            AttentionLayerPlan::Qwen35Linear(attn),
                            LayerAttentionState::Qwen35Linear(s),
                        ) => {
                            let normalized_attn = rms_norm_with_weight(
                                &hidden,
                                inputs.hidden_features,
                                1,
                                &attn.norm_values,
                                inputs.rms_norm_eps,
                            )?;
                            qwen35_linear_attention_decode_step(
                                attn,
                                &normalized_attn,
                                inputs.rms_norm_eps,
                                s,
                            )?
                        }
                        _ => unreachable!(
                            "two-phase loop should not encounter Standard or mismatched state"
                        ),
                    };
                    add_in_place(&mut hidden, &attention_output)?;
                }

                let normalized_ffn = rms_norm_with_weight(
                    &hidden,
                    inputs.hidden_features,
                    1,
                    &layer_plan.mlp.norm_values,
                    inputs.rms_norm_eps,
                )?;
                let mlp_output = mlp_sequence_inference_with_weights(
                    &layer_plan.mlp.weights,
                    &normalized_ffn,
                    1,
                    inputs.backend,
                )?;
                add_in_place(&mut hidden, &mlp_output)?;
            }

            let normalized_output = rms_norm_with_weight(
                &hidden,
                inputs.hidden_features,
                1,
                inputs.output_norm_values,
                inputs.rms_norm_eps,
            )?;
            let next_token_id = greedy_next_token_id(
                &normalized_output,
                0,
                inputs.hidden_features,
                inputs.output_weight_values,
                inputs.vocab_size,
            )?;

            generated_token_ids.push(next_token_id);
            if *current_token_count < inputs.total_sequence_length {
                all_token_ids[*current_token_count] = next_token_id;
                *current_token_count += 1;
            }

            if inputs.eos_token_id.is_some_and(|eos| eos == next_token_id) {
                break;
            }
        }
    }
    Ok(())
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

    /// Multi-layer integration test: verifies FullReprocess and TwoPhase produce
    /// identical token sequences on a synthetic 3-layer model.
    ///
    /// Layers: Qwen35Linear → Qwen35Full → Qwen35Linear (mimics real pattern).
    /// This tests the orchestration code (residual connections, MLP pass-through,
    /// embedding lookup, token sampling) that individual layer tests don't cover.
    #[cfg(feature = "link-system")]
    #[test]
    fn two_phase_matches_full_reprocess_multi_layer() {
        use super::super::plan::{
            MlpLayerPlan, Qwen35FullAttentionLayerPlan, Qwen35LinearAttentionLayerPlan,
        };
        use crate::backend::ensure_backends_loaded;
        use crate::inference::{MlpInferenceConfig, MlpWeights};

        let hidden = 8_usize;
        let ffn = 16_usize;
        let vocab = 6_usize;
        let head_count = 2_usize;
        let kv_head_count = 1_usize;
        let hd = 4_usize;
        let rms_norm_eps = 1e-5_f32;

        // Linear attention dims.
        let group_count = 2_usize;
        let time_step_rank = 4_usize;
        let state_size = 2_usize;
        let inner_size = time_step_rank * state_size;
        let conv_channels = inner_size + 2 * group_count * state_size;
        let conv_kernel = 2_usize;

        let make_mlp = || -> MlpLayerPlan {
            let config = MlpInferenceConfig::new(hidden, ffn).unwrap();
            MlpLayerPlan {
                weights: MlpWeights::deterministic(config),
                norm_values: vec![1.0_f32; hidden],
            }
        };

        let make_linear = |seed: usize| -> Qwen35LinearAttentionLayerPlan {
            let mut qkv_w = vec![0.0_f32; hidden * conv_channels];
            for i in 0..hidden.min(conv_channels) {
                qkv_w[i * conv_channels + i] = 1.0;
            }
            let mut gate_w = vec![0.0_f32; hidden * inner_size];
            for i in 0..hidden.min(inner_size) {
                gate_w[i * inner_size + i] = 0.5;
            }
            let alpha_w: Vec<f32> = (0..hidden * time_step_rank)
                .map(|i| ((i + seed) % 13) as f32 * 0.005)
                .collect();
            let beta_w: Vec<f32> = (0..hidden * time_step_rank)
                .map(|i| ((i + seed + 3) % 11) as f32 * 0.005)
                .collect();
            let mut conv_w = vec![0.0_f32; conv_channels * conv_kernel];
            for ch in 0..conv_channels {
                conv_w[ch * conv_kernel + (conv_kernel - 1)] = 1.0;
            }
            let mut ssm_out_w = vec![0.0_f32; inner_size * hidden];
            for i in 0..inner_size.min(hidden) {
                ssm_out_w[i * hidden + i] = 1.0;
            }

            Qwen35LinearAttentionLayerPlan {
                norm_values: vec![1.0_f32; hidden],
                qkv_weight_values: qkv_w,
                gate_weight_values: gate_w,
                alpha_weight_values: alpha_w,
                beta_weight_values: beta_w,
                conv_weight_values: conv_w,
                dt_bias_values: vec![0.0_f32; time_step_rank],
                ssm_a_values: vec![-1.0_f32; time_step_rank],
                ssm_norm_values: vec![1.0_f32; state_size],
                ssm_out_weight_values: ssm_out_w,
                state_size,
                group_count,
                time_step_rank,
                inner_size,
                conv_kernel,
            }
        };

        let make_full = || -> Qwen35FullAttentionLayerPlan {
            let query_features = head_count * hd;
            let kv_features = kv_head_count * hd;
            Qwen35FullAttentionLayerPlan {
                norm_values: vec![1.0_f32; hidden],
                q_norm_values: vec![1.0_f32; hd],
                k_norm_values: vec![1.0_f32; hd],
                q_weight_values: (0..hidden * query_features * 2)
                    .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
                    .collect(),
                k_weight_values: (0..hidden * kv_features)
                    .map(|i| ((i % 5) as f32 - 2.0) * 0.08)
                    .collect(),
                v_weight_values: (0..hidden * kv_features)
                    .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
                    .collect(),
                output_weight_values: (0..query_features * hidden)
                    .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
                    .collect(),
                head_count,
                kv_head_count,
                head_dimension: hd,
                attention_scale: 1.0 / (hd as f32).sqrt(),
                rope_n_dims: hd,
                rope_freq_base: 10000.0,
                rope_freq_scale: 1.0,
            }
        };

        // 3 layers: Linear → Full → Linear (mimics Qwen3.5 pattern).
        let layer_plans = vec![
            LayerPlan {
                attention: Some(AttentionLayerPlan::Qwen35Linear(make_linear(0))),
                mlp: make_mlp(),
            },
            LayerPlan {
                attention: Some(AttentionLayerPlan::Qwen35Full(make_full())),
                mlp: make_mlp(),
            },
            LayerPlan {
                attention: Some(AttentionLayerPlan::Qwen35Linear(make_linear(7))),
                mlp: make_mlp(),
            },
        ];

        // Asymmetric embeddings and output projection to avoid argmax ties.
        let token_embeddings: Vec<f32> = (0..vocab * hidden)
            .map(|i| ((i * 7 + 3) % 37) as f32 * 0.02 - 0.35)
            .collect();
        let output_weight: Vec<f32> = (0..hidden * vocab)
            .map(|i| ((i * 11 + 5) % 41) as f32 * 0.015 - 0.3)
            .collect();
        let output_norm = vec![1.0_f32; hidden];

        let prompt = vec![0_i32, 1, 2];
        let max_new_tokens = 3_usize;
        let total_seq = prompt.len() + max_new_tokens;

        ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        let full_output = generate_from_plans(
            &GenerationInputs {
                layer_plans: &layer_plans,
                token_embedding_values: &token_embeddings,
                output_weight_values: &output_weight,
                output_norm_values: &output_norm,
                hidden_features: hidden,
                vocab_size: vocab,
                rms_norm_eps,
                prompt_token_ids: &prompt,
                max_new_tokens,
                pad_token_id: 0,
                eos_token_id: None,
                backend: &backend,
                total_sequence_length: total_seq,
            },
            GenerationMode::FullReprocess,
        )
        .expect("FullReprocess should succeed");

        let two_phase_output = generate_from_plans(
            &GenerationInputs {
                layer_plans: &layer_plans,
                token_embedding_values: &token_embeddings,
                output_weight_values: &output_weight,
                output_norm_values: &output_norm,
                hidden_features: hidden,
                vocab_size: vocab,
                rms_norm_eps,
                prompt_token_ids: &prompt,
                max_new_tokens,
                pad_token_id: 0,
                eos_token_id: None,
                backend: &backend,
                total_sequence_length: total_seq,
            },
            GenerationMode::TwoPhase,
        )
        .expect("TwoPhase should succeed");

        assert_eq!(
            full_output.generated_token_ids.len(),
            two_phase_output.generated_token_ids.len(),
            "Both paths should generate the same number of tokens"
        );
        assert_eq!(
            full_output.generated_token_ids, two_phase_output.generated_token_ids,
            "FullReprocess and TwoPhase must produce identical token sequences.\n\
             FullReprocess: {:?}\n\
             TwoPhase:      {:?}",
            full_output.generated_token_ids, two_phase_output.generated_token_ids
        );
        assert_eq!(
            full_output.all_token_ids, two_phase_output.all_token_ids,
            "Full token sequences (prompt + generated) must match"
        );
    }

    /// Verify Auto mode selects FullReprocess when max_new_tokens is 0.
    #[cfg(feature = "link-system")]
    #[test]
    fn auto_mode_zero_tokens_uses_full_reprocess() {
        use super::super::plan::MlpLayerPlan;
        use crate::backend::ensure_backends_loaded;
        use crate::inference::{MlpInferenceConfig, MlpWeights};

        let hidden = 4_usize;
        let ffn = 8_usize;
        let vocab = 3_usize;

        let config = MlpInferenceConfig::new(hidden, ffn).unwrap();
        let layer_plans = vec![LayerPlan {
            attention: None,
            mlp: MlpLayerPlan {
                weights: MlpWeights::deterministic(config),
                norm_values: vec![1.0_f32; hidden],
            },
        }];

        let token_embeddings: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.1).collect();
        let output_weight: Vec<f32> = (0..hidden * vocab).map(|i| i as f32 * 0.05).collect();
        let output_norm = vec![1.0_f32; hidden];

        ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        let inputs = GenerationInputs {
            layer_plans: &layer_plans,
            token_embedding_values: &token_embeddings,
            output_weight_values: &output_weight,
            output_norm_values: &output_norm,
            hidden_features: hidden,
            vocab_size: vocab,
            rms_norm_eps: 1e-5,
            prompt_token_ids: &[0, 1],
            max_new_tokens: 0,
            pad_token_id: 0,
            eos_token_id: None,
            backend: &backend,
            total_sequence_length: 2,
        };

        let output = generate_from_plans(&inputs, GenerationMode::Auto)
            .expect("Auto with max_new_tokens=0 should succeed");
        assert!(
            output.generated_token_ids.is_empty(),
            "Should generate zero tokens"
        );
    }
}
