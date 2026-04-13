//! Token generation loop with greedy sampling.
//!
//! Public entry points: [`generate_token_ids_from_path`] and
//! [`generate_token_ids_from_model`].  Internally the loop dispatches via
//! [`GenerationMode`]: `TwoPhase` (prefill + incremental decode) when all
//! layers support cached state, or `FullReprocess` as a fallback.
//!
//! The per-layer processing logic (norm → attention → residual → norm → MLP →
//! residual) is shared via the [`AttentionStrategy`] trait, with three
//! implementations: [`InferenceStrategy`] (stateless full-reprocess),
//! [`PrefillStrategy`] (captures state), and [`DecodeStrategy`] (uses state).

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
use super::tensor_ops::{
    add_in_place, build_lm_head_graph, gather_embeddings, lm_head_sample_step,
    recommended_lm_head_memory, rms_norm_with_weight,
};
use crate::backend::ensure_backends_loaded;
use crate::inference::attention_inference_with_weights_on_backend_repeats_with_length;
use crate::metadata::resolve_transformer_metadata;
use crate::model::GgufModel;
use crate::tokenizer::tokenize_text_prompt;
use ggml_rs::{Backend, Context};
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
#[derive(Debug)]
pub(super) struct GenerationOutput {
    pub generated_token_ids: Vec<i32>,
    pub all_token_ids: Vec<i32>,
}

// ---------------------------------------------------------------------------
// AttentionStrategy trait and implementations
// ---------------------------------------------------------------------------

/// Dispatches attention computation for a single layer.
///
/// Three implementations cover the generation modes:
/// - [`InferenceStrategy`]: stateless, full-reprocess (supports all layer types)
/// - [`PrefillStrategy`]: captures per-layer state during prompt processing
/// - [`DecodeStrategy`]: uses cached state for single-token decode
///
/// For Qwen3.5 layers, `input` is un-normed; the norm is done in-graph.
/// For `Standard` attention and decode paths, the strategy applies host-side
/// norm internally before dispatching.
pub(super) trait AttentionStrategy {
    fn process_attention(
        &mut self,
        layer_idx: usize,
        attention: &AttentionLayerPlan,
        input: &[f32],
        seq_len: usize,
        rms_norm_eps: f32,
        backend: &Backend,
    ) -> Result<Vec<f32>, E2eError>;
}

/// Stateless strategy: dispatches to `*_inference` functions.
pub(super) struct InferenceStrategy;

impl AttentionStrategy for InferenceStrategy {
    fn process_attention(
        &mut self,
        _layer_idx: usize,
        attention: &AttentionLayerPlan,
        input: &[f32],
        seq_len: usize,
        rms_norm_eps: f32,
        backend: &Backend,
    ) -> Result<Vec<f32>, E2eError> {
        match attention {
            AttentionLayerPlan::Standard(attn) => {
                // Standard attention expects pre-normed input (host-side).
                let hidden_features = attn.norm_values.len();
                let normalized = rms_norm_with_weight(
                    input,
                    hidden_features,
                    seq_len,
                    &attn.norm_values,
                    rms_norm_eps,
                )?;
                attention_inference_with_weights_on_backend_repeats_with_length(
                    &attn.weights,
                    &normalized,
                    seq_len,
                    backend,
                    1,
                )
                .map_err(|source| {
                    E2eError::inference(
                        "attention_inference_with_weights_on_backend_repeats_with_length",
                        source,
                    )
                })
            }
            AttentionLayerPlan::Qwen35Full(attn) => qwen35_full_attention_inference(
                attn,
                input,
                seq_len,
                rms_norm_eps,
                attention.norm_values(),
                backend,
            ),
            AttentionLayerPlan::Qwen35Linear(attn) => qwen35_linear_attention_inference(
                attn,
                input,
                seq_len,
                rms_norm_eps,
                attention.norm_values(),
                backend,
            ),
        }
    }
}

/// Prefill strategy: dispatches to `*_prefill` functions, capturing state.
pub(super) struct PrefillStrategy<'a> {
    pub(super) state: &'a mut GenerationState,
}

impl AttentionStrategy for PrefillStrategy<'_> {
    fn process_attention(
        &mut self,
        layer_idx: usize,
        attention: &AttentionLayerPlan,
        input: &[f32],
        seq_len: usize,
        rms_norm_eps: f32,
        backend: &Backend,
    ) -> Result<Vec<f32>, E2eError> {
        match (attention, &mut self.state.layers[layer_idx]) {
            (AttentionLayerPlan::Qwen35Full(attn), LayerAttentionState::Qwen35Full(s)) => {
                qwen35_full_attention_prefill(
                    attn,
                    input,
                    seq_len,
                    rms_norm_eps,
                    attention.norm_values(),
                    s,
                    backend,
                )
            }
            (AttentionLayerPlan::Qwen35Linear(attn), LayerAttentionState::Qwen35Linear(s)) => {
                qwen35_linear_attention_prefill(
                    attn,
                    input,
                    seq_len,
                    rms_norm_eps,
                    attention.norm_values(),
                    s,
                    backend,
                )
            }
            _ => Err(E2eError::UnsupportedTwoPhase),
        }
    }
}

/// Decode strategy: dispatches to `*_decode_step` functions using cached state.
pub(super) struct DecodeStrategy<'a> {
    pub(super) state: &'a mut GenerationState,
}

impl AttentionStrategy for DecodeStrategy<'_> {
    fn process_attention(
        &mut self,
        layer_idx: usize,
        attention: &AttentionLayerPlan,
        input: &[f32],
        seq_len: usize,
        rms_norm_eps: f32,
        backend: &Backend,
    ) -> Result<Vec<f32>, E2eError> {
        debug_assert_eq!(seq_len, 1, "DecodeStrategy expects single-token input");
        let _ = seq_len;

        // Decode path: host-side norm before dispatch.
        // The hidden_features dimension is inferred from the norm weight vector.
        let norm_weight = attention.norm_values();
        let hidden_features = norm_weight.len();
        let normalized =
            rms_norm_with_weight(input, hidden_features, 1, norm_weight, rms_norm_eps)?;

        match (attention, &mut self.state.layers[layer_idx]) {
            (AttentionLayerPlan::Qwen35Full(attn), LayerAttentionState::Qwen35Full(s)) => {
                qwen35_full_attention_decode_step(attn, &normalized, rms_norm_eps, s, backend)
            }
            (AttentionLayerPlan::Qwen35Linear(attn), LayerAttentionState::Qwen35Linear(s)) => {
                qwen35_linear_attention_decode_step(attn, &normalized, rms_norm_eps, s, backend)
            }
            _ => Err(E2eError::UnsupportedTwoPhase),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared layer processing and sampling
// ---------------------------------------------------------------------------

/// Process all layers using the given attention strategy.
///
/// Attention norm is now done in-graph by each strategy (or host-side for
/// decode/Standard). MLP norm is done in-graph by the MLP function. The
/// generation loop passes un-normed `hidden` throughout.
pub(super) fn process_all_layers(
    hidden: &mut [f32],
    layer_plans: &[LayerPlan],
    strategy: &mut impl AttentionStrategy,
    _hidden_features: usize,
    seq_len: usize,
    rms_norm_eps: f32,
    backend: &Backend,
) -> Result<(), E2eError> {
    for (layer_idx, layer_plan) in layer_plans.iter().enumerate() {
        if let Some(attention) = &layer_plan.attention {
            let attention_output = strategy.process_attention(
                layer_idx,
                attention,
                hidden,
                seq_len,
                rms_norm_eps,
                backend,
            )?;
            add_in_place(hidden, &attention_output)?;
        }

        let mlp_output = mlp_sequence_inference_with_weights(
            &layer_plan.mlp.weights,
            hidden,
            seq_len,
            &layer_plan.mlp.norm_values,
            rms_norm_eps,
            backend,
        )?;
        add_in_place(hidden, &mlp_output)?;
    }
    Ok(())
}

/// Extract the last token's hidden state from a `[seq_len × hidden_features]` buffer
/// and run a graph-level LM head sampling step.
fn graph_sample_at(
    hidden: &[f32],
    token_index: usize,
    inputs: &GenerationInputs<'_>,
    x_in: &ggml_rs::Tensor<'_, f32>,
    logits_t: &ggml_rs::Tensor<'_, f32>,
    lm_graph: &mut ggml_rs::Graph<'_>,
) -> Result<i32, E2eError> {
    let offset = checked_mul(token_index, inputs.hidden_features)?;
    let last_hidden = &hidden[offset..offset + inputs.hidden_features];
    lm_head_sample_step(last_hidden, x_in, logits_t, lm_graph, inputs.backend)
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

    if matches!(effective_mode, GenerationMode::TwoPhase) {
        let has_unsupported = inputs
            .layer_plans
            .iter()
            .any(|p| matches!(p.attention, Some(AttentionLayerPlan::Standard(_))));
        if has_unsupported {
            return Err(E2eError::UnsupportedTwoPhase);
        }
    }

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
    let mut strategy = InferenceStrategy;

    // Persistent LM head: build graph and upload weights once.
    let lm_ctx_size = recommended_lm_head_memory(inputs.hidden_features, inputs.vocab_size)?;
    let lm_ctx = Context::new_no_alloc_bytes(lm_ctx_size)
        .map_err(|source| E2eError::ggml("Context::new_no_alloc_bytes(lm_head)", source))?;
    let (w_out, norm_w, x_in, logits_t, mut lm_graph) = build_lm_head_graph(
        &lm_ctx,
        inputs.hidden_features,
        inputs.vocab_size,
        inputs.rms_norm_eps,
    )?;
    let _lm_buffer = lm_ctx
        .allocate_tensors(inputs.backend)
        .map_err(|source| E2eError::ggml("allocate_tensors(lm_head)", source))?;
    w_out
        .write_data_backend(inputs.output_weight_values)
        .map_err(|source| E2eError::ggml("write<W_OUT>", source))?;
    norm_w
        .write_data_backend(inputs.output_norm_values)
        .map_err(|source| E2eError::ggml("write<NORM_W>", source))?;

    for _step in 0..inputs.max_new_tokens {
        let active_token_ids = &all_token_ids[..*current_token_count];
        let mut hidden = gather_embeddings(
            inputs.token_embedding_values,
            inputs.hidden_features,
            inputs.vocab_size,
            active_token_ids,
        )?;

        process_all_layers(
            &mut hidden,
            inputs.layer_plans,
            &mut strategy,
            inputs.hidden_features,
            *current_token_count,
            inputs.rms_norm_eps,
            inputs.backend,
        )?;

        let last_index = current_token_count
            .checked_sub(1)
            .ok_or(E2eError::EmptyPrompt)?;
        let next_token_id =
            graph_sample_at(&hidden, last_index, inputs, &x_in, &logits_t, &mut lm_graph)?;

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

    if inputs.max_new_tokens == 0 {
        return Ok(());
    }

    let mut state = GenerationState::new(inputs.layer_plans, inputs.total_sequence_length)?;

    // Persistent LM head: build graph and upload weights once.
    let lm_ctx_size = recommended_lm_head_memory(inputs.hidden_features, inputs.vocab_size)?;
    let lm_ctx = Context::new_no_alloc_bytes(lm_ctx_size)
        .map_err(|source| E2eError::ggml("Context::new_no_alloc_bytes(lm_head)", source))?;
    let (w_out, norm_w, x_in, logits_t, mut lm_graph) = build_lm_head_graph(
        &lm_ctx,
        inputs.hidden_features,
        inputs.vocab_size,
        inputs.rms_norm_eps,
    )?;
    let _lm_buffer = lm_ctx
        .allocate_tensors(inputs.backend)
        .map_err(|source| E2eError::ggml("allocate_tensors(lm_head)", source))?;
    w_out
        .write_data_backend(inputs.output_weight_values)
        .map_err(|source| E2eError::ggml("write<W_OUT>", source))?;
    norm_w
        .write_data_backend(inputs.output_norm_values)
        .map_err(|source| E2eError::ggml("write<NORM_W>", source))?;

    // Phase 1: Prefill — process all prompt tokens at once, capturing state.
    let prompt_ids = &all_token_ids[..prompt_token_count];
    let mut hidden = gather_embeddings(
        inputs.token_embedding_values,
        inputs.hidden_features,
        inputs.vocab_size,
        prompt_ids,
    )?;

    {
        let mut strategy = PrefillStrategy { state: &mut state };
        process_all_layers(
            &mut hidden,
            inputs.layer_plans,
            &mut strategy,
            inputs.hidden_features,
            prompt_token_count,
            inputs.rms_norm_eps,
            inputs.backend,
        )?;
    }

    let last_index = prompt_token_count
        .checked_sub(1)
        .ok_or(E2eError::EmptyPrompt)?;
    let first_token_id =
        graph_sample_at(&hidden, last_index, inputs, &x_in, &logits_t, &mut lm_graph)?;

    generated_token_ids.push(first_token_id);
    all_token_ids[prompt_token_count] = first_token_id;
    *current_token_count = prompt_token_count + 1;

    if inputs.eos_token_id.is_some_and(|eos| eos == first_token_id) {
        return Ok(());
    }

    // Phase 2: Decode — one token at a time using cached state.
    let mut strategy = DecodeStrategy { state: &mut state };
    for _step in 1..inputs.max_new_tokens {
        let new_token_id = all_token_ids[*current_token_count - 1];
        let mut hidden = gather_embeddings(
            inputs.token_embedding_values,
            inputs.hidden_features,
            inputs.vocab_size,
            &[new_token_id],
        )?;

        process_all_layers(
            &mut hidden,
            inputs.layer_plans,
            &mut strategy,
            inputs.hidden_features,
            1,
            inputs.rms_norm_eps,
            inputs.backend,
        )?;

        // Decode phase: hidden is [1 × hidden_features], so token_index=0.
        let next_token_id = graph_sample_at(&hidden, 0, inputs, &x_in, &logits_t, &mut lm_graph)?;

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

pub(super) fn greedy_next_token_id(
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

    /// Verify TwoPhase mode with max_new_tokens=0 returns empty output.
    #[cfg(feature = "link-system")]
    #[test]
    fn two_phase_zero_tokens_returns_empty() {
        use super::super::plan::MlpLayerPlan;
        use super::super::plan::Qwen35LinearAttentionLayerPlan;
        use crate::backend::ensure_backends_loaded;
        use crate::inference::{MlpInferenceConfig, MlpWeights};

        let hidden = 4_usize;
        let ffn = 8_usize;
        let vocab = 3_usize;
        let inner_size = 4_usize;
        let conv_channels = inner_size + 2; // inner_size + 2 * group_count * state_size (both 1)

        let config = MlpInferenceConfig::new(hidden, ffn).unwrap();
        let layer_plans = vec![LayerPlan {
            attention: Some(AttentionLayerPlan::Qwen35Linear(
                Qwen35LinearAttentionLayerPlan {
                    norm_values: vec![1.0; hidden],
                    qkv_weight_values: vec![0.1; hidden * conv_channels],
                    gate_weight_values: vec![0.1; hidden * inner_size],
                    alpha_weight_values: vec![0.01; hidden * 2],
                    beta_weight_values: vec![0.01; hidden * 2],
                    conv_weight_values: vec![1.0; conv_channels * 2],
                    dt_bias_values: vec![0.0; 2],
                    ssm_a_values: vec![-1.0; 2],
                    ssm_norm_values: vec![1.0; 1],
                    ssm_out_weight_values: vec![0.1; inner_size * hidden],
                    state_size: 1,
                    group_count: 1,
                    time_step_rank: 2,
                    inner_size,
                    conv_kernel: 2,
                },
            )),
            mlp: MlpLayerPlan {
                weights: MlpWeights::deterministic(config),
                norm_values: vec![1.0; hidden],
            },
        }];

        let token_embeddings: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.1).collect();
        let output_weight: Vec<f32> = (0..hidden * vocab).map(|i| i as f32 * 0.05).collect();
        let output_norm = vec![1.0_f32; hidden];

        ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        let output = generate_from_plans(
            &GenerationInputs {
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
            },
            GenerationMode::TwoPhase,
        )
        .expect("TwoPhase with max_new_tokens=0 should succeed");
        assert!(
            output.generated_token_ids.is_empty(),
            "Should generate zero tokens"
        );
    }

    /// Verify TwoPhase mode with Standard attention returns error, not panic.
    #[cfg(feature = "link-system")]
    #[test]
    fn two_phase_with_standard_attention_returns_error() {
        use super::super::plan::{MlpLayerPlan, StandardAttentionLayerPlan};
        use crate::backend::ensure_backends_loaded;
        use crate::inference::{
            AttentionInferenceConfig, AttentionWeights, MlpInferenceConfig, MlpWeights,
        };

        let hidden = 4_usize;
        let ffn = 8_usize;
        let vocab = 3_usize;

        let attn_config = AttentionInferenceConfig::new(hidden, 1).unwrap();
        let mlp_config = MlpInferenceConfig::new(hidden, ffn).unwrap();
        let layer_plans = vec![LayerPlan {
            attention: Some(AttentionLayerPlan::Standard(StandardAttentionLayerPlan {
                weights: AttentionWeights::deterministic(attn_config),
                norm_values: vec![1.0; hidden],
            })),
            mlp: MlpLayerPlan {
                weights: MlpWeights::deterministic(mlp_config),
                norm_values: vec![1.0; hidden],
            },
        }];

        let token_embeddings: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.1).collect();
        let output_weight: Vec<f32> = (0..hidden * vocab).map(|i| i as f32 * 0.05).collect();
        let output_norm = vec![1.0_f32; hidden];

        ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        let result = generate_from_plans(
            &GenerationInputs {
                layer_plans: &layer_plans,
                token_embedding_values: &token_embeddings,
                output_weight_values: &output_weight,
                output_norm_values: &output_norm,
                hidden_features: hidden,
                vocab_size: vocab,
                rms_norm_eps: 1e-5,
                prompt_token_ids: &[0, 1],
                max_new_tokens: 3,
                pad_token_id: 0,
                eos_token_id: None,
                backend: &backend,
                total_sequence_length: 5,
            },
            GenerationMode::TwoPhase,
        );
        assert!(
            result.is_err(),
            "TwoPhase with Standard attention should return Err"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(err, E2eError::UnsupportedTwoPhase),
            "Expected UnsupportedTwoPhase, got: {err}"
        );
    }
}
