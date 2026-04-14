//! Construction paths for [`GenerationSession`].
//!
//! Both [`new()`](super::GenerationSession::new) and
//! [`resume()`](super::GenerationSession::resume) share substantial model-loading
//! logic, extracted into [`ResolvedModel`].

use super::super::checkpoint::{GenerationCheckpoint, ModelFingerprint};
use super::super::config::{E2eGenerationConfig, MixedLayerPolicy};
use super::super::decode::decode_norm_tensor;
use super::super::error::{E2eError, GgmlResultExt};
use super::super::generation::GenerationMode;
use super::super::numeric::{checked_mul, validate_token_id};
use super::super::plan::{AttentionLayerPlan, LayerPlan};
use super::super::planner::build_layer_plans;
use super::super::resolve::resolve_global_tensor_names;
use super::super::state::GenerationState;
use super::GenerationSession;
use crate::backend::{LlamaBackend, ensure_backends_loaded};
use crate::metadata::resolve_transformer_metadata;
use crate::model::GgufModel;
use ggml_rs::Backend;

/// Resolved model data shared between `new()` and `resume()`.
struct ResolvedModel {
    layer_plans: Vec<LayerPlan>,
    token_embedding_values: Vec<f32>,
    output_weight_values: Vec<f32>,
    output_norm_values: Vec<f32>,
    hidden_features: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
}

impl ResolvedModel {
    /// Load and validate model tensors from a GGUF file.
    fn resolve(
        model: &GgufModel,
        total_sequence_length: usize,
        mixed_layer_policy: MixedLayerPolicy,
    ) -> Result<Self, E2eError> {
        let metadata = resolve_transformer_metadata(model)
            .map_err(|source| E2eError::metadata("resolve_transformer_metadata", source))?;
        let hidden_features = metadata.embedding_length();
        let rms_norm_eps = metadata.attention_layer_norm_rms_epsilon();
        let global_names = resolve_global_tensor_names(model)?;

        let token_embedding_values = model
            .tensor_values::<f32>(&global_names.token_embedding)
            .map_err(|source| {
                E2eError::model("GgufModel::tensor_values(token_embedding)", source)
            })?;
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
            values
        } else {
            token_embedding_values.clone()
        };

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
            mixed_layer_policy,
        )?;

        Ok(Self {
            layer_plans,
            token_embedding_values,
            output_weight_values,
            output_norm_values,
            hidden_features,
            vocab_size,
            rms_norm_eps,
        })
    }
}

/// Determine effective generation mode from layer plan types.
fn determine_mode(layer_plans: &[LayerPlan], max_new_tokens: usize) -> GenerationMode {
    let has_standard = layer_plans
        .iter()
        .any(|p| matches!(p.attention, Some(AttentionLayerPlan::Standard(_))));
    if has_standard || max_new_tokens == 0 {
        GenerationMode::FullReprocess
    } else {
        GenerationMode::TwoPhase
    }
}

impl GenerationSession {
    /// Create a new generation session from a loaded model and config.
    pub fn new(model: &GgufModel, config: &E2eGenerationConfig) -> Result<Self, E2eError> {
        let prompt_token_count = config.prompt_token_ids.len();
        if prompt_token_count == 0 {
            return Err(E2eError::EmptyPrompt);
        }

        let total_sequence_length = prompt_token_count
            .checked_add(config.max_new_tokens)
            .ok_or(E2eError::MemorySizeOverflow)?;

        // Validate context length before loading model
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

        let resolved =
            ResolvedModel::resolve(model, total_sequence_length, config.mixed_layer_policy)?;

        let _ = validate_token_id(config.pad_token_id, resolved.vocab_size)?;
        if let Some(eos_token_id) = config.eos_token_id {
            let _ = validate_token_id(eos_token_id, resolved.vocab_size)?;
        }
        for &token_id in &config.prompt_token_ids {
            let _ = validate_token_id(token_id, resolved.vocab_size)?;
        }

        let effective_mode = determine_mode(&resolved.layer_plans, config.max_new_tokens);
        let state = GenerationState::new(&resolved.layer_plans, total_sequence_length)?;
        let fingerprint = ModelFingerprint::from_plans(
            &resolved.layer_plans,
            resolved.hidden_features,
            resolved.vocab_size,
            resolved.rms_norm_eps,
        );

        let mut all_token_ids = vec![config.pad_token_id; total_sequence_length];
        all_token_ids[..prompt_token_count].copy_from_slice(&config.prompt_token_ids);

        ensure_backends_loaded();
        let backend = Backend::new(config.backend.into()).ggml_ctx("Backend::new")?;

        Ok(Self {
            layer_plans: resolved.layer_plans,
            token_embedding_values: resolved.token_embedding_values,
            output_weight_values: resolved.output_weight_values,
            output_norm_values: resolved.output_norm_values,
            hidden_features: resolved.hidden_features,
            vocab_size: resolved.vocab_size,
            rms_norm_eps: resolved.rms_norm_eps,
            prompt_token_ids: config.prompt_token_ids.clone(),
            all_token_ids,
            generated_token_ids: Vec::with_capacity(config.max_new_tokens),
            current_token_count: prompt_token_count,
            max_new_tokens: config.max_new_tokens,
            total_sequence_length,
            pad_token_id: config.pad_token_id,
            eos_token_id: config.eos_token_id,
            state,
            effective_mode,
            prefill_done: false,
            finished: config.max_new_tokens == 0,
            fingerprint,
            persistent_resources: None,
            #[cfg(test)]
            persistent_resources_disabled: false,
            backend,
        })
    }

    /// Resume a session from a saved checkpoint.
    ///
    /// The model must be compatible with the one used to create the checkpoint
    /// (same layer count, types, and dimensions). Only `backend` and
    /// `mixed_layer_policy` are taken from the caller — all other parameters
    /// (prompt, max_new_tokens, EOS, pad) are restored from the checkpoint.
    ///
    /// # Note on backend switching
    ///
    /// Cross-backend resume (e.g. CPU → Metal) is allowed but **not
    /// guaranteed** to produce identical tokens due to floating-point
    /// precision differences in greedy argmax.
    pub fn resume(
        model: &GgufModel,
        backend: LlamaBackend,
        mixed_layer_policy: MixedLayerPolicy,
        checkpoint: GenerationCheckpoint,
    ) -> Result<Self, E2eError> {
        // Validate checkpoint internal consistency first
        checkpoint.inner.validate_invariants()?;

        let resolved = ResolvedModel::resolve(
            model,
            checkpoint.inner.total_sequence_length,
            mixed_layer_policy,
        )?;

        let current_fingerprint = ModelFingerprint::from_plans(
            &resolved.layer_plans,
            resolved.hidden_features,
            resolved.vocab_size,
            resolved.rms_norm_eps,
        );

        // Validate checkpoint compatibility
        checkpoint
            .inner
            .fingerprint
            .validate_against(&current_fingerprint)?;

        // Restore state from checkpoint DTOs
        let state = checkpoint.inner.restore_state()?;

        let cp = &checkpoint.inner;

        // Rebuild all_token_ids
        let mut all_token_ids = vec![cp.pad_token_id; cp.total_sequence_length];
        all_token_ids[..cp.prompt_token_count].copy_from_slice(&cp.prompt_token_ids);
        for (i, &token_id) in cp.generated_token_ids.iter().enumerate() {
            all_token_ids[cp.prompt_token_count + i] = token_id;
        }

        let effective_mode = determine_mode(&resolved.layer_plans, cp.max_new_tokens);

        ensure_backends_loaded();
        let backend = Backend::new(backend.into()).ggml_ctx("Backend::new")?;

        Ok(Self {
            layer_plans: resolved.layer_plans,
            token_embedding_values: resolved.token_embedding_values,
            output_weight_values: resolved.output_weight_values,
            output_norm_values: resolved.output_norm_values,
            hidden_features: resolved.hidden_features,
            vocab_size: resolved.vocab_size,
            rms_norm_eps: resolved.rms_norm_eps,
            prompt_token_ids: cp.prompt_token_ids.clone(),
            all_token_ids,
            generated_token_ids: cp.generated_token_ids.clone(),
            current_token_count: cp.current_token_count,
            max_new_tokens: cp.max_new_tokens,
            total_sequence_length: cp.total_sequence_length,
            pad_token_id: cp.pad_token_id,
            eos_token_id: cp.eos_token_id,
            state,
            effective_mode,
            prefill_done: cp.prefill_done,
            finished: cp.finished,
            fingerprint: current_fingerprint,
            persistent_resources: None,
            #[cfg(test)]
            persistent_resources_disabled: false,
            backend,
        })
    }
}
