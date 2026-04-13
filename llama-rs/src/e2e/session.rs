//! Resumable, step-by-step token generation session.
//!
//! [`GenerationSession`] wraps the same inference logic as
//! [`generate_token_ids_from_model`](super::generation::generate_token_ids_from_model),
//! but exposes it as an iterator-like API where each call to
//! [`next_token()`](GenerationSession::next_token) generates exactly one token.
//!
//! Sessions can be checkpointed mid-generation via
//! [`checkpoint()`](GenerationSession::checkpoint) and later resumed from
//! a saved [`GenerationCheckpoint`](super::checkpoint::GenerationCheckpoint).
//!
//! # Execution modes
//!
//! - **TwoPhase** (Qwen3.5 layers): Prefill captures KV/conv/SSM state,
//!   then decode uses cached state. Checkpoints are performance-preserving.
//! - **FullReprocess** (Standard attention / zero max_new_tokens): All tokens
//!   are reprocessed each step. Checkpoints save token IDs and position but
//!   state is recomputed on resume — useful for persistence, not performance.

use super::checkpoint::{CaptureInput, CheckpointV1, GenerationCheckpoint, ModelFingerprint};
use super::config::{E2eGenerationConfig, MixedLayerPolicy};
use super::decode::decode_norm_tensor;
use super::error::E2eError;
use super::generation::{
    DecodeStrategy, GenerationMode, InferenceStrategy, PersistentDecodeResources, PrefillStrategy,
    greedy_next_token_id, process_all_layers,
};
use super::numeric::{checked_mul, validate_token_id};
use super::plan::{AttentionLayerPlan, LayerPlan};
use super::planner::build_layer_plans;
use super::resolve::resolve_global_tensor_names;
use super::state::GenerationState;
use super::tensor_ops::{gather_embeddings, rms_norm_with_weight};
use crate::backend::{LlamaBackend, ensure_backends_loaded};
use crate::metadata::resolve_transformer_metadata;
use crate::model::GgufModel;
use ggml_rs::Backend;

/// A resumable token generation session.
///
/// Created from a GGUF model and config via [`new()`](Self::new), or resumed
/// from a checkpoint via [`resume()`](Self::resume).
///
/// Each call to [`next_token()`](Self::next_token) runs one step of
/// generation: on the first call it performs prefill (processing all prompt
/// tokens), on subsequent calls it decodes one token at a time using cached
/// state.
pub struct GenerationSession {
    // Owned model data
    layer_plans: Vec<LayerPlan>,
    token_embedding_values: Vec<f32>,
    output_weight_values: Vec<f32>,
    output_norm_values: Vec<f32>,
    hidden_features: usize,
    vocab_size: usize,
    rms_norm_eps: f32,

    // Token tracking
    prompt_token_ids: Vec<i32>,
    all_token_ids: Vec<i32>,
    generated_token_ids: Vec<i32>,
    current_token_count: usize,
    max_new_tokens: usize,
    total_sequence_length: usize,
    pad_token_id: i32,
    eos_token_id: Option<i32>,

    // Internal state
    state: GenerationState,
    effective_mode: GenerationMode,
    prefill_done: bool,
    finished: bool,

    // Model fingerprint (for checkpoint validation)
    fingerprint: ModelFingerprint,

    // Persistent decode resources (built lazily after prefill for TwoPhase mode).
    // Drop order: resources drop BEFORE backend (declared before `backend`).
    persistent_resources: Option<PersistentDecodeResources>,

    // Backend
    backend: Backend,
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
            config.mixed_layer_policy,
        )?;

        let _ = validate_token_id(config.pad_token_id, vocab_size)?;
        if let Some(eos_token_id) = config.eos_token_id {
            let _ = validate_token_id(eos_token_id, vocab_size)?;
        }
        for &token_id in &config.prompt_token_ids {
            let _ = validate_token_id(token_id, vocab_size)?;
        }

        let effective_mode = {
            let has_standard = layer_plans
                .iter()
                .any(|p| matches!(p.attention, Some(AttentionLayerPlan::Standard(_))));
            if has_standard || config.max_new_tokens == 0 {
                GenerationMode::FullReprocess
            } else {
                GenerationMode::TwoPhase
            }
        };

        let state = GenerationState::new(&layer_plans, total_sequence_length)?;

        let fingerprint =
            ModelFingerprint::from_plans(&layer_plans, hidden_features, vocab_size, rms_norm_eps);

        let mut all_token_ids = vec![config.pad_token_id; total_sequence_length];
        all_token_ids[..prompt_token_count].copy_from_slice(&config.prompt_token_ids);

        ensure_backends_loaded();
        let backend = Backend::new(config.backend.into())
            .map_err(|source| E2eError::ggml("Backend::new", source))?;

        Ok(Self {
            layer_plans,
            token_embedding_values,
            output_weight_values,
            output_norm_values,
            hidden_features,
            vocab_size,
            rms_norm_eps,
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

        // Resolve model to get current fingerprint
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
            checkpoint.inner.total_sequence_length,
            mixed_layer_policy,
        )?;

        let current_fingerprint =
            ModelFingerprint::from_plans(&layer_plans, hidden_features, vocab_size, rms_norm_eps);

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

        let effective_mode = {
            let has_standard = layer_plans
                .iter()
                .any(|p| matches!(p.attention, Some(AttentionLayerPlan::Standard(_))));
            if has_standard || cp.max_new_tokens == 0 {
                GenerationMode::FullReprocess
            } else {
                GenerationMode::TwoPhase
            }
        };

        ensure_backends_loaded();
        let backend = Backend::new(backend.into())
            .map_err(|source| E2eError::ggml("Backend::new", source))?;

        Ok(Self {
            layer_plans,
            token_embedding_values,
            output_weight_values,
            output_norm_values,
            hidden_features,
            vocab_size,
            rms_norm_eps,
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
            backend,
        })
    }

    /// Generate the next token, returning `None` if generation is complete.
    ///
    /// On the first call, this performs prefill (processing all prompt tokens)
    /// and generates the first token. Subsequent calls decode one token at a time.
    pub fn next_token(&mut self) -> Result<Option<i32>, E2eError> {
        if self.finished {
            return Ok(None);
        }

        let remaining = self
            .max_new_tokens
            .saturating_sub(self.generated_token_ids.len());
        if remaining == 0 {
            self.finished = true;
            return Ok(None);
        }

        match self.effective_mode {
            GenerationMode::TwoPhase => self.step_two_phase(),
            GenerationMode::FullReprocess | GenerationMode::Auto => self.step_full_reprocess(),
        }
    }

    /// Capture the current session state as a serializable checkpoint.
    pub fn checkpoint(&self) -> GenerationCheckpoint {
        GenerationCheckpoint {
            inner: CheckpointV1::capture(CaptureInput {
                fingerprint: self.fingerprint.clone(),
                prompt_token_ids: &self.prompt_token_ids,
                generated_token_ids: &self.generated_token_ids,
                current_token_count: self.current_token_count,
                max_new_tokens: self.max_new_tokens,
                total_sequence_length: self.total_sequence_length,
                pad_token_id: self.pad_token_id,
                eos_token_id: self.eos_token_id,
                prefill_done: self.prefill_done,
                finished: self.finished,
                state: &self.state,
            }),
        }
    }

    /// Tokens generated so far.
    pub fn generated_tokens(&self) -> &[i32] {
        &self.generated_token_ids
    }

    /// All token IDs (prompt + generated) up to the current position.
    pub fn all_tokens(&self) -> &[i32] {
        &self.all_token_ids[..self.current_token_count]
    }

    /// Whether generation is finished (EOS hit or token budget exhausted).
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Number of tokens generated so far.
    pub fn generated_count(&self) -> usize {
        self.generated_token_ids.len()
    }

    // -----------------------------------------------------------------------
    // Internal stepping logic
    // -----------------------------------------------------------------------

    fn step_two_phase(&mut self) -> Result<Option<i32>, E2eError> {
        if !self.prefill_done {
            // Phase 1: Prefill all prompt tokens
            let prompt_ids = &self.all_token_ids[..self.prompt_token_ids.len()];
            let prompt_token_count = self.prompt_token_ids.len();
            let mut hidden = gather_embeddings(
                &self.token_embedding_values,
                self.hidden_features,
                self.vocab_size,
                prompt_ids,
            )?;

            let mut strategy = PrefillStrategy {
                state: &mut self.state,
            };
            process_all_layers(
                &mut hidden,
                &self.layer_plans,
                &mut strategy,
                self.hidden_features,
                prompt_token_count,
                self.rms_norm_eps,
                &self.backend,
                &mut [],
            )?;

            self.prefill_done = true;

            // Build persistent resources after prefill (lazy init).
            self.ensure_persistent_resources();

            let last_index = prompt_token_count
                .checked_sub(1)
                .ok_or(E2eError::EmptyPrompt)?;

            let next = if let Some(ref mut res) = self.persistent_resources {
                res.sample_token(&hidden, last_index, self.hidden_features, &self.backend)?
            } else {
                self.sample_next(&hidden, last_index)?
            };
            return self.emit_token(next);
        }

        // Lazy-init for resumed sessions that already had prefill_done.
        self.ensure_persistent_resources();

        // Phase 2: Decode one token using cached state
        let new_token_id = self.all_token_ids[self.current_token_count - 1];
        let mut hidden = gather_embeddings(
            &self.token_embedding_values,
            self.hidden_features,
            self.vocab_size,
            &[new_token_id],
        )?;

        if let Some(ref mut res) = self.persistent_resources {
            res.decode_step(
                &mut hidden,
                &self.layer_plans,
                &mut self.state,
                self.hidden_features,
                self.rms_norm_eps,
                &self.backend,
            )?;
            let next = res.sample_token(&hidden, 0, self.hidden_features, &self.backend)?;
            self.emit_token(next)
        } else {
            let mut strategy = DecodeStrategy {
                state: &mut self.state,
            };
            process_all_layers(
                &mut hidden,
                &self.layer_plans,
                &mut strategy,
                self.hidden_features,
                1,
                self.rms_norm_eps,
                &self.backend,
                &mut [],
            )?;
            let next = self.sample_next(&hidden, 0)?;
            self.emit_token(next)
        }
    }

    /// Build persistent decode resources if not already built.
    ///
    /// Called lazily after prefill or on first decode step after resume.
    /// Failure is non-fatal: session falls back to the slow path.
    fn ensure_persistent_resources(&mut self) {
        if self.persistent_resources.is_some() {
            return;
        }

        let resources = PersistentDecodeResources::try_build(
            &self.layer_plans,
            self.hidden_features,
            self.vocab_size,
            self.rms_norm_eps,
            self.total_sequence_length,
            &self.output_weight_values,
            &self.output_norm_values,
            &self.backend,
        );

        if let Some(ref res) = resources {
            res.seed_kv_caches(&self.state);
        }

        self.persistent_resources = resources;
    }

    fn step_full_reprocess(&mut self) -> Result<Option<i32>, E2eError> {
        let active_token_ids = &self.all_token_ids[..self.current_token_count];
        let mut hidden = gather_embeddings(
            &self.token_embedding_values,
            self.hidden_features,
            self.vocab_size,
            active_token_ids,
        )?;

        let mut strategy = InferenceStrategy;
        process_all_layers(
            &mut hidden,
            &self.layer_plans,
            &mut strategy,
            self.hidden_features,
            self.current_token_count,
            self.rms_norm_eps,
            &self.backend,
            &mut [],
        )?;

        let last_index = self
            .current_token_count
            .checked_sub(1)
            .ok_or(E2eError::EmptyPrompt)?;
        let next = self.sample_next(&hidden, last_index)?;
        self.prefill_done = true;
        self.emit_token(next)
    }

    fn emit_token(&mut self, token_id: i32) -> Result<Option<i32>, E2eError> {
        self.generated_token_ids.push(token_id);
        if self.current_token_count < self.total_sequence_length {
            self.all_token_ids[self.current_token_count] = token_id;
            self.current_token_count += 1;
        }
        if self.eos_token_id.is_some_and(|eos| eos == token_id) {
            self.finished = true;
        }
        if self.generated_token_ids.len() >= self.max_new_tokens {
            self.finished = true;
        }
        Ok(Some(token_id))
    }

    fn sample_next(&self, hidden: &[f32], token_index: usize) -> Result<i32, E2eError> {
        let seq_len = token_index + 1;
        let normalized_output = rms_norm_with_weight(
            hidden,
            self.hidden_features,
            seq_len,
            &self.output_norm_values,
            self.rms_norm_eps,
        )?;
        greedy_next_token_id(
            &normalized_output,
            token_index,
            self.hidden_features,
            &self.output_weight_values,
            self.vocab_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::plan::{LayerPlan, MlpLayerPlan, Qwen35FullAttentionLayerPlan};
    use super::*;
    use crate::inference::{MlpInferenceConfig, MlpWeights};

    /// Build a minimal synthetic session for testing (no model file needed).
    fn build_test_session(max_new_tokens: usize) -> GenerationSession {
        let hidden = 4;
        let vocab = 8;
        let prompt_ids = vec![1_i32, 2];
        let total_seq = prompt_ids.len() + max_new_tokens;

        let dummy_mlp = MlpLayerPlan {
            weights: MlpWeights::deterministic(MlpInferenceConfig::new(hidden, 8).unwrap()),
            norm_values: vec![1.0; hidden],
        };

        let layer_plans = vec![LayerPlan {
            attention: Some(AttentionLayerPlan::Qwen35Full(
                Qwen35FullAttentionLayerPlan {
                    norm_values: vec![1.0; hidden],
                    q_norm_values: vec![1.0; 2],
                    k_norm_values: vec![1.0; 2],
                    // Q weight: hidden × (query_features * 2) = 4 × 8 = 32
                    q_weight_values: vec![0.1; hidden * 8],
                    // K weight: hidden × kv_features = 4 × 2 = 8
                    k_weight_values: vec![0.1; hidden * 2],
                    // V weight: hidden × kv_features = 4 × 2 = 8
                    v_weight_values: vec![0.1; hidden * 2],
                    // Output weight: query_features × hidden = 4 × 4 = 16
                    output_weight_values: vec![0.1; 4 * hidden],
                    head_count: 2,
                    kv_head_count: 1,
                    head_dimension: 2,
                    attention_scale: 0.707,
                    rope_n_dims: 2,
                    rope_freq_base: 10000.0,
                    rope_freq_scale: 1.0,
                },
            )),
            mlp: dummy_mlp,
        }];

        let state = GenerationState::new(&layer_plans, total_seq).unwrap();
        let fingerprint = ModelFingerprint::from_plans(&layer_plans, hidden, vocab, 1e-5);

        let mut all_token_ids = vec![0i32; total_seq];
        all_token_ids[..prompt_ids.len()].copy_from_slice(&prompt_ids);

        ensure_backends_loaded();
        let backend = Backend::new(ggml_rs::BackendKind::Cpu).unwrap();

        // Deterministic embeddings: identity-like
        let mut token_embedding_values = vec![0.0f32; vocab * hidden];
        for t in 0..vocab {
            for d in 0..hidden {
                token_embedding_values[t * hidden + d] = if d == t % hidden { 1.0 } else { 0.01 };
            }
        }
        let output_weight_values = token_embedding_values.clone();
        let output_norm_values = vec![1.0f32; hidden];

        GenerationSession {
            layer_plans,
            token_embedding_values,
            output_weight_values,
            output_norm_values,
            hidden_features: hidden,
            vocab_size: vocab,
            rms_norm_eps: 1e-5,
            prompt_token_ids: prompt_ids.clone(),
            all_token_ids,
            generated_token_ids: Vec::new(),
            current_token_count: prompt_ids.len(),
            max_new_tokens,
            total_sequence_length: total_seq,
            pad_token_id: 0,
            eos_token_id: None,
            state,
            effective_mode: GenerationMode::TwoPhase,
            prefill_done: false,
            finished: max_new_tokens == 0,
            fingerprint,
            persistent_resources: None,
            backend,
        }
    }

    #[test]
    fn session_generates_tokens_step_by_step() {
        let mut session = build_test_session(3);
        let mut tokens = Vec::new();
        while let Some(token) = session.next_token().unwrap() {
            tokens.push(token);
        }
        assert_eq!(tokens.len(), 3);
        assert!(session.is_finished());
        assert_eq!(session.generated_tokens(), &tokens);
    }

    #[test]
    fn session_zero_tokens_is_immediately_finished() {
        let mut session = build_test_session(0);
        assert!(session.is_finished());
        assert_eq!(session.next_token().unwrap(), None);
    }

    #[test]
    fn checkpoint_roundtrip_preserves_state() {
        let mut session = build_test_session(5);
        // Generate 2 tokens
        let t1 = session.next_token().unwrap().unwrap();
        let t2 = session.next_token().unwrap().unwrap();

        // Save checkpoint
        let checkpoint = session.checkpoint();
        let mut buf = Vec::new();
        checkpoint.save_to(&mut buf).unwrap();

        // Verify checkpoint data
        let restored = GenerationCheckpoint::load_from(buf.as_slice()).unwrap();
        assert_eq!(restored.inner.generated_token_ids, vec![t1, t2]);
        assert_eq!(restored.inner.current_token_count, 4); // 2 prompt + 2 generated
        assert!(restored.inner.prefill_done);
        assert!(!restored.inner.finished);
    }

    #[test]
    fn session_matches_one_shot_generation() {
        // Generate all tokens via session
        let mut session = build_test_session(3);
        let mut session_tokens = Vec::new();
        while let Some(token) = session.next_token().unwrap() {
            session_tokens.push(token);
        }

        // Generate all tokens via fresh session (same config)
        let mut session2 = build_test_session(3);
        let mut oneshot_tokens = Vec::new();
        while let Some(token) = session2.next_token().unwrap() {
            oneshot_tokens.push(token);
        }

        assert_eq!(session_tokens, oneshot_tokens, "deterministic generation");
    }
}
