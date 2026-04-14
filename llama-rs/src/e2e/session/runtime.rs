//! Runtime stepping logic for [`GenerationSession`].
//!
//! Contains the per-token decode loop, persistent resource management,
//! and sampling — everything that runs after the session is constructed.

use super::super::error::E2eError;
use super::super::generation::{
    DecodeStrategy, GenerationMode, InferenceStrategy, LayerPassConfig, LmHeadResources,
    PersistentDecodeResources, PrefillStrategy, greedy_next_token_id, process_all_layers,
};
use super::super::tensor_ops::{gather_embeddings, rms_norm_with_weight};
use super::GenerationSession;

impl GenerationSession {
    /// Generate the next token, returning `None` when generation is complete.
    ///
    /// On the first call this performs a full prefill over the prompt tokens;
    /// subsequent calls decode one token at a time from cached state.
    pub fn next_token(&mut self) -> Result<Option<i32>, E2eError> {
        if self.finished {
            return Ok(None);
        }

        match self.effective_mode {
            GenerationMode::FullReprocess => self.step_full_reprocess(),
            GenerationMode::TwoPhase | GenerationMode::Auto => self.step_two_phase(),
        }
    }

    /// Two-phase stepping: prefill on first call, then incremental decode.
    fn step_two_phase(&mut self) -> Result<Option<i32>, E2eError> {
        if !self.prefill_done {
            // Phase 1: Prefill (process all prompt tokens at once)
            let prompt_token_count = self.prompt_token_ids.len();
            let active_token_ids = &self.all_token_ids[..prompt_token_count];
            let mut hidden = gather_embeddings(
                &self.token_embedding_values,
                self.hidden_features,
                self.vocab_size,
                active_token_ids,
            )?;

            let layer_config = LayerPassConfig {
                layer_plans: &self.layer_plans,
                rms_norm_eps: self.rms_norm_eps,
                backend: &self.backend,
            };
            let mut strategy = PrefillStrategy {
                state: &mut self.state,
            };
            process_all_layers(
                &mut hidden,
                &layer_config,
                &mut strategy,
                prompt_token_count,
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
            let layer_config = LayerPassConfig {
                layer_plans: &self.layer_plans,
                rms_norm_eps: self.rms_norm_eps,
                backend: &self.backend,
            };
            res.decode_step(
                &mut hidden,
                &layer_config,
                &mut self.state,
                self.hidden_features,
            )?;
            let next = res.sample_token(&hidden, 0, self.hidden_features, &self.backend)?;
            self.emit_token(next)
        } else {
            let layer_config = LayerPassConfig {
                layer_plans: &self.layer_plans,
                rms_norm_eps: self.rms_norm_eps,
                backend: &self.backend,
            };
            let mut strategy = DecodeStrategy {
                state: &mut self.state,
            };
            process_all_layers(&mut hidden, &layer_config, &mut strategy, 1, &mut [])?;
            let next = self.sample_next(&hidden, 0)?;
            self.emit_token(next)
        }
    }

    /// Build persistent decode resources if not already built.
    ///
    /// Called lazily after prefill or on first decode step after resume.
    /// Failure is non-fatal: session falls back to the slow path.
    ///
    /// Exposed as `pub(super)` so that root-level tests can call it directly.
    pub(super) fn ensure_persistent_resources(&mut self) {
        #[cfg(test)]
        if self.persistent_resources_disabled {
            return;
        }
        if self.persistent_resources.is_some() {
            return;
        }

        let resources = LmHeadResources::try_build(
            self.hidden_features,
            self.vocab_size,
            self.rms_norm_eps,
            &self.output_weight_values,
            &self.output_norm_values,
            &self.backend,
        )
        .map(|lm_head| {
            PersistentDecodeResources::try_build(
                &self.layer_plans,
                lm_head,
                self.rms_norm_eps,
                self.total_sequence_length,
                &self.backend,
            )
        });

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

        let layer_config = LayerPassConfig {
            layer_plans: &self.layer_plans,
            rms_norm_eps: self.rms_norm_eps,
            backend: &self.backend,
        };
        let mut strategy = InferenceStrategy;
        process_all_layers(
            &mut hidden,
            &layer_config,
            &mut strategy,
            self.current_token_count,
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
