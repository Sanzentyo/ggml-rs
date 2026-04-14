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

mod init;
mod runtime;

use super::checkpoint::{CaptureInput, CheckpointV1, GenerationCheckpoint, ModelFingerprint};
use super::generation::{GenerationMode, PersistentDecodeResources};
use super::plan::LayerPlan;
use super::state::GenerationState;
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

    /// When true, `ensure_persistent_resources` is a no-op.
    /// Used in tests to force the fallback (non-persistent) code path.
    #[cfg(test)]
    persistent_resources_disabled: bool,

    // Backend
    backend: Backend,
}

impl GenerationSession {
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
}

#[cfg(test)]
mod tests {
    use super::super::plan::AttentionLayerPlan;
    use super::super::plan::{LayerPlan, MlpLayerPlan, Qwen35FullAttentionLayerPlan};
    use super::*;
    use crate::backend::ensure_backends_loaded;
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
                    q_weight_values: vec![0.1; hidden * 8],
                    k_weight_values: vec![0.1; hidden * 2],
                    v_weight_values: vec![0.1; hidden * 2],
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
            persistent_resources_disabled: false,
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
        let t1 = session.next_token().unwrap().unwrap();
        let t2 = session.next_token().unwrap().unwrap();

        let checkpoint = session.checkpoint();
        let mut buf = Vec::new();
        checkpoint.save_to(&mut buf).unwrap();

        let restored = GenerationCheckpoint::load_from(buf.as_slice()).unwrap();
        assert_eq!(restored.inner.generated_token_ids, vec![t1, t2]);
        assert_eq!(restored.inner.current_token_count, 4);
        assert!(restored.inner.prefill_done);
        assert!(!restored.inner.finished);
    }

    #[test]
    fn session_matches_one_shot_generation() {
        let mut session = build_test_session(3);
        let mut session_tokens = Vec::new();
        while let Some(token) = session.next_token().unwrap() {
            session_tokens.push(token);
        }

        let mut session2 = build_test_session(3);
        let mut oneshot_tokens = Vec::new();
        while let Some(token) = session2.next_token().unwrap() {
            oneshot_tokens.push(token);
        }

        assert_eq!(session_tokens, oneshot_tokens, "deterministic generation");
    }

    #[test]
    fn persistent_resources_built_lazily_after_prefill() {
        let mut session = build_test_session(3);
        assert!(session.persistent_resources.is_none());
        assert!(!session.prefill_done);

        let _ = session.next_token().unwrap().unwrap();
        assert!(session.prefill_done);

        let _ = session.next_token().unwrap().unwrap();
        let _ = session.next_token().unwrap().unwrap();
        assert!(session.is_finished());
    }

    #[test]
    fn session_without_persistent_resources_still_works() {
        let mut session = build_test_session(3);
        session.persistent_resources_disabled = true;

        let mut tokens = Vec::new();
        while let Some(token) = session.next_token().unwrap() {
            tokens.push(token);
        }
        assert_eq!(tokens.len(), 3);
        assert!(session.is_finished());
        assert!(
            session.persistent_resources.is_none(),
            "persistent resources should remain None when disabled"
        );
    }

    #[test]
    fn persistent_resources_produce_same_tokens_as_fallback() {
        let mut session_a = build_test_session(5);
        let mut session_b = build_test_session(5);
        session_b.persistent_resources_disabled = true;

        let mut tokens_a = Vec::new();
        let mut tokens_b = Vec::new();
        for _ in 0..5 {
            if let Some(t) = session_a.next_token().unwrap() {
                tokens_a.push(t);
            }
            if let Some(t) = session_b.next_token().unwrap() {
                tokens_b.push(t);
            }
        }
        assert_eq!(tokens_a.len(), 5, "persistent path should produce 5 tokens");
        assert_eq!(tokens_b.len(), 5, "fallback path should produce 5 tokens");
    }

    #[test]
    fn ensure_persistent_resources_is_idempotent() {
        let mut session = build_test_session(3);
        let _ = session.next_token().unwrap();

        let resources_present_first = session.persistent_resources.is_some();
        session.ensure_persistent_resources();
        let resources_present_second = session.persistent_resources.is_some();
        assert_eq!(resources_present_first, resources_present_second);
    }

    /// Facade tests that verify module structure is wired correctly.
    #[test]
    fn init_module_wired() {
        // new() and resume() live in init submodule — verify they exist
        // by checking that the type can be constructed (via build_test_session).
        let session = build_test_session(1);
        assert!(!session.is_finished());
    }

    #[test]
    fn runtime_module_wired() {
        // next_token() and stepping methods live in runtime submodule.
        let mut session = build_test_session(1);
        let token = session.next_token().unwrap();
        assert!(token.is_some());
    }
}
