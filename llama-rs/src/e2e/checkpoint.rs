//! Serializable checkpoint for resumable generation sessions.
//!
//! The checkpoint captures all state needed to resume token generation from
//! where it left off. It uses a separate DTO layer (`CheckpointV1`) that is
//! independent of the internal runtime state types, so internal refactoring
//! does not break serialized checkpoint compatibility.
//!
//! Binary format uses `postcard` (compact, stable wire format) with a version
//! envelope for forward compatibility.
//!
//! # Module structure
//!
//! - [`dto`]: Versioned DTO types and structural validation
//! - [`runtime`]: Runtime state ↔ DTO conversion, capture, and restore

mod dto;
mod runtime;

use super::error::E2eError;
use std::io::{Read, Write};

// Items with `pub(in crate::e2e)` in submodules are directly accessible
// at crate::e2e level without re-exports. We only re-export for local test use.
pub(in crate::e2e) use dto::{CheckpointV1, ModelFingerprint};
#[cfg(test)]
pub(super) use dto::{LayerStateDto, LayerTypeTag};
pub(in crate::e2e) use runtime::CaptureInput;

/// Magic bytes identifying a llama-rs checkpoint file.
const CHECKPOINT_MAGIC: [u8; 4] = *b"LRCK";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// An opaque, serializable snapshot of a generation session.
///
/// Created via [`GenerationSession::checkpoint()`] and restored via
/// [`GenerationSession::resume()`].
#[derive(Debug, Clone)]
pub struct GenerationCheckpoint {
    pub(super) inner: CheckpointV1,
}

impl GenerationCheckpoint {
    /// Serialize the checkpoint to a writer in binary format.
    pub fn save_to(&self, mut writer: impl Write) -> Result<(), E2eError> {
        writer.write_all(&CHECKPOINT_MAGIC)?;
        let bytes = postcard::to_stdvec(&self.inner)
            .map_err(|e| E2eError::CheckpointDeserialize(format!("serialize: {e}")))?;
        writer.write_all(&bytes)?;
        Ok(())
    }

    /// Deserialize a checkpoint from a reader.
    pub fn load_from(mut reader: impl Read) -> Result<Self, E2eError> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != CHECKPOINT_MAGIC {
            return Err(E2eError::CheckpointDeserialize(
                "invalid magic bytes (not a llama-rs checkpoint)".into(),
            ));
        }
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        let inner: CheckpointV1 = postcard::from_bytes(&buf)
            .map_err(|e| E2eError::CheckpointDeserialize(format!("deserialize: {e}")))?;
        if inner.version != dto::CHECKPOINT_VERSION {
            return Err(E2eError::CheckpointVersionMismatch {
                file_version: inner.version,
                expected_version: dto::CHECKPOINT_VERSION,
            });
        }
        inner.validate_invariants()?;
        Ok(Self { inner })
    }

    /// The prompt token IDs that started this generation.
    pub fn prompt_token_ids(&self) -> &[i32] {
        &self.inner.prompt_token_ids
    }

    /// Tokens generated so far.
    pub fn generated_token_ids(&self) -> &[i32] {
        &self.inner.generated_token_ids
    }
}

// ---------------------------------------------------------------------------
// Facade tests: save/load roundtrip through the public API
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fingerprint() -> ModelFingerprint {
        ModelFingerprint {
            layer_count: 2,
            hidden_features: 64,
            vocab_size: 1000,
            rms_norm_eps_bits: 1e-5_f32.to_bits(),
            layer_types: vec![
                LayerTypeTag::Qwen35Full {
                    kv_head_count: 2,
                    head_dimension: 8,
                },
                LayerTypeTag::Qwen35Linear {
                    conv_kernel: 3,
                    conv_channels: 16,
                    time_step_rank: 4,
                    state_size: 2,
                },
            ],
        }
    }

    fn sample_checkpoint() -> GenerationCheckpoint {
        let kv_state = LayerStateDto::Qwen35Full {
            k_cache: vec![1.0; 16],
            v_cache: vec![2.0; 16],
            cached_len: 1,
            kv_features: 16,
        };
        let lin_state = LayerStateDto::Qwen35Linear {
            conv_buffer: vec![3.0; 32],
            ssm_states: vec![4.0; 16],
            conv_valid: 2,
            conv_channels: 16,
            conv_kernel: 3,
        };
        GenerationCheckpoint {
            inner: CheckpointV1 {
                version: dto::CHECKPOINT_VERSION,
                fingerprint: sample_fingerprint(),
                prompt_token_ids: vec![1, 2, 3],
                generated_token_ids: vec![42, 43],
                current_token_count: 5,
                prompt_token_count: 3,
                max_new_tokens: 10,
                total_sequence_length: 13,
                pad_token_id: 0,
                eos_token_id: Some(2),
                prefill_done: true,
                finished: false,
                layer_states: vec![kv_state, lin_state],
            },
        }
    }

    #[test]
    fn roundtrip_save_load() {
        let original = sample_checkpoint();
        let mut buf = Vec::new();
        original.save_to(&mut buf).unwrap();

        let restored = GenerationCheckpoint::load_from(buf.as_slice()).unwrap();
        assert_eq!(
            restored.inner.prompt_token_ids,
            original.inner.prompt_token_ids
        );
        assert_eq!(
            restored.inner.generated_token_ids,
            original.inner.generated_token_ids
        );
        assert_eq!(
            restored.inner.current_token_count,
            original.inner.current_token_count
        );
        assert_eq!(restored.inner.fingerprint, original.inner.fingerprint);
        assert_eq!(restored.inner.layer_states.len(), 2);
    }

    #[test]
    fn reject_bad_magic() {
        let bad_data = b"BADDsome data here";
        let result = GenerationCheckpoint::load_from(bad_data.as_slice());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("invalid magic bytes"), "got: {err}");
    }

    fn sample_checkpoint_with_standard() -> GenerationCheckpoint {
        let kv_head_count = 2;
        let head_dimension = 4;
        let kv_features = kv_head_count * head_dimension;
        let total_sequence_length = 13;
        let cached_len = 3;
        let data_len = cached_len * kv_features;
        let std_state = LayerStateDto::Standard {
            k_cache: (0..data_len).map(|i| i as f32 * 0.1).collect(),
            v_cache: (0..data_len).map(|i| i as f32 * 0.2).collect(),
            cached_len,
            kv_features,
        };
        let fingerprint = ModelFingerprint {
            layer_count: 1,
            hidden_features: 64,
            vocab_size: 1000,
            rms_norm_eps_bits: 1e-5_f32.to_bits(),
            layer_types: vec![LayerTypeTag::Standard {
                kv_head_count,
                head_dimension,
            }],
        };
        GenerationCheckpoint {
            inner: CheckpointV1 {
                version: dto::CHECKPOINT_VERSION,
                fingerprint,
                prompt_token_ids: vec![1, 2, 3],
                generated_token_ids: vec![42, 43],
                current_token_count: 5,
                prompt_token_count: 3,
                max_new_tokens: 10,
                total_sequence_length,
                pad_token_id: 0,
                eos_token_id: Some(2),
                prefill_done: true,
                finished: false,
                layer_states: vec![std_state],
            },
        }
    }

    #[test]
    fn standard_checkpoint_roundtrip_restores_kv() {
        let original = sample_checkpoint_with_standard();
        let mut buf = Vec::new();
        original.save_to(&mut buf).unwrap();

        let restored = GenerationCheckpoint::load_from(buf.as_slice()).unwrap();
        match (
            &original.inner.layer_states[0],
            &restored.inner.layer_states[0],
        ) {
            (
                LayerStateDto::Standard {
                    k_cache: ok,
                    v_cache: ov,
                    cached_len: oc,
                    kv_features: of,
                },
                LayerStateDto::Standard {
                    k_cache: rk,
                    v_cache: rv,
                    cached_len: rc,
                    kv_features: rf,
                },
            ) => {
                assert_eq!(ok, rk, "k_cache mismatch");
                assert_eq!(ov, rv, "v_cache mismatch");
                assert_eq!(oc, rc, "cached_len mismatch");
                assert_eq!(of, rf, "kv_features mismatch");
            }
            _ => panic!("expected Standard layer state"),
        }
    }

    #[test]
    fn standard_capture_roundtrip_via_from_impl() {
        use super::super::state::StandardAttentionState;

        let kv_features = 8;
        let cached_len = 3;
        let total_seq = 13;
        let runtime = super::super::state::LayerAttentionState::Standard(StandardAttentionState {
            k_cache: (0..total_seq * kv_features)
                .map(|i| i as f32 * 0.1)
                .collect(),
            v_cache: (0..total_seq * kv_features)
                .map(|i| i as f32 * 0.2)
                .collect(),
            cached_len,
            kv_features,
        });

        // From impl trims to cached_len * kv_features
        let dto = LayerStateDto::from(&runtime);
        match &dto {
            LayerStateDto::Standard { k_cache, .. } => {
                assert_eq!(k_cache.len(), cached_len * kv_features);
            }
            _ => panic!("expected Standard"),
        }

        // Build a checkpoint with the trimmed DTO
        let cp = GenerationCheckpoint {
            inner: CheckpointV1 {
                version: dto::CHECKPOINT_VERSION,
                fingerprint: ModelFingerprint {
                    layer_count: 1,
                    hidden_features: 64,
                    vocab_size: 1000,
                    rms_norm_eps_bits: 1e-5_f32.to_bits(),
                    layer_types: vec![LayerTypeTag::Standard {
                        kv_head_count: 1,
                        head_dimension: kv_features,
                    }],
                },
                prompt_token_ids: vec![1, 2, 3],
                generated_token_ids: vec![42],
                current_token_count: 4,
                prompt_token_count: 3,
                max_new_tokens: 10,
                total_sequence_length: total_seq,
                pad_token_id: 0,
                eos_token_id: None,
                prefill_done: true,
                finished: false,
                layer_states: vec![dto],
            },
        };

        // Round-trip through save/load
        let mut buf = Vec::new();
        cp.save_to(&mut buf).unwrap();
        let restored = GenerationCheckpoint::load_from(buf.as_slice()).unwrap();

        // Validate that restore reconstructs full-size caches
        let state = restored.inner.restore_state().unwrap();
        match &state.layers[0] {
            super::super::state::LayerAttentionState::Standard(kv) => {
                assert_eq!(kv.k_cache.len(), total_seq * kv_features);
                assert_eq!(kv.cached_len, cached_len);
                // First cached_len * kv_features elements match original
                let data_len = cached_len * kv_features;
                for i in 0..data_len {
                    assert_eq!(kv.k_cache[i], i as f32 * 0.1, "k_cache[{i}] mismatch");
                }
                // Remaining are zeroed
                for i in data_len..kv.k_cache.len() {
                    assert_eq!(kv.k_cache[i], 0.0, "k_cache[{i}] should be zero");
                }
            }
            _ => panic!("expected Standard"),
        }
    }
}
