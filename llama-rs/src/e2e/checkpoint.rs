//! Serializable checkpoint for resumable generation sessions.
//!
//! The checkpoint captures all state needed to resume token generation from
//! where it left off. It uses a separate DTO layer (`CheckpointV1`) that is
//! independent of the internal runtime state types, so internal refactoring
//! does not break serialized checkpoint compatibility.
//!
//! Binary format uses `postcard` (compact, stable wire format) with a version
//! envelope for forward compatibility.

use super::error::E2eError;
use super::plan::AttentionLayerPlan;
use super::state::{GenerationState, LayerAttentionState, LinearAttentionState};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

/// Current checkpoint format version.
const CHECKPOINT_VERSION: u32 = 1;

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
        if inner.version != CHECKPOINT_VERSION {
            return Err(E2eError::CheckpointVersionMismatch {
                file_version: inner.version,
                expected_version: CHECKPOINT_VERSION,
            });
        }
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
// DTO types (serde, versioned, decoupled from runtime state)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct CheckpointV1 {
    pub version: u32,
    pub fingerprint: ModelFingerprint,
    pub prompt_token_ids: Vec<i32>,
    pub generated_token_ids: Vec<i32>,
    pub current_token_count: usize,
    pub prompt_token_count: usize,
    pub max_new_tokens: usize,
    pub total_sequence_length: usize,
    pub pad_token_id: i32,
    pub eos_token_id: Option<i32>,
    pub prefill_done: bool,
    pub finished: bool,
    pub layer_states: Vec<LayerStateDto>,
}

/// Model structure fingerprint for compatibility validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ModelFingerprint {
    pub layer_count: usize,
    pub hidden_features: usize,
    pub vocab_size: usize,
    pub rms_norm_eps_bits: u32,
    pub layer_types: Vec<LayerTypeTag>,
}

/// Discriminant tag for each layer's attention type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum LayerTypeTag {
    Standard,
    Qwen35Full {
        kv_head_count: usize,
        head_dimension: usize,
    },
    Qwen35Linear {
        conv_kernel: usize,
        conv_channels: usize,
        time_step_rank: usize,
        state_size: usize,
    },
    None,
}

/// Per-layer serializable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) enum LayerStateDto {
    Standard,
    Qwen35Full {
        k_cache: Vec<f32>,
        v_cache: Vec<f32>,
        cached_len: usize,
        kv_features: usize,
    },
    Qwen35Linear {
        conv_buffer: Vec<f32>,
        ssm_states: Vec<f32>,
        conv_valid: usize,
        conv_channels: usize,
        conv_kernel: usize,
    },
    None,
}

// ---------------------------------------------------------------------------
// Conversion: runtime state ↔ DTO
// ---------------------------------------------------------------------------

impl From<&LayerAttentionState> for LayerStateDto {
    fn from(state: &LayerAttentionState) -> Self {
        match state {
            LayerAttentionState::Standard => LayerStateDto::Standard,
            LayerAttentionState::Qwen35Full(kv) => LayerStateDto::Qwen35Full {
                // Only serialize the populated portion of the cache
                k_cache: kv.k_cache[..kv.cached_len * kv.kv_features].to_vec(),
                v_cache: kv.v_cache[..kv.cached_len * kv.kv_features].to_vec(),
                cached_len: kv.cached_len,
                kv_features: kv.kv_features,
            },
            LayerAttentionState::Qwen35Linear(lin) => LayerStateDto::Qwen35Linear {
                conv_buffer: lin.conv_buffer.clone(),
                ssm_states: lin.ssm_states.clone(),
                conv_valid: lin.conv_valid,
                conv_channels: lin.conv_channels,
                conv_kernel: lin.conv_kernel,
            },
            LayerAttentionState::None => LayerStateDto::None,
        }
    }
}

impl LayerStateDto {
    /// Convert DTO back into runtime state, re-allocating full-size caches.
    pub(super) fn into_runtime_state(
        self,
        total_sequence_length: usize,
    ) -> Result<LayerAttentionState, E2eError> {
        match self {
            LayerStateDto::Standard => Ok(LayerAttentionState::Standard),
            LayerStateDto::Qwen35Full {
                k_cache: k_data,
                v_cache: v_data,
                cached_len,
                kv_features,
            } => {
                if kv_features == 0 {
                    return Err(E2eError::CheckpointDeserialize(
                        "kv_features must be > 0".into(),
                    ));
                }
                let cache_size = total_sequence_length
                    .checked_mul(kv_features)
                    .ok_or(E2eError::MemorySizeOverflow)?;
                let data_len = cached_len
                    .checked_mul(kv_features)
                    .ok_or(E2eError::MemorySizeOverflow)?;
                if k_data.len() != data_len || v_data.len() != data_len {
                    return Err(E2eError::CheckpointDeserialize(format!(
                        "KV cache data length mismatch: expected {data_len}, got k={} v={}",
                        k_data.len(),
                        v_data.len()
                    )));
                }
                let mut k_cache = vec![0.0f32; cache_size];
                let mut v_cache = vec![0.0f32; cache_size];
                k_cache[..data_len].copy_from_slice(&k_data);
                v_cache[..data_len].copy_from_slice(&v_data);
                Ok(LayerAttentionState::Qwen35Full(
                    super::state::Qwen35FullAttentionState {
                        k_cache,
                        v_cache,
                        cached_len,
                        kv_features,
                    },
                ))
            }
            LayerStateDto::Qwen35Linear {
                conv_buffer,
                ssm_states,
                conv_valid,
                conv_channels,
                conv_kernel,
            } => Ok(LayerAttentionState::Qwen35Linear(LinearAttentionState {
                conv_buffer,
                ssm_states,
                conv_valid,
                conv_channels,
                conv_kernel,
            })),
            LayerStateDto::None => Ok(LayerAttentionState::None),
        }
    }
}

// ---------------------------------------------------------------------------
// Fingerprint construction and validation
// ---------------------------------------------------------------------------

impl ModelFingerprint {
    /// Build a fingerprint from resolved layer plans and model parameters.
    pub(super) fn from_plans(
        layer_plans: &[super::plan::LayerPlan],
        hidden_features: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
    ) -> Self {
        let layer_types = layer_plans
            .iter()
            .map(|plan| match &plan.attention {
                Some(AttentionLayerPlan::Standard(_)) => LayerTypeTag::Standard,
                Some(AttentionLayerPlan::Qwen35Full(attn)) => LayerTypeTag::Qwen35Full {
                    kv_head_count: attn.kv_head_count,
                    head_dimension: attn.head_dimension,
                },
                Some(AttentionLayerPlan::Qwen35Linear(attn)) => {
                    let conv_channels = attn.inner_size + 2 * attn.group_count * attn.state_size;
                    LayerTypeTag::Qwen35Linear {
                        conv_kernel: attn.conv_kernel,
                        conv_channels,
                        time_step_rank: attn.time_step_rank,
                        state_size: attn.state_size,
                    }
                }
                None => LayerTypeTag::None,
            })
            .collect();
        Self {
            layer_count: layer_plans.len(),
            hidden_features,
            vocab_size,
            rms_norm_eps_bits: rms_norm_eps.to_bits(),
            layer_types,
        }
    }

    /// Validate that a checkpoint fingerprint is compatible with the current model.
    pub(super) fn validate_against(&self, other: &Self) -> Result<(), E2eError> {
        if self.layer_count != other.layer_count {
            return Err(E2eError::CheckpointModelMismatch {
                reason: format!(
                    "layer count: checkpoint has {}, model has {}",
                    self.layer_count, other.layer_count
                ),
            });
        }
        if self.hidden_features != other.hidden_features {
            return Err(E2eError::CheckpointModelMismatch {
                reason: format!(
                    "hidden features: checkpoint has {}, model has {}",
                    self.hidden_features, other.hidden_features
                ),
            });
        }
        if self.vocab_size != other.vocab_size {
            return Err(E2eError::CheckpointModelMismatch {
                reason: format!(
                    "vocab size: checkpoint has {}, model has {}",
                    self.vocab_size, other.vocab_size
                ),
            });
        }
        if self.rms_norm_eps_bits != other.rms_norm_eps_bits {
            return Err(E2eError::CheckpointModelMismatch {
                reason: "rms_norm_eps differs".into(),
            });
        }
        for (i, (ours, theirs)) in self.layer_types.iter().zip(&other.layer_types).enumerate() {
            if ours != theirs {
                return Err(E2eError::CheckpointModelMismatch {
                    reason: format!(
                        "layer {i} type mismatch: checkpoint={ours:?}, model={theirs:?}"
                    ),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Checkpoint construction
// ---------------------------------------------------------------------------

/// Input data for constructing a checkpoint. Groups session fields to avoid
/// a long argument list.
pub(super) struct CaptureInput<'a> {
    pub fingerprint: ModelFingerprint,
    pub prompt_token_ids: &'a [i32],
    pub generated_token_ids: &'a [i32],
    pub current_token_count: usize,
    pub max_new_tokens: usize,
    pub total_sequence_length: usize,
    pub pad_token_id: i32,
    pub eos_token_id: Option<i32>,
    pub prefill_done: bool,
    pub finished: bool,
    pub state: &'a GenerationState,
}

impl CheckpointV1 {
    pub(super) fn capture(input: CaptureInput<'_>) -> Self {
        let layer_states = input.state.layers.iter().map(LayerStateDto::from).collect();
        Self {
            version: CHECKPOINT_VERSION,
            fingerprint: input.fingerprint,
            prompt_token_ids: input.prompt_token_ids.to_vec(),
            generated_token_ids: input.generated_token_ids.to_vec(),
            current_token_count: input.current_token_count,
            prompt_token_count: input.prompt_token_ids.len(),
            max_new_tokens: input.max_new_tokens,
            total_sequence_length: input.total_sequence_length,
            pad_token_id: input.pad_token_id,
            eos_token_id: input.eos_token_id,
            prefill_done: input.prefill_done,
            finished: input.finished,
            layer_states,
        }
    }

    /// Restore `GenerationState` from checkpoint DTOs.
    pub(super) fn restore_state(&self) -> Result<GenerationState, E2eError> {
        let layers: Result<Vec<_>, _> = self
            .layer_states
            .iter()
            .cloned()
            .map(|dto| dto.into_runtime_state(self.total_sequence_length))
            .collect();
        Ok(GenerationState { layers: layers? })
    }
}

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
                version: CHECKPOINT_VERSION,
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

    #[test]
    fn fingerprint_mismatch_layer_count() {
        let fp1 = sample_fingerprint();
        let mut fp2 = sample_fingerprint();
        fp2.layer_count = 3;
        let result = fp1.validate_against(&fp2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("layer count"));
    }

    #[test]
    fn fingerprint_mismatch_hidden_features() {
        let fp1 = sample_fingerprint();
        let mut fp2 = sample_fingerprint();
        fp2.hidden_features = 128;
        let result = fp1.validate_against(&fp2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("hidden features"));
    }

    #[test]
    fn fingerprint_mismatch_layer_type() {
        let fp1 = sample_fingerprint();
        let mut fp2 = sample_fingerprint();
        fp2.layer_types[0] = LayerTypeTag::Standard;
        let result = fp1.validate_against(&fp2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("layer 0"));
    }

    #[test]
    fn kv_state_dto_trims_unused_cache() {
        use super::super::state::Qwen35FullAttentionState;
        let mut kv = Qwen35FullAttentionState::new(16, 2, 4).unwrap();
        kv.append_batch(&[1.0; 8], &[2.0; 8], 1).unwrap();
        // Runtime cache is 16 * 8 = 128, but only 1 token (8 floats) is populated
        assert_eq!(kv.k_cache.len(), 128);

        let dto: LayerStateDto = (&LayerAttentionState::Qwen35Full(kv)).into();
        if let LayerStateDto::Qwen35Full { k_cache, .. } = &dto {
            // DTO should only contain the populated portion
            assert_eq!(k_cache.len(), 8);
        } else {
            panic!("expected Qwen35Full DTO");
        }
    }

    #[test]
    fn kv_state_dto_roundtrip_restores_full_cache() {
        let dto = LayerStateDto::Qwen35Full {
            k_cache: vec![1.0; 8],
            v_cache: vec![2.0; 8],
            cached_len: 1,
            kv_features: 8,
        };
        let state = dto.into_runtime_state(16).unwrap();
        if let LayerAttentionState::Qwen35Full(kv) = &state {
            // Full cache should be re-allocated to total_sequence_length * kv_features
            assert_eq!(kv.k_cache.len(), 128);
            assert_eq!(kv.cached_len, 1);
            // First 8 elements populated, rest zero
            assert_eq!(&kv.k_cache[..8], &[1.0; 8]);
            assert!(kv.k_cache[8..].iter().all(|&v| v == 0.0));
        } else {
            panic!("expected Qwen35Full state");
        }
    }
}
