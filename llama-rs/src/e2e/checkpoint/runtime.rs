//! Runtime state ↔ DTO conversion, capture, and restore.
//!
//! This module owns the mapping between in-memory runtime state
//! (`LayerAttentionState`) and the serializable DTOs defined in `dto`.
//! It also provides the `CaptureInput` builder and the
//! `CheckpointV1::capture` / `CheckpointV1::restore_state` entry-points.

use super::super::error::E2eError;
use super::super::state::{
    GenerationState, LayerAttentionState, LinearAttentionState, StandardAttentionState,
};
use super::dto::{CHECKPOINT_VERSION, CheckpointV1, LayerStateDto, ModelFingerprint};

// ---------------------------------------------------------------------------
// Conversion: runtime state → DTO
// ---------------------------------------------------------------------------

impl From<&LayerAttentionState> for LayerStateDto {
    fn from(state: &LayerAttentionState) -> Self {
        match state {
            LayerAttentionState::Standard(kv) => LayerStateDto::Standard {
                k_cache: kv.k_cache[..kv.cached_len * kv.kv_features].to_vec(),
                v_cache: kv.v_cache[..kv.cached_len * kv.kv_features].to_vec(),
                cached_len: kv.cached_len,
                kv_features: kv.kv_features,
            },
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

// ---------------------------------------------------------------------------
// Conversion: DTO → runtime state
// ---------------------------------------------------------------------------

/// Validated KV cache pair restored from checkpoint data.
struct RestoredKvCache {
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    cached_len: usize,
    kv_features: usize,
}

/// Validate and re-allocate a KV cache pair from checkpoint data.
///
/// Checks `kv_features > 0`, `cached_len ≤ total_sequence_length`, and that
/// the serialised data lengths match `cached_len * kv_features`. Returns
/// full-size zero-initialised caches with the serialised portion copied in.
fn restore_kv_cache(
    k_data: Vec<f32>,
    v_data: Vec<f32>,
    cached_len: usize,
    kv_features: usize,
    total_sequence_length: usize,
    label: &str,
) -> Result<RestoredKvCache, E2eError> {
    if kv_features == 0 {
        return Err(E2eError::CheckpointDeserialize(format!(
            "{label} kv_features must be > 0"
        )));
    }
    if cached_len > total_sequence_length {
        return Err(E2eError::CheckpointDeserialize(format!(
            "{label} cached_len ({cached_len}) exceeds total_sequence_length ({total_sequence_length})"
        )));
    }
    let cache_size = total_sequence_length
        .checked_mul(kv_features)
        .ok_or(E2eError::MemorySizeOverflow)?;
    let data_len = cached_len
        .checked_mul(kv_features)
        .ok_or(E2eError::MemorySizeOverflow)?;
    if k_data.len() != data_len || v_data.len() != data_len {
        return Err(E2eError::CheckpointDeserialize(format!(
            "{label} KV data length mismatch: expected {data_len}, got k={} v={}",
            k_data.len(),
            v_data.len()
        )));
    }
    let mut k_cache = vec![0.0f32; cache_size];
    let mut v_cache = vec![0.0f32; cache_size];
    k_cache[..data_len].copy_from_slice(&k_data);
    v_cache[..data_len].copy_from_slice(&v_data);
    Ok(RestoredKvCache {
        k_cache,
        v_cache,
        cached_len,
        kv_features,
    })
}

impl LayerStateDto {
    /// Convert DTO back into runtime state, re-allocating full-size caches.
    pub(in crate::e2e) fn into_runtime_state(
        self,
        total_sequence_length: usize,
    ) -> Result<LayerAttentionState, E2eError> {
        match self {
            LayerStateDto::Standard {
                k_cache,
                v_cache,
                cached_len,
                kv_features,
            } => {
                let kv = restore_kv_cache(
                    k_cache,
                    v_cache,
                    cached_len,
                    kv_features,
                    total_sequence_length,
                    "standard",
                )?;
                Ok(LayerAttentionState::Standard(StandardAttentionState {
                    k_cache: kv.k_cache,
                    v_cache: kv.v_cache,
                    cached_len: kv.cached_len,
                    kv_features: kv.kv_features,
                }))
            }
            LayerStateDto::Qwen35Full {
                k_cache,
                v_cache,
                cached_len,
                kv_features,
            } => {
                let kv = restore_kv_cache(
                    k_cache,
                    v_cache,
                    cached_len,
                    kv_features,
                    total_sequence_length,
                    "qwen35_full",
                )?;
                Ok(LayerAttentionState::Qwen35Full(
                    super::super::state::Qwen35FullAttentionState {
                        k_cache: kv.k_cache,
                        v_cache: kv.v_cache,
                        cached_len: kv.cached_len,
                        kv_features: kv.kv_features,
                        gpu_scoring_failed: false,
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
// Checkpoint capture and restore
// ---------------------------------------------------------------------------

/// Input data for constructing a checkpoint. Groups session fields to avoid
/// a long argument list.
pub(in crate::e2e) struct CaptureInput<'a> {
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
    pub(in crate::e2e) fn capture(input: CaptureInput<'_>) -> Self {
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
    pub(in crate::e2e) fn restore_state(&self) -> Result<GenerationState, E2eError> {
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
    use super::super::dto::{
        CHECKPOINT_VERSION, CheckpointV1, LayerStateDto, LayerTypeTag, ModelFingerprint,
    };
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

    #[test]
    fn kv_state_dto_trims_unused_cache() {
        use super::super::super::state::Qwen35FullAttentionState;
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

    #[test]
    fn standard_capture_roundtrip_via_from_impl() {
        let kv_features = 8;
        let cached_len = 3;
        let total_seq = 13;
        let runtime = LayerAttentionState::Standard(StandardAttentionState {
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

        // Restore into full-size runtime state
        let restored = dto.into_runtime_state(total_seq).unwrap();
        match &restored {
            LayerAttentionState::Standard(kv) => {
                assert_eq!(kv.k_cache.len(), total_seq * kv_features);
                assert_eq!(kv.cached_len, cached_len);
                let data_len = cached_len * kv_features;
                for i in 0..data_len {
                    assert_eq!(kv.k_cache[i], i as f32 * 0.1, "k_cache[{i}] mismatch");
                }
                for i in data_len..kv.k_cache.len() {
                    assert_eq!(kv.k_cache[i], 0.0, "k_cache[{i}] should be zero");
                }
            }
            _ => panic!("expected Standard"),
        }
    }

    #[test]
    fn capture_builds_checkpoint_from_runtime_state() {
        let kv_state =
            LayerAttentionState::Qwen35Full(super::super::super::state::Qwen35FullAttentionState {
                k_cache: vec![1.0; 128],
                v_cache: vec![2.0; 128],
                cached_len: 1,
                kv_features: 16,
                gpu_scoring_failed: false,
            });
        let lin_state = LayerAttentionState::Qwen35Linear(LinearAttentionState {
            conv_buffer: vec![3.0; 32],
            ssm_states: vec![4.0; 16],
            conv_valid: 2,
            conv_channels: 16,
            conv_kernel: 3,
        });
        let gen_state = GenerationState {
            layers: vec![kv_state, lin_state],
        };
        let input = CaptureInput {
            fingerprint: sample_fingerprint(),
            prompt_token_ids: &[1, 2, 3],
            generated_token_ids: &[42, 43],
            current_token_count: 5,
            max_new_tokens: 10,
            total_sequence_length: 16,
            pad_token_id: 0,
            eos_token_id: Some(2),
            prefill_done: true,
            finished: false,
            state: &gen_state,
        };
        let cp = CheckpointV1::capture(input);
        assert_eq!(cp.version, CHECKPOINT_VERSION);
        assert_eq!(cp.prompt_token_count, 3);
        assert_eq!(cp.layer_states.len(), 2);
        cp.validate_invariants().unwrap();
    }

    #[test]
    fn restore_state_rebuilds_generation_state() {
        let cp = CheckpointV1 {
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
            layer_states: vec![
                LayerStateDto::Qwen35Full {
                    k_cache: vec![1.0; 16],
                    v_cache: vec![2.0; 16],
                    cached_len: 1,
                    kv_features: 16,
                },
                LayerStateDto::Qwen35Linear {
                    conv_buffer: vec![3.0; 32],
                    ssm_states: vec![4.0; 16],
                    conv_valid: 2,
                    conv_channels: 16,
                    conv_kernel: 3,
                },
            ],
        };
        let state = cp.restore_state().unwrap();
        assert_eq!(state.layers.len(), 2);
        match &state.layers[0] {
            LayerAttentionState::Qwen35Full(kv) => {
                // Full cache re-allocated to total_sequence_length * kv_features
                assert_eq!(kv.k_cache.len(), 13 * 16);
                assert_eq!(kv.cached_len, 1);
            }
            _ => panic!("expected Qwen35Full"),
        }
        match &state.layers[1] {
            LayerAttentionState::Qwen35Linear(lin) => {
                assert_eq!(lin.conv_buffer.len(), 32);
                assert_eq!(lin.conv_valid, 2);
            }
            _ => panic!("expected Qwen35Linear"),
        }
    }

    #[test]
    fn restore_rejects_cached_len_exceeding_sequence_length() {
        // cached_len=5 but total_sequence_length=4 → must fail
        let dto = LayerStateDto::Standard {
            k_cache: vec![1.0; 40],
            v_cache: vec![2.0; 40],
            cached_len: 5,
            kv_features: 8,
        };
        let err = dto.into_runtime_state(4).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("cached_len"),
            "error should mention cached_len: {msg}"
        );
    }

    #[test]
    fn restore_rejects_cached_len_exceeding_sequence_length_qwen35() {
        let dto = LayerStateDto::Qwen35Full {
            k_cache: vec![1.0; 24],
            v_cache: vec![2.0; 24],
            cached_len: 3,
            kv_features: 8,
        };
        let err = dto.into_runtime_state(2).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("cached_len"),
            "error should mention cached_len: {msg}"
        );
    }
}
