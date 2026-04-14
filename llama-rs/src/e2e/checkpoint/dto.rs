//! Versioned DTO types and validation for checkpoint serialization.
//!
//! This module owns the wire-format types (`CheckpointV1`, `ModelFingerprint`,
//! `LayerTypeTag`, `LayerStateDto`) that are decoupled from runtime state.
//! `validate_invariants` lives here because it validates serialized shape.

use super::super::error::E2eError;
use super::super::plan::AttentionLayerPlan;
use serde::{Deserialize, Serialize};

/// Current checkpoint format version.
pub(in crate::e2e) const CHECKPOINT_VERSION: u32 = 2;

// ---------------------------------------------------------------------------
// DTO types (serde, versioned, decoupled from runtime state)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(in crate::e2e) struct CheckpointV1 {
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
pub(in crate::e2e) struct ModelFingerprint {
    pub layer_count: usize,
    pub hidden_features: usize,
    pub vocab_size: usize,
    pub rms_norm_eps_bits: u32,
    pub layer_types: Vec<LayerTypeTag>,
}

/// Discriminant tag for each layer's attention type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(in crate::e2e) enum LayerTypeTag {
    Standard {
        kv_head_count: usize,
        head_dimension: usize,
    },
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
pub(in crate::e2e) enum LayerStateDto {
    Standard {
        k_cache: Vec<f32>,
        v_cache: Vec<f32>,
        cached_len: usize,
        kv_features: usize,
    },
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
// ModelFingerprint: construction + compatibility validation
// ---------------------------------------------------------------------------

impl ModelFingerprint {
    /// Build a fingerprint from resolved layer plans and model parameters.
    pub(in crate::e2e) fn from_plans(
        layer_plans: &[super::super::plan::LayerPlan],
        hidden_features: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
    ) -> Self {
        let layer_types = layer_plans
            .iter()
            .map(|plan| match &plan.attention {
                Some(attn @ AttentionLayerPlan::Standard(_)) => LayerTypeTag::Standard {
                    kv_head_count: attn.kv_head_count(),
                    head_dimension: attn.head_dimension(),
                },
                Some(attn @ AttentionLayerPlan::Qwen35Full(_)) => LayerTypeTag::Qwen35Full {
                    kv_head_count: attn.kv_head_count(),
                    head_dimension: attn.head_dimension(),
                },
                Some(AttentionLayerPlan::Qwen35Linear(lin)) => LayerTypeTag::Qwen35Linear {
                    conv_kernel: lin.conv_kernel,
                    conv_channels: lin.conv_channels().unwrap_or(0),
                    time_step_rank: lin.time_step_rank,
                    state_size: lin.state_size,
                },
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
    pub(in crate::e2e) fn validate_against(&self, other: &Self) -> Result<(), E2eError> {
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
// CheckpointV1: structural validation
// ---------------------------------------------------------------------------

impl CheckpointV1 {
    /// Validate internal invariants of a deserialized checkpoint.
    ///
    /// Prevents panics from malformed or corrupted data by checking all
    /// structural invariants before constructing a session.
    pub(in crate::e2e) fn validate_invariants(&self) -> Result<(), E2eError> {
        let err = |msg: String| E2eError::CheckpointDeserialize(msg);

        if self.prompt_token_count != self.prompt_token_ids.len() {
            return Err(err(format!(
                "prompt_token_count ({}) != prompt_token_ids.len() ({})",
                self.prompt_token_count,
                self.prompt_token_ids.len()
            )));
        }
        let expected_token_count = self
            .prompt_token_count
            .checked_add(self.generated_token_ids.len())
            .ok_or(E2eError::MemorySizeOverflow)?;
        if self.current_token_count != expected_token_count {
            return Err(err(format!(
                "current_token_count ({}) != prompt + generated ({})",
                self.current_token_count, expected_token_count
            )));
        }
        if self.current_token_count > self.total_sequence_length {
            return Err(err(format!(
                "current_token_count ({}) > total_sequence_length ({})",
                self.current_token_count, self.total_sequence_length
            )));
        }
        if self.generated_token_ids.len() > self.max_new_tokens {
            return Err(err(format!(
                "generated count ({}) > max_new_tokens ({})",
                self.generated_token_ids.len(),
                self.max_new_tokens
            )));
        }
        if self.layer_states.len() != self.fingerprint.layer_count {
            return Err(err(format!(
                "layer_states.len() ({}) != fingerprint.layer_count ({})",
                self.layer_states.len(),
                self.fingerprint.layer_count
            )));
        }
        if self.prompt_token_count == 0 {
            return Err(err("prompt_token_count must be > 0".into()));
        }
        // Validate token IDs are in vocab range
        let vocab = self.fingerprint.vocab_size;
        for &token_id in self
            .prompt_token_ids
            .iter()
            .chain(&self.generated_token_ids)
        {
            if token_id < 0 || token_id as usize >= vocab {
                return Err(err(format!(
                    "token id {token_id} out of vocab range [0, {vocab})"
                )));
            }
        }
        if self.pad_token_id < 0 || self.pad_token_id as usize >= vocab {
            return Err(err(format!(
                "pad_token_id {} out of vocab range [0, {vocab})",
                self.pad_token_id
            )));
        }
        if let Some(eos) = self.eos_token_id
            && (eos < 0 || eos as usize >= vocab)
        {
            return Err(err(format!(
                "eos_token_id {eos} out of vocab range [0, {vocab})"
            )));
        }

        // Per-layer DTO validation: check internal consistency of each state
        for (i, dto) in self.layer_states.iter().enumerate() {
            match dto {
                LayerStateDto::Qwen35Full {
                    k_cache,
                    v_cache,
                    cached_len,
                    kv_features,
                } => {
                    if *kv_features == 0 {
                        return Err(err(format!(
                            "layer {i}: Qwen35Full kv_features must be > 0"
                        )));
                    }
                    if *cached_len > self.total_sequence_length {
                        return Err(err(format!(
                            "layer {i}: cached_len ({cached_len}) > total_sequence_length ({})",
                            self.total_sequence_length
                        )));
                    }
                    let expected_data = cached_len
                        .checked_mul(*kv_features)
                        .ok_or(E2eError::MemorySizeOverflow)?;
                    if k_cache.len() != expected_data || v_cache.len() != expected_data {
                        return Err(err(format!(
                            "layer {i}: KV data length mismatch: expected {expected_data}, got k={} v={}",
                            k_cache.len(),
                            v_cache.len()
                        )));
                    }
                }
                LayerStateDto::Qwen35Linear {
                    conv_buffer,
                    conv_valid,
                    conv_channels,
                    conv_kernel,
                    ssm_states,
                } => {
                    if *conv_channels == 0 {
                        return Err(err(format!(
                            "layer {i}: Qwen35Linear conv_channels must be > 0"
                        )));
                    }
                    if *conv_kernel < 2 {
                        return Err(err(format!(
                            "layer {i}: Qwen35Linear conv_kernel must be >= 2"
                        )));
                    }
                    let max_rows = conv_kernel - 1;
                    if *conv_valid > max_rows {
                        return Err(err(format!(
                            "layer {i}: conv_valid ({conv_valid}) > max rows ({max_rows})"
                        )));
                    }
                    let expected_buf = max_rows
                        .checked_mul(*conv_channels)
                        .ok_or(E2eError::MemorySizeOverflow)?;
                    if conv_buffer.len() != expected_buf {
                        return Err(err(format!(
                            "layer {i}: conv_buffer length mismatch: expected {expected_buf}, got {}",
                            conv_buffer.len()
                        )));
                    }
                    if ssm_states.is_empty() {
                        return Err(err(format!(
                            "layer {i}: Qwen35Linear ssm_states must not be empty"
                        )));
                    }
                }
                LayerStateDto::Standard {
                    cached_len,
                    kv_features,
                    k_cache,
                    v_cache,
                } => {
                    if *cached_len > 0 && *kv_features == 0 {
                        return Err(E2eError::CheckpointDeserialize(format!(
                            "layer {i}: Standard kv_features must be > 0 when cached_len > 0"
                        )));
                    }
                    if *cached_len > self.total_sequence_length {
                        return Err(E2eError::CheckpointDeserialize(format!(
                            "layer {i}: Standard cached_len ({cached_len}) > total_sequence_length ({})",
                            self.total_sequence_length
                        )));
                    }
                    if *kv_features > 0 {
                        let expected_data = cached_len
                            .checked_mul(*kv_features)
                            .ok_or(E2eError::MemorySizeOverflow)?;
                        if k_cache.len() != expected_data {
                            return Err(E2eError::CheckpointDeserialize(format!(
                                "layer {i}: Standard k_cache length mismatch: expected {expected_data}, got {}",
                                k_cache.len()
                            )));
                        }
                        if v_cache.len() != expected_data {
                            return Err(E2eError::CheckpointDeserialize(format!(
                                "layer {i}: Standard v_cache length mismatch: expected {expected_data}, got {}",
                                v_cache.len()
                            )));
                        }
                    }
                }
                LayerStateDto::None => {}
            }
        }

        Ok(())
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

    #[test]
    fn fingerprint_equality() {
        let a = sample_fingerprint();
        let b = sample_fingerprint();
        assert_eq!(a, b);
    }

    #[test]
    fn fingerprint_validate_against_rejects_layer_count() {
        let a = sample_fingerprint();
        let mut b = sample_fingerprint();
        b.layer_count = 99;
        let err = a.validate_against(&b).unwrap_err().to_string();
        assert!(err.contains("layer count"), "got: {err}");
    }

    #[test]
    fn fingerprint_validate_against_rejects_hidden_features() {
        let a = sample_fingerprint();
        let mut b = sample_fingerprint();
        b.hidden_features = 999;
        let err = a.validate_against(&b).unwrap_err().to_string();
        assert!(err.contains("hidden features"), "got: {err}");
    }

    #[test]
    fn fingerprint_validate_against_rejects_vocab_size() {
        let a = sample_fingerprint();
        let mut b = sample_fingerprint();
        b.vocab_size = 9999;
        let err = a.validate_against(&b).unwrap_err().to_string();
        assert!(err.contains("vocab size"), "got: {err}");
    }

    #[test]
    fn fingerprint_validate_against_rejects_eps() {
        let a = sample_fingerprint();
        let mut b = sample_fingerprint();
        b.rms_norm_eps_bits = 1e-3_f32.to_bits();
        let err = a.validate_against(&b).unwrap_err().to_string();
        assert!(err.contains("rms_norm_eps"), "got: {err}");
    }

    #[test]
    fn fingerprint_validate_against_rejects_layer_type_mismatch() {
        let a = sample_fingerprint();
        let mut b = sample_fingerprint();
        b.layer_types[0] = LayerTypeTag::None;
        let err = a.validate_against(&b).unwrap_err().to_string();
        assert!(err.contains("type mismatch"), "got: {err}");
    }

    fn sample_checkpoint_v1() -> CheckpointV1 {
        CheckpointV1 {
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
        }
    }

    #[test]
    fn validate_invariants_accepts_valid() {
        sample_checkpoint_v1().validate_invariants().unwrap();
    }

    #[test]
    fn validate_rejects_prompt_count_mismatch() {
        let mut cp = sample_checkpoint_v1();
        cp.prompt_token_count = 99;
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("prompt_token_count"), "got: {err}");
    }

    #[test]
    fn validate_rejects_token_count_mismatch() {
        let mut cp = sample_checkpoint_v1();
        cp.current_token_count = 99;
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("current_token_count"), "got: {err}");
    }

    #[test]
    fn validate_rejects_overflow_sequence() {
        let mut cp = sample_checkpoint_v1();
        cp.total_sequence_length = 4; // less than current_token_count (5)
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("total_sequence_length"), "got: {err}");
    }

    #[test]
    fn validate_rejects_generated_exceeds_max() {
        let mut cp = sample_checkpoint_v1();
        cp.max_new_tokens = 1;
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("max_new_tokens"), "got: {err}");
    }

    #[test]
    fn validate_rejects_layer_count_mismatch() {
        let mut cp = sample_checkpoint_v1();
        cp.layer_states.pop();
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("layer_count"), "got: {err}");
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        let mut cp = sample_checkpoint_v1();
        cp.prompt_token_ids.clear();
        cp.prompt_token_count = 0;
        cp.current_token_count = 2;
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("prompt_token_count must be > 0"), "got: {err}");
    }

    #[test]
    fn validate_rejects_out_of_vocab_token() {
        let mut cp = sample_checkpoint_v1();
        cp.prompt_token_ids[0] = 9999;
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("out of vocab range"), "got: {err}");
    }

    #[test]
    fn validate_rejects_negative_token() {
        let mut cp = sample_checkpoint_v1();
        cp.prompt_token_ids[0] = -1;
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("out of vocab range"), "got: {err}");
    }

    #[test]
    fn validate_rejects_bad_pad_token() {
        let mut cp = sample_checkpoint_v1();
        cp.pad_token_id = 9999;
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("pad_token_id"), "got: {err}");
    }

    #[test]
    fn validate_rejects_bad_eos_token() {
        let mut cp = sample_checkpoint_v1();
        cp.eos_token_id = Some(9999);
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("eos_token_id"), "got: {err}");
    }

    #[test]
    fn validate_rejects_zero_kv_features_with_data() {
        let mut cp = sample_checkpoint_v1();
        cp.layer_states[0] = LayerStateDto::Qwen35Full {
            k_cache: vec![1.0],
            v_cache: vec![2.0],
            cached_len: 1,
            kv_features: 0,
        };
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("kv_features must be > 0"), "got: {err}");
    }

    #[test]
    fn validate_rejects_kv_cache_length_mismatch() {
        let mut cp = sample_checkpoint_v1();
        cp.layer_states[0] = LayerStateDto::Qwen35Full {
            k_cache: vec![1.0; 5],
            v_cache: vec![2.0; 16],
            cached_len: 1,
            kv_features: 16,
        };
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("KV data length mismatch"), "got: {err}");
    }

    #[test]
    fn validate_rejects_conv_buffer_length_mismatch() {
        let mut cp = sample_checkpoint_v1();
        cp.layer_states[1] = LayerStateDto::Qwen35Linear {
            conv_buffer: vec![3.0; 5],
            ssm_states: vec![4.0; 16],
            conv_valid: 2,
            conv_channels: 16,
            conv_kernel: 3,
        };
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("conv_buffer length mismatch"), "got: {err}");
    }

    #[test]
    fn validate_rejects_empty_ssm_states() {
        let mut cp = sample_checkpoint_v1();
        cp.layer_states[1] = LayerStateDto::Qwen35Linear {
            conv_buffer: vec![3.0; 32],
            ssm_states: vec![],
            conv_valid: 2,
            conv_channels: 16,
            conv_kernel: 3,
        };
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(err.contains("ssm_states must not be empty"), "got: {err}");
    }

    fn sample_checkpoint_with_standard_v1() -> CheckpointV1 {
        let kv_head_count = 2;
        let head_dimension = 4;
        let kv_features = kv_head_count * head_dimension;
        let cached_len = 3;
        let data_len = cached_len * kv_features;
        CheckpointV1 {
            version: CHECKPOINT_VERSION,
            fingerprint: ModelFingerprint {
                layer_count: 1,
                hidden_features: 64,
                vocab_size: 1000,
                rms_norm_eps_bits: 1e-5_f32.to_bits(),
                layer_types: vec![LayerTypeTag::Standard {
                    kv_head_count,
                    head_dimension,
                }],
            },
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
            layer_states: vec![LayerStateDto::Standard {
                k_cache: (0..data_len).map(|i| i as f32 * 0.1).collect(),
                v_cache: (0..data_len).map(|i| i as f32 * 0.2).collect(),
                cached_len,
                kv_features,
            }],
        }
    }

    #[test]
    fn standard_validate_rejects_cached_len_overflow() {
        let mut cp = sample_checkpoint_with_standard_v1();
        if let LayerStateDto::Standard {
            ref mut cached_len, ..
        } = cp.layer_states[0]
        {
            *cached_len = cp.total_sequence_length + 1;
        }
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(
            err.contains("cached_len"),
            "expected cached_len overflow error, got: {err}"
        );
    }

    #[test]
    fn standard_validate_rejects_cache_length_mismatch() {
        let mut cp = sample_checkpoint_with_standard_v1();
        if let LayerStateDto::Standard {
            ref mut k_cache, ..
        } = cp.layer_states[0]
        {
            k_cache.truncate(5);
        }
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(
            err.contains("k_cache length mismatch"),
            "expected k_cache length error, got: {err}"
        );
    }

    #[test]
    fn standard_validate_rejects_zero_kv_features_with_data() {
        let mut cp = sample_checkpoint_with_standard_v1();
        if let LayerStateDto::Standard {
            ref mut kv_features,
            ..
        } = cp.layer_states[0]
        {
            *kv_features = 0;
        }
        let err = cp.validate_invariants().unwrap_err().to_string();
        assert!(
            err.contains("kv_features must be > 0"),
            "expected kv_features error, got: {err}"
        );
    }
}
