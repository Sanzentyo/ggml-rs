//! Per-layer runtime state for autoregressive token-by-token generation.
//!
//! The generation loop operates in two phases:
//! 1. **Prefill**: Process all prompt tokens at once, capturing per-layer state.
//! 2. **Decode**: Process one new token at a time using cached state.

use super::error::E2eError;
use super::numeric::checked_mul;
use super::plan::{AttentionLayerPlan, LayerPlan};

/// Per-layer attention state for autoregressive generation.
#[derive(Debug, Clone)]
pub(super) enum LayerAttentionState {
    /// Standard attention: no incremental decode (full reprocess each step).
    Standard,
    /// Qwen3.5 full attention: KV cache (post-RoPE K, raw V).
    Qwen35Full(Qwen35FullAttentionState),
    /// Qwen3.5 linear attention: conv buffer + SSM recurrence states.
    Qwen35Linear(LinearAttentionState),
    /// No attention in this layer (MLP-only).
    None,
}

/// KV cache for Qwen3.5 full attention layers.
///
/// Stores post-norm, post-RoPE K values and raw V values for all processed
/// tokens. During decode, the new token's K/V are appended, then the query
/// attends to the full cache (including the newly appended entry).
#[derive(Debug, Clone)]
pub(super) struct Qwen35FullAttentionState {
    /// Post-RoPE K values, flat `[max_tokens × kv_features]`.
    pub(super) k_cache: Vec<f32>,
    /// Raw V values, flat `[max_tokens × kv_features]`.
    pub(super) v_cache: Vec<f32>,
    /// Number of tokens currently in the cache.
    pub(super) cached_len: usize,
    /// Features per KV token (`kv_head_count × head_dimension`).
    pub(super) kv_features: usize,
}

/// Conv buffer + SSM recurrence states for Qwen3.5 linear attention.
///
/// The conv buffer stores the last `d_conv - 1` raw QKV activations (before
/// convolution). During decode, the new token's QKV is convolved using this
/// buffer, and the oldest entry is shifted out.
///
/// The SSM states persist across tokens — the delta-net recurrence updates
/// them at each step.
#[derive(Debug, Clone)]
pub(super) struct LinearAttentionState {
    /// Last `d_conv - 1` raw QKV rows, flat `[(d_conv-1) × conv_channels]`.
    pub(super) conv_buffer: Vec<f32>,
    /// Delta-net recurrence states, flat `[time_step_rank × state_size × state_size]`.
    pub(super) ssm_states: Vec<f32>,
    /// Number of valid rows in conv_buffer (may be less than `d_conv - 1`
    /// when the total processed sequence is shorter than that).
    pub(super) conv_valid: usize,
    /// Conv channels for this layer.
    pub(super) conv_channels: usize,
    /// Conv kernel size.
    pub(super) conv_kernel: usize,
}

/// All per-layer states for the two-phase autoregressive generation loop.
#[derive(Debug, Clone)]
pub(super) struct GenerationState {
    pub(super) layers: Vec<LayerAttentionState>,
}

impl Qwen35FullAttentionState {
    pub(super) fn new(
        max_tokens: usize,
        kv_head_count: usize,
        head_dimension: usize,
    ) -> Result<Self, E2eError> {
        let kv_features = checked_mul(kv_head_count, head_dimension)?;
        let cache_size = checked_mul(max_tokens, kv_features)?;
        Ok(Self {
            k_cache: vec![0.0; cache_size],
            v_cache: vec![0.0; cache_size],
            cached_len: 0,
            kv_features,
        })
    }

    /// Append K and V for `count` tokens at once.
    pub(super) fn append_batch(
        &mut self,
        k_values: &[f32],
        v_values: &[f32],
        count: usize,
    ) -> Result<(), E2eError> {
        let batch_size = checked_mul(count, self.kv_features)?;
        if k_values.len() != batch_size || v_values.len() != batch_size {
            return Err(E2eError::BufferLengthMismatch {
                expected: batch_size,
                actual: k_values.len().min(v_values.len()),
            });
        }
        let offset = checked_mul(self.cached_len, self.kv_features)?;
        if offset + batch_size > self.k_cache.len() {
            return Err(E2eError::SequenceTooLong {
                requested: self.cached_len + count,
                context_length: self.k_cache.len() / self.kv_features,
            });
        }
        self.k_cache[offset..offset + batch_size].copy_from_slice(k_values);
        self.v_cache[offset..offset + batch_size].copy_from_slice(v_values);
        self.cached_len += count;
        Ok(())
    }

    /// Get a K slice for a specific token.
    #[allow(dead_code)] // Used in tests; public API for consumers
    pub(super) fn k_at(&self, token: usize) -> &[f32] {
        let offset = token * self.kv_features;
        &self.k_cache[offset..offset + self.kv_features]
    }

    /// Get a V slice for a specific token.
    #[allow(dead_code)] // Used in tests; public API for consumers
    pub(super) fn v_at(&self, token: usize) -> &[f32] {
        let offset = token * self.kv_features;
        &self.v_cache[offset..offset + self.kv_features]
    }

    /// Number of tokens currently in the cache.
    pub(super) fn token_count(&self) -> usize {
        self.cached_len
    }

    /// Get a K slice for a specific token and KV head.
    pub(super) fn k_head_at(&self, token: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let token_offset = token * self.kv_features + kv_head * head_dim;
        &self.k_cache[token_offset..token_offset + head_dim]
    }

    /// Get a V slice for a specific token and KV head.
    pub(super) fn v_head_at(&self, token: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let token_offset = token * self.kv_features + kv_head * head_dim;
        &self.v_cache[token_offset..token_offset + head_dim]
    }
}

impl LinearAttentionState {
    pub(super) fn new(
        conv_kernel: usize,
        conv_channels: usize,
        time_step_rank: usize,
        state_size: usize,
    ) -> Result<Self, E2eError> {
        let buffer_rows = conv_kernel.saturating_sub(1);
        let buffer_size = checked_mul(buffer_rows, conv_channels)?;
        let ssm_size = checked_mul(time_step_rank, checked_mul(state_size, state_size)?)?;
        Ok(Self {
            conv_buffer: vec![0.0; buffer_size],
            ssm_states: vec![0.0; ssm_size],
            conv_valid: 0,
            conv_channels,
            conv_kernel,
        })
    }

    /// Push a new QKV row into the conv buffer, shifting out the oldest if full.
    pub(super) fn push_conv_row(&mut self, row: &[f32]) -> Result<(), E2eError> {
        if row.len() != self.conv_channels {
            return Err(E2eError::BufferLengthMismatch {
                expected: self.conv_channels,
                actual: row.len(),
            });
        }
        let max_rows = self.conv_kernel.saturating_sub(1);
        if max_rows == 0 {
            return Ok(());
        }
        if self.conv_valid < max_rows {
            let offset = checked_mul(self.conv_valid, self.conv_channels)?;
            self.conv_buffer[offset..offset + self.conv_channels].copy_from_slice(row);
            self.conv_valid += 1;
        } else {
            self.conv_buffer.copy_within(self.conv_channels.., 0);
            let offset = checked_mul(max_rows - 1, self.conv_channels)?;
            self.conv_buffer[offset..offset + self.conv_channels].copy_from_slice(row);
        }
        Ok(())
    }

    /// Capture the last `d_conv - 1` rows from a full QKV buffer after prefill.
    pub(super) fn capture_conv_buffer(
        &mut self,
        qkv: &[f32],
        sequence_length: usize,
    ) -> Result<(), E2eError> {
        let max_rows = self.conv_kernel.saturating_sub(1);
        let rows_to_copy = sequence_length.min(max_rows);
        let src_start = checked_mul(sequence_length - rows_to_copy, self.conv_channels)?;
        let dst_start = checked_mul(max_rows - rows_to_copy, self.conv_channels)?;
        let copy_len = checked_mul(rows_to_copy, self.conv_channels)?;
        // Zero-fill unused prefix (for prompts shorter than d_conv-1)
        self.conv_buffer[..dst_start].fill(0.0);
        self.conv_buffer[dst_start..dst_start + copy_len]
            .copy_from_slice(&qkv[src_start..src_start + copy_len]);
        self.conv_valid = rows_to_copy;
        Ok(())
    }

    /// Capture SSM states from a flat states buffer after prefill.
    pub(super) fn capture_ssm_states(&mut self, states: &[f32]) {
        self.ssm_states[..states.len()].copy_from_slice(states);
    }
}

impl GenerationState {
    /// Initialize empty state for all layers based on their plan types.
    pub(super) fn new(
        layer_plans: &[LayerPlan],
        total_sequence_length: usize,
    ) -> Result<Self, E2eError> {
        let mut layers = Vec::with_capacity(layer_plans.len());
        for plan in layer_plans {
            let state = match &plan.attention {
                Some(AttentionLayerPlan::Standard(_)) => LayerAttentionState::Standard,
                Some(AttentionLayerPlan::Qwen35Full(attn)) => {
                    LayerAttentionState::Qwen35Full(Qwen35FullAttentionState::new(
                        total_sequence_length,
                        attn.kv_head_count,
                        attn.head_dimension,
                    )?)
                }
                Some(AttentionLayerPlan::Qwen35Linear(attn)) => {
                    let conv_channels = attn.inner_size
                        + checked_mul(checked_mul(attn.group_count, attn.state_size)?, 2)?;
                    LayerAttentionState::Qwen35Linear(LinearAttentionState::new(
                        attn.conv_kernel,
                        conv_channels,
                        attn.time_step_rank,
                        attn.state_size,
                    )?)
                }
                None => LayerAttentionState::None,
            };
            layers.push(state);
        }
        Ok(Self { layers })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_append_and_retrieve() {
        let mut state = Qwen35FullAttentionState::new(4, 2, 3).unwrap();
        assert_eq!(state.kv_features, 6);
        assert_eq!(state.cached_len, 0);

        let k = vec![1.0_f32; 6];
        let v = vec![2.0_f32; 6];
        state.append_batch(&k, &v, 1).unwrap();
        assert_eq!(state.cached_len, 1);
        assert_eq!(state.k_at(0), &[1.0; 6]);
        assert_eq!(state.v_at(0), &[2.0; 6]);
    }

    #[test]
    fn conv_buffer_push_and_shift() {
        let mut state = LinearAttentionState::new(3, 2, 1, 1).unwrap();
        // kernel=3, so buffer holds 2 rows of 2 channels each
        assert_eq!(state.conv_buffer.len(), 4);
        assert_eq!(state.conv_valid, 0);

        state.push_conv_row(&[1.0, 2.0]).unwrap();
        assert_eq!(state.conv_valid, 1);
        assert_eq!(&state.conv_buffer[0..2], &[1.0, 2.0]);

        state.push_conv_row(&[3.0, 4.0]).unwrap();
        assert_eq!(state.conv_valid, 2);
        assert_eq!(&state.conv_buffer, &[1.0, 2.0, 3.0, 4.0]);

        // Buffer full, should shift
        state.push_conv_row(&[5.0, 6.0]).unwrap();
        assert_eq!(state.conv_valid, 2);
        assert_eq!(&state.conv_buffer, &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn capture_conv_buffer_short_sequence() {
        let mut state = LinearAttentionState::new(4, 2, 1, 1).unwrap();
        // kernel=4, buffer holds 3 rows
        let qkv = vec![10.0_f32, 20.0]; // 1 token
        state.capture_conv_buffer(&qkv, 1).unwrap();
        assert_eq!(state.conv_valid, 1);
        // First 2 rows should be zero, last row should be the token
        assert_eq!(&state.conv_buffer, &[0.0, 0.0, 0.0, 0.0, 10.0, 20.0]);
    }
}
