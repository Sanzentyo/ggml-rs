//! Attention dispatch strategies for the generation loop.
//!
//! Three strategies implement [`AttentionStrategy`]:
//! - [`InferenceStrategy`] — stateless full-reprocess
//! - [`PrefillStrategy`] — captures state during prefill
//! - [`DecodeStrategy`] — uses cached state for single-token decode

use crate::e2e::attention::{
    qwen35_full_attention_decode_step, qwen35_full_attention_inference,
    qwen35_full_attention_prefill, standard_attention_decode_step, standard_attention_inference,
    standard_attention_prefill,
};
use crate::e2e::error::E2eError;
use crate::e2e::linear_attention::{
    qwen35_linear_attention_decode_step, qwen35_linear_attention_inference,
    qwen35_linear_attention_prefill,
};
use crate::e2e::plan::AttentionLayerPlan;
use crate::e2e::state::{GenerationState, LayerAttentionState};
use crate::e2e::tensor_ops::rms_norm_with_weight;
use ggml_rs::Backend;

/// Trait for dispatching per-layer attention computation.
///
/// For Qwen3.5 layers, `input` is un-normed; the norm is done in-graph.
/// For `Standard` attention and decode paths, the strategy applies host-side
/// norm internally before dispatching.
pub(in crate::e2e) trait AttentionStrategy {
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
pub(in crate::e2e) struct InferenceStrategy;

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
            AttentionLayerPlan::Standard(attn) => standard_attention_inference(
                attn,
                input,
                seq_len,
                rms_norm_eps,
                &attn.norm_values,
                backend,
            ),
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
pub(in crate::e2e) struct PrefillStrategy<'a> {
    pub(in crate::e2e) state: &'a mut GenerationState,
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
            (AttentionLayerPlan::Standard(attn), LayerAttentionState::Standard(s)) => {
                standard_attention_prefill(
                    attn,
                    input,
                    seq_len,
                    rms_norm_eps,
                    &attn.norm_values,
                    s,
                    backend,
                )
            }
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
pub(in crate::e2e) struct DecodeStrategy<'a> {
    pub(in crate::e2e) state: &'a mut GenerationState,
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
            (AttentionLayerPlan::Standard(attn), LayerAttentionState::Standard(s)) => {
                standard_attention_decode_step(attn, &normalized, rms_norm_eps, s, backend)
            }
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
