//! Qwen3.5 full (standard-style) attention with gated Q and NeoX RoPE.
//!
//! Provides full-sequence inference, prefill (capturing KV cache), and
//! single-token decode step using cached state.
//!
//! Implementation is split across coherent submodules:
//! - [`persistent`]: Backend-resident KV cache and GPU-accelerated scoring
//! - [`projection`]: QKV projection, deinterleaving, and preparation
//! - [`shared`]: RoPE utilities and flash-attention pipeline helpers
//! - [`qwen35_full`]: Qwen3.5 gated attention (fused graph + decode)
//! - [`standard`]: Standard (non-gated) attention (fused graph + decode)

mod persistent;
mod projection;
mod qwen35_full;
pub(in crate::e2e) mod shared;
mod standard;

// Re-exports: keep existing import paths stable for e2e consumers
// (generation.rs, tensor_ops.rs).
pub(super) use persistent::{
    PersistentKvCache, PersistentScoringContext, build_persistent_kv_cache,
};
pub(super) use projection::{QkvProjections, full_attention_hidden_features, prepare_qkv_from_raw};
pub(super) use qwen35_full::{
    full_attention_decode_core, qwen35_full_attention_decode_step, qwen35_full_attention_inference,
    qwen35_full_attention_prefill,
};
pub(super) use standard::{
    standard_attention_decode_step, standard_attention_inference, standard_attention_prefill,
};

#[cfg(test)]
mod tests {
    use super::*;
    // Test-only imports from submodules (not re-exported to avoid visibility widening).
    use super::projection::{deinterleave_q_gate, project_and_prepare_qkv};
    use super::shared::{RopeParams, apply_neox_rope_in_place};
    // External types used by tests.
    use super::super::state::Qwen35FullAttentionState;
    use ggml_rs::Backend;

    #[test]
    fn qwen35_full_attention_qgate_split_is_head_interleaved() {
        let q_full: Vec<f32> = vec![
            1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0,
        ];
        let (q_values, q_gate) = deinterleave_q_gate(&q_full, 1, 2, 3).unwrap();
        assert_eq!(q_values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(q_gate, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn qwen35_full_attention_qgate_split_multi_token() {
        let q_full: Vec<f32> = vec![
            1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0, 5.0, 6.0, 50.0, 60.0, 7.0, 8.0, 70.0, 80.0,
        ];
        let (q_values, q_gate) = deinterleave_q_gate(&q_full, 2, 2, 2).unwrap();
        assert_eq!(q_values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(q_gate, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn rope_identity_at_position_zero() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let original = values.clone();
        apply_neox_rope_in_place(
            &mut values,
            1,
            1,
            4,
            &RopeParams {
                n_rot: 4,
                freq_base: 10000.0,
                freq_scale: 1.0,
                position_offset: 0,
            },
        )
        .unwrap();
        for (a, b) in values.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
        }
    }

    #[test]
    fn rope_rotates_at_nonzero_position() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0];
        apply_neox_rope_in_place(
            &mut values,
            2,
            1,
            4,
            &RopeParams {
                n_rot: 4,
                freq_base: 1.0,
                freq_scale: 1.0,
                position_offset: 0,
            },
        )
        .unwrap();

        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[1] - 2.0).abs() < 1e-6);
        assert!((values[2] - 3.0).abs() < 1e-6);
        assert!((values[3] - 4.0).abs() < 1e-6);

        let cos1 = 1.0_f32.cos();
        let sin1 = 1.0_f32.sin();
        assert!(
            (values[4] - cos1).abs() < 1e-6,
            "expected {cos1}, got {}",
            values[4]
        );
        assert!((values[5]).abs() < 1e-6);
        assert!(
            (values[6] - sin1).abs() < 1e-6,
            "expected {sin1}, got {}",
            values[6]
        );
        assert!((values[7]).abs() < 1e-6);
    }

    #[test]
    fn rope_preserves_dims_beyond_n_rot() {
        let mut values = [
            1.0_f32, 2.0, 3.0, 4.0, 99.0, 88.0, 1.0, 2.0, 3.0, 4.0, 99.0, 88.0,
        ];
        apply_neox_rope_in_place(
            &mut values,
            2,
            1,
            6,
            &RopeParams {
                n_rot: 4,
                freq_base: 10000.0,
                freq_scale: 1.0,
                position_offset: 0,
            },
        )
        .unwrap();
        assert!((values[4] - 99.0).abs() < 1e-6);
        assert!((values[5] - 88.0).abs() < 1e-6);
        assert!((values[10] - 99.0).abs() < 1e-6);
        assert!((values[11] - 88.0).abs() < 1e-6);
    }

    #[test]
    fn rope_multi_head_applies_same_rotation_per_head() {
        let mut buf = [
            0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        apply_neox_rope_in_place(
            &mut buf,
            2,
            2,
            4,
            &RopeParams {
                n_rot: 4,
                freq_base: 1.0,
                freq_scale: 1.0,
                position_offset: 0,
            },
        )
        .unwrap();
        assert_eq!(&buf[8..12], &buf[12..16]);
    }

    #[test]
    fn rope_position_offset_matches_sequential() {
        // RoPE at offset=2 for 1 token should match position 2 from a 3-token batch.
        let mut batch = vec![0.0_f32; 3 * 4]; // 3 tokens, hd=4
        batch[2 * 4] = 1.0;
        batch[2 * 4 + 1] = 2.0;
        batch[2 * 4 + 2] = 3.0;
        batch[2 * 4 + 3] = 4.0;
        apply_neox_rope_in_place(
            &mut batch,
            3,
            1,
            4,
            &RopeParams {
                n_rot: 4,
                freq_base: 10000.0,
                freq_scale: 1.0,
                position_offset: 0,
            },
        )
        .unwrap();

        let mut single = vec![1.0, 2.0, 3.0, 4.0];
        apply_neox_rope_in_place(
            &mut single,
            1,
            1,
            4,
            &RopeParams {
                n_rot: 4,
                freq_base: 10000.0,
                freq_scale: 1.0,
                position_offset: 2,
            },
        )
        .unwrap();

        for (i, (a, b)) in single.iter().zip(&batch[8..12]).enumerate() {
            assert!((a - b).abs() < 1e-6, "dim {i}: offset={a} vs batch={b}");
        }
    }

    #[test]
    fn full_attention_prefill_then_decode_matches_full_reprocess() {
        // Build a small deterministic plan: 2 heads, 1 kv_head (GQA), hd=4.
        let head_count = 2;
        let kv_head_count = 1;
        let hd = 4;
        let query_features = head_count * hd; // 8
        let kv_features = kv_head_count * hd; // 4
        let hidden = 6;

        // Q weight: hidden → query_features*2 (Q+Gate interleaved)
        let q_weight: Vec<f32> = (0..hidden * query_features * 2)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let k_weight: Vec<f32> = (0..hidden * kv_features)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.08)
            .collect();
        let v_weight: Vec<f32> = (0..hidden * kv_features)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let output_weight: Vec<f32> = (0..query_features * hidden)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
            .collect();
        let q_norm = vec![1.0_f32; hd];
        let k_norm = vec![1.0_f32; hd];

        let plan = super::super::plan::Qwen35FullAttentionLayerPlan {
            norm_values: vec![1.0; hidden],
            q_norm_values: q_norm,
            k_norm_values: k_norm,
            q_weight_values: q_weight,
            k_weight_values: k_weight,
            v_weight_values: v_weight,
            output_weight_values: output_weight,
            head_count,
            kv_head_count,
            head_dimension: hd,
            attention_scale: 1.0 / (hd as f32).sqrt(),
            rope_n_dims: hd,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
        };

        // 3-token prompt + 1 decode token.
        let prompt: Vec<f32> = (0..3 * hidden).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let new_token: Vec<f32> = (0..hidden).map(|i| (i as f32 + 50.0) * 0.05).collect();

        crate::backend::ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        // Full reprocess: 4 tokens at once.
        let full_input: Vec<f32> = prompt.iter().chain(new_token.iter()).copied().collect();
        let norm_weight = &plan.norm_values;
        let full_output =
            qwen35_full_attention_inference(&plan, &full_input, 4, 1e-5, norm_weight, &backend)
                .unwrap();
        let expected = &full_output[3 * hidden..4 * hidden];

        // Prefill 3 tokens, then decode 1.
        let mut state = Qwen35FullAttentionState::new(4, kv_head_count, hd).unwrap();
        let _prefill_out = qwen35_full_attention_prefill(
            &plan,
            &prompt,
            3,
            1e-5,
            norm_weight,
            &mut state,
            &backend,
        )
        .unwrap();

        // Decode path: apply host-side rms_norm + weight to match the in-graph
        // norm that inference/prefill now perform.
        let normalized_token = super::super::tensor_ops::rms_norm_with_weight(
            &new_token,
            hidden,
            1,
            norm_weight,
            1e-5,
        )
        .unwrap();
        let decode_out =
            qwen35_full_attention_decode_step(&plan, &normalized_token, 1e-5, &mut state, &backend)
                .unwrap();

        for (i, (a, b)) in decode_out.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "feature {i}: decode={a} vs full={b}, diff={}",
                (a - b).abs()
            );
        }
    }

    /// Verifies that `decode_scoring_gpu` (flash_attn_ext path) produces the
    /// same gated head outputs as the host scoring loop inside
    /// `full_attention_decode_core`.
    ///
    /// Uses GQA (H > Hkv) with multiple cached tokens (Tkv > 1) to exercise
    /// the KV cache permutation and grouped-query layout.
    #[test]
    fn gpu_scoring_matches_host_scoring() {
        let head_count = 4;
        let kv_head_count = 2; // GQA: 2 groups
        let hd = 4;
        let query_features = head_count * hd; // 16
        let kv_features = kv_head_count * hd; // 8
        let hidden = 6;

        // Deterministic weights.
        let q_weight: Vec<f32> = (0..hidden * query_features * 2)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let k_weight: Vec<f32> = (0..hidden * kv_features)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.08)
            .collect();
        let v_weight: Vec<f32> = (0..hidden * kv_features)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let output_weight: Vec<f32> = (0..query_features * hidden)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
            .collect();
        let q_norm = vec![1.0_f32; hd];
        let k_norm = vec![1.0_f32; hd];

        let plan = super::super::plan::Qwen35FullAttentionLayerPlan {
            norm_values: vec![1.0; hidden],
            q_norm_values: q_norm,
            k_norm_values: k_norm,
            q_weight_values: q_weight,
            k_weight_values: k_weight,
            v_weight_values: v_weight,
            output_weight_values: output_weight,
            head_count,
            kv_head_count,
            head_dimension: hd,
            attention_scale: 1.0 / (hd as f32).sqrt(),
            rope_n_dims: hd,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
        };

        // 5-token prompt + 1 decode token — ensures Tkv > 1 for cache.
        let prompt: Vec<f32> = (0..5 * hidden).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let decode_token: Vec<f32> = (0..hidden).map(|i| (i as f32 + 50.0) * 0.05).collect();

        crate::backend::ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        // Prefill to populate KV cache with 5 tokens.
        let norm_weight = &plan.norm_values;
        let mut state_host = Qwen35FullAttentionState::new(6, kv_head_count, hd).unwrap();
        let _prefill = qwen35_full_attention_prefill(
            &plan,
            &prompt,
            5,
            1e-5,
            norm_weight,
            &mut state_host,
            &backend,
        )
        .unwrap();

        // Clone state so both paths start from the same KV cache snapshot.
        let mut state_gpu = state_host.clone();

        // Project decode token (shared for both paths).
        let normalized = super::super::tensor_ops::rms_norm_with_weight(
            &decode_token,
            hidden,
            1,
            norm_weight,
            1e-5,
        )
        .unwrap();

        // Host scoring: backend = None.
        let prepared_host =
            project_and_prepare_qkv(&plan, &normalized, 1, 1e-5, Some(&backend)).unwrap();
        let host_out =
            full_attention_decode_core(prepared_host, &plan, &mut state_host, None, None, None)
                .unwrap();

        // GPU scoring: backend = Some.
        let prepared_gpu =
            project_and_prepare_qkv(&plan, &normalized, 1, 1e-5, Some(&backend)).unwrap();
        let gpu_out = full_attention_decode_core(
            prepared_gpu,
            &plan,
            &mut state_gpu,
            Some(&backend),
            None,
            None,
        )
        .unwrap();

        assert_eq!(host_out.len(), gpu_out.len());
        for (i, (h, g)) in host_out.iter().zip(gpu_out.iter()).enumerate() {
            assert!(
                (h - g).abs() < 1e-4,
                "head_output[{i}]: host={h} vs gpu={g}, diff={}",
                (h - g).abs()
            );
        }
    }

    /// Helper: build a small `StandardAttentionLayerPlan` with RoPE + GQA.
    fn sample_standard_plan(
        hidden: usize,
        h: usize,
        hkv: usize,
        hd: usize,
    ) -> (
        super::super::plan::StandardAttentionLayerPlan,
        crate::inference::AttentionInferenceConfig,
    ) {
        use crate::inference::{
            AttentionHeadDimension, AttentionInferenceConfig, AttentionLayout, AttentionMaskPolicy,
            RopeConfig, RotaryEmbedding,
        };

        let layout = AttentionLayout::from_projection_dimensions(hidden, h, hkv, hd).unwrap();
        let config = AttentionInferenceConfig::from_layout(layout, 1)
            .unwrap()
            .with_rotary(RotaryEmbedding::Llama(RopeConfig {
                dimensions: AttentionHeadDimension::new(hd).unwrap(),
                base: 10000.0,
                scale: 1.0,
                original_context: None,
            }))
            .with_mask(AttentionMaskPolicy::Causal { past_tokens: 0 })
            .with_attention_scale(1.0 / (hd as f32).sqrt());

        let weights = crate::inference::AttentionWeights::deterministic(config);
        let norm_values = vec![1.0_f32; hidden];

        let plan = super::super::plan::StandardAttentionLayerPlan {
            weights,
            norm_values,
        };
        (plan, config)
    }

    #[test]
    fn standard_attention_prefill_captures_state() {
        let hidden = 8;
        let h = 4;
        let hkv = 2; // GQA: 4 query heads, 2 KV heads
        let hd = 4;
        let kvf = hkv * hd;
        let prompt_len = 3;
        let max_tokens = 16;

        let (plan, _config) = sample_standard_plan(hidden, h, hkv, hd);

        crate::backend::ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        let input: Vec<f32> = (0..prompt_len * hidden)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        let mut state =
            super::super::state::StandardAttentionState::new(max_tokens, hkv, hd).unwrap();

        let _output = standard_attention_prefill(
            &plan,
            &input,
            prompt_len,
            1e-5,
            &plan.norm_values,
            &mut state,
            &backend,
        )
        .unwrap();

        // token_count matches prompt length
        assert_eq!(state.token_count(), prompt_len);
        // kv_features matches config
        assert_eq!(state.kv_features, kvf);

        // Populated prefix is non-zero (at least one value differs from 0.0).
        let prefix_len = prompt_len * kvf;
        let k_prefix = &state.k_cache[..prefix_len];
        let v_prefix = &state.v_cache[..prefix_len];
        assert!(
            k_prefix.iter().any(|&v| v != 0.0),
            "K cache prefix should contain non-zero values after prefill"
        );
        assert!(
            v_prefix.iter().any(|&v| v != 0.0),
            "V cache prefix should contain non-zero values after prefill"
        );

        // Unused tail stays zero.
        let k_tail = &state.k_cache[prefix_len..];
        let v_tail = &state.v_cache[prefix_len..];
        assert!(
            k_tail.iter().all(|&v| v == 0.0),
            "K cache tail should remain zero"
        );
        assert!(
            v_tail.iter().all(|&v| v == 0.0),
            "V cache tail should remain zero"
        );
    }

    #[test]
    fn standard_attention_decode_matches_reprocess() {
        // Uses RoPE + GQA (4 query heads, 2 KV heads) — exercises position_offset
        // and KV-head grouping in the decode path.
        let hidden = 8;
        let h = 4;
        let hkv = 2;
        let hd = 4;
        let prompt_len = 3;
        let max_tokens = 16;

        let (plan, _config) = sample_standard_plan(hidden, h, hkv, hd);

        crate::backend::ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        let prompt: Vec<f32> = (0..prompt_len * hidden)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let new_token: Vec<f32> = (0..hidden).map(|i| (i as f32 + 50.0) * 0.05).collect();

        // Path A: Full reprocess of [prompt + new_token] — extract last-token slice.
        let full_input: Vec<f32> = prompt.iter().chain(new_token.iter()).copied().collect();
        let full_output = standard_attention_inference(
            &plan,
            &full_input,
            prompt_len + 1,
            1e-5,
            &plan.norm_values,
            &backend,
        )
        .unwrap();
        let expected = &full_output[prompt_len * hidden..(prompt_len + 1) * hidden];

        // Path B: Prefill prompt, then decode new_token.
        let mut state =
            super::super::state::StandardAttentionState::new(max_tokens, hkv, hd).unwrap();

        let _prefill_out = standard_attention_prefill(
            &plan,
            &prompt,
            prompt_len,
            1e-5,
            &plan.norm_values,
            &mut state,
            &backend,
        )
        .unwrap();

        // Decode path receives pre-normed input (matching DecodeStrategy behavior).
        let normalized_token = super::super::tensor_ops::rms_norm_with_weight(
            &new_token,
            hidden,
            1,
            &plan.norm_values,
            1e-5,
        )
        .unwrap();
        let decode_out =
            standard_attention_decode_step(&plan, &normalized_token, 1e-5, &mut state, &backend)
                .unwrap();

        assert_eq!(decode_out.len(), expected.len());
        for (i, (a, b)) in decode_out.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "feature {i}: decode={a} vs full={b}, diff={}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn standard_attention_decode_errors_when_cache_full() {
        let hidden = 8;
        let h = 2;
        let hkv = 2;
        let hd = 4;
        let max_tokens = 3; // small capacity

        let (plan, _config) = sample_standard_plan(hidden, h, hkv, hd);

        crate::backend::ensure_backends_loaded();
        let backend =
            Backend::new(ggml_rs::BackendKind::Cpu).expect("CPU backend should be available");

        // Prefill exactly to capacity.
        let input: Vec<f32> = (0..max_tokens * hidden)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let mut state =
            super::super::state::StandardAttentionState::new(max_tokens, hkv, hd).unwrap();

        let _out = standard_attention_prefill(
            &plan,
            &input,
            max_tokens,
            1e-5,
            &plan.norm_values,
            &mut state,
            &backend,
        )
        .unwrap();

        assert_eq!(state.token_count(), max_tokens);

        // Next decode should fail — cache is full.
        let token: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.01).collect();
        let normalized = super::super::tensor_ops::rms_norm_with_weight(
            &token,
            hidden,
            1,
            &plan.norm_values,
            1e-5,
        )
        .unwrap();
        let result = standard_attention_decode_step(&plan, &normalized, 1e-5, &mut state, &backend);
        assert!(
            result.is_err(),
            "decode should fail when KV cache is at capacity"
        );
    }
}
