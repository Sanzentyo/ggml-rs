use super::error::E2eError;
use crate::backend::LlamaBackend;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct E2eGenerationConfig {
    pub backend: LlamaBackend,
    pub prompt_token_ids: Vec<i32>,
    pub max_new_tokens: usize,
    pub pad_token_id: i32,
    pub eos_token_id: Option<i32>,
    pub mixed_layer_policy: MixedLayerPolicy,
}

impl E2eGenerationConfig {
    pub fn new(
        backend: LlamaBackend,
        prompt_token_ids: Vec<i32>,
        max_new_tokens: usize,
    ) -> Result<Self, E2eError> {
        if prompt_token_ids.is_empty() {
            return Err(E2eError::EmptyPrompt);
        }
        Ok(Self {
            backend,
            prompt_token_ids,
            max_new_tokens,
            pad_token_id: 0,
            eos_token_id: None,
            mixed_layer_policy: MixedLayerPolicy::Strict,
        })
    }

    pub const fn with_pad_token_id(mut self, pad_token_id: i32) -> Self {
        self.pad_token_id = pad_token_id;
        self
    }

    pub const fn with_eos_token_id(mut self, eos_token_id: Option<i32>) -> Self {
        self.eos_token_id = eos_token_id;
        self
    }

    pub const fn with_mixed_layer_policy(mut self, mixed_layer_policy: MixedLayerPolicy) -> Self {
        self.mixed_layer_policy = mixed_layer_policy;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedLayerPolicy {
    Strict,
    SkipUnsupportedAttention,
}

#[derive(Debug, Clone)]
pub struct E2eGenerationReport {
    pub backend_name: String,
    pub prompt_token_count: usize,
    pub generated_token_ids: Vec<i32>,
    pub all_token_ids: Vec<i32>,
    pub attention_layer_count: usize,
    pub mlp_only_layer_count: usize,
    pub elapsed: Duration,
}

impl E2eGenerationReport {
    pub fn avg_generated_token_ms(&self) -> f64 {
        if self.generated_token_ids.is_empty() {
            0.0
        } else {
            self.elapsed.as_secs_f64() * 1000.0 / self.generated_token_ids.len() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::LlamaBackend;

    // ── E2eGenerationConfig ─────────────────────────────────────
    #[test]
    fn config_new_valid() {
        let cfg = E2eGenerationConfig::new(LlamaBackend::Cpu, vec![1, 2, 3], 10).unwrap();
        assert_eq!(cfg.prompt_token_ids, &[1, 2, 3]);
        assert_eq!(cfg.max_new_tokens, 10);
        assert_eq!(cfg.pad_token_id, 0);
        assert_eq!(cfg.eos_token_id, None);
        assert_eq!(cfg.mixed_layer_policy, MixedLayerPolicy::Strict);
    }

    #[test]
    fn config_empty_prompt_rejected() {
        let err = E2eGenerationConfig::new(LlamaBackend::Cpu, vec![], 10).unwrap_err();
        assert!(matches!(err, E2eError::EmptyPrompt));
    }

    #[test]
    fn config_builder_chaining() {
        let cfg = E2eGenerationConfig::new(LlamaBackend::Cpu, vec![1], 5)
            .unwrap()
            .with_pad_token_id(42)
            .with_eos_token_id(Some(99))
            .with_mixed_layer_policy(MixedLayerPolicy::SkipUnsupportedAttention);
        assert_eq!(cfg.pad_token_id, 42);
        assert_eq!(cfg.eos_token_id, Some(99));
        assert_eq!(
            cfg.mixed_layer_policy,
            MixedLayerPolicy::SkipUnsupportedAttention
        );
    }

    // ── E2eGenerationReport ─────────────────────────────────────
    #[test]
    fn avg_token_ms_empty_is_zero() {
        let report = E2eGenerationReport {
            backend_name: "test".into(),
            prompt_token_count: 0,
            generated_token_ids: vec![],
            all_token_ids: vec![],
            attention_layer_count: 0,
            mlp_only_layer_count: 0,
            elapsed: Duration::from_millis(100),
        };
        assert_eq!(report.avg_generated_token_ms(), 0.0);
    }

    #[test]
    fn avg_token_ms_normal() {
        let report = E2eGenerationReport {
            backend_name: "test".into(),
            prompt_token_count: 5,
            generated_token_ids: vec![1, 2, 3, 4],
            all_token_ids: vec![0; 9],
            attention_layer_count: 1,
            mlp_only_layer_count: 0,
            elapsed: Duration::from_millis(400),
        };
        // 400ms / 4 tokens = 100ms per token
        assert!((report.avg_generated_token_ms() - 100.0).abs() < 0.01);
    }

    #[test]
    fn avg_token_ms_zero_duration() {
        let report = E2eGenerationReport {
            backend_name: "test".into(),
            prompt_token_count: 0,
            generated_token_ids: vec![1],
            all_token_ids: vec![1],
            attention_layer_count: 0,
            mlp_only_layer_count: 0,
            elapsed: Duration::ZERO,
        };
        assert_eq!(report.avg_generated_token_ms(), 0.0);
    }
}
