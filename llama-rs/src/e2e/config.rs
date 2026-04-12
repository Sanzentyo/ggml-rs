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
