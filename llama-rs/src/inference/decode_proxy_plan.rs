use super::{
    AttentionDecodeCache, AttentionDecodeProxyReport, AttentionMaskPolicy, AttentionWeights,
    InferenceError, LlamaBackend, attention_decode_proxy_with_cache_repeats_inner,
    build_attention_decode_cache,
};
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct BackendUnset;
#[derive(Debug, Clone, Copy)]
pub struct BackendSet;
#[derive(Debug, Clone, Copy)]
pub struct RepeatsUnset;
#[derive(Debug, Clone, Copy)]
pub struct RepeatsSet;

/// Type-state builder for [`AttentionDecodePlan`].
#[derive(Debug, Clone, Copy)]
pub struct AttentionDecodePlanBuilder<BackendState, RepeatsState> {
    backend_kind: Option<LlamaBackend>,
    repeats: Option<usize>,
    past_tokens: Option<usize>,
    _state: PhantomData<(BackendState, RepeatsState)>,
}

/// ADT for decode-proxy execution policy.
#[derive(Debug, Clone, Copy)]
pub struct AttentionDecodePlan {
    backend_kind: LlamaBackend,
    repeats: usize,
    past_tokens: Option<usize>,
}

impl AttentionDecodePlan {
    pub fn builder() -> AttentionDecodePlanBuilder<BackendUnset, RepeatsUnset> {
        AttentionDecodePlanBuilder {
            backend_kind: None,
            repeats: None,
            past_tokens: None,
            _state: PhantomData,
        }
    }

    pub const fn backend_kind(self) -> LlamaBackend {
        self.backend_kind
    }

    pub const fn repeats(self) -> usize {
        self.repeats
    }

    pub const fn past_tokens(self) -> Option<usize> {
        self.past_tokens
    }

    fn resolved_past_tokens(self, weights: &AttentionWeights) -> usize {
        self.past_tokens.unwrap_or(match weights.config.mask {
            AttentionMaskPolicy::None => 0,
            AttentionMaskPolicy::Causal { past_tokens } => past_tokens,
        })
    }

    pub fn execute<'a, S>(self, source: S) -> Result<AttentionDecodeProxyReport, InferenceError>
    where
        S: AttentionDecodeSource<'a>,
    {
        source.execute(self)
    }
}

/// Trait-based static dispatch for decode-proxy input source shape.
pub trait AttentionDecodeSource<'a> {
    fn execute(
        self,
        plan: AttentionDecodePlan,
    ) -> Result<AttentionDecodeProxyReport, InferenceError>;
}

#[derive(Debug, Clone, Copy)]
pub struct AttentionDecodeCacheInput<'a> {
    weights: &'a AttentionWeights,
    query_input: &'a [f32],
    cache: &'a AttentionDecodeCache,
}

impl<'a> AttentionDecodeCacheInput<'a> {
    pub const fn new(
        weights: &'a AttentionWeights,
        query_input: &'a [f32],
        cache: &'a AttentionDecodeCache,
    ) -> Self {
        Self {
            weights,
            query_input,
            cache,
        }
    }
}

impl<'a> AttentionDecodeSource<'a> for AttentionDecodeCacheInput<'a> {
    fn execute(
        self,
        plan: AttentionDecodePlan,
    ) -> Result<AttentionDecodeProxyReport, InferenceError> {
        let past_tokens = plan.resolved_past_tokens(self.weights);
        attention_decode_proxy_with_cache_repeats_inner(
            self.weights,
            self.query_input,
            self.cache,
            plan.backend_kind,
            plan.repeats,
            past_tokens,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AttentionDecodeWeightsInput<'a> {
    weights: &'a AttentionWeights,
    query_input: &'a [f32],
    key_value_input: &'a [f32],
    key_value_length: usize,
}

impl<'a> AttentionDecodeWeightsInput<'a> {
    pub const fn new(
        weights: &'a AttentionWeights,
        query_input: &'a [f32],
        key_value_input: &'a [f32],
        key_value_length: usize,
    ) -> Self {
        Self {
            weights,
            query_input,
            key_value_input,
            key_value_length,
        }
    }
}

impl<'a> AttentionDecodeSource<'a> for AttentionDecodeWeightsInput<'a> {
    fn execute(
        self,
        plan: AttentionDecodePlan,
    ) -> Result<AttentionDecodeProxyReport, InferenceError> {
        let cache = build_attention_decode_cache(
            self.weights,
            self.key_value_input,
            self.key_value_length,
        )?;
        let past_tokens = plan.resolved_past_tokens(self.weights);
        attention_decode_proxy_with_cache_repeats_inner(
            self.weights,
            self.query_input,
            &cache,
            plan.backend_kind,
            plan.repeats,
            past_tokens,
        )
    }
}

impl<RepeatsState> AttentionDecodePlanBuilder<BackendUnset, RepeatsState> {
    pub fn backend(
        self,
        backend_kind: LlamaBackend,
    ) -> AttentionDecodePlanBuilder<BackendSet, RepeatsState> {
        AttentionDecodePlanBuilder {
            backend_kind: Some(backend_kind),
            repeats: self.repeats,
            past_tokens: self.past_tokens,
            _state: PhantomData,
        }
    }
}

impl<BackendState> AttentionDecodePlanBuilder<BackendState, RepeatsUnset> {
    pub fn repeats(self, repeats: usize) -> AttentionDecodePlanBuilder<BackendState, RepeatsSet> {
        AttentionDecodePlanBuilder {
            backend_kind: self.backend_kind,
            repeats: Some(repeats),
            past_tokens: self.past_tokens,
            _state: PhantomData,
        }
    }
}

impl<BackendState, RepeatsState> AttentionDecodePlanBuilder<BackendState, RepeatsState> {
    pub const fn past_tokens(mut self, past_tokens: usize) -> Self {
        self.past_tokens = Some(past_tokens);
        self
    }
}

impl AttentionDecodePlanBuilder<BackendSet, RepeatsSet> {
    pub fn build(self) -> Result<AttentionDecodePlan, InferenceError> {
        let repeats = self.repeats.expect("builder invariant: repeats is present");
        if repeats == 0 {
            return Err(InferenceError::InvalidRepeats);
        }
        Ok(AttentionDecodePlan {
            backend_kind: self
                .backend_kind
                .expect("builder invariant: backend_kind is present"),
            repeats,
            past_tokens: self.past_tokens,
        })
    }
}
