use super::stepwise_decode::{
    bench_stepwise_sweep_with_block_mlp, bench_stepwise_with_block_mlp,
    decode_stepwise_with_block_mlp,
};
use super::{
    AttentionDecodeCache, AttentionDecodeStepwiseBenchReport,
    AttentionDecodeStepwiseBenchSweepReport, AttentionDecodeStepwiseConfig,
    AttentionDecodeStepwiseReport, AttentionWeights, InferenceError, LlamaBackend, MlpWeights,
};
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct BackendUnset;
#[derive(Debug, Clone, Copy)]
pub struct BackendSet;
#[derive(Debug, Clone, Copy)]
pub struct ConfigUnset;
#[derive(Debug, Clone, Copy)]
pub struct ConfigSet;

/// Type-state builder for [`DecodeStepPlan`].
#[derive(Debug, Clone, Copy)]
pub struct DecodeStepPlanBuilder<BackendState, ConfigState> {
    backend_kind: Option<LlamaBackend>,
    stepwise: Option<AttentionDecodeStepwiseConfig>,
    warmup_repeats_per_step: usize,
    _state: PhantomData<(BackendState, ConfigState)>,
}

/// ADT that owns decode-step execution policy.
#[derive(Debug, Clone, Copy)]
pub struct DecodeStepPlan {
    backend_kind: LlamaBackend,
    stepwise: AttentionDecodeStepwiseConfig,
    warmup_repeats_per_step: usize,
}

impl DecodeStepPlan {
    pub fn builder() -> DecodeStepPlanBuilder<BackendUnset, ConfigUnset> {
        DecodeStepPlanBuilder {
            backend_kind: None,
            stepwise: None,
            warmup_repeats_per_step: 0,
            _state: PhantomData,
        }
    }

    pub const fn backend_kind(self) -> LlamaBackend {
        self.backend_kind
    }

    pub const fn stepwise(self) -> AttentionDecodeStepwiseConfig {
        self.stepwise
    }

    pub const fn warmup_repeats_per_step(self) -> usize {
        self.warmup_repeats_per_step
    }

    pub fn execute_single(
        &self,
        weights: &AttentionWeights,
        query_input: &[f32],
        cache: &AttentionDecodeCache,
        block_mlp_weights: Option<&MlpWeights>,
    ) -> Result<AttentionDecodeStepwiseReport, InferenceError> {
        decode_stepwise_with_block_mlp(
            weights,
            query_input,
            cache,
            self.backend_kind,
            self.stepwise,
            block_mlp_weights,
        )
    }

    pub fn bench_single(
        &self,
        weights: &AttentionWeights,
        query_input: &[f32],
        cache: &AttentionDecodeCache,
        block_mlp_weights: Option<&MlpWeights>,
    ) -> Result<AttentionDecodeStepwiseBenchReport, InferenceError> {
        bench_stepwise_with_block_mlp(
            weights,
            query_input,
            cache,
            self.backend_kind,
            self.stepwise,
            self.warmup_repeats_per_step,
            block_mlp_weights,
        )
    }

    pub fn bench_sweep(
        &self,
        weights: &AttentionWeights,
        query_input: &[f32],
        cache: &AttentionDecodeCache,
        block_mlp_weights: &[&MlpWeights],
    ) -> Result<AttentionDecodeStepwiseBenchSweepReport, InferenceError> {
        bench_stepwise_sweep_with_block_mlp(
            weights,
            query_input,
            cache,
            self.backend_kind,
            self.stepwise,
            self.warmup_repeats_per_step,
            block_mlp_weights,
        )
    }

    pub fn bench<'a, B>(
        &self,
        weights: &'a AttentionWeights,
        query_input: &'a [f32],
        cache: &'a AttentionDecodeCache,
        block_mlp_set: B,
    ) -> Result<B::Report, InferenceError>
    where
        B: DecodeStepBenchSet<'a>,
    {
        block_mlp_set.bench(self, weights, query_input, cache)
    }
}

/// Trait-based static dispatch for block-MLP benchmark set shape.
pub trait DecodeStepBenchSet<'a> {
    type Report;

    fn bench(
        self,
        plan: &DecodeStepPlan,
        weights: &'a AttentionWeights,
        query_input: &'a [f32],
        cache: &'a AttentionDecodeCache,
    ) -> Result<Self::Report, InferenceError>;
}

impl<'a> DecodeStepBenchSet<'a> for Option<&'a MlpWeights> {
    type Report = AttentionDecodeStepwiseBenchReport;

    fn bench(
        self,
        plan: &DecodeStepPlan,
        weights: &'a AttentionWeights,
        query_input: &'a [f32],
        cache: &'a AttentionDecodeCache,
    ) -> Result<Self::Report, InferenceError> {
        plan.bench_single(weights, query_input, cache, self)
    }
}

impl<'a> DecodeStepBenchSet<'a> for &'a [&'a MlpWeights] {
    type Report = AttentionDecodeStepwiseBenchSweepReport;

    fn bench(
        self,
        plan: &DecodeStepPlan,
        weights: &'a AttentionWeights,
        query_input: &'a [f32],
        cache: &'a AttentionDecodeCache,
    ) -> Result<Self::Report, InferenceError> {
        plan.bench_sweep(weights, query_input, cache, self)
    }
}

impl<ConfigState> DecodeStepPlanBuilder<BackendUnset, ConfigState> {
    pub fn backend(
        self,
        backend_kind: LlamaBackend,
    ) -> DecodeStepPlanBuilder<BackendSet, ConfigState> {
        DecodeStepPlanBuilder {
            backend_kind: Some(backend_kind),
            stepwise: self.stepwise,
            warmup_repeats_per_step: self.warmup_repeats_per_step,
            _state: PhantomData,
        }
    }
}

impl<BackendState> DecodeStepPlanBuilder<BackendState, ConfigUnset> {
    pub fn stepwise(
        self,
        stepwise: AttentionDecodeStepwiseConfig,
    ) -> DecodeStepPlanBuilder<BackendState, ConfigSet> {
        DecodeStepPlanBuilder {
            backend_kind: self.backend_kind,
            stepwise: Some(stepwise),
            warmup_repeats_per_step: self.warmup_repeats_per_step,
            _state: PhantomData,
        }
    }
}

impl<BackendState, ConfigState> DecodeStepPlanBuilder<BackendState, ConfigState> {
    pub const fn warmup_repeats_per_step(mut self, warmup_repeats_per_step: usize) -> Self {
        self.warmup_repeats_per_step = warmup_repeats_per_step;
        self
    }
}

impl DecodeStepPlanBuilder<BackendSet, ConfigSet> {
    pub fn build(self) -> DecodeStepPlan {
        DecodeStepPlan {
            backend_kind: self
                .backend_kind
                .expect("builder invariant: backend_kind is present"),
            stepwise: self
                .stepwise
                .expect("builder invariant: stepwise is present"),
            warmup_repeats_per_step: self.warmup_repeats_per_step,
        }
    }
}
