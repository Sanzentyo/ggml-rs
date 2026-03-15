//! Stepwise decode execution path and benchmark reports.

use super::attention_ops::{
    BalancedHeadConcat, HeadConcatMetadata, HeadConcatStrategy, LeftFoldHeadConcat,
    LlamaRotaryApplier, RotaryApplier,
};
use super::{
    AttentionDecodeCache, AttentionMaskPolicy, AttentionWeights, InferenceError, MlpWeights,
    RotaryEmbedding, fill_causal_mask_values,
    recommended_attention_backend_memory_bytes_for_lengths,
};
use crate::backend::{LlamaBackend, ensure_backends_loaded};
use ggml_rs::{Backend, Context, Length, Shape2D};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
/// Output payload and execution metadata for stepwise decode-like proxy inference.
pub struct AttentionDecodeStepwiseReport {
    pub backend_name: String,
    pub hidden_features: usize,
    pub query_length: usize,
    pub key_value_start: usize,
    pub steps: usize,
    pub repeats_per_step: usize,
    pub output: Vec<f32>,
}

#[derive(Debug, Clone)]
/// Stepwise decode benchmark report with explicit warmup/bench phase timing.
pub struct AttentionDecodeStepwiseBenchReport {
    pub warmup_repeats_per_step: usize,
    pub setup_duration: Duration,
    pub bench_duration: Duration,
    pub execution: AttentionDecodeStepwiseReport,
}

impl AttentionDecodeStepwiseBenchReport {
    pub fn avg_token_ms(&self) -> f64 {
        let token_iters = self
            .execution
            .steps
            .checked_mul(self.execution.repeats_per_step)
            .unwrap_or(1)
            .max(1);
        self.bench_duration.as_secs_f64() * 1000.0 / token_iters as f64
    }

    pub fn setup_ms(&self) -> f64 {
        self.setup_duration.as_secs_f64() * 1000.0
    }
}

#[derive(Debug, Clone)]
/// Batched stepwise decode benchmark reports sharing one setup path.
pub struct AttentionDecodeStepwiseBenchSweepReport {
    pub warmup_repeats_per_step: usize,
    pub setup_duration: Duration,
    pub entries: Vec<AttentionDecodeStepwiseBenchReport>,
}

impl AttentionDecodeStepwiseBenchSweepReport {
    pub fn setup_ms(&self) -> f64 {
        self.setup_duration.as_secs_f64() * 1000.0
    }

    pub fn amortized_setup_ms(&self) -> f64 {
        let entry_count = self.entries.len().max(1);
        self.setup_ms() / entry_count as f64
    }
}

#[derive(Debug, Clone, Copy)]
/// Configuration for stepwise decode-like proxy execution.
pub struct AttentionDecodeStepwiseConfig {
    pub key_value_start: usize,
    pub steps: usize,
    pub past_start: usize,
    pub repeats_per_step: usize,
    pub layer_repeat: usize,
    pub include_kv_projection: bool,
    pub include_kv_cache_write: bool,
    pub include_kv_cache_write_to_cache: bool,
    pub include_block_scope: bool,
    pub synchronize_per_step: bool,
    pub readback_per_step: bool,
    pub use_position_deltas: bool,
    pub use_mask_deltas: bool,
    pub elide_mask_host_buffer: bool,
    pub fuse_output_projection: bool,
    pub precompute_static_kv_head_views: bool,
    pub use_balanced_head_concat: bool,
    pub use_head_output_staging_buffer: bool,
    pub use_fused_block_gate_up_projection: bool,
}

impl AttentionDecodeStepwiseConfig {
    /// Creates a stepwise decode configuration with conservative defaults.
    ///
    /// Optional cost scopes (`kv_projection`, `block_scope`) and backend timing
    /// knobs (`synchronize_per_step`, `readback_per_step`) are disabled by default.
    pub const fn new(
        key_value_start: usize,
        steps: usize,
        past_start: usize,
        repeats_per_step: usize,
    ) -> Self {
        Self {
            key_value_start,
            steps,
            past_start,
            repeats_per_step,
            layer_repeat: 1,
            include_kv_projection: false,
            include_kv_cache_write: false,
            include_kv_cache_write_to_cache: false,
            include_block_scope: false,
            synchronize_per_step: false,
            readback_per_step: false,
            use_position_deltas: true,
            use_mask_deltas: true,
            elide_mask_host_buffer: false,
            fuse_output_projection: false,
            precompute_static_kv_head_views: true,
            use_balanced_head_concat: false,
            use_head_output_staging_buffer: false,
            use_fused_block_gate_up_projection: false,
        }
    }

    pub const fn with_kv_projection(mut self, include_kv_projection: bool) -> Self {
        self.include_kv_projection = include_kv_projection;
        self
    }

    pub const fn with_layer_repeat(mut self, layer_repeat: usize) -> Self {
        self.layer_repeat = layer_repeat;
        self
    }

    pub const fn with_kv_cache_write(mut self, include_kv_cache_write: bool) -> Self {
        self.include_kv_cache_write = include_kv_cache_write;
        self
    }

    pub const fn with_kv_cache_write_to_cache(
        mut self,
        include_kv_cache_write_to_cache: bool,
    ) -> Self {
        self.include_kv_cache_write_to_cache = include_kv_cache_write_to_cache;
        self
    }

    pub const fn with_block_scope(mut self, include_block_scope: bool) -> Self {
        self.include_block_scope = include_block_scope;
        self
    }

    pub const fn with_sync_per_step(mut self, synchronize_per_step: bool) -> Self {
        self.synchronize_per_step = synchronize_per_step;
        self
    }

    pub const fn with_readback_per_step(mut self, readback_per_step: bool) -> Self {
        self.readback_per_step = readback_per_step;
        self
    }

    pub const fn with_position_deltas(mut self, use_position_deltas: bool) -> Self {
        self.use_position_deltas = use_position_deltas;
        self
    }

    pub const fn with_mask_deltas(mut self, use_mask_deltas: bool) -> Self {
        self.use_mask_deltas = use_mask_deltas;
        self
    }

    pub const fn with_mask_host_buffer_elision(mut self, elide_mask_host_buffer: bool) -> Self {
        self.elide_mask_host_buffer = elide_mask_host_buffer;
        self
    }

    pub const fn with_fused_output_projection(mut self, fuse_output_projection: bool) -> Self {
        self.fuse_output_projection = fuse_output_projection;
        self
    }

    pub const fn with_static_kv_head_view_precompute(
        mut self,
        precompute_static_kv_head_views: bool,
    ) -> Self {
        self.precompute_static_kv_head_views = precompute_static_kv_head_views;
        self
    }

    pub const fn with_balanced_head_concat(mut self, use_balanced_head_concat: bool) -> Self {
        self.use_balanced_head_concat = use_balanced_head_concat;
        self
    }

    pub const fn with_head_output_staging_buffer(
        mut self,
        use_head_output_staging_buffer: bool,
    ) -> Self {
        self.use_head_output_staging_buffer = use_head_output_staging_buffer;
        self
    }

    pub const fn with_fused_block_gate_up_projection(
        mut self,
        use_fused_block_gate_up_projection: bool,
    ) -> Self {
        self.use_fused_block_gate_up_projection = use_fused_block_gate_up_projection;
        self
    }
}

#[derive(Default)]
struct KvCacheWriteNodes<'ctx> {
    shared_k: Option<ggml_rs::Tensor<'ctx>>,
    shared_v: Option<ggml_rs::Tensor<'ctx>>,
    step_k: Option<Vec<ggml_rs::Tensor<'ctx>>>,
    step_v: Option<Vec<ggml_rs::Tensor<'ctx>>>,
}

trait KvCacheWriteStrategy {
    fn graph_count(&self, steps: usize) -> usize;
    fn graph_index_for_step(&self, step: usize) -> usize;
    fn bench_needs_kv_reset(&self, warmup_repeats_per_step: usize) -> bool;
    fn expand_graph_nodes<'ctx>(
        &self,
        graph: &mut ggml_rs::Graph<'ctx>,
        graph_step: usize,
        write_nodes: &KvCacheWriteNodes<'ctx>,
    );
}

#[derive(Debug, Clone, Copy, Default)]
struct SharedKvCacheWriteStrategy;

impl KvCacheWriteStrategy for SharedKvCacheWriteStrategy {
    fn graph_count(&self, _steps: usize) -> usize {
        1
    }

    fn graph_index_for_step(&self, _step: usize) -> usize {
        0
    }

    fn bench_needs_kv_reset(&self, warmup_repeats_per_step: usize) -> bool {
        warmup_repeats_per_step == 0
    }

    fn expand_graph_nodes<'ctx>(
        &self,
        graph: &mut ggml_rs::Graph<'ctx>,
        _graph_step: usize,
        write_nodes: &KvCacheWriteNodes<'ctx>,
    ) {
        if let Some(kv_cache_write_k) = write_nodes.shared_k.as_ref() {
            graph.build_forward_expand(kv_cache_write_k);
        }
        if let Some(kv_cache_write_v) = write_nodes.shared_v.as_ref() {
            graph.build_forward_expand(kv_cache_write_v);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct StepSpecificKvCacheWriteStrategy;

impl KvCacheWriteStrategy for StepSpecificKvCacheWriteStrategy {
    fn graph_count(&self, steps: usize) -> usize {
        steps
    }

    fn graph_index_for_step(&self, step: usize) -> usize {
        step
    }

    fn bench_needs_kv_reset(&self, _warmup_repeats_per_step: usize) -> bool {
        true
    }

    fn expand_graph_nodes<'ctx>(
        &self,
        graph: &mut ggml_rs::Graph<'ctx>,
        graph_step: usize,
        write_nodes: &KvCacheWriteNodes<'ctx>,
    ) {
        if let Some(step_nodes) = write_nodes.step_k.as_ref()
            && let Some(kv_cache_write_k) = step_nodes.get(graph_step)
        {
            graph.build_forward_expand(kv_cache_write_k);
        }
        if let Some(step_nodes) = write_nodes.step_v.as_ref()
            && let Some(kv_cache_write_v) = step_nodes.get(graph_step)
        {
            graph.build_forward_expand(kv_cache_write_v);
        }
    }
}

struct StepwiseGraphBuildInput<'ctx> {
    steps: usize,
    y: &'ctx ggml_rs::Tensor<'ctx>,
    projected_k_step: Option<&'ctx ggml_rs::Tensor<'ctx>>,
    projected_v_step: Option<&'ctx ggml_rs::Tensor<'ctx>>,
    head_prereq_nodes: &'ctx [ggml_rs::Tensor<'ctx>],
    kv_cache_write_nodes: &'ctx KvCacheWriteNodes<'ctx>,
}

trait StepwiseGraphBuilder {
    fn build_graphs<'ctx>(
        &self,
        ctx: &'ctx Context,
        input: StepwiseGraphBuildInput<'ctx>,
    ) -> Result<Vec<ggml_rs::Graph<'ctx>>, InferenceError>;
}

struct KvPolicyStepwiseGraphBuilder<'strategy> {
    kv_cache_write_strategy: &'strategy dyn KvCacheWriteStrategy,
}

impl<'strategy> KvPolicyStepwiseGraphBuilder<'strategy> {
    fn new(kv_cache_write_strategy: &'strategy dyn KvCacheWriteStrategy) -> Self {
        Self {
            kv_cache_write_strategy,
        }
    }
}

impl StepwiseGraphBuilder for KvPolicyStepwiseGraphBuilder<'_> {
    fn build_graphs<'ctx>(
        &self,
        ctx: &'ctx Context,
        input: StepwiseGraphBuildInput<'ctx>,
    ) -> Result<Vec<ggml_rs::Graph<'ctx>>, InferenceError> {
        let graph_count = self.kv_cache_write_strategy.graph_count(input.steps);
        let mut graphs = Vec::with_capacity(graph_count);
        for graph_step in 0..graph_count {
            let mut graph = ctx
                .new_graph()
                .map_err(|source| InferenceError::ggml("Context::new_graph", source))?;
            for head_output_write in input.head_prereq_nodes {
                graph.build_forward_expand(head_output_write);
            }
            graph.build_forward_expand(input.y);
            if let Some(projected_k_step) = input.projected_k_step {
                graph.build_forward_expand(projected_k_step);
            }
            if let Some(projected_v_step) = input.projected_v_step {
                graph.build_forward_expand(projected_v_step);
            }
            self.kv_cache_write_strategy.expand_graph_nodes(
                &mut graph,
                graph_step,
                input.kv_cache_write_nodes,
            );
            graphs.push(graph);
        }
        Ok(graphs)
    }
}

#[derive(Debug, Clone)]
struct StepGraphSchedule {
    graph_indices: Vec<usize>,
}

impl StepGraphSchedule {
    fn new(kv_cache_write_strategy: &dyn KvCacheWriteStrategy, steps: usize) -> Self {
        let graph_indices = (0..steps)
            .map(|step| kv_cache_write_strategy.graph_index_for_step(step))
            .collect();
        Self { graph_indices }
    }

    fn debug_validate(&self, graph_count: usize) {
        debug_assert!(
            self.graph_indices
                .iter()
                .all(|&graph_index| graph_index < graph_count),
            "stepwise graph schedule must stay within graph range",
        );
    }

    fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.graph_indices.iter().copied().enumerate()
    }
}

#[derive(Debug, Clone, Copy)]
enum HeadOutputProjectionMode {
    PerHead,
    FusedConcat { balanced_concat: bool },
    FusedStaging,
}

impl HeadOutputProjectionMode {
    const fn from_flags(
        fuse_output_projection: bool,
        use_head_output_staging_buffer: bool,
        use_balanced_head_concat: bool,
    ) -> Self {
        if !fuse_output_projection {
            Self::PerHead
        } else if use_head_output_staging_buffer {
            Self::FusedStaging
        } else {
            Self::FusedConcat {
                balanced_concat: use_balanced_head_concat,
            }
        }
    }
}

struct HeadOutputAssembler<'ctx> {
    mode: HeadOutputProjectionMode,
    output_projection: Option<ggml_rs::Tensor<'ctx>>,
    head_outputs: Vec<ggml_rs::Tensor<'ctx>>,
    head_output_staging: Option<ggml_rs::Tensor<'ctx>>,
    head_output_staging_writes: Vec<ggml_rs::Tensor<'ctx>>,
    concat_metadata: HeadConcatMetadata,
    hidden_features: usize,
    query_head_count: usize,
    kv_head_count: usize,
    query_length: usize,
}

#[derive(Debug, Clone, Copy)]
struct HeadOutputAssemblerInit {
    query_features: usize,
    query_length: usize,
    query_head_count: usize,
    hidden_features: usize,
    kv_head_count: usize,
    concat_metadata: HeadConcatMetadata,
}

#[derive(Debug, Clone, Copy)]
struct HeadOutputAccumulateArgs {
    query_offset: usize,
    head_dimension: usize,
    q_row_stride: usize,
    o_row_stride: usize,
}

impl<'ctx> HeadOutputAssembler<'ctx> {
    fn new(
        ctx: &'ctx Context,
        mode: HeadOutputProjectionMode,
        init: HeadOutputAssemblerInit,
    ) -> Result<Self, InferenceError> {
        let head_output_staging = if matches!(mode, HeadOutputProjectionMode::FusedStaging) {
            Some(
                ctx.new_f32_tensor_2d_shape(Shape2D::new(init.query_features, init.query_length))
                    .map_err(|source| {
                        InferenceError::ggml(
                            "Context::new_f32_tensor_2d_shape<HEAD_OUTPUT_STAGING>",
                            source,
                        )
                    })?,
            )
        } else {
            None
        };
        let head_outputs = if matches!(mode, HeadOutputProjectionMode::FusedConcat { .. }) {
            Vec::with_capacity(init.query_head_count)
        } else {
            Vec::new()
        };
        let head_output_staging_writes = if head_output_staging.is_some() {
            Vec::with_capacity(init.query_head_count)
        } else {
            Vec::new()
        };
        Ok(Self {
            mode,
            output_projection: None,
            head_outputs,
            head_output_staging,
            head_output_staging_writes,
            concat_metadata: init.concat_metadata,
            hidden_features: init.hidden_features,
            query_head_count: init.query_head_count,
            kv_head_count: init.kv_head_count,
            query_length: init.query_length,
        })
    }

    fn graph_prereq_nodes(&self) -> &[ggml_rs::Tensor<'ctx>] {
        &self.head_output_staging_writes
    }

    fn accumulate(
        &mut self,
        ctx: &'ctx Context,
        w_o: &ggml_rs::Tensor<'ctx>,
        head_output: ggml_rs::Tensor<'ctx>,
        args: HeadOutputAccumulateArgs,
    ) -> Result<(), InferenceError> {
        match self.mode {
            HeadOutputProjectionMode::PerHead => {
                let w_o_head = ctx
                    .view_2d(
                        w_o,
                        args.head_dimension,
                        self.hidden_features,
                        args.o_row_stride,
                        args.query_offset,
                    )
                    .map_err(|source| InferenceError::ggml("Context::view_2d(W_O_HEAD)", source))?;
                let projected = ctx.mul_mat(&w_o_head, &head_output).map_err(|source| {
                    InferenceError::ggml("Context::mul_mat(W_O_HEAD*HEAD)", source)
                })?;
                self.output_projection = Some(if let Some(acc) = self.output_projection.take() {
                    ctx.add(&acc, &projected)
                        .map_err(|source| InferenceError::ggml("Context::add(head_acc)", source))?
                } else {
                    projected
                });
            }
            HeadOutputProjectionMode::FusedConcat { .. } => {
                self.head_outputs.push(head_output);
            }
            HeadOutputProjectionMode::FusedStaging => {
                let head_output_staging = self
                    .head_output_staging
                    .as_ref()
                    .ok_or_else(|| self.invalid_layout_error())?;
                let head_output_slot = ctx
                    .view_2d(
                        head_output_staging,
                        args.head_dimension,
                        self.query_length,
                        args.q_row_stride,
                        args.query_offset,
                    )
                    .map_err(|source| {
                        InferenceError::ggml("Context::view_2d(HEAD_OUTPUT_STAGING_SLOT)", source)
                    })?;
                let head_output_write =
                    ctx.cpy(&head_output, &head_output_slot).map_err(|source| {
                        InferenceError::ggml(
                            "Context::cpy(HEAD_OUTPUT->HEAD_OUTPUT_STAGING_SLOT)",
                            source,
                        )
                    })?;
                self.head_output_staging_writes.push(head_output_write);
            }
        }
        Ok(())
    }

    fn finalize(
        &mut self,
        ctx: &'ctx Context,
        w_o: &ggml_rs::Tensor<'ctx>,
    ) -> Result<ggml_rs::Tensor<'ctx>, InferenceError> {
        match self.mode {
            HeadOutputProjectionMode::PerHead => self
                .output_projection
                .take()
                .ok_or_else(|| self.invalid_layout_error()),
            HeadOutputProjectionMode::FusedConcat { balanced_concat } => {
                let head_outputs = std::mem::take(&mut self.head_outputs);
                let concatenated = if balanced_concat {
                    BalancedHeadConcat.concat(ctx, head_outputs, 0, self.concat_metadata)?
                } else {
                    LeftFoldHeadConcat.concat(ctx, head_outputs, 0, self.concat_metadata)?
                };
                ctx.mul_mat(w_o, &concatenated)
                    .map_err(|source| InferenceError::ggml("Context::mul_mat(W_O*HEADS)", source))
            }
            HeadOutputProjectionMode::FusedStaging => {
                let head_output_staging = self
                    .head_output_staging
                    .as_ref()
                    .ok_or_else(|| self.invalid_layout_error())?;
                ctx.mul_mat(w_o, head_output_staging).map_err(|source| {
                    InferenceError::ggml("Context::mul_mat(W_O*HEAD_STAGING)", source)
                })
            }
        }
    }

    fn invalid_layout_error(&self) -> InferenceError {
        InferenceError::InvalidAttentionLayout {
            hidden_features: self.hidden_features,
            query_head_count: self.query_head_count,
            kv_head_count: self.kv_head_count,
        }
    }
}

trait SequenceStateUpdater {
    fn initialize_mask<'ctx>(
        &mut self,
        mask: Option<&ggml_rs::Tensor<'ctx>>,
        past_start: usize,
    ) -> Result<(), InferenceError>;

    fn update_positions<'ctx>(
        &mut self,
        positions_q: Option<&ggml_rs::Tensor<'ctx>>,
        step_past_tokens: usize,
    ) -> Result<(), InferenceError>;

    fn update_mask<'ctx>(
        &mut self,
        mask: Option<&ggml_rs::Tensor<'ctx>>,
        step: usize,
        step_past_tokens: usize,
    ) -> Result<(), InferenceError>;
}

#[derive(Debug, Clone)]
struct DeltaSequenceStateUpdater {
    incremental_position_update: bool,
    incremental_mask_update: bool,
    query_length: usize,
    key_value_length: usize,
    step_positions_q_values: Option<Vec<i32>>,
    step_mask_values: Option<Vec<f32>>,
    initial_mask_values: Option<Vec<f32>>,
}

impl DeltaSequenceStateUpdater {
    fn new(
        positions_enabled: bool,
        mask_enabled: bool,
        query_length: usize,
        key_value_length: usize,
        incremental_position_update: bool,
        incremental_mask_update: bool,
        elide_mask_host_buffer: bool,
    ) -> Self {
        let step_positions_q_values = if positions_enabled && !incremental_position_update {
            Some(vec![0_i32; query_length])
        } else {
            None
        };
        let step_mask_values =
            if mask_enabled && (!incremental_mask_update || !elide_mask_host_buffer) {
                Some(vec![0.0_f32; query_length * key_value_length])
            } else {
                None
            };
        let initial_mask_values =
            if incremental_mask_update && mask_enabled && step_mask_values.is_none() {
                Some(vec![0.0_f32; query_length * key_value_length])
            } else {
                None
            };
        Self {
            incremental_position_update,
            incremental_mask_update,
            query_length,
            key_value_length,
            step_positions_q_values,
            step_mask_values,
            initial_mask_values,
        }
    }
}

impl SequenceStateUpdater for DeltaSequenceStateUpdater {
    fn initialize_mask<'ctx>(
        &mut self,
        mask: Option<&ggml_rs::Tensor<'ctx>>,
        past_start: usize,
    ) -> Result<(), InferenceError> {
        if !self.incremental_mask_update {
            return Ok(());
        }
        if let Some(mask) = mask {
            if let Some(mask_values) = self.step_mask_values.as_mut() {
                fill_causal_mask_values(
                    mask_values,
                    self.query_length,
                    self.key_value_length,
                    past_start,
                );
                mask.write_data_backend(mask_values).map_err(|source| {
                    InferenceError::ggml("Tensor::write_data_backend<CAUSAL_MASK>", source)
                })?;
            } else if let Some(initial_mask_values) = self.initial_mask_values.as_mut() {
                fill_causal_mask_values(
                    initial_mask_values,
                    self.query_length,
                    self.key_value_length,
                    past_start,
                );
                mask.write_data_backend(initial_mask_values)
                    .map_err(|source| {
                        InferenceError::ggml("Tensor::write_data_backend<CAUSAL_MASK>", source)
                    })?;
            }
        }
        Ok(())
    }

    fn update_positions<'ctx>(
        &mut self,
        positions_q: Option<&ggml_rs::Tensor<'ctx>>,
        step_past_tokens: usize,
    ) -> Result<(), InferenceError> {
        if let Some(positions_q) = positions_q {
            if self.incremental_position_update {
                let position = i32::try_from(step_past_tokens)
                    .map_err(|_| InferenceError::MemorySizeOverflow)?;
                positions_q
                    .write_data_backend_at(0, &[position])
                    .map_err(|source| {
                        InferenceError::ggml("Tensor::write_data_backend_at<QUERY_POS>", source)
                    })?;
            } else if let Some(values) = self.step_positions_q_values.as_mut() {
                for (index, value) in values.iter_mut().enumerate() {
                    let position = step_past_tokens
                        .checked_add(index)
                        .ok_or(InferenceError::MemorySizeOverflow)?;
                    *value =
                        i32::try_from(position).map_err(|_| InferenceError::MemorySizeOverflow)?;
                }
                positions_q.write_data_backend(values).map_err(|source| {
                    InferenceError::ggml("Tensor::write_data_backend<QUERY_POS>", source)
                })?;
            }
        }
        Ok(())
    }

    fn update_mask<'ctx>(
        &mut self,
        mask: Option<&ggml_rs::Tensor<'ctx>>,
        step: usize,
        step_past_tokens: usize,
    ) -> Result<(), InferenceError> {
        if let Some(mask) = mask {
            if self.incremental_mask_update {
                if step > 0 {
                    let newly_visible_key = step_past_tokens;
                    if newly_visible_key < self.key_value_length {
                        if let Some(mask_values) = self.step_mask_values.as_mut() {
                            mask_values[newly_visible_key] = 0.0;
                        }
                        mask.write_data_backend_at(newly_visible_key, &[0.0_f32])
                            .map_err(|source| {
                                InferenceError::ggml(
                                    "Tensor::write_data_backend_at<CAUSAL_MASK>",
                                    source,
                                )
                            })?;
                    }
                }
            } else if let Some(mask_values) = self.step_mask_values.as_mut() {
                fill_causal_mask_values(
                    mask_values,
                    self.query_length,
                    self.key_value_length,
                    step_past_tokens,
                );
                mask.write_data_backend(mask_values).map_err(|source| {
                    InferenceError::ggml("Tensor::write_data_backend<CAUSAL_MASK>", source)
                })?;
            }
        }
        Ok(())
    }
}

/// Runs stepwise decode-like attention with persistent graph/tensor allocations.
///
/// This runner keeps one backend/context/graph allocation and advances decode state
/// by updating causal-mask and query-position tensors for each step.
pub(crate) fn decode_stepwise_with_block_mlp(
    weights: &AttentionWeights,
    query_input: &[f32],
    cache: &AttentionDecodeCache,
    backend_kind: LlamaBackend,
    stepwise: AttentionDecodeStepwiseConfig,
    block_mlp_weights: Option<&MlpWeights>,
) -> Result<AttentionDecodeStepwiseReport, InferenceError> {
    let (report, _bench_duration, _setup_duration) = execute_stepwise_with_block_mlp_internal(
        weights,
        query_input,
        cache,
        backend_kind,
        stepwise,
        0,
        block_mlp_weights,
    )?;
    Ok(report)
}

/// Bench-oriented stepwise runner that reuses backend/context/graph allocation
/// across warmup and measured phases.
pub(crate) fn bench_stepwise_with_block_mlp(
    weights: &AttentionWeights,
    query_input: &[f32],
    cache: &AttentionDecodeCache,
    backend_kind: LlamaBackend,
    stepwise: AttentionDecodeStepwiseConfig,
    warmup_repeats_per_step: usize,
    block_mlp_weights: Option<&MlpWeights>,
) -> Result<AttentionDecodeStepwiseBenchReport, InferenceError> {
    let (execution, bench_duration, setup_duration) = execute_stepwise_with_block_mlp_internal(
        weights,
        query_input,
        cache,
        backend_kind,
        stepwise,
        warmup_repeats_per_step,
        block_mlp_weights,
    )?;
    Ok(AttentionDecodeStepwiseBenchReport {
        warmup_repeats_per_step,
        setup_duration,
        bench_duration,
        execution,
    })
}

/// Bench-oriented stepwise layer sweep runner sharing one setup path.
///
/// This API reuses backend/context/graph allocation across all provided
/// block-MLP weight sets and reports one shared setup duration.
pub(crate) fn bench_stepwise_sweep_with_block_mlp(
    weights: &AttentionWeights,
    query_input: &[f32],
    cache: &AttentionDecodeCache,
    backend_kind: LlamaBackend,
    stepwise: AttentionDecodeStepwiseConfig,
    warmup_repeats_per_step: usize,
    block_mlp_weights: &[&MlpWeights],
) -> Result<AttentionDecodeStepwiseBenchSweepReport, InferenceError> {
    if block_mlp_weights.is_empty() {
        return Err(InferenceError::InvalidInputLength {
            expected: 1,
            actual: 0,
        });
    }
    let (executions, bench_durations, setup_duration) = execute_stepwise_sweep_internal(
        weights,
        query_input,
        cache,
        backend_kind,
        stepwise,
        warmup_repeats_per_step,
        Some(block_mlp_weights),
    )?;
    let entries = executions
        .into_iter()
        .zip(bench_durations)
        .map(
            |(execution, bench_duration)| AttentionDecodeStepwiseBenchReport {
                warmup_repeats_per_step,
                setup_duration: Duration::ZERO,
                bench_duration,
                execution,
            },
        )
        .collect();
    Ok(AttentionDecodeStepwiseBenchSweepReport {
        warmup_repeats_per_step,
        setup_duration,
        entries,
    })
}

fn execute_stepwise_with_block_mlp_internal(
    weights: &AttentionWeights,
    query_input: &[f32],
    cache: &AttentionDecodeCache,
    backend_kind: LlamaBackend,
    stepwise: AttentionDecodeStepwiseConfig,
    warmup_repeats_per_step: usize,
    block_mlp_weights: Option<&MlpWeights>,
) -> Result<(AttentionDecodeStepwiseReport, Duration, Duration), InferenceError> {
    let block_mlp_weight_runs_storage = block_mlp_weights.map(|weights| vec![weights]);
    let (mut reports, mut bench_durations, setup_duration) = execute_stepwise_sweep_internal(
        weights,
        query_input,
        cache,
        backend_kind,
        stepwise,
        warmup_repeats_per_step,
        block_mlp_weight_runs_storage.as_deref(),
    )?;
    if reports.len() != 1 || bench_durations.len() != 1 {
        return Err(InferenceError::InvalidInputLength {
            expected: 1,
            actual: reports.len().max(bench_durations.len()),
        });
    }
    let report = reports.pop().ok_or(InferenceError::InvalidInputLength {
        expected: 1,
        actual: 0,
    })?;
    let bench_duration = bench_durations
        .pop()
        .ok_or(InferenceError::InvalidInputLength {
            expected: 1,
            actual: 0,
        })?;
    Ok((report, bench_duration, setup_duration))
}

fn execute_stepwise_sweep_internal(
    weights: &AttentionWeights,
    query_input: &[f32],
    cache: &AttentionDecodeCache,
    backend_kind: LlamaBackend,
    stepwise: AttentionDecodeStepwiseConfig,
    warmup_repeats_per_step: usize,
    block_mlp_weight_runs: Option<&[&MlpWeights]>,
) -> Result<(Vec<AttentionDecodeStepwiseReport>, Vec<Duration>, Duration), InferenceError> {
    let key_value_start = stepwise.key_value_start;
    let steps = stepwise.steps;
    let past_start = stepwise.past_start;
    let repeats_per_step = stepwise.repeats_per_step;
    let layer_repeat = stepwise.layer_repeat;
    let include_kv_projection = stepwise.include_kv_projection;
    let include_kv_cache_write = stepwise.include_kv_cache_write;
    let include_kv_cache_write_to_cache = stepwise.include_kv_cache_write_to_cache;
    let include_block_scope = stepwise.include_block_scope;
    let synchronize_per_step = stepwise.synchronize_per_step;
    let readback_per_step = stepwise.readback_per_step;
    let use_position_deltas = stepwise.use_position_deltas;
    let use_mask_deltas = stepwise.use_mask_deltas;
    let elide_mask_host_buffer = stepwise.elide_mask_host_buffer;
    let fuse_output_projection = stepwise.fuse_output_projection;
    let precompute_static_kv_head_views = stepwise.precompute_static_kv_head_views;
    let use_balanced_head_concat = stepwise.use_balanced_head_concat;
    let use_head_output_staging_buffer = stepwise.use_head_output_staging_buffer;
    let use_fused_block_gate_up_projection = stepwise.use_fused_block_gate_up_projection;
    if key_value_start == 0 {
        return Err(InferenceError::InvalidInputLength {
            expected: 1,
            actual: key_value_start,
        });
    }
    if steps == 0 {
        return Err(InferenceError::InvalidInputLength {
            expected: 1,
            actual: steps,
        });
    }
    if repeats_per_step == 0 {
        return Err(InferenceError::InvalidRepeats);
    }
    if layer_repeat == 0 {
        return Err(InferenceError::InvalidInputLength {
            expected: 1,
            actual: layer_repeat,
        });
    }
    if include_kv_cache_write_to_cache && !include_kv_cache_write {
        return Err(InferenceError::InvalidInputLength {
            expected: 1,
            actual: 0,
        });
    }
    let config = weights.config;
    if !matches!(config.mask, AttentionMaskPolicy::Causal { .. }) {
        return Err(InferenceError::InvalidAttentionShape {
            hidden_features: config.hidden_features(),
            sequence_length: config.sequence_length(),
        });
    }
    let hidden_features = config.hidden_features();
    let query_length = config.sequence_length();
    let block_mlp_runs: Vec<Option<&MlpWeights>> = if let Some(weight_runs) = block_mlp_weight_runs
    {
        weight_runs.iter().copied().map(Some).collect()
    } else {
        vec![None]
    };
    let has_block_mlp_weights = block_mlp_runs.iter().any(Option::is_some);
    let has_missing_block_mlp_weights = block_mlp_runs.iter().any(Option::is_none);
    if has_block_mlp_weights && has_missing_block_mlp_weights {
        return Err(InferenceError::InvalidInputLength {
            expected: 1,
            actual: 0,
        });
    }
    let graph_block_mlp_weights = block_mlp_runs.iter().flatten().copied().next();
    if let Some(graph_block_mlp_weights) = graph_block_mlp_weights {
        for block_mlp_weights in block_mlp_runs.iter().flatten() {
            if block_mlp_weights.hidden_features != hidden_features {
                return Err(InferenceError::InvalidInputLength {
                    expected: hidden_features,
                    actual: block_mlp_weights.hidden_features,
                });
            }
            if block_mlp_weights.ffn_features != graph_block_mlp_weights.ffn_features {
                return Err(InferenceError::InvalidInputLength {
                    expected: graph_block_mlp_weights.ffn_features,
                    actual: block_mlp_weights.ffn_features,
                });
            }
        }
    }
    let key_value_length = cache.key_value_length();
    let cache_kv_features = cache.kv_features();
    if cache_kv_features != config.kv_features() {
        return Err(InferenceError::InvalidInputLength {
            expected: config.kv_features(),
            actual: cache_kv_features,
        });
    }
    let final_visible_length = key_value_start
        .checked_add(steps - 1)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    if final_visible_length > key_value_length {
        return Err(InferenceError::InvalidInputLength {
            expected: key_value_length,
            actual: final_visible_length,
        });
    }

    let expected_query_len = hidden_features
        .checked_mul(query_length)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    if query_input.len() != expected_query_len {
        return Err(InferenceError::InvalidInputLength {
            expected: expected_query_len,
            actual: query_input.len(),
        });
    }

    let kv_features = config.kv_features();
    let expected_projected_len = kv_features
        .checked_mul(key_value_length)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    if cache.projected_k_values.len() != expected_projected_len {
        return Err(InferenceError::InvalidInputLength {
            expected: expected_projected_len,
            actual: cache.projected_k_values.len(),
        });
    }
    if cache.projected_v_values.len() != expected_projected_len {
        return Err(InferenceError::InvalidInputLength {
            expected: expected_projected_len,
            actual: cache.projected_v_values.len(),
        });
    }

    let setup_start = Instant::now();
    ensure_backends_loaded();
    let backend = Backend::new(backend_kind.into())
        .map_err(|source| InferenceError::ggml("Backend::new", source))?;
    let backend_name = backend
        .name()
        .map_err(|source| InferenceError::ggml("Backend::name", source))?
        .to_string();

    let ctx_size = recommended_attention_backend_memory_bytes_for_lengths(
        config,
        query_length,
        key_value_length,
    )?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| InferenceError::ggml("Context::new_no_alloc_bytes", source))?;

    let query_features = config.query_features();

    let w_q = ctx
        .new_f32_tensor_2d_shape(Shape2D::new(hidden_features, query_features))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_Q>", source))?;
    let w_o = ctx
        .new_f32_tensor_2d_shape(Shape2D::new(query_features, hidden_features))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_O>", source))?;
    let x_q = ctx
        .new_f32_tensor_2d_shape(Shape2D::new(hidden_features, query_length))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<X_Q>", source))?;
    let k = ctx
        .new_f32_tensor_2d_shape(Shape2D::new(kv_features, key_value_length))
        .map_err(|source| {
            InferenceError::ggml("Context::new_f32_tensor_2d_shape<K_CACHE>", source)
        })?;
    let v = ctx
        .new_f32_tensor_2d_shape(Shape2D::new(kv_features, key_value_length))
        .map_err(|source| {
            InferenceError::ggml("Context::new_f32_tensor_2d_shape<V_CACHE>", source)
        })?;

    let q = ctx
        .mul_mat(&w_q, &x_q)
        .map_err(|source| InferenceError::ggml("Context::mul_mat(Q)", source))?;

    let (positions_q, positions_k) = if matches!(config.rotary, RotaryEmbedding::Llama(_)) {
        let positions_q = ctx
            .new_i32_tensor_1d_len(Length::new(query_length))
            .map_err(|source| {
                InferenceError::ggml("Context::new_i32_tensor_1d_len<QUERY_POS>", source)
            })?;
        let positions_k = ctx
            .new_i32_tensor_1d_len(Length::new(key_value_length))
            .map_err(|source| {
                InferenceError::ggml("Context::new_i32_tensor_1d_len<KV_POS>", source)
            })?;
        (Some(positions_q), Some(positions_k))
    } else {
        (None, None)
    };
    let mask = Some(
        ctx.new_f32_tensor_2d_shape(Shape2D::new(key_value_length, query_length))
            .map_err(|source| {
                InferenceError::ggml("Context::new_f32_tensor_2d_shape<CAUSAL_MASK>", source)
            })?,
    );

    let bytes_per_element = std::mem::size_of::<f32>();
    let kv_cache_row_stride = kv_features
        .checked_mul(bytes_per_element)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let step_specific_kv_cache_writes =
        include_kv_projection && include_kv_cache_write && include_kv_cache_write_to_cache;
    let shared_kv_cache_write_strategy = SharedKvCacheWriteStrategy;
    let step_specific_kv_cache_write_strategy = StepSpecificKvCacheWriteStrategy;
    let kv_cache_write_strategy: &dyn KvCacheWriteStrategy = if step_specific_kv_cache_writes {
        &step_specific_kv_cache_write_strategy
    } else {
        &shared_kv_cache_write_strategy
    };
    let precompute_kv_head_views =
        precompute_static_kv_head_views && !step_specific_kv_cache_writes;

    let (w_k, w_v, projected_k_step, projected_v_step, kv_cache_write_nodes) =
        if include_kv_projection {
            let w_k = ctx
                .new_f32_tensor_2d_shape(Shape2D::new(hidden_features, kv_features))
                .map_err(|source| {
                    InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_K>", source)
                })?;
            let w_v = ctx
                .new_f32_tensor_2d_shape(Shape2D::new(hidden_features, kv_features))
                .map_err(|source| {
                    InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_V>", source)
                })?;
            let projected_k_step = ctx
                .mul_mat(&w_k, &x_q)
                .map_err(|source| InferenceError::ggml("Context::mul_mat(K_STEP)", source))?;
            let projected_v_step = ctx
                .mul_mat(&w_v, &x_q)
                .map_err(|source| InferenceError::ggml("Context::mul_mat(V_STEP)", source))?;
            let kv_cache_write_nodes = if include_kv_cache_write {
                if step_specific_kv_cache_writes {
                    let mut kv_cache_write_k_steps = Vec::with_capacity(steps);
                    let mut kv_cache_write_v_steps = Vec::with_capacity(steps);
                    for step in 0..steps {
                        let step_slot = key_value_start
                            .checked_add(step)
                            .and_then(|value| value.checked_sub(1))
                            .ok_or(InferenceError::MemorySizeOverflow)?;
                        let step_offset = step_slot
                            .checked_mul(kv_features)
                            .and_then(|value| value.checked_mul(bytes_per_element))
                            .ok_or(InferenceError::MemorySizeOverflow)?;

                        let k_write_slot = ctx
                            .view_2d(
                                &k,
                                kv_features,
                                query_length,
                                kv_cache_row_stride,
                                step_offset,
                            )
                            .map_err(|source| {
                                InferenceError::ggml("Context::view_2d<K_CACHE_WRITE_SLOT>", source)
                            })?;
                        let v_write_slot = ctx
                            .view_2d(
                                &v,
                                kv_features,
                                query_length,
                                kv_cache_row_stride,
                                step_offset,
                            )
                            .map_err(|source| {
                                InferenceError::ggml("Context::view_2d<V_CACHE_WRITE_SLOT>", source)
                            })?;
                        let kv_cache_write_k =
                            ctx.cpy(&projected_k_step, &k_write_slot)
                                .map_err(|source| {
                                    InferenceError::ggml(
                                        "Context::cpy(K_STEP->K_CACHE_SLOT)",
                                        source,
                                    )
                                })?;
                        let kv_cache_write_v =
                            ctx.cpy(&projected_v_step, &v_write_slot)
                                .map_err(|source| {
                                    InferenceError::ggml(
                                        "Context::cpy(V_STEP->V_CACHE_SLOT)",
                                        source,
                                    )
                                })?;
                        kv_cache_write_k_steps.push(kv_cache_write_k);
                        kv_cache_write_v_steps.push(kv_cache_write_v);
                    }
                    KvCacheWriteNodes {
                        step_k: Some(kv_cache_write_k_steps),
                        step_v: Some(kv_cache_write_v_steps),
                        ..KvCacheWriteNodes::default()
                    }
                } else {
                    let k_write_slot = ctx
                        .new_f32_tensor_2d_shape(Shape2D::new(kv_features, query_length))
                        .map_err(|source| {
                            InferenceError::ggml(
                                "Context::new_f32_tensor_2d_shape<K_WRITE_SLOT>",
                                source,
                            )
                        })?;
                    let v_write_slot = ctx
                        .new_f32_tensor_2d_shape(Shape2D::new(kv_features, query_length))
                        .map_err(|source| {
                            InferenceError::ggml(
                                "Context::new_f32_tensor_2d_shape<V_WRITE_SLOT>",
                                source,
                            )
                        })?;
                    let kv_cache_write_k =
                        ctx.cpy(&projected_k_step, &k_write_slot)
                            .map_err(|source| {
                                InferenceError::ggml("Context::cpy(K_STEP->K_WRITE_SLOT)", source)
                            })?;
                    let kv_cache_write_v =
                        ctx.cpy(&projected_v_step, &v_write_slot)
                            .map_err(|source| {
                                InferenceError::ggml("Context::cpy(V_STEP->V_WRITE_SLOT)", source)
                            })?;
                    KvCacheWriteNodes {
                        shared_k: Some(kv_cache_write_k),
                        shared_v: Some(kv_cache_write_v),
                        ..KvCacheWriteNodes::default()
                    }
                }
            } else {
                KvCacheWriteNodes::default()
            };
            (
                Some(w_k),
                Some(w_v),
                Some(projected_k_step),
                Some(projected_v_step),
                kv_cache_write_nodes,
            )
        } else {
            (None, None, None, None, KvCacheWriteNodes::default())
        };

    let head_output_projection_mode = HeadOutputProjectionMode::from_flags(
        fuse_output_projection,
        use_head_output_staging_buffer,
        use_balanced_head_concat,
    );
    let mut head_output_assembler = HeadOutputAssembler::new(
        &ctx,
        head_output_projection_mode,
        HeadOutputAssemblerInit {
            query_features,
            query_length,
            query_head_count: config.query_head_count(),
            hidden_features,
            kv_head_count: config.kv_head_count(),
            concat_metadata: HeadConcatMetadata::from_config(config),
        },
    )?;
    let q_row_stride = query_features
        .checked_mul(bytes_per_element)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let kv_row_stride = kv_cache_row_stride;
    let o_row_stride = q_row_stride;
    let attention_scale = 1.0 / (config.head_dimension() as f32).sqrt();
    let kv_group_size = config.layout.kv_group_size();
    let rotary_applier = LlamaRotaryApplier;
    let mut rotated_k_heads = Vec::with_capacity(config.kv_head_count());
    let mut transposed_v_heads = Vec::with_capacity(config.kv_head_count());
    let mut static_kv_head_graph = if precompute_kv_head_views {
        Some(ctx.new_graph().map_err(|source| {
            InferenceError::ggml("Context::new_graph<KV_HEAD_PRECOMPUTE>", source)
        })?)
    } else {
        None
    };

    for kv_head in 0..config.kv_head_count() {
        let kv_offset = kv_head
            .checked_mul(config.head_dimension())
            .and_then(|value| value.checked_mul(bytes_per_element))
            .ok_or(InferenceError::MemorySizeOverflow)?;
        let k_head = ctx
            .view_2d(
                &k,
                config.head_dimension(),
                key_value_length,
                kv_row_stride,
                kv_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(K_HEAD)", source))?;
        let v_head = ctx
            .view_2d(
                &v,
                config.head_dimension(),
                key_value_length,
                kv_row_stride,
                kv_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(V_HEAD)", source))?;

        let rotated_k_head = rotary_applier.apply_single_with_sequence(
            &ctx,
            &k_head,
            positions_k.as_ref(),
            config,
            key_value_length,
        )?;
        let v_t = ctx
            .transpose(&v_head)
            .map_err(|source| InferenceError::ggml("Context::transpose(V_HEAD)", source))?;
        let v_t = ctx
            .cont(&v_t)
            .map_err(|source| InferenceError::ggml("Context::cont(V_HEAD)", source))?;

        if let Some(precompute_graph) = static_kv_head_graph.as_mut() {
            let rotated_k_cached = ctx
                .new_f32_tensor_2d_shape(Shape2D::new(config.head_dimension(), key_value_length))
                .map_err(|source| {
                    InferenceError::ggml(
                        "Context::new_f32_tensor_2d_shape<K_ROTATED_CACHED>",
                        source,
                    )
                })?;
            let transposed_v_cached = ctx
                .new_f32_tensor_2d_shape(Shape2D::new(key_value_length, config.head_dimension()))
                .map_err(|source| {
                    InferenceError::ggml(
                        "Context::new_f32_tensor_2d_shape<V_TRANSPOSED_CACHED>",
                        source,
                    )
                })?;
            let rotate_write = ctx
                .cpy(&rotated_k_head, &rotated_k_cached)
                .map_err(|source| {
                    InferenceError::ggml("Context::cpy(K_ROTATED->K_ROTATED_CACHED)", source)
                })?;
            let v_transpose_write = ctx.cpy(&v_t, &transposed_v_cached).map_err(|source| {
                InferenceError::ggml("Context::cpy(VT->V_TRANSPOSED_CACHED)", source)
            })?;
            precompute_graph.build_forward_expand(&rotate_write);
            precompute_graph.build_forward_expand(&v_transpose_write);
            rotated_k_heads.push(rotated_k_cached);
            transposed_v_heads.push(transposed_v_cached);
        } else {
            rotated_k_heads.push(rotated_k_head);
            transposed_v_heads.push(v_t);
        }
    }

    let q_heads_source = rotary_applier.apply_multi_head_with_sequence(
        &ctx,
        &q,
        positions_q.as_ref(),
        config,
        config.query_head_count(),
        query_length,
    )?;

    for head in 0..config.query_head_count() {
        let query_offset = head
            .checked_mul(config.head_dimension())
            .and_then(|value| value.checked_mul(bytes_per_element))
            .ok_or(InferenceError::MemorySizeOverflow)?;
        let kv_head = head / kv_group_size;

        let q_head = ctx
            .view_2d(
                &q_heads_source,
                config.head_dimension(),
                query_length,
                q_row_stride,
                query_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(Q_HEAD)", source))?;
        let k_head =
            rotated_k_heads
                .get(kv_head)
                .ok_or(InferenceError::InvalidAttentionLayout {
                    hidden_features,
                    query_head_count: config.query_head_count(),
                    kv_head_count: config.kv_head_count(),
                })?;
        let v_t =
            transposed_v_heads
                .get(kv_head)
                .ok_or(InferenceError::InvalidAttentionLayout {
                    hidden_features,
                    query_head_count: config.query_head_count(),
                    kv_head_count: config.kv_head_count(),
                })?;

        let scores = ctx
            .mul_mat(k_head, &q_head)
            .map_err(|source| InferenceError::ggml("Context::mul_mat(K_HEAD*Q_HEAD)", source))?;
        let probabilities = ctx
            .soft_max_ext(&scores, mask.as_ref(), attention_scale, 0.0)
            .map_err(|source| InferenceError::ggml("Context::soft_max_ext(causal)", source))?;
        let head_output = ctx
            .mul_mat(v_t, &probabilities)
            .map_err(|source| InferenceError::ggml("Context::mul_mat(VT*P)", source))?;
        head_output_assembler.accumulate(
            &ctx,
            &w_o,
            head_output,
            HeadOutputAccumulateArgs {
                query_offset,
                head_dimension: config.head_dimension(),
                q_row_stride,
                o_row_stride,
            },
        )?;
    }

    let y_attention = head_output_assembler.finalize(&ctx, &w_o)?;
    let mut block_mlp_split_tensors = None;
    let mut block_mlp_fused_tensors = None;
    let y = if include_block_scope {
        let residual = ctx
            .add(&x_q, &y_attention)
            .map_err(|source| InferenceError::ggml("Context::add(residual_attn)", source))?;
        let norm = ctx
            .rms_norm(&residual, 1e-5)
            .map_err(|source| InferenceError::ggml("Context::rms_norm(block)", source))?;
        let mlp = if let Some(block_mlp_weights) = graph_block_mlp_weights {
            let ffn_features = block_mlp_weights.ffn_features;
            let w_down = ctx
                .new_f32_tensor_2d_shape(Shape2D::new(ffn_features, hidden_features))
                .map_err(|source| {
                    InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_DOWN_BLOCK>", source)
                })?;
            let (gate, up) = if use_fused_block_gate_up_projection {
                let gate_up_features = ffn_features
                    .checked_mul(2)
                    .ok_or(InferenceError::MemorySizeOverflow)?;
                let w_gate_up = ctx
                    .new_f32_tensor_2d_shape(Shape2D::new(hidden_features, gate_up_features))
                    .map_err(|source| {
                        InferenceError::ggml(
                            "Context::new_f32_tensor_2d_shape<W_GATE_UP_BLOCK>",
                            source,
                        )
                    })?;
                let gate_up = ctx.mul_mat(&w_gate_up, &norm).map_err(|source| {
                    InferenceError::ggml("Context::mul_mat(GATE_UP_BLOCK)", source)
                })?;
                let gate_up_row_stride = gate_up_features
                    .checked_mul(bytes_per_element)
                    .ok_or(InferenceError::MemorySizeOverflow)?;
                let up_offset = ffn_features
                    .checked_mul(bytes_per_element)
                    .ok_or(InferenceError::MemorySizeOverflow)?;
                let gate = ctx
                    .view_2d(&gate_up, ffn_features, query_length, gate_up_row_stride, 0)
                    .map_err(|source| {
                        InferenceError::ggml("Context::view_2d(GATE_BLOCK_FUSED)", source)
                    })?;
                let up = ctx
                    .view_2d(
                        &gate_up,
                        ffn_features,
                        query_length,
                        gate_up_row_stride,
                        up_offset,
                    )
                    .map_err(|source| {
                        InferenceError::ggml("Context::view_2d(UP_BLOCK_FUSED)", source)
                    })?;
                block_mlp_fused_tensors = Some((w_gate_up, w_down));
                (gate, up)
            } else {
                let w_gate = ctx
                    .new_f32_tensor_2d_shape(Shape2D::new(hidden_features, ffn_features))
                    .map_err(|source| {
                        InferenceError::ggml(
                            "Context::new_f32_tensor_2d_shape<W_GATE_BLOCK>",
                            source,
                        )
                    })?;
                let w_up = ctx
                    .new_f32_tensor_2d_shape(Shape2D::new(hidden_features, ffn_features))
                    .map_err(|source| {
                        InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_UP_BLOCK>", source)
                    })?;
                let gate = ctx.mul_mat(&w_gate, &norm).map_err(|source| {
                    InferenceError::ggml("Context::mul_mat(GATE_BLOCK)", source)
                })?;
                let up = ctx
                    .mul_mat(&w_up, &norm)
                    .map_err(|source| InferenceError::ggml("Context::mul_mat(UP_BLOCK)", source))?;
                block_mlp_split_tensors = Some((w_gate, w_up, w_down));
                (gate, up)
            };
            let activated = ctx
                .silu(&gate)
                .map_err(|source| InferenceError::ggml("Context::silu(block_gate)", source))?;
            let fused = ctx
                .mul(&activated, &up)
                .map_err(|source| InferenceError::ggml("Context::mul(block_fused)", source))?;
            ctx.mul_mat(&w_down, &fused)
                .map_err(|source| InferenceError::ggml("Context::mul_mat(MLP_OUT_BLOCK)", source))?
        } else {
            // Lightweight fallback when no model-derived MLP is provided.
            let gate = ctx
                .mul_mat(&w_q, &norm)
                .map_err(|source| InferenceError::ggml("Context::mul_mat(GATE)", source))?;
            let w_o_t = ctx
                .transpose(&w_o)
                .map_err(|source| InferenceError::ggml("Context::transpose(W_O)", source))?;
            let w_o_t = ctx
                .cont(&w_o_t)
                .map_err(|source| InferenceError::ggml("Context::cont(W_O_T)", source))?;
            let up = ctx
                .mul_mat(&w_o_t, &norm)
                .map_err(|source| InferenceError::ggml("Context::mul_mat(UP)", source))?;
            let activated = ctx
                .silu(&gate)
                .map_err(|source| InferenceError::ggml("Context::silu(block_gate)", source))?;
            let fused = ctx
                .mul(&activated, &up)
                .map_err(|source| InferenceError::ggml("Context::mul(block_fused)", source))?;
            ctx.mul_mat(&w_o, &fused)
                .map_err(|source| InferenceError::ggml("Context::mul_mat(MLP_OUT)", source))?
        };
        ctx.add(&residual, &mlp)
            .map_err(|source| InferenceError::ggml("Context::add(residual_mlp)", source))?
    } else {
        y_attention
    };

    let mut graphs = KvPolicyStepwiseGraphBuilder::new(kv_cache_write_strategy).build_graphs(
        &ctx,
        StepwiseGraphBuildInput {
            steps,
            y: &y,
            projected_k_step: projected_k_step.as_ref(),
            projected_v_step: projected_v_step.as_ref(),
            head_prereq_nodes: head_output_assembler.graph_prereq_nodes(),
            kv_cache_write_nodes: &kv_cache_write_nodes,
        },
    )?;
    let step_graph_schedule = StepGraphSchedule::new(kv_cache_write_strategy, steps);
    step_graph_schedule.debug_validate(graphs.len());
    let _buffer = ctx
        .allocate_tensors(&backend)
        .map_err(|source| InferenceError::ggml("Context::allocate_tensors", source))?;

    w_q.write_data_backend(weights.q_values())
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W_Q>", source))?;
    w_o.write_data_backend(weights.o_values())
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W_O>", source))?;
    if let Some(w_k) = w_k.as_ref() {
        w_k.write_data_backend(weights.k_values())
            .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W_K>", source))?;
    }
    if let Some(w_v) = w_v.as_ref() {
        w_v.write_data_backend(weights.v_values())
            .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W_V>", source))?;
    }
    let upload_block_mlp_weights =
        |block_mlp_weights: Option<&MlpWeights>| -> Result<(), InferenceError> {
            if let Some((w_gate, w_up, w_down)) = block_mlp_split_tensors.as_ref() {
                let block_mlp_weights =
                    block_mlp_weights.ok_or(InferenceError::InvalidInputLength {
                        expected: 1,
                        actual: 0,
                    })?;
                w_gate
                    .write_data_backend(block_mlp_weights.gate_values())
                    .map_err(|source| {
                        InferenceError::ggml("Tensor::write_data_backend<W_GATE_BLOCK>", source)
                    })?;
                w_up.write_data_backend(block_mlp_weights.up_values())
                    .map_err(|source| {
                        InferenceError::ggml("Tensor::write_data_backend<W_UP_BLOCK>", source)
                    })?;
                w_down
                    .write_data_backend(block_mlp_weights.down_values())
                    .map_err(|source| {
                        InferenceError::ggml("Tensor::write_data_backend<W_DOWN_BLOCK>", source)
                    })?;
            } else if let Some((w_gate_up, w_down)) = block_mlp_fused_tensors.as_ref() {
                let block_mlp_weights =
                    block_mlp_weights.ok_or(InferenceError::InvalidInputLength {
                        expected: 1,
                        actual: 0,
                    })?;
                let mut gate_up_values = Vec::with_capacity(
                    block_mlp_weights
                        .gate_values()
                        .len()
                        .checked_add(block_mlp_weights.up_values().len())
                        .ok_or(InferenceError::MemorySizeOverflow)?,
                );
                gate_up_values.extend_from_slice(block_mlp_weights.gate_values());
                gate_up_values.extend_from_slice(block_mlp_weights.up_values());
                w_gate_up
                    .write_data_backend(&gate_up_values)
                    .map_err(|source| {
                        InferenceError::ggml("Tensor::write_data_backend<W_GATE_UP_BLOCK>", source)
                    })?;
                w_down
                    .write_data_backend(block_mlp_weights.down_values())
                    .map_err(|source| {
                        InferenceError::ggml("Tensor::write_data_backend<W_DOWN_BLOCK>", source)
                    })?;
            } else if block_mlp_weights.is_some() {
                return Err(InferenceError::InvalidInputLength {
                    expected: 0,
                    actual: 1,
                });
            }
            Ok(())
        };
    x_q.write_data_backend(query_input)
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<X_Q>", source))?;
    if let Some(positions_k) = positions_k.as_ref() {
        let positions_values: Result<Vec<i32>, InferenceError> = (0..key_value_length)
            .map(|index| i32::try_from(index).map_err(|_| InferenceError::MemorySizeOverflow))
            .collect();
        positions_k
            .write_data_backend(&positions_values?)
            .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<KV_POS>", source))?;
    }

    let incremental_position_update = use_position_deltas
        && query_length == 1
        && past_start.checked_add(1) == Some(key_value_start);
    let incremental_mask_update =
        use_mask_deltas && query_length == 1 && past_start.checked_add(1) == Some(key_value_start);
    let mut sequence_state_updater = DeltaSequenceStateUpdater::new(
        positions_q.is_some(),
        mask.is_some(),
        query_length,
        key_value_length,
        incremental_position_update,
        incremental_mask_update,
        elide_mask_host_buffer,
    );

    let mut execute_phase =
        |phase_repeats_per_step: usize, reset_kv_state: bool| -> Result<(), InferenceError> {
            if phase_repeats_per_step == 0 {
                return Ok(());
            }
            let per_step_compute_count = phase_repeats_per_step
                .checked_mul(layer_repeat)
                .ok_or(InferenceError::MemorySizeOverflow)?;

            if reset_kv_state {
                k.write_data_backend(&cache.projected_k_values)
                    .map_err(|source| {
                        InferenceError::ggml("Tensor::write_data_backend<K_CACHE>", source)
                    })?;
                v.write_data_backend(&cache.projected_v_values)
                    .map_err(|source| {
                        InferenceError::ggml("Tensor::write_data_backend<V_CACHE>", source)
                    })?;

                if let Some(precompute_graph) = static_kv_head_graph.as_mut() {
                    backend.compute(precompute_graph).map_err(|source| {
                        InferenceError::ggml("Backend::compute<KV_HEAD_PRECOMPUTE>", source)
                    })?;
                }
            }

            sequence_state_updater.initialize_mask(mask.as_ref(), past_start)?;

            for (step, graph_index) in step_graph_schedule.iter() {
                let step_past_tokens = past_start
                    .checked_add(step)
                    .ok_or(InferenceError::MemorySizeOverflow)?;
                sequence_state_updater.update_positions(positions_q.as_ref(), step_past_tokens)?;
                sequence_state_updater.update_mask(mask.as_ref(), step, step_past_tokens)?;
                let graph = graphs
                    .get_mut(graph_index)
                    .expect("stepwise graph index must stay in range");
                for _ in 0..per_step_compute_count {
                    backend
                        .compute(graph)
                        .map_err(|source| InferenceError::ggml("Backend::compute", source))?;
                }
                if synchronize_per_step {
                    backend
                        .synchronize()
                        .map_err(|source| InferenceError::ggml("Backend::synchronize", source))?;
                }
                if readback_per_step {
                    let _ = y.read_data_backend::<f32>().map_err(|source| {
                        InferenceError::ggml("Tensor::read_data_backend<Y_STEP>", source)
                    })?;
                }
            }
            Ok(())
        };

    let setup_duration = setup_start.elapsed();
    let mut executions = Vec::with_capacity(block_mlp_runs.len());
    let mut bench_durations = Vec::with_capacity(block_mlp_runs.len());
    for block_mlp_weights in block_mlp_runs {
        upload_block_mlp_weights(block_mlp_weights)?;
        execute_phase(warmup_repeats_per_step, true)?;
        let bench_needs_kv_reset =
            kv_cache_write_strategy.bench_needs_kv_reset(warmup_repeats_per_step);
        let bench_start = Instant::now();
        execute_phase(repeats_per_step, bench_needs_kv_reset)?;
        let bench_duration = bench_start.elapsed();

        let output = y
            .read_data_backend::<f32>()
            .map_err(|source| InferenceError::ggml("Tensor::read_data_backend<Y>", source))?;
        executions.push(AttentionDecodeStepwiseReport {
            backend_name: backend_name.clone(),
            hidden_features,
            query_length,
            key_value_start,
            steps,
            repeats_per_step,
            output,
        });
        bench_durations.push(bench_duration);
    }
    Ok((executions, bench_durations, setup_duration))
}
