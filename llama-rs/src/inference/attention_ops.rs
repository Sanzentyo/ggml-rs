use super::{AttentionInferenceConfig, InferenceError, RotaryEmbedding};
use ggml_rs::{Context, RopeExtParams, Tensor};

#[derive(Debug, Clone, Copy)]
pub(super) struct HeadConcatMetadata {
    hidden_features: usize,
    query_head_count: usize,
    kv_head_count: usize,
}

impl HeadConcatMetadata {
    pub(super) const fn from_config(config: AttentionInferenceConfig) -> Self {
        Self {
            hidden_features: config.hidden_features(),
            query_head_count: config.query_head_count(),
            kv_head_count: config.kv_head_count(),
        }
    }
}

pub(super) trait HeadConcatStrategy {
    fn concat<'ctx>(
        &self,
        ctx: &'ctx Context,
        tensors: Vec<Tensor<'ctx>>,
        dim: usize,
        metadata: HeadConcatMetadata,
    ) -> Result<Tensor<'ctx>, InferenceError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct LeftFoldHeadConcat;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct BalancedHeadConcat;

impl HeadConcatStrategy for LeftFoldHeadConcat {
    fn concat<'ctx>(
        &self,
        ctx: &'ctx Context,
        tensors: Vec<Tensor<'ctx>>,
        dim: usize,
        metadata: HeadConcatMetadata,
    ) -> Result<Tensor<'ctx>, InferenceError> {
        let mut tensors = tensors.into_iter();
        let first = tensors
            .next()
            .ok_or_else(|| invalid_attention_layout(metadata))?;
        tensors.try_fold(first, |acc, tensor| {
            ctx.concat(&acc, &tensor, dim).map_err(|source| {
                InferenceError::ggml("Context::concat(head_outputs_left_fold)", source)
            })
        })
    }
}

impl HeadConcatStrategy for BalancedHeadConcat {
    fn concat<'ctx>(
        &self,
        ctx: &'ctx Context,
        mut tensors: Vec<Tensor<'ctx>>,
        dim: usize,
        metadata: HeadConcatMetadata,
    ) -> Result<Tensor<'ctx>, InferenceError> {
        if tensors.is_empty() {
            return Err(invalid_attention_layout(metadata));
        }
        while tensors.len() > 1 {
            let mut next_level = Vec::with_capacity(tensors.len().div_ceil(2));
            let mut level_iter = tensors.into_iter();
            while let Some(lhs) = level_iter.next() {
                if let Some(rhs) = level_iter.next() {
                    let merged = ctx.concat(&lhs, &rhs, dim).map_err(|source| {
                        InferenceError::ggml("Context::concat(head_outputs_balanced)", source)
                    })?;
                    next_level.push(merged);
                } else {
                    next_level.push(lhs);
                }
            }
            tensors = next_level;
        }
        tensors
            .pop()
            .ok_or_else(|| invalid_attention_layout(metadata))
    }
}

pub(super) trait RotaryApplier {
    fn apply_single_with_sequence<'ctx>(
        &self,
        ctx: &'ctx Context,
        tensor: &Tensor<'ctx>,
        positions: Option<&Tensor<'ctx>>,
        config: AttentionInferenceConfig,
        sequence_length: usize,
    ) -> Result<Tensor<'ctx>, InferenceError>;

    fn apply_multi_head_with_sequence<'ctx>(
        &self,
        ctx: &'ctx Context,
        tensor: &Tensor<'ctx>,
        positions: Option<&Tensor<'ctx>>,
        config: AttentionInferenceConfig,
        head_count: usize,
        sequence_length: usize,
    ) -> Result<Tensor<'ctx>, InferenceError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct LlamaRotaryApplier;

impl RotaryApplier for LlamaRotaryApplier {
    fn apply_single_with_sequence<'ctx>(
        &self,
        ctx: &'ctx Context,
        tensor: &Tensor<'ctx>,
        positions: Option<&Tensor<'ctx>>,
        config: AttentionInferenceConfig,
        sequence_length: usize,
    ) -> Result<Tensor<'ctx>, InferenceError> {
        apply_rotary_with_head_count(
            ctx,
            tensor,
            positions,
            config,
            1,
            sequence_length,
            RopeContextNames {
                contiguous: "Context::cont(rope_single)",
                reshape_3d: "Context::reshape_3d(rope_single)",
                rope_ext: "Context::rope_ext(single_head)",
                reshape_2d: "Context::reshape_2d(rope_single)",
            },
        )
    }

    fn apply_multi_head_with_sequence<'ctx>(
        &self,
        ctx: &'ctx Context,
        tensor: &Tensor<'ctx>,
        positions: Option<&Tensor<'ctx>>,
        config: AttentionInferenceConfig,
        head_count: usize,
        sequence_length: usize,
    ) -> Result<Tensor<'ctx>, InferenceError> {
        apply_rotary_with_head_count(
            ctx,
            tensor,
            positions,
            config,
            head_count,
            sequence_length,
            RopeContextNames {
                contiguous: "Context::cont(rope_multi_head)",
                reshape_3d: "Context::reshape_3d(rope_multi_head)",
                rope_ext: "Context::rope_ext(multi_head)",
                reshape_2d: "Context::reshape_2d(rope_multi_head)",
            },
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct RopeContextNames {
    contiguous: &'static str,
    reshape_3d: &'static str,
    rope_ext: &'static str,
    reshape_2d: &'static str,
}

fn apply_rotary_with_head_count<'ctx>(
    ctx: &'ctx Context,
    tensor: &Tensor<'ctx>,
    positions: Option<&Tensor<'ctx>>,
    config: AttentionInferenceConfig,
    head_count: usize,
    sequence_length: usize,
    names: RopeContextNames,
) -> Result<Tensor<'ctx>, InferenceError> {
    match (config.rotary, positions) {
        (RotaryEmbedding::Disabled, _) => Ok(*tensor),
        (RotaryEmbedding::Llama(params), Some(positions)) => {
            let rope_dimensions = params.dimensions.get();
            if rope_dimensions > config.head_dimension() {
                return Err(InferenceError::InvalidRopeDimensions {
                    rope_dimensions,
                    head_dimension: config.head_dimension(),
                });
            }
            let n_dims =
                i32::try_from(rope_dimensions).map_err(|_| InferenceError::MemorySizeOverflow)?;
            let n_ctx_orig = params
                .original_context
                .and_then(|value| i32::try_from(value.get()).ok())
                .unwrap_or(0);
            let rope_params = RopeExtParams {
                n_dims,
                n_ctx_orig,
                freq_base: params.base,
                freq_scale: params.scale,
                ..RopeExtParams::default()
            };
            let contiguous = ctx
                .cont(tensor)
                .map_err(|source| InferenceError::ggml(names.contiguous, source))?;
            let reshaped = ctx
                .reshape_3d(
                    &contiguous,
                    config.head_dimension(),
                    head_count,
                    sequence_length,
                )
                .map_err(|source| InferenceError::ggml(names.reshape_3d, source))?;
            let rotated = ctx
                .rope_ext(&reshaped, positions, None, rope_params)
                .map_err(|source| InferenceError::ggml(names.rope_ext, source))?;
            let total_features = config
                .head_dimension()
                .checked_mul(head_count)
                .ok_or(InferenceError::MemorySizeOverflow)?;
            ctx.reshape_2d(&rotated, total_features, sequence_length)
                .map_err(|source| InferenceError::ggml(names.reshape_2d, source))
        }
        (RotaryEmbedding::Llama(_), None) => Err(InferenceError::InvalidAttentionShape {
            hidden_features: config.hidden_features(),
            sequence_length,
        }),
    }
}

fn invalid_attention_layout(metadata: HeadConcatMetadata) -> InferenceError {
    InferenceError::InvalidAttentionLayout {
        hidden_features: metadata.hidden_features,
        query_head_count: metadata.query_head_count,
        kv_head_count: metadata.kv_head_count,
    }
}
