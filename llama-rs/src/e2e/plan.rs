//! Layer plan ADTs describing the structure and weights of each transformer layer.
//!
//! Plans are built once from GGUF tensors by the [`super::planner`] module and
//! then consumed by the generation loop.

use crate::inference::{AttentionWeights, MlpWeights};

#[derive(Debug, Clone)]
pub(super) struct LayerPlan {
    pub attention: Option<AttentionLayerPlan>,
    pub mlp: MlpLayerPlan,
}

#[derive(Debug, Clone)]
pub(super) enum AttentionLayerPlan {
    Standard(StandardAttentionLayerPlan),
    Qwen35Full(Qwen35FullAttentionLayerPlan),
    Qwen35Linear(Qwen35LinearAttentionLayerPlan),
}

#[derive(Debug, Clone)]
pub(super) struct StandardAttentionLayerPlan {
    pub weights: AttentionWeights<f32>,
    pub norm_values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct Qwen35FullAttentionLayerPlan {
    pub norm_values: Vec<f32>,
    pub q_norm_values: Vec<f32>,
    pub k_norm_values: Vec<f32>,
    pub q_weight_values: Vec<f32>,
    pub k_weight_values: Vec<f32>,
    pub v_weight_values: Vec<f32>,
    pub output_weight_values: Vec<f32>,
    pub head_count: usize,
    pub kv_head_count: usize,
    pub head_dimension: usize,
    pub attention_scale: f32,
    pub rope_n_dims: usize,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
}

#[derive(Debug, Clone)]
pub(super) struct Qwen35LinearAttentionLayerPlan {
    pub norm_values: Vec<f32>,
    pub qkv_weight_values: Vec<f32>,
    pub gate_weight_values: Vec<f32>,
    pub alpha_weight_values: Vec<f32>,
    pub beta_weight_values: Vec<f32>,
    pub conv_weight_values: Vec<f32>,
    pub dt_bias_values: Vec<f32>,
    pub ssm_a_values: Vec<f32>,
    pub ssm_norm_values: Vec<f32>,
    pub ssm_out_weight_values: Vec<f32>,
    pub state_size: usize,
    pub group_count: usize,
    pub time_step_rank: usize,
    pub inner_size: usize,
    pub conv_kernel: usize,
}

impl AttentionLayerPlan {
    pub(super) fn norm_values(&self) -> &[f32] {
        match self {
            Self::Standard(a) => &a.norm_values,
            Self::Qwen35Full(a) => &a.norm_values,
            Self::Qwen35Linear(a) => &a.norm_values,
        }
    }

    /// True when this variant is [`Standard`](Self::Standard).
    pub(super) fn is_standard(&self) -> bool {
        matches!(self, Self::Standard(_))
    }

    /// KV head count, unified across attention variants.
    pub(super) fn kv_head_count(&self) -> usize {
        match self {
            Self::Standard(a) => a.weights.config.layout.kv_head_count(),
            Self::Qwen35Full(a) => a.kv_head_count,
            Self::Qwen35Linear(a) => a.group_count,
        }
    }

    /// Per-head dimension, unified across attention variants.
    pub(super) fn head_dimension(&self) -> usize {
        match self {
            Self::Standard(a) => a.weights.config.layout.head_dimension(),
            Self::Qwen35Full(a) => a.head_dimension,
            Self::Qwen35Linear(a) => a.state_size,
        }
    }
}

impl Qwen35LinearAttentionLayerPlan {
    /// Convolution channel count: `inner_size + 2 × group_count × state_size`.
    pub(super) fn conv_channels(&self) -> usize {
        self.inner_size + 2 * self.group_count * self.state_size
    }
}

#[derive(Debug, Clone)]
pub(super) struct MlpLayerPlan {
    pub weights: MlpWeights<f32>,
    pub norm_values: Vec<f32>,
}
