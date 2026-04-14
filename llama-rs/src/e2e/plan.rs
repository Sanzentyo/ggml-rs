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

    /// Returns `true` if this layer uses standard (non-Qwen3.5) attention.
    pub(super) fn is_standard(&self) -> bool {
        matches!(self, Self::Standard(_))
    }
}

#[derive(Debug, Clone)]
pub(super) struct MlpLayerPlan {
    pub weights: MlpWeights<f32>,
    pub norm_values: Vec<f32>,
}
