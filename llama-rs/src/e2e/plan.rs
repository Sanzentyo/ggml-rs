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

#[cfg(test)]
impl Qwen35FullAttentionLayerPlan {
    /// Build a plan with deterministic pseudo-random weights for testing.
    ///
    /// The weight pattern is reproducible across tests and matches the
    /// seed formulas used in bench_graphs and generation fixtures.
    pub(super) fn deterministic(
        hidden: usize,
        head_count: usize,
        kv_head_count: usize,
        hd: usize,
    ) -> Self {
        let query_features = head_count * hd;
        let kv_features = kv_head_count * hd;
        Self {
            norm_values: vec![1.0_f32; hidden],
            q_norm_values: vec![1.0_f32; hd],
            k_norm_values: vec![1.0_f32; hd],
            q_weight_values: (0..hidden * query_features * 2)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
                .collect(),
            k_weight_values: (0..hidden * kv_features)
                .map(|i| ((i % 5) as f32 - 2.0) * 0.08)
                .collect(),
            v_weight_values: (0..hidden * kv_features)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
                .collect(),
            output_weight_values: (0..query_features * hidden)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
                .collect(),
            head_count,
            kv_head_count,
            head_dimension: hd,
            attention_scale: 1.0 / (hd as f32).sqrt(),
            rope_n_dims: hd,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
        }
    }
}

impl Qwen35LinearAttentionLayerPlan {
    /// Convolution channel count: `inner_size + 2 × group_count × state_size`.
    ///
    /// Uses checked arithmetic to prevent overflow on pathological inputs.
    pub(super) fn conv_channels(&self) -> Result<usize, super::E2eError> {
        use super::numeric::checked_mul;
        let gs = checked_mul(self.group_count, self.state_size)?;
        Ok(self.inner_size + checked_mul(gs, 2)?)
    }
}

#[derive(Debug, Clone)]
pub(super) struct MlpLayerPlan {
    pub weights: MlpWeights<f32>,
    pub norm_values: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{AttentionInferenceConfig, AttentionLayout, AttentionWeights};

    /// Helper: build a `StandardAttentionLayerPlan` with the given layout.
    fn standard_plan(qh: usize, kvh: usize, hd: usize) -> AttentionLayerPlan {
        let layout = AttentionLayout::from_projection_dimensions(qh * hd, qh, kvh, hd).unwrap();
        let config = AttentionInferenceConfig::from_layout(layout, 1).unwrap();
        AttentionLayerPlan::Standard(StandardAttentionLayerPlan {
            weights: AttentionWeights::deterministic(config),
            norm_values: vec![1.0; qh * hd],
        })
    }

    /// Helper: build a `Qwen35FullAttentionLayerPlan` with the given dimensions.
    fn full_plan(head_count: usize, kv_head_count: usize, head_dim: usize) -> AttentionLayerPlan {
        AttentionLayerPlan::Qwen35Full(Qwen35FullAttentionLayerPlan {
            norm_values: vec![1.0; head_count * head_dim],
            q_norm_values: vec![],
            k_norm_values: vec![],
            q_weight_values: vec![],
            k_weight_values: vec![],
            v_weight_values: vec![],
            output_weight_values: vec![],
            head_count,
            kv_head_count,
            head_dimension: head_dim,
            attention_scale: 1.0,
            rope_n_dims: head_dim,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
        })
    }

    /// Helper: build a `Qwen35LinearAttentionLayerPlan` with given dimensions.
    fn linear_plan(group_count: usize, state_size: usize, inner_size: usize) -> AttentionLayerPlan {
        AttentionLayerPlan::Qwen35Linear(Qwen35LinearAttentionLayerPlan {
            norm_values: vec![1.0; inner_size],
            qkv_weight_values: vec![],
            gate_weight_values: vec![],
            alpha_weight_values: vec![],
            beta_weight_values: vec![],
            conv_weight_values: vec![],
            dt_bias_values: vec![],
            ssm_a_values: vec![],
            ssm_norm_values: vec![],
            ssm_out_weight_values: vec![],
            state_size,
            group_count,
            time_step_rank: 4,
            inner_size,
            conv_kernel: 4,
        })
    }

    // ── norm_values ─────────────────────────────────────────────
    #[test]
    fn norm_values_standard() {
        let plan = standard_plan(4, 2, 8);
        assert_eq!(plan.norm_values().len(), 32);
    }

    #[test]
    fn norm_values_full() {
        let plan = full_plan(4, 2, 8);
        assert_eq!(plan.norm_values().len(), 32);
    }

    #[test]
    fn norm_values_linear() {
        let plan = linear_plan(2, 4, 16);
        assert_eq!(plan.norm_values().len(), 16);
    }

    // ── kv_head_count ───────────────────────────────────────────
    #[test]
    fn kv_head_count_standard() {
        let plan = standard_plan(8, 2, 16);
        assert_eq!(plan.kv_head_count(), 2);
    }

    #[test]
    fn kv_head_count_full() {
        let plan = full_plan(4, 2, 8);
        assert_eq!(plan.kv_head_count(), 2);
    }

    #[test]
    fn kv_head_count_linear_is_group_count() {
        let plan = linear_plan(3, 4, 16);
        assert_eq!(plan.kv_head_count(), 3);
    }

    // ── head_dimension ──────────────────────────────────────────
    #[test]
    fn head_dimension_standard() {
        let plan = standard_plan(4, 2, 16);
        assert_eq!(plan.head_dimension(), 16);
    }

    #[test]
    fn head_dimension_full() {
        let plan = full_plan(4, 2, 8);
        assert_eq!(plan.head_dimension(), 8);
    }

    #[test]
    fn head_dimension_linear_is_state_size() {
        let plan = linear_plan(2, 7, 16);
        assert_eq!(plan.head_dimension(), 7);
    }

    // ── is_standard ─────────────────────────────────────────────
    #[test]
    fn is_standard_true() {
        assert!(standard_plan(4, 2, 8).is_standard());
    }

    #[test]
    fn is_standard_false_for_full() {
        assert!(!full_plan(4, 2, 8).is_standard());
    }

    #[test]
    fn is_standard_false_for_linear() {
        assert!(!linear_plan(2, 4, 16).is_standard());
    }

    // ── conv_channels ───────────────────────────────────────────
    #[test]
    fn conv_channels_basic() {
        // inner_size + 2 × group_count × state_size = 8 + 2×2×4 = 24
        if let AttentionLayerPlan::Qwen35Linear(lin) = linear_plan(2, 4, 8) {
            assert_eq!(lin.conv_channels().unwrap(), 24);
        } else {
            panic!("expected Qwen35Linear");
        }
    }

    #[test]
    fn conv_channels_qwen35_dimensions() {
        // inner_size=3584, group_count=4, state_size=128
        // → 3584 + 2×4×128 = 3584 + 1024 = 4608
        if let AttentionLayerPlan::Qwen35Linear(lin) = linear_plan(4, 128, 3584) {
            assert_eq!(lin.conv_channels().unwrap(), 4608);
        } else {
            panic!("expected Qwen35Linear");
        }
    }

    #[test]
    fn conv_channels_overflow() {
        let plan = Qwen35LinearAttentionLayerPlan {
            norm_values: vec![],
            qkv_weight_values: vec![],
            gate_weight_values: vec![],
            alpha_weight_values: vec![],
            beta_weight_values: vec![],
            conv_weight_values: vec![],
            dt_bias_values: vec![],
            ssm_a_values: vec![],
            ssm_norm_values: vec![],
            ssm_out_weight_values: vec![],
            state_size: usize::MAX,
            group_count: usize::MAX,
            time_step_rank: 1,
            inner_size: 0,
            conv_kernel: 4,
        };
        assert!(plan.conv_channels().is_err());
    }
}
