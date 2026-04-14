//! Shared helpers for attention modules: KV cache access, RoPE utilities,
//! host-side attention scoring, and flash-attention pipeline.

use crate::e2e::error::{E2eError, GgmlResultExt};
use crate::e2e::numeric::{checked_mul, dot, softmax_prefix};
use crate::e2e::state::{Qwen35FullAttentionState, StandardAttentionState};
use ggml_rs::{Context, DynTensor, Length, Shape2D, Tensor};

// ---------------------------------------------------------------------------
// GQA head validation
// ---------------------------------------------------------------------------

/// Validate that a GQA head configuration is valid: both counts must be
/// positive and `head_count` must be a multiple of `kv_head_count`.
pub(in crate::e2e) fn validate_gqa_heads(
    head_count: usize,
    kv_head_count: usize,
) -> Result<(), E2eError> {
    if head_count == 0 || kv_head_count == 0 || !head_count.is_multiple_of(kv_head_count) {
        return Err(E2eError::InvalidGqaHeadConfig {
            head_count,
            kv_head_count,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Graph norm input (shared context + RMS-norm boilerplate)
// ---------------------------------------------------------------------------

/// Tensors produced by [`graph_norm_input`]: the raw input, norm weight,
/// and the pre-normed result that downstream projections consume.
pub(in crate::e2e) struct NormInput<'ctx> {
    /// Un-normed input `[hidden × seq_len]` — caller uploads data here.
    pub(in crate::e2e) x_raw: Tensor<'ctx, f32>,
    /// RMS-norm weight `[hidden]` — caller uploads the layer norm weight.
    pub(in crate::e2e) norm_w: Tensor<'ctx, f32>,
    /// Normed result: `rms_norm(x_raw, eps) * norm_w`.
    pub(in crate::e2e) x: Tensor<'ctx, f32>,
}

/// Create the raw input tensor, norm weight tensor, and pre-normed input
/// within an already-allocated ggml `Context`.
pub(in crate::e2e) fn graph_norm_input<'ctx>(
    ctx: &'ctx Context,
    hidden: usize,
    seq_len: usize,
    rms_norm_eps: f32,
) -> Result<NormInput<'ctx>, E2eError> {
    let x_raw = ctx
        .new_tensor_2d::<f32>(Shape2D::new(hidden, seq_len))
        .ggml_ctx("new<X>")?;
    let norm_w = ctx
        .new_tensor_1d::<f32>(Length::new(hidden))
        .ggml_ctx("new<attn_norm_w>")?;
    let x_normed = ctx.rms_norm(&x_raw, rms_norm_eps).ggml_ctx("rms_norm(X)")?;
    let x = ctx.mul(&x_normed, &norm_w).ggml_ctx("mul(X_norm)")?;
    Ok(NormInput { x_raw, norm_w, x })
}

// ---------------------------------------------------------------------------
// KV cache read-only access (for host-side attention scoring)
// ---------------------------------------------------------------------------

/// Read-only view into a KV cache for host-side attention scoring.
///
/// Implemented by both `StandardAttentionState` and `Qwen35FullAttentionState`,
/// allowing the scoring loop to be shared across attention variants.
pub(in crate::e2e) trait KvCacheView {
    /// Get a K slice for a specific token and KV head.
    fn k_head_at(&self, token: usize, kv_head: usize, head_dim: usize) -> &[f32];
    /// Get a V slice for a specific token and KV head.
    fn v_head_at(&self, token: usize, kv_head: usize, head_dim: usize) -> &[f32];
}

impl KvCacheView for StandardAttentionState {
    fn k_head_at(&self, token: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        self.k_head_at(token, kv_head, head_dim)
    }
    fn v_head_at(&self, token: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        self.v_head_at(token, kv_head, head_dim)
    }
}

impl KvCacheView for Qwen35FullAttentionState {
    fn k_head_at(&self, token: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        self.k_head_at(token, kv_head, head_dim)
    }
    fn v_head_at(&self, token: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        self.v_head_at(token, kv_head, head_dim)
    }
}

// ---------------------------------------------------------------------------
// Host-side attention scoring
// ---------------------------------------------------------------------------

/// Host-side multi-head attention scoring against a KV cache.
///
/// For each query head, scores all cached keys via dot-product, applies
/// softmax, and produces a weighted sum of values (grouped query attention).
/// Returns `head_outputs` of length `head_count * head_dimension`.
///
/// This is the fallback scoring path used by both standard and Qwen3.5 full
/// attention decode steps when GPU scoring is unavailable. Qwen3.5 applies
/// additional per-head sigmoid gating on the returned outputs.
pub(super) fn host_attention_scoring(
    q_values: &[f32],
    head_count: usize,
    kv_head_count: usize,
    head_dimension: usize,
    attention_scale: f32,
    total_tokens: usize,
    cache: &impl KvCacheView,
) -> Vec<f32> {
    debug_assert_eq!(
        q_values.len(),
        head_count * head_dimension,
        "q_values length must equal head_count * head_dimension"
    );
    debug_assert!(
        kv_head_count > 0 && head_count.is_multiple_of(kv_head_count),
        "head_count must be a positive multiple of kv_head_count"
    );

    let groups = head_count / kv_head_count;
    let query_features = head_count * head_dimension;
    let mut head_outputs = vec![0.0_f32; query_features];

    for head in 0..head_count {
        let kv_head = head / groups;
        let q = &q_values[head * head_dimension..(head + 1) * head_dimension];

        let mut scores = vec![f32::NEG_INFINITY; total_tokens];
        for (source, score) in scores.iter_mut().enumerate().take(total_tokens) {
            let k = cache.k_head_at(source, kv_head, head_dimension);
            *score = dot(q, k) * attention_scale;
        }
        let weights = softmax_prefix(&scores, total_tokens);

        let dst = &mut head_outputs[head * head_dimension..(head + 1) * head_dimension];
        for (source, weight) in weights.iter().copied().enumerate() {
            let v = cache.v_head_at(source, kv_head, head_dimension);
            for (index, d) in dst.iter_mut().enumerate().take(head_dimension) {
                *d += v[index] * weight;
            }
        }
    }
    head_outputs
}

// ---------------------------------------------------------------------------
// NeoX-style RoPE
// ---------------------------------------------------------------------------

/// NeoX-style RoPE configuration shared across Q and K rotations.
#[derive(Debug, Clone, Copy)]
pub(in crate::e2e) struct RopeParams {
    /// Number of dimensions to rotate (must be even, ≤ head_dimension).
    pub n_rot: usize,
    /// Base frequency for position encoding (e.g. 10000.0).
    pub freq_base: f32,
    /// Frequency scaling factor (typically 1.0).
    pub freq_scale: f32,
    /// Position offset for decode (0 for prefill, prompt_len for decode).
    pub position_offset: usize,
}

/// Apply NeoX-style rotary position embedding in-place.
///
/// For each token at position `rope.position_offset + pos`, rotates dimension pairs
/// `(x[k], x[k + n_rot/2])` for `k` in `0..n_rot/2` using angle
/// `theta_k = pos * freq_base^(-2k / n_rot)`.
/// Dimensions beyond `n_rot` are left unchanged.
pub(in crate::e2e) fn apply_neox_rope_in_place(
    values: &mut [f32],
    sequence_length: usize,
    head_count: usize,
    head_dimension: usize,
    rope: &RopeParams,
) -> Result<(), E2eError> {
    let total_features = checked_mul(head_count, head_dimension)?;
    let expected_len = checked_mul(sequence_length, total_features)?;
    if values.len() != expected_len {
        return Err(E2eError::BufferLengthMismatch {
            expected: expected_len,
            actual: values.len(),
        });
    }
    debug_assert!(rope.n_rot <= head_dimension && rope.n_rot.is_multiple_of(2));

    let half_rot = rope.n_rot / 2;
    let theta_scale = rope.freq_base.powf(-2.0 / rope.n_rot as f32);

    let cache_size = checked_mul(sequence_length, half_rot)?;
    let mut cos_cache = vec![0.0_f32; cache_size];
    let mut sin_cache = vec![0.0_f32; cache_size];
    for pos in 0..sequence_length {
        let mut theta = (rope.position_offset + pos) as f32;
        for k in 0..half_rot {
            let cache_idx = pos * half_rot + k;
            let angle = theta * rope.freq_scale;
            cos_cache[cache_idx] = angle.cos();
            sin_cache[cache_idx] = angle.sin();
            theta *= theta_scale;
        }
    }

    for pos in 0..sequence_length {
        let token_base = pos * total_features;
        for head in 0..head_count {
            let head_base = token_base + head * head_dimension;
            for k in 0..half_rot {
                let cache_idx = pos * half_rot + k;
                let cos_t = cos_cache[cache_idx];
                let sin_t = sin_cache[cache_idx];
                let idx0 = head_base + k;
                let idx1 = head_base + k + half_rot;
                let x0 = values[idx0];
                let x1 = values[idx1];
                values[idx0] = x0 * cos_t - x1 * sin_t;
                values[idx1] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared flash-attention pipeline helpers
// ---------------------------------------------------------------------------

/// Configuration for the shared flash-attention pipeline.
///
/// Carries only the dimensional and scalar parameters; the heavy tensors
/// (Q, K, V, mask, gate, output weight) are passed as arguments.
pub(super) struct FlashAttentionConfig {
    pub d: usize,
    pub h: usize,
    pub hkv: usize,
    pub t: usize,
    pub qf: usize,
    pub attention_scale: f32,
}

/// Shared flash-attention pipeline: reshape → permute → cont → flash_attn_ext
/// → optional gating → output projection.
///
/// Accepts Q `[D, H, T]`, K `[D, Hkv, T]`, V `[D, Hkv, T]` (already after
/// any per-head norm and RoPE), plus optional gate `[D, H, T]` and causal
/// mask `[T, T, 1, 1]`.  Returns the output tensor `[hidden, T]`.
pub(super) fn run_flash_attention_pipeline<'ctx>(
    ctx: &'ctx Context,
    cfg: &FlashAttentionConfig,
    qkv: (&Tensor<'ctx, f32>, &Tensor<'ctx, f32>, &Tensor<'ctx, f32>),
    mask: Option<&DynTensor<'ctx>>,
    gate: Option<&Tensor<'ctx, f32>>,
    w_out: &Tensor<'ctx, f32>,
) -> Result<Tensor<'ctx, f32>, E2eError> {
    let (q, k, v) = qkv;
    let FlashAttentionConfig {
        d,
        h,
        hkv,
        t,
        qf,
        attention_scale,
    } = *cfg;
    debug_assert_eq!(qf, d * h, "qf must equal d * h");

    // Reshape to 4D for flash_attn_ext: [D, H/Hkv, T] → [D, H/Hkv, T, 1].
    let q_4d = ctx.reshape_4d(q, d, h, t, 1).ggml_ctx("reshape_4d(Q)")?;
    let k_4d = ctx.reshape_4d(k, d, hkv, t, 1).ggml_ctx("reshape_4d(K)")?;
    let v_3d = ctx.reshape_3d(v, d, hkv, t).ggml_ctx("reshape_3d(V)")?;
    let v_4d = ctx
        .reshape_4d(&v_3d, d, hkv, t, 1)
        .ggml_ctx("reshape_4d(V)")?;

    // Permute [D, H, T, 1] → [D, T, H, 1] + cont for flash_attn_ext.
    let q_perm = ctx.permute(&q_4d, 0, 2, 1, 3).ggml_ctx("permute(Q)")?;
    let k_perm = ctx.permute(&k_4d, 0, 2, 1, 3).ggml_ctx("permute(K)")?;
    let v_perm = ctx.permute(&v_4d, 0, 2, 1, 3).ggml_ctx("permute(V)")?;

    let q_c = ctx.cont(&q_perm).ggml_ctx("cont(Q)")?;
    let k_c = ctx.cont(&k_perm).ggml_ctx("cont(K)")?;
    let v_c = ctx.cont(&v_perm).ggml_ctx("cont(V)")?;

    // Flash attention.
    let attn = ctx
        .flash_attn_ext(&q_c, &k_c, &v_c, mask, attention_scale, 0.0, 0.0)
        .ggml_ctx("flash_attn_ext")?; // [D, H, T, 1]

    // Optional gating: sigmoid(gate) ⊙ attn (Qwen3.5 full attention).
    let scored = if let Some(gate) = gate {
        let gate_4d = ctx
            .reshape_4d(gate, d, h, t, 1)
            .ggml_ctx("reshape_4d(Gate)")?;
        let gate_sig = ctx.sigmoid(&gate_4d).ggml_ctx("sigmoid(Gate)")?;
        ctx.mul(&attn, &gate_sig).ggml_ctx("mul(attn, gate)")?
    } else {
        attn
    };

    // Output projection: [D, H, T, 1] → [H*D, T] → mul_mat(W_out) → [hidden, T].
    let scored_2d = ctx
        .reshape_2d(&scored, qf, t)
        .ggml_ctx("reshape_2d(scored)")?;
    ctx.mul_mat(w_out, &scored_2d).ggml_ctx("mul_mat(output)")
}

/// Optional per-head RMS norm + weight scaling on a 3D tensor `[D, H, T]`.
///
/// If `norm_weight` is Some, applies `rms_norm(x, eps) * weight` per head.
/// Otherwise returns the input tensor unchanged.
pub(super) fn apply_optional_per_head_norm<'ctx>(
    ctx: &'ctx Context,
    x: Tensor<'ctx, f32>,
    norm_weight: Option<&Tensor<'ctx, f32>>,
    rms_norm_eps: f32,
    label_norm: &'static str,
    label_mul: &'static str,
) -> Result<Tensor<'ctx, f32>, E2eError> {
    match norm_weight {
        Some(nw) => {
            let normed = ctx.rms_norm(&x, rms_norm_eps).ggml_ctx(label_norm)?;
            ctx.mul(&normed, nw).ggml_ctx(label_mul)
        }
        None => Ok(x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::e2e::error::E2eError;

    #[test]
    fn validate_gqa_heads_standard_gqa() {
        assert!(validate_gqa_heads(8, 2).is_ok());
    }

    #[test]
    fn validate_gqa_heads_equal_heads() {
        assert!(validate_gqa_heads(4, 4).is_ok());
    }

    #[test]
    fn validate_gqa_heads_mqa_single_kv() {
        assert!(validate_gqa_heads(8, 1).is_ok());
    }

    #[test]
    fn validate_gqa_heads_zero_head_count() {
        match validate_gqa_heads(0, 4) {
            Err(E2eError::InvalidGqaHeadConfig {
                head_count,
                kv_head_count,
            }) => {
                assert_eq!(head_count, 0);
                assert_eq!(kv_head_count, 4);
            }
            other => panic!("expected InvalidGqaHeadConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_gqa_heads_zero_kv_head_count() {
        match validate_gqa_heads(8, 0) {
            Err(E2eError::InvalidGqaHeadConfig {
                head_count,
                kv_head_count,
            }) => {
                assert_eq!(head_count, 8);
                assert_eq!(kv_head_count, 0);
            }
            other => panic!("expected InvalidGqaHeadConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_gqa_heads_not_divisible() {
        match validate_gqa_heads(7, 3) {
            Err(E2eError::InvalidGqaHeadConfig {
                head_count,
                kv_head_count,
            }) => {
                assert_eq!(head_count, 7);
                assert_eq!(kv_head_count, 3);
            }
            other => panic!("expected InvalidGqaHeadConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_gqa_heads_both_zero() {
        assert!(matches!(
            validate_gqa_heads(0, 0),
            Err(E2eError::InvalidGqaHeadConfig { .. })
        ));
    }
}
