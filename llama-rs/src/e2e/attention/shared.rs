//! Shared helpers for attention modules: RoPE utilities and flash-attention pipeline.

use crate::e2e::error::E2eError;
use crate::e2e::numeric::checked_mul;
use ggml_rs::{Context, DynTensor, Tensor};

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
    let q_4d = ctx
        .reshape_4d(q, d, h, t, 1)
        .map_err(|source| E2eError::ggml("reshape_4d(Q)", source))?;
    let k_4d = ctx
        .reshape_4d(k, d, hkv, t, 1)
        .map_err(|source| E2eError::ggml("reshape_4d(K)", source))?;
    let v_3d = ctx
        .reshape_3d(v, d, hkv, t)
        .map_err(|source| E2eError::ggml("reshape_3d(V)", source))?;
    let v_4d = ctx
        .reshape_4d(&v_3d, d, hkv, t, 1)
        .map_err(|source| E2eError::ggml("reshape_4d(V)", source))?;

    // Permute [D, H, T, 1] → [D, T, H, 1] + cont for flash_attn_ext.
    let q_perm = ctx
        .permute(&q_4d, 0, 2, 1, 3)
        .map_err(|source| E2eError::ggml("permute(Q)", source))?;
    let k_perm = ctx
        .permute(&k_4d, 0, 2, 1, 3)
        .map_err(|source| E2eError::ggml("permute(K)", source))?;
    let v_perm = ctx
        .permute(&v_4d, 0, 2, 1, 3)
        .map_err(|source| E2eError::ggml("permute(V)", source))?;

    let q_c = ctx
        .cont(&q_perm)
        .map_err(|source| E2eError::ggml("cont(Q)", source))?;
    let k_c = ctx
        .cont(&k_perm)
        .map_err(|source| E2eError::ggml("cont(K)", source))?;
    let v_c = ctx
        .cont(&v_perm)
        .map_err(|source| E2eError::ggml("cont(V)", source))?;

    // Flash attention.
    let attn = ctx
        .flash_attn_ext(&q_c, &k_c, &v_c, mask, attention_scale, 0.0, 0.0)
        .map_err(|source| E2eError::ggml("flash_attn_ext", source))?; // [D, H, T, 1]

    // Optional gating: sigmoid(gate) ⊙ attn (Qwen3.5 full attention).
    let scored = if let Some(gate) = gate {
        let gate_4d = ctx
            .reshape_4d(gate, d, h, t, 1)
            .map_err(|source| E2eError::ggml("reshape_4d(Gate)", source))?;
        let gate_sig = ctx
            .sigmoid(&gate_4d)
            .map_err(|source| E2eError::ggml("sigmoid(Gate)", source))?;
        ctx.mul(&attn, &gate_sig)
            .map_err(|source| E2eError::ggml("mul(attn, gate)", source))?
    } else {
        attn
    };

    // Output projection: [D, H, T, 1] → [H*D, T] → mul_mat(W_out) → [hidden, T].
    let scored_2d = ctx
        .reshape_2d(&scored, qf, t)
        .map_err(|source| E2eError::ggml("reshape_2d(scored)", source))?;
    ctx.mul_mat(w_out, &scored_2d)
        .map_err(|source| E2eError::ggml("mul_mat(output)", source))
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
            let normed = ctx
                .rms_norm(&x, rms_norm_eps)
                .map_err(|source| E2eError::ggml(label_norm, source))?;
            ctx.mul(&normed, nw)
                .map_err(|source| E2eError::ggml(label_mul, source))
        }
        None => Ok(x),
    }
}
