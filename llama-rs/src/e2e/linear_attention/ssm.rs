//! SSM recurrence logic and scratch buffers for linear attention.

use crate::e2e::error::E2eError;
use crate::e2e::numeric::checked_mul;
use crate::e2e::plan::Qwen35LinearAttentionLayerPlan;
use crate::e2e::tensor_ops::per_head_l2_norm;

/// Reusable scratch buffers for SSM recurrence, avoiding per-head allocation.
pub(super) struct SsmScratch {
    pub(super) sk: Vec<f32>,
    pub(super) delta: Vec<f32>,
    pub(super) out: Vec<f32>,
}

impl SsmScratch {
    pub(super) fn new(state_size: usize) -> Self {
        Self {
            sk: vec![0.0_f32; state_size],
            delta: vec![0.0_f32; state_size],
            out: vec![0.0_f32; state_size],
        }
    }

    pub(super) fn clear(&mut self) {
        self.sk.fill(0.0);
        self.delta.fill(0.0);
        self.out.fill(0.0);
    }
}

/// Reusable scratch buffers for [`super::linear_attention_decode_core`], avoiding
/// per-call and per-head heap allocations during autoregressive decode.
///
/// Created once per decode session and passed into every decode step.
/// Bundles the SSM recurrence scratch, the per-call output buffer, and a
/// temporary buffer for RMS normalization.
pub(in crate::e2e) struct LinearDecodeScratch {
    pub(super) ssm: SsmScratch,
    /// Per-call output accumulator, sized to `inner_size`.
    pub(super) output: Vec<f32>,
    /// Temporary for RMS norm result, sized to `state_size`.
    pub(super) norm_buf: Vec<f32>,
}

impl LinearDecodeScratch {
    /// Create scratch buffers for a linear attention layer with the given
    /// `state_size` and `inner_size`.
    pub(in crate::e2e) fn new(state_size: usize, inner_size: usize) -> Self {
        Self {
            ssm: SsmScratch::new(state_size),
            output: vec![0.0_f32; inner_size],
            norm_buf: vec![0.0_f32; state_size],
        }
    }
}

/// Single SSM recurrence step: decay → sk → delta → update → read.
///
/// Operates on one head's `state_size × state_size` state matrix. Results are
/// written into `scratch.out`; the caller is responsible for post-processing
/// (RMS norm, SiLU gating, etc.).
///
/// The loop order is preserved exactly to match the original implementation and
/// maintain floating-point parity.
#[allow(clippy::too_many_arguments)]
pub(super) fn ssm_recurrence_step(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    _z: &[f32],
    decay: f32,
    beta_value: f32,
    state_size: usize,
    scale: f32,
    scratch: &mut SsmScratch,
) {
    debug_assert_eq!(state.len(), state_size * state_size);
    debug_assert_eq!(q.len(), state_size);
    debug_assert_eq!(k.len(), state_size);
    debug_assert_eq!(v.len(), state_size);

    scratch.clear();

    // Decay state and accumulate sk (row-major order preserved).
    for (row, row_slice) in state.chunks_exact_mut(state_size).enumerate() {
        let k_row = k[row];
        for (col, s) in row_slice.iter_mut().enumerate() {
            *s *= decay;
            scratch.sk[col] += *s * k_row;
        }
    }

    // Delta = (v - sk) * beta
    scratch
        .delta
        .iter_mut()
        .zip(v.iter().zip(scratch.sk.iter()))
        .for_each(|(d, (&vi, &ski))| *d = (vi - ski) * beta_value);

    // State update: state[row][col] += k[row] * delta[col]
    for (row_slice, &k_row) in state.chunks_exact_mut(state_size).zip(k.iter()) {
        row_slice
            .iter_mut()
            .zip(scratch.delta.iter())
            .for_each(|(s, &d)| *s += k_row * d);
    }

    // Read output: row-major traversal for contiguous access + auto-vectorization.
    for row in 0..state_size {
        let qr_scaled = q[row] * scale;
        let row_slice = &state[row * state_size..(row + 1) * state_size];
        scratch
            .out
            .iter_mut()
            .zip(row_slice.iter())
            .for_each(|(o, &s)| *o += s * qr_scaled);
    }
}

/// Split conv output into Q, K, V regions and L2-normalize Q and K heads.
///
/// Returns `(q_heads, k_heads)` — the caller retains ownership of `conv` for
/// V slicing (avoids an extra copy).
pub(super) fn split_and_norm_qk(
    conv: &[f32],
    sequence_length: usize,
    attention: &Qwen35LinearAttentionLayerPlan,
    conv_channels: usize,
    rms_norm_eps: f32,
) -> Result<(Vec<f32>, Vec<f32>), E2eError> {
    let qk_features = checked_mul(attention.group_count, attention.state_size)?;
    let mut q_raw = vec![0.0_f32; checked_mul(sequence_length, qk_features)?];
    let mut k_raw = vec![0.0_f32; q_raw.len()];
    debug_assert_eq!(conv.len(), checked_mul(sequence_length, conv_channels)?);
    for ((conv_row, q_dst), k_dst) in conv
        .chunks_exact(conv_channels)
        .zip(q_raw.chunks_exact_mut(qk_features))
        .zip(k_raw.chunks_exact_mut(qk_features))
    {
        q_dst.copy_from_slice(&conv_row[..qk_features]);
        k_dst.copy_from_slice(&conv_row[qk_features..qk_features * 2]);
    }

    let q_heads = per_head_l2_norm(
        &q_raw,
        sequence_length,
        attention.group_count,
        attention.state_size,
        rms_norm_eps,
    )?;
    let k_heads = per_head_l2_norm(
        &k_raw,
        sequence_length,
        attention.group_count,
        attention.state_size,
        rms_norm_eps,
    )?;

    Ok((q_heads, k_heads))
}
