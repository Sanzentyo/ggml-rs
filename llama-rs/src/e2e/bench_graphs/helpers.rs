//! Shared infrastructure for e2e microbenchmarks.
//!
//! Contains warmup/measure helpers, result reporting, plan builders,
//! and synthetic input generators used by all bench submodules.

use super::super::plan::{Qwen35FullAttentionLayerPlan, Qwen35LinearAttentionLayerPlan};
use crate::inference::MlpWeights;
use ggml_rs::{Backend, BackendKind};
use std::time::Instant;

/// Warmup + measure pattern returning average milliseconds.
pub(super) fn bench_fn(
    warmup: usize,
    iters: usize,
    mut f: impl FnMut() -> Result<Vec<f32>, super::super::error::E2eError>,
) -> f64 {
    for _ in 0..warmup {
        f().expect("warmup should succeed");
    }
    let start = Instant::now();
    for _ in 0..iters {
        f().expect("bench iteration should succeed");
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1000.0 / iters as f64
}

pub(super) struct BenchResult {
    pub(super) label: &'static str,
    pub(super) scale: &'static str,
    pub(super) seq_len: usize,
    pub(super) backend_name: &'static str,
    pub(super) avg_ms: f64,
}

pub(super) fn print_results(results: &[BenchResult]) {
    println!("\n{:-<80}", "");
    println!(
        "{:<32} {:>6} {:>8} {:>8} {:>12}",
        "Graph", "Scale", "SeqLen", "Backend", "Avg (ms)"
    );
    println!("{:-<80}", "");
    for r in results {
        println!(
            "{:<32} {:>6} {:>8} {:>8} {:>12.3}",
            r.label, r.scale, r.seq_len, r.backend_name, r.avg_ms
        );
    }
    println!("{:-<80}", "");
}

pub(super) fn available_backends() -> Vec<(&'static str, Backend)> {
    crate::backend::ensure_backends_loaded();
    let mut backends = vec![];
    if let Ok(cpu) = Backend::new(BackendKind::Cpu) {
        backends.push(("CPU", cpu));
    }
    if let Ok(metal) = Backend::new(BackendKind::Metal) {
        backends.push(("Metal", metal));
    }
    backends
}

// -------------------------------------------------------------------
// Helpers to build synthetic plans at various scales.
// -------------------------------------------------------------------

pub(super) fn build_full_attention_plan(
    hidden: usize,
    head_count: usize,
    kv_head_count: usize,
    hd: usize,
) -> Qwen35FullAttentionLayerPlan {
    Qwen35FullAttentionLayerPlan::deterministic(hidden, head_count, kv_head_count, hd)
}

pub(super) fn build_linear_attention_plan(
    hidden: usize,
    inner_size: usize,
    group_count: usize,
    time_step_rank: usize,
    state_size: usize,
    conv_kernel: usize,
) -> Qwen35LinearAttentionLayerPlan {
    let conv_channels = inner_size + 2 * group_count * state_size;
    let mut conv_weight = vec![0.0_f32; conv_channels * conv_kernel];
    for ch in 0..conv_channels {
        conv_weight[ch * conv_kernel + (conv_kernel - 1)] = 1.0;
    }
    Qwen35LinearAttentionLayerPlan {
        norm_values: vec![1.0_f32; hidden],
        qkv_weight_values: (0..hidden * conv_channels)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
            .collect(),
        gate_weight_values: (0..hidden * inner_size)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.01)
            .collect(),
        alpha_weight_values: (0..hidden * time_step_rank)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.01)
            .collect(),
        beta_weight_values: (0..hidden * time_step_rank)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
            .collect(),
        conv_weight_values: conv_weight,
        dt_bias_values: vec![0.0_f32; time_step_rank],
        ssm_a_values: vec![-1.0_f32; time_step_rank],
        ssm_norm_values: vec![1.0_f32; state_size],
        ssm_out_weight_values: (0..inner_size * hidden)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.01)
            .collect(),
        state_size,
        group_count,
        time_step_rank,
        inner_size,
        conv_kernel,
    }
}

pub(super) fn build_mlp_weights(hidden: usize, ffn: usize) -> MlpWeights<f32> {
    let config = crate::inference::MlpInferenceConfig::new(hidden, ffn)
        .expect("MlpInferenceConfig construction should succeed");
    MlpWeights::deterministic(config)
}

pub(super) fn synthetic_input(features: usize, seq_len: usize) -> Vec<f32> {
    (0..features * seq_len)
        .map(|i| ((i + 5) % 19) as f32 * 0.125)
        .collect()
}
