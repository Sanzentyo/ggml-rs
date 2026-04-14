//! Inference graph benchmarks: full attention, linear attention, MLP, and combined.

use super::super::attention::qwen35_full_attention_inference;
use super::super::linear_attention::qwen35_linear_attention_inference;
use super::super::mlp::mlp_sequence_inference_with_weights;
use super::helpers::{
    BenchResult, available_backends, bench_fn, build_full_attention_plan,
    build_linear_attention_plan, build_mlp_weights, print_results, synthetic_input,
};

/// Benchmark the fully-fused full attention graph (norm -> proj -> deinterleave
/// -> QK norm -> RoPE -> flash_attn -> gate -> output proj).
#[test]
#[ignore]
fn bench_e2e_graphs_full_attention() {
    let warmup = 3;
    let iters = 20;
    let backends = available_backends();
    let mut results = Vec::new();

    // Small scale (test-size).
    let (hidden, heads, kv_heads, hd) = (6, 2, 1, 4);
    let plan_small = build_full_attention_plan(hidden, heads, kv_heads, hd);
    for seq_len in [4, 16] {
        let input = synthetic_input(hidden, seq_len);
        let norm_w = &plan_small.norm_values;
        for &(bname, ref backend) in &backends {
            let avg = bench_fn(warmup, iters, || {
                qwen35_full_attention_inference(&plan_small, &input, seq_len, 1e-5, norm_w, backend)
            });
            results.push(BenchResult {
                label: "full_attention_fused",
                scale: "small",
                seq_len,
                backend_name: bname,
                avg_ms: avg,
            });
        }
    }

    // Realistic scale (Qwen3.5 0.6B dimensions).
    let (hidden, heads, kv_heads, hd) = (1536, 8, 4, 128);
    let plan_large = build_full_attention_plan(hidden, heads, kv_heads, hd);
    for seq_len in [1, 4, 16, 64] {
        let input = synthetic_input(hidden, seq_len);
        let norm_w = &plan_large.norm_values;
        for &(bname, ref backend) in &backends {
            let avg = bench_fn(warmup, iters, || {
                qwen35_full_attention_inference(&plan_large, &input, seq_len, 1e-5, norm_w, backend)
            });
            results.push(BenchResult {
                label: "full_attention_fused",
                scale: "qwen35",
                seq_len,
                backend_name: bname,
                avg_ms: avg,
            });
        }
    }

    print_results(&results);
}

/// Benchmark the fused linear attention graph (norm -> 4 projections -> conv -> silu)
/// plus host-side SSM recurrence.
#[test]
#[ignore]
fn bench_e2e_graphs_linear_attention() {
    let warmup = 3;
    let iters = 20;
    let backends = available_backends();
    let mut results = Vec::new();

    // Small scale.
    let (hidden, inner, gc, tsr, ss, ck) = (8, 8, 2, 4, 2, 2);
    let plan_small = build_linear_attention_plan(hidden, inner, gc, tsr, ss, ck);
    for seq_len in [4, 16] {
        let input = synthetic_input(hidden, seq_len);
        let norm_w = &plan_small.norm_values;
        for &(bname, ref backend) in &backends {
            let avg = bench_fn(warmup, iters, || {
                qwen35_linear_attention_inference(
                    &plan_small,
                    &input,
                    seq_len,
                    1e-5,
                    norm_w,
                    backend,
                )
            });
            results.push(BenchResult {
                label: "linear_attention_fused",
                scale: "small",
                seq_len,
                backend_name: bname,
                avg_ms: avg,
            });
        }
    }

    // Realistic scale (Qwen3.5 0.6B: inner_size=1536, group_count=4,
    // time_step_rank=96, state_size=16, conv_kernel=4).
    let (hidden, inner, gc, tsr, ss, ck) = (1536, 1536, 4, 96, 16, 4);
    let plan_large = build_linear_attention_plan(hidden, inner, gc, tsr, ss, ck);
    for seq_len in [1, 4, 16, 64] {
        let input = synthetic_input(hidden, seq_len);
        let norm_w = &plan_large.norm_values;
        for &(bname, ref backend) in &backends {
            let avg = bench_fn(warmup, iters, || {
                qwen35_linear_attention_inference(
                    &plan_large,
                    &input,
                    seq_len,
                    1e-5,
                    norm_w,
                    backend,
                )
            });
            results.push(BenchResult {
                label: "linear_attention_fused",
                scale: "qwen35",
                seq_len,
                backend_name: bname,
                avg_ms: avg,
            });
        }
    }

    print_results(&results);
}

/// Benchmark linear attention prefill (with state capture).
///
/// This exercises the optimized path: tail-only `qkv_pre_conv` readback
/// (item 51) and direct SSM state write (item 52).
#[test]
#[ignore]
fn bench_e2e_graphs_linear_attention_prefill() {
    use super::super::linear_attention::{
        linear_attention_conv_channels, qwen35_linear_attention_prefill,
    };
    use super::super::state::LinearAttentionState;

    let warmup = 3;
    let iters = 20;
    let backends = available_backends();
    let mut results = Vec::new();

    // Realistic scale (Qwen3.5 0.6B).
    let (hidden, inner, gc, tsr, ss, ck) = (1536, 1536, 4, 96, 16, 4);
    let plan = build_linear_attention_plan(hidden, inner, gc, tsr, ss, ck);
    let conv_channels =
        linear_attention_conv_channels(&plan).expect("conv_channels should be computable");

    for seq_len in [1, 4, 16, 64, 256] {
        let input = synthetic_input(hidden, seq_len);
        let norm_w = &plan.norm_values;
        for &(bname, ref backend) in &backends {
            let avg = bench_fn(warmup, iters, || {
                let mut state = LinearAttentionState::new(ck, conv_channels, tsr, ss)
                    .expect("state creation should succeed");
                qwen35_linear_attention_prefill(
                    &plan, &input, seq_len, 1e-5, norm_w, &mut state, backend,
                )
            });
            results.push(BenchResult {
                label: "linear_attn_prefill",
                scale: "qwen35",
                seq_len,
                backend_name: bname,
                avg_ms: avg,
            });
        }
    }

    print_results(&results);
}

/// Phase-breakdown comparison: QKV projection + conv vs QK split vs SSM
/// recurrence vs output projection.
///
/// Isolates the four major phases of linear attention prefill to show where
/// wall-clock time is spent at realistic (Qwen3.5 0.6B) dimensions.
#[test]
#[ignore]
fn bench_e2e_linear_attention_phase_breakdown() {
    use super::super::linear_attention::{
        bench_linear_attention_phases, linear_attention_conv_channels,
    };
    use super::super::state::LinearAttentionState;

    let warmup = 3;
    let iters = 10;
    let backends = available_backends();

    // Qwen3.5 0.6B dimensions.
    let (hidden, inner, gc, tsr, ss, ck) = (1536, 1536, 4, 96, 16, 4);
    let plan = build_linear_attention_plan(hidden, inner, gc, tsr, ss, ck);
    let conv_channels =
        linear_attention_conv_channels(&plan).expect("conv_channels should be computable");

    println!();
    println!(
        "{:<16} {:>8} {:>10} {:>12} {:>12} {:>12} {:>12}",
        "Backend", "SeqLen", "Total(ms)", "Proj+Conv", "QK Split", "SSM Recur", "OutProj"
    );
    println!("{}", "-".repeat(90));

    for seq_len in [1, 4, 16, 64, 256] {
        let input = synthetic_input(hidden, seq_len);
        let norm_w = &plan.norm_values;
        for &(bname, ref backend) in &backends {
            // Warmup.
            for _ in 0..warmup {
                let mut state =
                    LinearAttentionState::new(ck, conv_channels, tsr, ss).expect("state creation");
                let _ = bench_linear_attention_phases(
                    &plan, &input, seq_len, 1e-5, norm_w, &mut state, backend,
                );
            }
            // Measure.
            let mut totals = [0.0_f64; 4];
            for _ in 0..iters {
                let mut state =
                    LinearAttentionState::new(ck, conv_channels, tsr, ss).expect("state creation");
                let t = bench_linear_attention_phases(
                    &plan, &input, seq_len, 1e-5, norm_w, &mut state, backend,
                )
                .expect("phase bench should succeed");
                totals[0] += t.proj_conv_ms;
                totals[1] += t.qk_split_norm_ms;
                totals[2] += t.ssm_recurrence_ms;
                totals[3] += t.output_proj_ms;
            }
            let n = iters as f64;
            let avgs: Vec<f64> = totals.iter().map(|t| t / n).collect();
            let total: f64 = avgs.iter().sum();
            println!(
                "{:<16} {:>8} {:>10.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3}",
                bname, seq_len, total, avgs[0], avgs[1], avgs[2], avgs[3]
            );
        }
    }
}

/// Benchmark the fused MLP graph (norm -> gate -> up -> silu -> mul -> down).
#[test]
#[ignore]
fn bench_e2e_graphs_mlp() {
    let warmup = 3;
    let iters = 20;
    let backends = available_backends();
    let mut results = Vec::new();

    // Small scale.
    let (hidden, ffn) = (8, 16);
    let weights_small = build_mlp_weights(hidden, ffn);
    let norm_small = vec![1.0_f32; hidden];
    for seq_len in [4, 16] {
        let input = synthetic_input(hidden, seq_len);
        for &(bname, ref backend) in &backends {
            let avg = bench_fn(warmup, iters, || {
                mlp_sequence_inference_with_weights(
                    &weights_small,
                    &input,
                    seq_len,
                    &norm_small,
                    1e-5,
                    backend,
                )
            });
            results.push(BenchResult {
                label: "mlp_fused",
                scale: "small",
                seq_len,
                backend_name: bname,
                avg_ms: avg,
            });
        }
    }

    // Realistic scale (Qwen3.5 0.6B: hidden=1536, ffn=8960).
    let (hidden, ffn) = (1536, 8960);
    let weights_large = build_mlp_weights(hidden, ffn);
    let norm_large = vec![1.0_f32; hidden];
    for seq_len in [1, 4, 16, 64] {
        let input = synthetic_input(hidden, seq_len);
        for &(bname, ref backend) in &backends {
            let avg = bench_fn(warmup, iters, || {
                mlp_sequence_inference_with_weights(
                    &weights_large,
                    &input,
                    seq_len,
                    &norm_large,
                    1e-5,
                    backend,
                )
            });
            results.push(BenchResult {
                label: "mlp_fused",
                scale: "qwen35",
                seq_len,
                backend_name: bname,
                avg_ms: avg,
            });
        }
    }

    print_results(&results);
}

/// Combined benchmark: run all graph types at Qwen3.5 scale, both backends.
#[test]
#[ignore]
fn bench_e2e_graphs_combined() {
    let warmup = 3;
    let iters = 20;
    let backends = available_backends();
    let mut results = Vec::new();

    // Full attention (Qwen3.5 0.6B).
    let full_plan = build_full_attention_plan(1536, 8, 4, 128);
    // Linear attention (Qwen3.5 0.6B).
    let linear_plan = build_linear_attention_plan(1536, 1536, 4, 96, 16, 4);
    // MLP (Qwen3.5 0.6B).
    let mlp_weights = build_mlp_weights(1536, 8960);
    let mlp_norm = vec![1.0_f32; 1536];

    for seq_len in [1, 4, 16, 64] {
        let input = synthetic_input(1536, seq_len);
        for &(bname, ref backend) in &backends {
            // Full attention.
            let avg = bench_fn(warmup, iters, || {
                qwen35_full_attention_inference(
                    &full_plan,
                    &input,
                    seq_len,
                    1e-5,
                    &full_plan.norm_values,
                    backend,
                )
            });
            results.push(BenchResult {
                label: "full_attention_fused",
                scale: "qwen35",
                seq_len,
                backend_name: bname,
                avg_ms: avg,
            });

            // Linear attention.
            let avg = bench_fn(warmup, iters, || {
                qwen35_linear_attention_inference(
                    &linear_plan,
                    &input,
                    seq_len,
                    1e-5,
                    &linear_plan.norm_values,
                    backend,
                )
            });
            results.push(BenchResult {
                label: "linear_attention_fused",
                scale: "qwen35",
                seq_len,
                backend_name: bname,
                avg_ms: avg,
            });

            // MLP.
            let avg = bench_fn(warmup, iters, || {
                mlp_sequence_inference_with_weights(
                    &mlp_weights,
                    &input,
                    seq_len,
                    &mlp_norm,
                    1e-5,
                    backend,
                )
            });
            results.push(BenchResult {
                label: "mlp_fused",
                scale: "qwen35",
                seq_len,
                backend_name: bname,
                avg_ms: avg,
            });
        }
    }

    print_results(&results);
}
