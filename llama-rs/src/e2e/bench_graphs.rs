//! Microbenchmarks for fused e2e compute graphs (CPU vs Metal).
//!
//! These benchmarks measure the wall-clock time of each fused graph type at
//! two scales: a small "test" size for quick iteration, and a larger "realistic"
//! size matching Qwen3.5 (0.6B) model dimensions.
//!
//! Run with:
//! ```sh
//! GGML_RS_LIB_DIR=... DYLD_FALLBACK_LIBRARY_PATH=... \
//!   cargo test --workspace --features link-system --release \
//!   -- bench_e2e_graphs --nocapture --ignored
//! ```
//!
//! All benchmarks are `#[ignore]`d to keep normal `cargo test` fast.

#[cfg(test)]
mod tests {
    use super::super::attention::qwen35_full_attention_inference;
    use super::super::linear_attention::qwen35_linear_attention_inference;
    use super::super::mlp::mlp_sequence_inference_with_weights;
    use super::super::plan::{Qwen35FullAttentionLayerPlan, Qwen35LinearAttentionLayerPlan};
    use crate::inference::MlpWeights;
    use ggml_rs::{Backend, BackendKind};
    use std::time::Instant;

    /// Warmup + measure pattern returning average milliseconds.
    fn bench_fn(
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

    // -------------------------------------------------------------------
    // Helpers to build synthetic plans at various scales.
    // -------------------------------------------------------------------

    fn build_full_attention_plan(
        hidden: usize,
        head_count: usize,
        kv_head_count: usize,
        hd: usize,
    ) -> Qwen35FullAttentionLayerPlan {
        let query_features = head_count * hd;
        let kv_features = kv_head_count * hd;
        Qwen35FullAttentionLayerPlan {
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

    fn build_linear_attention_plan(
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

    fn build_mlp_weights(hidden: usize, ffn: usize) -> MlpWeights<f32> {
        let config = crate::inference::MlpInferenceConfig::new(hidden, ffn)
            .expect("MlpInferenceConfig construction should succeed");
        MlpWeights::deterministic(config)
    }

    fn synthetic_input(features: usize, seq_len: usize) -> Vec<f32> {
        (0..features * seq_len)
            .map(|i| ((i + 5) % 19) as f32 * 0.125)
            .collect()
    }

    // -------------------------------------------------------------------
    // Benchmark driver
    // -------------------------------------------------------------------

    struct BenchResult {
        label: &'static str,
        scale: &'static str,
        seq_len: usize,
        backend_name: &'static str,
        avg_ms: f64,
    }

    fn print_results(results: &[BenchResult]) {
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

    fn available_backends() -> Vec<(&'static str, Backend)> {
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
    // Tests (ignored by default; run with `--ignored --nocapture`)
    // -------------------------------------------------------------------

    /// Benchmark the fully-fused full attention graph (norm → proj → deinterleave
    /// → QK norm → RoPE → flash_attn → gate → output proj).
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
                    qwen35_full_attention_inference(
                        &plan_small,
                        &input,
                        seq_len,
                        1e-5,
                        norm_w,
                        backend,
                    )
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
                    qwen35_full_attention_inference(
                        &plan_large,
                        &input,
                        seq_len,
                        1e-5,
                        norm_w,
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
            }
        }

        print_results(&results);
    }

    /// Benchmark the fused linear attention graph (norm → 4 projections → conv → silu)
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
                    let mut state = LinearAttentionState::new(ck, conv_channels, tsr, ss)
                        .expect("state creation");
                    let _ = bench_linear_attention_phases(
                        &plan, &input, seq_len, 1e-5, norm_w, &mut state, backend,
                    );
                }
                // Measure.
                let mut totals = [0.0_f64; 4];
                for _ in 0..iters {
                    let mut state = LinearAttentionState::new(ck, conv_channels, tsr, ss)
                        .expect("state creation");
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

    /// Benchmark the fused MLP graph (norm → gate → up → silu → mul → down).
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

    // -------------------------------------------------------------------
    // LM head benchmark: host-side vs one-shot graph vs persistent graph
    // -------------------------------------------------------------------

    #[test]
    #[ignore]
    fn bench_lm_head_qwen35() {
        use super::super::generation::greedy_next_token_id;
        use super::super::tensor_ops::{
            build_lm_head_graph, lm_head_graph, lm_head_sample_step, recommended_lm_head_memory,
            rms_norm_with_weight,
        };
        use ggml_rs::Context;

        crate::backend::ensure_backends_loaded();

        let hidden = 1536_usize;
        let vocab = 151936_usize;
        let eps = 1e-5_f32;
        let warmup = 2;
        let iters = 10;

        let norm_weight: Vec<f32> = (0..hidden).map(|i| 0.8 + (i as f32) * 0.0001).collect();
        let output_weight: Vec<f32> = (0..hidden * vocab)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let hidden_state: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let mut results: Vec<BenchResult> = Vec::new();

        // 1. Host-side: rms_norm + greedy_next_token_id
        let avg = bench_fn(warmup, iters, || {
            let normed = rms_norm_with_weight(&hidden_state, hidden, 1, &norm_weight, eps)?;
            let _tok = greedy_next_token_id(&normed, 0, hidden, &output_weight, vocab)?;
            Ok(normed)
        });
        results.push(BenchResult {
            label: "lm_head_host",
            scale: "qwen35",
            seq_len: 1,
            backend_name: "host",
            avg_ms: avg,
        });

        // Backend benchmarks
        let backends: Vec<(&str, Backend)> = {
            let mut v = vec![("cpu", Backend::new(BackendKind::Cpu).expect("CPU"))];
            if let Ok(metal) = Backend::new(BackendKind::Metal) {
                v.push(("metal", metal));
            }
            v
        };

        for &(bname, ref backend) in &backends {
            // 2. One-shot graph (cold: allocate + upload + compute each iteration)
            let avg = bench_fn(warmup, iters, || {
                lm_head_graph(
                    &hidden_state,
                    &norm_weight,
                    &output_weight,
                    hidden,
                    vocab,
                    eps,
                    backend,
                )
            });
            results.push(BenchResult {
                label: "lm_head_graph_cold",
                scale: "qwen35",
                seq_len: 1,
                backend_name: bname,
                avg_ms: avg,
            });

            // 3. Persistent graph (warm: weights pre-uploaded, measure step only)
            let ctx_size = recommended_lm_head_memory(hidden, vocab).expect("mem");
            let ctx = Context::new_no_alloc_bytes(ctx_size).expect("ctx");
            let mut parts = build_lm_head_graph(&ctx, hidden, vocab, eps).expect("build");
            let _buf = ctx.allocate_tensors(backend).expect("alloc");
            parts
                .w_out
                .write_data_backend(&output_weight)
                .expect("write w_out");
            parts
                .norm_w
                .write_data_backend(&norm_weight)
                .expect("write norm_w");

            let avg = bench_fn(warmup, iters, || {
                let tok = lm_head_sample_step(
                    &hidden_state,
                    &parts.x_in,
                    &parts.logits,
                    &mut parts.graph,
                    backend,
                )?;
                Ok(vec![tok as f32])
            });
            results.push(BenchResult {
                label: "lm_head_graph_warm",
                scale: "qwen35",
                seq_len: 1,
                backend_name: bname,
                avg_ms: avg,
            });
        }

        print_results(&results);
    }

    // -------------------------------------------------------------------
    // Item 57: Causal Depthwise Conv vs QKV Packing Micro-Benchmarks
    //
    // Compares packed QKV (single matmul) vs separate Q/K/V projections,
    // and host conv vs graph conv (ssm_conv), with layout prep overhead.
    //
    // Run with:
    //   cargo test --workspace --features link-system --release \
    //     -- bench_conv_vs_qkv --nocapture --ignored
    // -------------------------------------------------------------------

    /// Build a single-graph benchmark for N parallel projections from shared input.
    ///
    /// Returns average ms for: allocate + upload + compute + read.
    fn bench_batch_projections(
        input: &[f32],
        seq_len: usize,
        hidden: usize,
        out_features_list: &[usize],
        warmup: usize,
        iters: usize,
        backend: &Backend,
    ) -> f64 {
        use super::super::tensor_ops::{ProjectionSpec, build_batch_projections, upload_weight};
        use ggml_rs::{Bytes, Context, Shape2D};

        // Pre-generate deterministic weight data for each projection.
        let weights: Vec<Vec<f32>> = out_features_list
            .iter()
            .enumerate()
            .map(|(idx, &out_f)| {
                (0..hidden * out_f)
                    .map(|i| ((i + idx * 37) % 19) as f32 * 0.01)
                    .collect()
            })
            .collect();

        let specs: Vec<ProjectionSpec> = out_features_list
            .iter()
            .enumerate()
            .map(|(idx, &out_f)| ProjectionSpec {
                weight_label: match idx {
                    0 => "bench<W0>",
                    1 => "bench<W1>",
                    2 => "bench<W2>",
                    _ => "bench<WN>",
                },
                matmul_label: match idx {
                    0 => "bench<mm0>",
                    1 => "bench<mm1>",
                    2 => "bench<mm2>",
                    _ => "bench<mmN>",
                },
                out_features: out_f,
            })
            .collect();

        bench_fn(warmup, iters, || {
            // Estimate memory: sum of matmul memories.
            let mut total_bytes = 0_usize;
            for &out_f in out_features_list {
                let mem = Context::recommended_backend_matmul_memory::<f32>(
                    Shape2D::new(hidden, out_f),
                    Shape2D::new(hidden, seq_len),
                )
                .expect("mem estimate");
                total_bytes += mem.get();
            }
            total_bytes += 4 * 1024 * 1024; // slack

            let ctx = Context::new_no_alloc_bytes(Bytes::new(total_bytes)).expect("ctx");
            let x_in = ctx
                .new_tensor_2d::<f32>(Shape2D::new(hidden, seq_len))
                .expect("x_in");
            let projs = build_batch_projections(&ctx, &x_in, hidden, &specs)
                .expect("build_batch_projections");

            let mut graph = ctx.new_graph().expect("graph");
            for p in &projs {
                graph.build_forward_expand(&p.y);
            }
            let _buf = ctx.allocate_tensors(backend).expect("alloc");

            upload_weight(&x_in, input, "bench<X>").expect("upload X");
            for (p, w_data) in projs.iter().zip(weights.iter()) {
                upload_weight(&p.w, w_data, "bench<W>").expect("upload W");
            }

            backend.compute(&mut graph).expect("compute");

            // Read last projection output as representative result.
            projs
                .last()
                .unwrap()
                .y
                .read_data_backend()
                .map_err(|e| super::super::error::E2eError::ggml("read", e))
        })
    }

    /// Build a layout-prep-only benchmark: transpose + cont + pad.
    ///
    /// Measures the overhead of converting from channels-fast [conv_channels, seq_len]
    /// to the padded time-fast layout [padded_len, conv_channels, 1] needed by ssm_conv.
    fn bench_layout_prep(
        seq_len: usize,
        conv_channels: usize,
        kernel_size: usize,
        warmup: usize,
        iters: usize,
        backend: &Backend,
    ) -> f64 {
        use super::super::tensor_ops::upload_weight;
        use ggml_rs::{Bytes, Context, Shape2D};

        let pad = kernel_size.saturating_sub(1);

        // Input: channels-fast layout [conv_channels, seq_len] (as if from mul_mat output).
        let input_data: Vec<f32> = (0..conv_channels * seq_len)
            .map(|i| (i % 23) as f32 * 0.05)
            .collect();
        let zero_data = vec![0.0_f32; pad * conv_channels];

        bench_fn(warmup, iters, || {
            let tensor_bytes = std::mem::size_of::<f32>()
                * (conv_channels * seq_len + pad * conv_channels + (pad + seq_len) * conv_channels);
            let ctx = Context::new_no_alloc_bytes(Bytes::new(tensor_bytes + 4 * 1024 * 1024))
                .expect("ctx");

            let qkv_out = ctx
                .new_tensor_2d::<f32>(Shape2D::new(conv_channels, seq_len))
                .expect("qkv_out");
            let qkv_t = ctx.transpose(&qkv_out).expect("transpose");
            let qkv_cont = ctx.cont(&qkv_t).expect("cont");

            let result = if pad > 0 {
                let zeros = ctx
                    .new_tensor_2d::<f32>(Shape2D::new(pad, conv_channels))
                    .expect("zeros");
                let padded = ctx.concat(&zeros, &qkv_cont, 0).expect("concat");
                ctx.reshape_3d(&padded, pad + seq_len, conv_channels, 1)
                    .expect("reshape")
            } else {
                ctx.reshape_3d(&qkv_cont, seq_len, conv_channels, 1)
                    .expect("reshape")
            };

            let mut graph = ctx.new_graph().expect("graph");
            graph.build_forward_expand(&result);
            let _buf = ctx.allocate_tensors(backend).expect("alloc");

            upload_weight(&qkv_out, &input_data, "bench<qkv>").expect("upload");
            if pad > 0 {
                // zeros tensor is the 2nd tensor created; find via its data size.
                // Since we can't easily reference it after reshape, just rely on
                // the allocator having zero-initialized the buffer.
                // (The benchmark measures layout ops, not data correctness.)
            }
            let _ = &zero_data; // keep alive for reference

            backend.compute(&mut graph).expect("compute");

            result
                .read_data_backend()
                .map_err(|e| super::super::error::E2eError::ggml("read", e))
        })
    }

    /// Item 57: Benchmark comparing causal depthwise conv and QKV packing strategies.
    ///
    /// Measures at Qwen3.5 0.6B scale:
    /// 1. packed_qkv — single matmul [hidden → conv_channels]
    /// 2. separate_qkv — 3 matmuls in one graph (inner, qk, qk)
    /// 3. layout_prep — transpose + cont + pad overhead
    /// 4. host_conv — causal_depthwise_conv (CPU scalar)
    /// 5. graph_conv — ssm_conv + silu (ggml graph, pre-prepared input)
    /// 6. full_fused — norm + 4 projections + transpose + pad + ssm_conv + silu
    #[test]
    #[ignore]
    fn bench_conv_vs_qkv_comparison() {
        use super::super::linear_attention::{causal_depthwise_conv, causal_depthwise_conv_graph};
        use super::super::tensor_ops::project_sequence_graph;

        let warmup = 3;
        let iters = 20;
        let backends = available_backends();
        let mut results: Vec<BenchResult> = Vec::new();

        // Qwen3.5 0.6B dimensions
        let hidden = 1536_usize;
        let inner_size = 1536_usize;
        let group_count = 4_usize;
        let state_size = 16_usize;
        let time_step_rank = 96_usize;
        let conv_kernel = 4_usize;
        let conv_channels = inner_size + 2 * group_count * state_size;
        let qk_features = group_count * state_size; // 64

        // Conv weight: [conv_channels × conv_kernel], identity-on-last-tap
        let mut conv_weight = vec![0.0_f32; conv_channels * conv_kernel];
        for ch in 0..conv_channels {
            conv_weight[ch * conv_kernel + (conv_kernel - 1)] = 1.0;
        }

        for seq_len in [64, 256, 1024] {
            let input = synthetic_input(hidden, seq_len);
            let conv_input = synthetic_input(conv_channels, seq_len);

            // --- 1. Packed QKV: single matmul [hidden → conv_channels] ---
            for &(bname, ref backend) in &backends {
                let weight: Vec<f32> = (0..hidden * conv_channels)
                    .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
                    .collect();
                let avg = bench_fn(warmup, iters, || {
                    project_sequence_graph(&input, seq_len, hidden, conv_channels, &weight, backend)
                });
                results.push(BenchResult {
                    label: "packed_qkv",
                    scale: "qwen35",
                    seq_len,
                    backend_name: bname,
                    avg_ms: avg,
                });
            }

            // --- 2. Separate Q/K/V: 3 matmuls in one graph ---
            //     inner_size + qk_features + qk_features = conv_channels
            for &(bname, ref backend) in &backends {
                let avg = bench_batch_projections(
                    &input,
                    seq_len,
                    hidden,
                    &[inner_size, qk_features, qk_features],
                    warmup,
                    iters,
                    backend,
                );
                results.push(BenchResult {
                    label: "separate_qkv",
                    scale: "qwen35",
                    seq_len,
                    backend_name: bname,
                    avg_ms: avg,
                });
            }

            // --- 3. Layout prep: transpose + cont + pad ---
            for &(bname, ref backend) in &backends {
                let avg =
                    bench_layout_prep(seq_len, conv_channels, conv_kernel, warmup, iters, backend);
                results.push(BenchResult {
                    label: "layout_prep",
                    scale: "qwen35",
                    seq_len,
                    backend_name: bname,
                    avg_ms: avg,
                });
            }

            // --- 4. Host conv: causal_depthwise_conv (CPU only) ---
            {
                let avg = bench_fn(warmup, iters, || {
                    causal_depthwise_conv(
                        &conv_input,
                        seq_len,
                        conv_channels,
                        conv_kernel,
                        &conv_weight,
                    )
                });
                results.push(BenchResult {
                    label: "host_conv",
                    scale: "qwen35",
                    seq_len,
                    backend_name: "host",
                    avg_ms: avg,
                });
            }

            // --- 5. Graph conv: ssm_conv + silu (pre-prepared input) ---
            for &(bname, ref backend) in &backends {
                let avg = bench_fn(warmup, iters, || {
                    causal_depthwise_conv_graph(
                        &conv_input,
                        seq_len,
                        conv_channels,
                        conv_kernel,
                        &conv_weight,
                        backend,
                    )
                });
                results.push(BenchResult {
                    label: "graph_conv",
                    scale: "qwen35",
                    seq_len,
                    backend_name: bname,
                    avg_ms: avg,
                });
            }

            // --- 6. Full fused: norm + 4 projections + conv pipeline ---
            {
                let plan = build_linear_attention_plan(
                    hidden,
                    inner_size,
                    group_count,
                    time_step_rank,
                    state_size,
                    conv_kernel,
                );
                let norm_w = &plan.norm_values;
                for &(bname, ref backend) in &backends {
                    let avg = bench_fn(warmup, iters, || {
                        qwen35_linear_attention_inference(
                            &plan, &input, seq_len, 1e-5, norm_w, backend,
                        )
                    });
                    results.push(BenchResult {
                        label: "full_fused",
                        scale: "qwen35",
                        seq_len,
                        backend_name: bname,
                        avg_ms: avg,
                    });
                }
            }
        }

        print_results(&results);

        // Also print a comparison summary table.
        println!("\n=== Conv vs QKV Packing Summary ===\n");
        println!(
            "{:>8} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "SeqLen",
            "Backend",
            "Packed",
            "Separate",
            "LayoutPrep",
            "HostConv",
            "GraphConv",
            "FullFused",
        );
        println!("{:-<100}", "");
        for seq_len in [64, 256, 1024] {
            for backend_name in ["CPU", "Metal"] {
                let find = |label: &str, bname: &str| -> Option<f64> {
                    results
                        .iter()
                        .find(|r| {
                            r.label == label && r.seq_len == seq_len && r.backend_name == bname
                        })
                        .map(|r| r.avg_ms)
                };
                let host_conv = results
                    .iter()
                    .find(|r| {
                        r.label == "host_conv" && r.seq_len == seq_len && r.backend_name == "host"
                    })
                    .map(|r| r.avg_ms);

                if let Some(packed) = find("packed_qkv", backend_name) {
                    println!(
                        "{:>8} {:>8} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3}",
                        seq_len,
                        backend_name,
                        packed,
                        find("separate_qkv", backend_name).unwrap_or(f64::NAN),
                        find("layout_prep", backend_name).unwrap_or(f64::NAN),
                        host_conv.unwrap_or(f64::NAN),
                        find("graph_conv", backend_name).unwrap_or(f64::NAN),
                        find("full_fused", backend_name).unwrap_or(f64::NAN),
                    );
                }
            }
        }
        println!();
    }
}
