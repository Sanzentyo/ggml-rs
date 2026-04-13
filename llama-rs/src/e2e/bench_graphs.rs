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
}
