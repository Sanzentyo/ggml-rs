//! LM head benchmark: host-side vs one-shot graph vs persistent graph.

use super::helpers::{BenchResult, bench_fn, print_results};
use ggml_rs::{Backend, BackendKind};

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
