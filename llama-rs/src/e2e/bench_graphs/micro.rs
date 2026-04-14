//! Causal depthwise conv vs QKV packing micro-benchmarks (item 57).
//!
//! Compares packed QKV (single matmul) vs separate Q/K/V projections,
//! and host conv vs graph conv (ssm_conv), with layout prep overhead.
//!
//! Run with:
//! ```sh
//! cargo test --workspace --features link-system --release \
//!   -- bench_conv_vs_qkv --nocapture --ignored
//! ```

use super::helpers::{
    BenchResult, available_backends, bench_fn, build_linear_attention_plan, print_results,
    synthetic_input,
};
use ggml_rs::Backend;

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
        let projs =
            build_batch_projections(&ctx, &x_in, hidden, &specs).expect("build_batch_projections");

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
        let ctx =
            Context::new_no_alloc_bytes(Bytes::new(tensor_bytes + 4 * 1024 * 1024)).expect("ctx");

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
            // zeros tensor is the 2nd tensor created; rely on allocator zero-init.
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
/// 1. packed_qkv -- single matmul [hidden -> conv_channels]
/// 2. separate_qkv -- 3 matmuls in one graph (inner, qk, qk)
/// 3. layout_prep -- transpose + cont + pad overhead
/// 4. host_conv -- causal_depthwise_conv (CPU scalar)
/// 5. graph_conv -- ssm_conv + silu (ggml graph, pre-prepared input)
/// 6. full_fused -- norm + 4 projections + transpose + pad + ssm_conv + silu
#[test]
#[ignore]
fn bench_conv_vs_qkv_comparison() {
    use super::super::linear_attention::{
        causal_depthwise_conv, causal_depthwise_conv_graph, qwen35_linear_attention_inference,
    };
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

    // Conv weight: [conv_channels x conv_kernel], identity-on-last-tap
    let mut conv_weight = vec![0.0_f32; conv_channels * conv_kernel];
    for ch in 0..conv_channels {
        conv_weight[ch * conv_kernel + (conv_kernel - 1)] = 1.0;
    }

    for seq_len in [64, 256, 1024] {
        let input = synthetic_input(hidden, seq_len);
        let conv_input = synthetic_input(conv_channels, seq_len);

        // --- 1. Packed QKV: single matmul [hidden -> conv_channels] ---
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
            for &(bname, ref backend) in &backends {
                let avg = bench_fn(warmup, iters, || {
                    qwen35_linear_attention_inference(&plan, &input, seq_len, 1e-5, backend)
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
                    .find(|r| r.label == label && r.seq_len == seq_len && r.backend_name == bname)
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
