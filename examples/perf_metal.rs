//! Safe equivalent of ggml `perf-metal` benchmark graph.

use clap::Parser;
use ggml_rs::{
    Backend, BackendKind, Bytes, Context, Error, Result, Shape2D, graph_overhead_custom,
    init_timing, tensor_overhead_bytes,
};
use std::time::Instant;

const DEFAULT_N_OP: usize = 1024;
const DEFAULT_N_ITER: usize = 128;
const MATRIX_DIM: usize = 8;
const CONTEXT_SLACK_BYTES: usize = 1024;

#[derive(Debug, Parser, Clone, Copy)]
#[command(name = "perf_metal")]
struct Cli {
    #[arg(value_name = "N_OP", default_value_t = DEFAULT_N_OP)]
    n_op: usize,
    #[arg(value_name = "N_ITER", default_value_t = DEFAULT_N_ITER)]
    n_iter: usize,
}

fn main() -> Result<()> {
    init_timing();
    Backend::load_all();

    let cli = Cli::parse();
    let n_op = cli.n_op.max(1);
    let n_iter = cli.n_iter.max(1);
    println!("main: n_op = {n_op}, n_iter = {n_iter}");

    let graph_capacity = n_op.checked_mul(4).ok_or(Error::Overflow)?;
    let tensor_slots = graph_capacity.checked_add(2).ok_or(Error::Overflow)?;
    let ctx_size = tensor_slots
        .checked_mul(tensor_overhead_bytes())
        .ok_or(Error::Overflow)?
        .checked_add(graph_overhead_custom(graph_capacity, false))
        .ok_or(Error::Overflow)?
        .checked_add(CONTEXT_SLACK_BYTES)
        .ok_or(Error::Overflow)?;
    let ctx = Context::new_no_alloc_bytes(Bytes::new(ctx_size))?;

    let shape = Shape2D::new(MATRIX_DIM, MATRIX_DIM);
    let t0 = ctx.new_tensor_2d::<f32>(shape)?;
    let t1 = ctx.new_tensor_2d::<f32>(shape)?;

    let data0 = vec![1.0_f32; MATRIX_DIM * MATRIX_DIM];
    let data1 = vec![1.0_f32 / MATRIX_DIM as f32; MATRIX_DIM * MATRIX_DIM];

    let backend = Backend::new(BackendKind::Metal)?;

    let mut graph = ctx.new_graph_custom(graph_capacity, false)?;
    let mut cur = ctx.mul_mat(&t0, &t1)?;
    cur = ctx.scale(&cur, 1.0)?;

    for _ in 0..n_op.saturating_sub(1) {
        cur = ctx.mul_mat(&cur, &t1)?;
        cur = ctx.scale(&cur, 1.0)?;
    }

    cur = ctx.scale(&cur, 42.0)?;
    graph.build_forward_expand(&cur);
    let _buffer = ctx.allocate_tensors(&backend)?;
    t0.write_data_backend(&data0)?;
    t1.write_data_backend(&data1)?;

    println!("main: graph nodes = {}", graph.node_count());

    backend.compute(&mut graph)?;
    backend.synchronize()?;

    let start = Instant::now();
    for _ in 0..n_iter {
        backend.compute(&mut graph)?;
    }
    backend.synchronize()?;

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let avg_ms = elapsed_ms / n_iter as f64;
    println!("main: time = {avg_ms:.6} ms");

    let result = graph.last_node_typed::<f32>()?;
    let values = result.read_data_backend()?;
    let checksum: f64 = values.iter().copied().map(f64::from).sum();
    println!("main: checksum = {checksum:.6}");

    for row in values.chunks_exact(MATRIX_DIM) {
        for value in row {
            print!("{value:.6} ");
        }
        println!();
    }

    Ok(())
}
