use clap::{Parser, ValueEnum};
use ggml_rs::{Backend, BackendKind, Context, Shape2D};
use std::time::Instant;

const ROWS_A: usize = 256;
const COLS_A: usize = 256;
const ROWS_B: usize = 256;
const COLS_B: usize = 256;
const SHAPE_A: Shape2D = Shape2D::new(COLS_A, ROWS_A);
const SHAPE_B: Shape2D = Shape2D::new(COLS_B, ROWS_B);

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendArg {
    Cpu,
    Metal,
}

impl BackendArg {
    const fn kind(self) -> BackendKind {
        match self {
            Self::Cpu => BackendKind::Cpu,
            Self::Metal => BackendKind::Metal,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "bench_matmul")]
struct Cli {
    #[arg(long = "iters", short = 'n', default_value_t = 30)]
    iters: usize,
    #[arg(value_enum)]
    backends: Vec<BackendArg>,
}

fn main() -> Result<(), ggml_rs::Error> {
    ggml_rs::init_timing();
    Backend::load_all();

    let mut cli = Cli::parse();
    cli.iters = cli.iters.max(1);
    let backends = if cli.backends.is_empty() {
        vec![BackendKind::Cpu, BackendKind::Metal]
    } else {
        cli.backends.into_iter().map(BackendArg::kind).collect()
    };
    for backend in backends {
        run_bench(backend, cli.iters)?;
    }
    Ok(())
}

fn run_bench(kind: BackendKind, iters: usize) -> Result<(), ggml_rs::Error> {
    let backend = Backend::new(kind)?;
    let backend_name = backend.name()?.to_string();

    let ctx_size = Context::recommended_backend_matmul_memory::<f32>(SHAPE_A, SHAPE_B)?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)?;

    let a = ctx.new_tensor_2d::<f32>(SHAPE_A)?;
    let b = ctx.new_tensor_2d::<f32>(SHAPE_B)?;
    let result = ctx.mul_mat(&a, &b)?;

    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&result);
    let _buffer = ctx.allocate_tensors(&backend)?;

    let data_a: Vec<f32> = (0..(ROWS_A * COLS_A))
        .map(|i| (i % 31) as f32 * 0.03125)
        .collect();
    let data_b: Vec<f32> = (0..(ROWS_B * COLS_B))
        .map(|i| (i % 17) as f32 * 0.0625)
        .collect();
    a.write_data_backend(&data_a)?;
    b.write_data_backend(&data_b)?;

    for _ in 0..3 {
        backend.compute(&mut graph)?;
    }

    let start = Instant::now();
    for _ in 0..iters {
        backend.compute(&mut graph)?;
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;

    let out = graph.last_node()?.read_data_backend::<f32>()?;
    let checksum: f64 = out.iter().take(16).map(|v| f64::from(*v)).sum();

    println!(
        "[{backend_name}] matmul {ROWS_A}x{COLS_A} · {ROWS_B}x{COLS_B} avg={avg_ms:.3} ms, checksum={checksum:.6}"
    );

    Ok(())
}
