use ggml_rs::{Backend, BackendKind, Context, Shape2D};
use std::time::Instant;

const ROWS_A: usize = 256;
const COLS_A: usize = 256;
const ROWS_B: usize = 256;
const COLS_B: usize = 256;
const SHAPE_A: Shape2D = Shape2D::new(COLS_A, ROWS_A);
const SHAPE_B: Shape2D = Shape2D::new(COLS_B, ROWS_B);

fn main() -> Result<(), ggml_rs::Error> {
    ggml_rs::init_timing();
    Backend::load_all();

    let (iters, backends) = parse_args(std::env::args().skip(1));
    for backend in backends {
        run_bench(backend, iters)?;
    }
    Ok(())
}

fn parse_args(args: impl Iterator<Item = String>) -> (usize, Vec<BackendKind>) {
    let mut iters = 30usize;
    let mut backends = Vec::new();

    let mut pending_iters = false;
    for arg in args {
        if pending_iters {
            if let Ok(value) = arg.parse::<usize>() {
                iters = value.max(1);
            }
            pending_iters = false;
            continue;
        }

        match arg.as_str() {
            "--iters" | "-n" => pending_iters = true,
            "cpu" | "CPU" => backends.push(BackendKind::Cpu),
            "metal" | "METAL" | "Metal" => backends.push(BackendKind::Metal),
            _ => {}
        }
    }

    if backends.is_empty() {
        (iters, vec![BackendKind::Cpu, BackendKind::Metal])
    } else {
        (iters, backends)
    }
}

fn run_bench(kind: BackendKind, iters: usize) -> Result<(), ggml_rs::Error> {
    let backend = Backend::new(kind)?;
    let backend_name = backend.name()?.to_string();

    let ctx_size = Context::recommended_backend_matmul_memory::<f32>(SHAPE_A, SHAPE_B)?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)?;

    let a = ctx.new_f32_tensor_2d_shape(SHAPE_A)?;
    let b = ctx.new_f32_tensor_2d_shape(SHAPE_B)?;
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
