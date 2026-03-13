use ggml_rs::{Backend, BackendKind, Context, Result, init_timing};

const ROWS_A: usize = 4;
const COLS_A: usize = 2;
const ROWS_B: usize = 3;
const COLS_B: usize = 2;

const MATRIX_A: [f32; ROWS_A * COLS_A] = [2.0, 8.0, 5.0, 1.0, 4.0, 2.0, 8.0, 6.0];
const MATRIX_B: [f32; ROWS_B * COLS_B] = [10.0, 5.0, 9.0, 9.0, 5.0, 4.0];

const EXPECTED: [f32; ROWS_A * ROWS_B] = [
    60.0, 55.0, 50.0, 110.0, //
    90.0, 54.0, 54.0, 126.0, //
    42.0, 29.0, 28.0, 64.0,
];

const TOLERANCE: f32 = 1e-4;

fn main() -> Result<()> {
    init_timing();
    Backend::load_all();

    let requested = parse_requested_backends(std::env::args().skip(1));
    for backend_kind in requested {
        run_backend(backend_kind)?;
    }

    Ok(())
}

fn parse_requested_backends(args: impl Iterator<Item = String>) -> Vec<BackendKind> {
    let mut parsed = Vec::new();

    for arg in args {
        match arg.as_str() {
            "cpu" | "CPU" => parsed.push(BackendKind::Cpu),
            "metal" | "METAL" | "Metal" => parsed.push(BackendKind::Metal),
            unknown => {
                eprintln!("unknown backend `{unknown}`, expected `cpu` or `metal`");
                std::process::exit(2);
            }
        }
    }

    if parsed.is_empty() {
        vec![BackendKind::Cpu, BackendKind::Metal]
    } else {
        parsed
    }
}

fn run_backend(kind: BackendKind) -> Result<()> {
    let backend = Backend::new(kind)?;
    let backend_name = backend.name()?;

    let ctx_size = Context::recommended_backend_matmul_memory_f32(ROWS_A, COLS_A, ROWS_B, COLS_B)?;
    let ctx = Context::new_no_alloc(ctx_size)?;

    let a = ctx.new_f32_tensor_2d(COLS_A, ROWS_A)?;
    let b = ctx.new_f32_tensor_2d(COLS_B, ROWS_B)?;
    let result = ctx.mul_mat(&a, &b)?;

    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&result);

    let _buffer = ctx.allocate_tensors(&backend)?;
    a.set_f32_backend(&MATRIX_A)?;
    b.set_f32_backend(&MATRIX_B)?;

    backend.compute(&mut graph)?;

    let output = graph.last_node()?;
    let values = output.to_vec_f32_backend()?;
    assert_expected(&values, backend_name);

    let (cols, rows) = output.shape_2d()?;
    println!("[{backend_name}] mul mat ({cols} x {rows}) OK");

    Ok(())
}

fn assert_expected(values: &[f32], backend_name: &str) {
    assert_eq!(values.len(), EXPECTED.len(), "unexpected output length");

    for (index, (actual, expected)) in values
        .iter()
        .copied()
        .zip(EXPECTED.iter().copied())
        .enumerate()
    {
        let delta = (actual - expected).abs();
        assert!(
            delta <= TOLERANCE,
            "[{backend_name}] mismatch at index {index}: expected {expected}, actual {actual}, delta {delta}"
        );
    }
}
