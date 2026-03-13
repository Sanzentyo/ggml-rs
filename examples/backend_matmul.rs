//! Safe CPU/Metal backend matmul parity example.

use ggml_rs::{Backend, BackendKind, Context, Result, Shape2D, StaticShape2D, init_timing};

const ROWS_A: usize = 4;
const COLS_A: usize = 2;
const ROWS_B: usize = 3;
const COLS_B: usize = 2;
const SHAPE_A: Shape2D = Shape2D::new(COLS_A, ROWS_A);
const SHAPE_B: Shape2D = Shape2D::new(COLS_B, ROWS_B);
type AShape = StaticShape2D<COLS_A, ROWS_A>;
type BShape = StaticShape2D<COLS_B, ROWS_B>;

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

    let ctx_size = Context::recommended_backend_matmul_memory_f32_shapes_bytes(SHAPE_A, SHAPE_B)?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)?;

    let a = ctx.new_f32_tensor_2d_typed::<AShape>()?;
    let b = ctx.new_f32_tensor_2d_typed::<BShape>()?;
    let result = ctx.mul_mat(a.inner(), b.inner())?;

    // Build graph first, then allocate backend memory for all tensors in the context.
    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&result);

    // Backend transfer APIs move host slices into backend-owned tensor storage.
    let _buffer = ctx.allocate_tensors(&backend)?;
    a.set_f32_backend(&MATRIX_A)?;
    b.set_f32_backend(&MATRIX_B)?;

    backend.compute(&mut graph)?;

    // `graph.last_node()` is the result tensor of this simple single-op graph.
    let output = graph.last_node()?;
    let values = output.to_vec_f32_backend()?;
    assert_expected(&values, backend_name);

    let shape = output.shape()?;
    println!(
        "[{backend_name}] mul mat ({} x {}) OK",
        shape.cols.get(),
        shape.rows.get()
    );

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
