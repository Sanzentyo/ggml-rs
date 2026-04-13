//! Backend multi-op graph example: matmul + bias add.
//!
//! Demonstrates the safe backend compute lifecycle:
//! 1. Create a no-alloc context (required for backend path)
//! 2. Define tensors and build a computation graph
//! 3. Allocate backend storage for all tensors
//! 4. Transfer input data from host to backend
//! 5. Execute the graph on CPU or Metal
//! 6. Read results back from backend to host
//!
//! Run:
//!   cargo run --example backend_ops --features link-system -- cpu
//!   cargo run --example backend_ops --features link-system -- metal

use clap::{Parser, ValueEnum};
use ggml_rs::{Backend, BackendKind, Bytes, Context, Length, Result, Shape2D, init_timing};

// W: 3×2 weight matrix, x: 2×1 input vector, b: 3-element bias
// result = W * x + b
// ggml mul_mat: result ne[0] = W.ne[1] = 3, ne[1] = x.ne[1] = 1 → shape [3, 1]
const W_DATA: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const X_DATA: [f32; 2] = [10.0, 20.0];
const B_DATA: [f32; 3] = [100.0, 200.0, 300.0];

// Expected: W*x = [1*10+2*20, 3*10+4*20, 5*10+6*20] = [50, 110, 170]
//           + b = [150, 310, 470]
const EXPECTED: [f32; 3] = [150.0, 310.0, 470.0];

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
#[command(name = "backend_ops", about = "Multi-op backend graph: matmul + bias")]
struct Cli {
    #[arg(value_enum)]
    backends: Vec<BackendArg>,
}

fn main() -> Result<()> {
    init_timing();
    Backend::load_all();

    let cli = Cli::parse();
    let requested = if cli.backends.is_empty() {
        vec![BackendKind::Cpu, BackendKind::Metal]
    } else {
        cli.backends.into_iter().map(BackendArg::kind).collect()
    };
    for kind in requested {
        if let Err(e) = run_backend(kind) {
            eprintln!("[{kind:?}] skipped: {e}");
        }
    }

    Ok(())
}

fn run_backend(kind: BackendKind) -> Result<()> {
    let backend = Backend::new(kind)?;
    let backend_name = backend.name()?;

    // No-alloc context: tensors are placeholders until backend allocates storage.
    let ctx = Context::new_no_alloc_bytes(Bytes::new(64 * 1024 * 1024))?;

    // Define tensors.
    let w = ctx.new_tensor_2d::<f32>(Shape2D::new(2, 3))?; // 3 rows × 2 cols
    let x = ctx.new_tensor_2d::<f32>(Shape2D::new(2, 1))?; // 1 row  × 2 cols
    let b = ctx.new_tensor_1d::<f32>(Length::new(3))?; // 3-element bias (broadcasts to [3,1])

    // Build computation graph: result = mul_mat(w, x) + b
    let wx = ctx.mul_mat(&w, &x)?; // [3×1]
    let result = ctx.add(&wx, &b)?; // [3×1]

    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&result);

    // Allocate backend memory for all context tensors at once.
    let _buffer = ctx.allocate_tensors(&backend)?;

    // Transfer host data → backend.
    w.write_data_backend(&W_DATA)?;
    x.write_data_backend(&X_DATA)?;
    b.write_data_backend(&B_DATA)?;

    // Execute the graph.
    backend.compute(&mut graph)?;

    // Ensure asynchronous backends (for example Metal) have completed
    // execution before reading results back to the host.
    backend.synchronize()?;

    // Read results back: backend → host.
    let output = result.read_data_backend()?;

    // Verify.
    assert_eq!(output.len(), EXPECTED.len());
    for (i, (&actual, &expected)) in output.iter().zip(EXPECTED.iter()).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta < 1e-4,
            "[{backend_name}] mismatch at {i}: expected {expected}, got {actual}"
        );
    }

    println!("[{backend_name}] matmul + bias: {output:?} ✓");
    Ok(())
}
