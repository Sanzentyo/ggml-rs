#![cfg(feature = "link-system")]

//! Rust-native parity test for upstream `simple-ctx` style matmul.

use ggml_rs::{Context, Shape2D, ThreadCount};

const ROWS_A: usize = 4;
const COLS_A: usize = 2;
const ROWS_B: usize = 3;
const COLS_B: usize = 2;
const SHAPE_A: Shape2D = Shape2D::new(COLS_A, ROWS_A);
const SHAPE_B: Shape2D = Shape2D::new(COLS_B, ROWS_B);

const MATRIX_A: [f32; ROWS_A * COLS_A] = [2.0, 8.0, 5.0, 1.0, 4.0, 2.0, 8.0, 6.0];
const MATRIX_B: [f32; ROWS_B * COLS_B] = [10.0, 5.0, 9.0, 9.0, 5.0, 4.0];
const EXPECTED: [f32; ROWS_A * ROWS_B] = [
    60.0, 55.0, 50.0, 110.0, //
    90.0, 54.0, 54.0, 126.0, //
    42.0, 29.0, 28.0, 64.0,
];

#[test]
fn matmul_simple_ctx_parity() -> Result<(), ggml_rs::Error> {
    ggml_rs::init_timing();

    let mem = Context::recommended_matmul_memory::<f32>(SHAPE_A, SHAPE_B)?;
    let ctx = Context::new_bytes(mem)?;

    let a = ctx.new_f32_tensor_2d_shape(SHAPE_A)?;
    let b = ctx.new_f32_tensor_2d_shape(SHAPE_B)?;
    a.set_f32(&MATRIX_A)?;
    b.set_f32(&MATRIX_B)?;

    let result = ctx.mul_mat(&a, &b)?;
    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&result);
    ctx.compute_with_threads(&mut graph, ThreadCount::new(1))?;

    let out = graph.last_node()?.to_vec_f32()?;
    assert_eq!(out.len(), EXPECTED.len());
    for (actual, expected) in out.iter().zip(EXPECTED) {
        assert!((actual - expected).abs() <= 1e-4);
    }

    Ok(())
}
