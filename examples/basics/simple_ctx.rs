//! Safe equivalent of ggml `simple-ctx` style matrix multiplication.

use ggml_rs::{Context, Result, Shape2D, StaticShape2D, init_timing};

fn main() -> Result<()> {
    init_timing();

    const ROWS_A: usize = 4;
    const COLS_A: usize = 2;
    const ROWS_B: usize = 3;
    const COLS_B: usize = 2;
    const SHAPE_A: Shape2D = Shape2D::new(COLS_A, ROWS_A);
    const SHAPE_B: Shape2D = Shape2D::new(COLS_B, ROWS_B);
    type AShape = StaticShape2D<COLS_A, ROWS_A>;
    type BShape = StaticShape2D<COLS_B, ROWS_B>;

    let matrix_a: [f32; ROWS_A * COLS_A] = [2.0, 8.0, 5.0, 1.0, 4.0, 2.0, 8.0, 6.0];
    let matrix_b: [f32; ROWS_B * COLS_B] = [10.0, 5.0, 9.0, 9.0, 5.0, 4.0];

    let ctx_size = Context::recommended_matmul_memory::<f32>(SHAPE_A, SHAPE_B)?;
    let ctx = Context::new_bytes(ctx_size)?;

    let a = ctx.new_tensor_2d_typed::<f32, AShape>()?;
    let b = ctx.new_tensor_2d_typed::<f32, BShape>()?;
    a.write_data(&matrix_a)?;
    b.write_data(&matrix_b)?;

    // Legacy context execution path mirrors upstream simple-ctx behavior.
    let result = ctx.mul_mat(a.inner(), b.inner())?;
    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&result);
    ctx.compute(&mut graph, 1)?;

    let output = graph.last_node_typed::<f32>()?;
    let values = output.read_data()?;
    let shape = output.shape()?;
    let cols = shape.cols.get();
    let rows = shape.rows.get();

    println!("mul mat ({cols} x {rows}) (transposed result):");
    print!("[");
    for row in 0..rows {
        if row > 0 {
            println!();
        }

        for col in 0..cols {
            let idx = row * cols + col;
            print!(" {:.2}", values[idx]);
        }
    }
    println!(" ]");

    Ok(())
}
