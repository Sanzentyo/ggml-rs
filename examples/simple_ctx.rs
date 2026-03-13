use ggml_rs::{Context, Result, init_timing};

fn main() -> Result<()> {
    init_timing();

    const ROWS_A: usize = 4;
    const COLS_A: usize = 2;
    const ROWS_B: usize = 3;
    const COLS_B: usize = 2;

    let matrix_a: [f32; ROWS_A * COLS_A] = [2.0, 8.0, 5.0, 1.0, 4.0, 2.0, 8.0, 6.0];
    let matrix_b: [f32; ROWS_B * COLS_B] = [10.0, 5.0, 9.0, 9.0, 5.0, 4.0];

    let ctx_size = Context::recommended_matmul_memory_f32(ROWS_A, COLS_A, ROWS_B, COLS_B)?;
    let ctx = Context::new(ctx_size)?;

    let a = ctx.new_f32_tensor_2d(COLS_A, ROWS_A)?;
    let b = ctx.new_f32_tensor_2d(COLS_B, ROWS_B)?;
    a.set_f32(&matrix_a)?;
    b.set_f32(&matrix_b)?;

    let result = ctx.mul_mat(&a, &b)?;
    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&result);
    ctx.compute(&mut graph, 1)?;

    let output = graph.last_node()?;
    let values = output.to_vec_f32()?;
    let (cols, rows) = output.shape_2d()?;

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
