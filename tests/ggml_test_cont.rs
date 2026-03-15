#![cfg(feature = "link-system")]

//! Rust-native parity test for upstream `test-cont` behavior.

use ggml_rs::{Context, Length, ThreadCount};

#[test]
fn cont_after_transpose_matches_reference() -> Result<(), ggml_rs::Error> {
    ggml_rs::init_timing();

    let ctx = Context::new(2 * 1024 * 1024)?;
    let input = ctx.new_tensor_1d::<f32>(Length::new(2))?;
    input.write_data(&[1.0, 2.0])?;

    let transposed = ctx.transpose(&input)?;
    let contiguous = ctx.cont(&transposed)?;
    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&contiguous);
    ctx.compute_with_threads(&mut graph, ThreadCount::new(1))?;

    let out = graph.last_node()?.read_data::<f32>()?;
    assert_eq!(out, vec![1.0, 2.0]);

    Ok(())
}
