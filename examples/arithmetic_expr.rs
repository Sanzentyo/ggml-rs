use ggml_rs::{Context, Length};
use std::error::Error as StdError;

const LEN: usize = 4;
const A: [f32; LEN] = [1.0, 2.0, 3.0, 4.0];
const B: [f32; LEN] = [5.0, 6.0, 7.0, 8.0];
const C: [f32; LEN] = [2.0, 2.0, 2.0, 2.0];
const D: [f32; LEN] = [3.0, 3.0, 3.0, 3.0];
const TOLERANCE: f32 = 1e-6;

fn main() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let mem = 1024 * 1024;
    let ctx = Context::new(mem)?;

    let a = ctx.new_tensor_1d::<f32>(Length::new(LEN))?;
    let b = ctx.new_tensor_1d::<f32>(Length::new(LEN))?;
    let c = ctx.new_tensor_1d::<f32>(Length::new(LEN))?;
    let d = ctx.new_tensor_1d::<f32>(Length::new(LEN))?;

    a.write_data(&A)?;
    b.write_data(&B)?;
    c.write_data(&C)?;
    d.write_data(&D)?;

    let expr = ((ctx.expr(a) + ctx.expr(b))? * ctx.expr(c))? / ctx.expr(d);
    let out = expr?.into_tensor();

    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&out);
    ctx.compute(&mut graph, 1)?;

    let values = out.read_data::<f32>()?;
    let expected = expected_values();
    for (index, (actual, expected)) in values.iter().zip(expected.iter()).enumerate() {
        let delta = (actual - expected).abs();
        if delta > TOLERANCE {
            return Err(format!(
                "mismatch at index {index}: expected {expected}, actual {actual}, delta {delta}"
            )
            .into());
        }
    }

    println!("arithmetic expression OK: {values:?}");
    Ok(())
}

fn expected_values() -> [f32; LEN] {
    std::array::from_fn(|index| ((A[index] + B[index]) * C[index]) / D[index])
}
