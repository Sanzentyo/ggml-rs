use llama_rs::run_simple_ctx;
use std::error::Error as StdError;

fn main() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let report = run_simple_ctx()?;
    println!(
        "simple-ctx mul mat ({} x {}) OK: {:?}",
        report.cols, report.rows, report.values
    );

    Ok(())
}
