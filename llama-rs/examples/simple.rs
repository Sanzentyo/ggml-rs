use llama_rs::run_simple_ctx;
use std::error::Error as StdError;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    run().map_err(Into::into)
}

fn run() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let report = run_simple_ctx()?;
    println!(
        "simple-ctx mul mat ({} x {}) OK: {:?}",
        report.cols, report.rows, report.values
    );

    Ok(())
}

#[derive(Debug, Error)]
#[error(transparent)]
struct ExampleError(#[from] Box<dyn StdError>);
