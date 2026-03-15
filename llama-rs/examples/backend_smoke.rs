use clap::{Parser, ValueEnum};
use llama_rs::{LlamaBackend, run_backend_smoke};
use std::error::Error as StdError;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    run().map_err(Into::into)
}

fn run() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let cli = Cli::parse();
    let backends = if cli.backends.is_empty() {
        vec![BackendArg::Cpu, BackendArg::Metal]
    } else {
        cli.backends
    };

    for backend in backends.into_iter().map(Into::into) {
        let report = run_backend_smoke(backend)?;
        println!(
            "[{}] mul mat ({} x {}) OK",
            report.backend_name, report.cols, report.rows
        );
    }

    Ok(())
}

#[derive(Debug, Error)]
#[error(transparent)]
struct ExampleError(#[from] Box<dyn StdError>);

#[derive(Debug, Clone, Parser)]
#[command(about = "Run backend smoke checks", version, long_about = None)]
struct Cli {
    #[arg(value_enum)]
    backends: Vec<BackendArg>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendArg {
    Cpu,
    Metal,
}

impl From<BackendArg> for LlamaBackend {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Cpu => LlamaBackend::Cpu,
            BackendArg::Metal => LlamaBackend::Metal,
        }
    }
}
