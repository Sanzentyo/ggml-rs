//! Batched execution demo using `llama-rs` foundation APIs.

use clap::{Parser, ValueEnum};
use llama_rs::{
    BatchSize, BatchedConfig, BatchedWorkload, LlamaBackend, ReadbackEvery, RepeatCount,
    run_batched_matmul_with_workload,
};
use std::error::Error as StdError;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    run().map_err(Into::into)
}

fn run() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let cli = Cli::parse();
    let config = cli.config()?;
    let backends = if cli.backends.is_empty() {
        vec![BackendArg::Cpu, BackendArg::Metal]
    } else {
        cli.backends.clone()
    };
    let workload = BatchedWorkload::deterministic(config)?;
    for backend in backends.into_iter().map(Into::into) {
        let report = run_batched_matmul_with_workload(backend, &workload)?;
        println!(
            "[{}] batched matmul {}x{} · {}x{} batch={} repeats={} readback_every={} readbacks={} avg_item={:.3} ms, checksum={:.6}",
            report.backend_name,
            report.rows_a,
            report.cols_a,
            report.rows_b,
            report.cols_b,
            report.batch_size,
            report.repeats,
            report.readback_every,
            report.readback_samples,
            report.avg_item_ms,
            report.checksum
        );
    }

    Ok(())
}

#[derive(Debug, Error)]
#[error(transparent)]
struct ExampleError(#[from] Box<dyn StdError>);

#[derive(Debug, Clone, Parser)]
#[command(about = "Run batched matmul workload", version, long_about = None)]
struct Cli {
    #[arg(long, short = 'b')]
    batch: Option<usize>,
    #[arg(long, short = 'n')]
    repeats: Option<usize>,
    #[arg(long = "readback-every", short = 'r')]
    readback_every: Option<usize>,
    #[arg(long, short = 's')]
    size: Option<usize>,
    #[arg(value_enum)]
    backends: Vec<BackendArg>,
}

impl Cli {
    fn config(&self) -> Result<BatchedConfig, Box<dyn StdError>> {
        let mut config = BatchedConfig::default();
        if let Some(batch) = self.batch {
            config.batch_size = BatchSize::new(batch)?;
        }
        if let Some(repeats) = self.repeats {
            config.repeats = RepeatCount::new(repeats)?;
        }
        if let Some(readback_every) = self.readback_every {
            config.readback_every = ReadbackEvery::new(readback_every)?;
        }
        if let Some(size) = self.size {
            config.rows_a = size;
            config.cols_a = size;
            config.rows_b = size;
            config.cols_b = size;
        }
        Ok(config.validated()?)
    }
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
