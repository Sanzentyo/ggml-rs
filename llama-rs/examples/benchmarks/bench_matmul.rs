use clap::{Parser, ValueEnum};
use llama_rs::{LlamaBackend, MatmulBenchConfig, backend_matmul_bench};
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
        cli.backends.clone()
    };
    let config = cli.config();

    for backend in backends.into_iter().map(Into::into) {
        let report = backend_matmul_bench(backend, config)?;
        println!(
            "[{}] llama-rs matmul {}x{} · {}x{} warmup={} bench={} avg={:.3} ms, checksum={:.6}",
            report.backend_name,
            report.rows_a,
            report.cols_a,
            report.rows_b,
            report.cols_b,
            report.warmup_iters,
            report.bench_iters,
            report.avg_ms,
            report.checksum
        );
    }

    Ok(())
}

#[derive(Debug, Error)]
#[error(transparent)]
struct ExampleError(#[from] Box<dyn StdError>);

#[derive(Debug, Clone, Parser)]
#[command(about = "Benchmark backend matmul path", version, long_about = None)]
struct Cli {
    #[arg(long = "iters", short = 'n', default_value_t = MatmulBenchConfig::default().bench_iters)]
    bench_iters: usize,
    #[arg(long = "warmup", short = 'w', default_value_t = MatmulBenchConfig::default().warmup_iters)]
    warmup_iters: usize,
    #[arg(long = "size", short = 's')]
    size: Option<usize>,
    #[arg(value_enum)]
    backends: Vec<BackendArg>,
}

impl Cli {
    fn config(&self) -> MatmulBenchConfig {
        let mut config = MatmulBenchConfig {
            warmup_iters: self.warmup_iters,
            bench_iters: self.bench_iters,
            ..MatmulBenchConfig::default()
        };
        if let Some(size) = self.size {
            config.rows_a = size;
            config.cols_a = size;
            config.rows_b = size;
            config.cols_b = size;
        }
        config
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
