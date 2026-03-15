//! Synthetic GPT-2 scheduler-path runner mirroring `main-sched.cpp` behavior.

use clap::{Parser, ValueEnum};
use llama_rs::{LlamaBackend, SyntheticConfig, SyntheticError, run_sched};
use std::error::Error as StdError;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    run().map_err(Into::into)
}

fn run() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let cli = Cli::parse();
    let backends: Vec<LlamaBackend> = if cli.backends.is_empty() {
        vec![LlamaBackend::Cpu]
    } else {
        cli.backends.iter().copied().map(Into::into).collect()
    };
    let report = run_sched(cli.config(), &backends)?;
    println!("{}", report.to_kv_line());

    Ok(())
}

#[derive(Debug, Error)]
enum ExampleError {
    #[error(transparent)]
    Synthetic(#[from] SyntheticError),
    #[error(transparent)]
    Other(#[from] Box<dyn StdError>),
}

#[derive(Debug, Clone, Parser)]
#[command(about = "Run synthetic GPT-2 scheduler execution path", version)]
struct Cli {
    #[arg(long = "n-embd", default_value_t = 128)]
    n_embd: usize,
    #[arg(long = "n-vocab", default_value_t = 512)]
    n_vocab: usize,
    #[arg(long = "n-batch", default_value_t = 8)]
    n_batch: usize,
    #[arg(long = "n-predict", default_value_t = 32)]
    n_predict: usize,
    #[arg(long = "threads", short = 't', default_value_t = 1)]
    n_threads: usize,
    #[arg(long = "seed", short = 's', default_value_t = 42)]
    seed: u64,
    #[arg(value_enum)]
    backends: Vec<BackendArg>,
}

impl Cli {
    fn config(&self) -> SyntheticConfig {
        SyntheticConfig {
            n_embd: self.n_embd,
            n_vocab: self.n_vocab,
            n_batch: self.n_batch,
            n_predict: self.n_predict,
            n_threads: self.n_threads,
            seed: self.seed,
        }
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
