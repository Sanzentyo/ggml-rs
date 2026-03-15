//! Synthetic GPT-2 quantization runner mirroring `quantize.cpp` behavior.

use clap::Parser;
use llama_rs::{QuantizeConfig, SyntheticError, run_quantize};
use std::error::Error as StdError;
use std::path::PathBuf;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    run().map_err(Into::into)
}

fn run() -> Result<(), Box<dyn StdError>> {
    let cli = Cli::parse();
    let report = run_quantize(cli.config())?;
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
#[command(about = "Run synthetic GPT-2 quantization path", version)]
struct Cli {
    #[arg(long = "n-embd", default_value_t = 128)]
    n_embd: usize,
    #[arg(long = "n-vocab", default_value_t = 512)]
    n_vocab: usize,
    #[arg(long = "seed", short = 's', default_value_t = 42)]
    seed: u64,
    #[arg(
        long = "input",
        default_value = "target/benchmarks/gpt2_quantize_input_f32.bin"
    )]
    input: PathBuf,
    #[arg(
        long = "output",
        default_value = "target/benchmarks/gpt2_quantize_output_q8.bin"
    )]
    output: PathBuf,
}

impl Cli {
    fn config(&self) -> QuantizeConfig {
        QuantizeConfig {
            n_embd: self.n_embd,
            n_vocab: self.n_vocab,
            seed: self.seed,
            input_path: self.input.clone(),
            output_path: self.output.clone(),
        }
    }
}
