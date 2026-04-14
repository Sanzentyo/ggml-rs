//! Embedding probe demo for f32 tensor summary statistics.

use clap::Parser;
use llama_rs::{GgufModel, summarize_embedding_tensor};
use std::error::Error as StdError;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    run().map_err(Into::into)
}

fn run() -> Result<(), Box<dyn StdError>> {
    let cli = Cli::parse();

    let model = GgufModel::open(&cli.path)?;
    let stats = summarize_embedding_tensor(&model, &cli.tensor_name)?;

    println!("tensor:   {}", cli.tensor_name);
    println!("len:      {}", stats.len);
    println!("mean:     {:.6}", stats.mean);
    println!("l2_norm:  {:.6}", stats.l2_norm);
    println!("min:      {:.6}", stats.min);
    println!("max:      {:.6}", stats.max);

    Ok(())
}

#[derive(Debug, Error)]
#[error(transparent)]
struct ExampleError(#[from] Box<dyn StdError>);

#[derive(Debug, Parser)]
#[command(
    about = "Embedding probe for one GGUF tensor",
    version,
    long_about = None
)]
struct Cli {
    /// Path to the GGUF model file.
    path: String,
    /// Tensor name to summarize.
    #[arg(default_value = "tensor_0")]
    tensor_name: String,
}
