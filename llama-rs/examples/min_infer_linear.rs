//! Minimal linear inference demo using reusable GGUF-decoded weights.

use clap::{Parser, ValueEnum};
use llama_rs::{
    GgufModel, LinearInferenceConfig, LinearWeights, LlamaBackend,
    linear_inference_with_weights_repeats,
};
use std::error::Error as StdError;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    run().map_err(Into::into)
}

fn run() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let cli = Cli::parse();
    let model = GgufModel::open(&cli.path)?;
    let config = cli.linear_config()?;
    let input = make_input(config.in_features());
    let weights = LinearWeights::from_model(&model, &cli.weight_tensor, config)?;
    let report =
        linear_inference_with_weights_repeats(&weights, &input, cli.backend.into(), cli.repeats)?;

    let preview: Vec<f32> = report.output.iter().copied().take(8).collect();
    println!(
        "[{}] linear inference OK: in={} out={} repeats={} preview={preview:?}",
        report.backend_name, report.in_features, report.out_features, cli.repeats
    );
    Ok(())
}

#[derive(Debug, Error)]
#[error(transparent)]
struct ExampleError(#[from] Box<dyn StdError>);

#[derive(Debug, Clone, Parser)]
#[command(about = "Minimal linear inference from GGUF weights", version, long_about = None)]
struct Cli {
    /// Path to model GGUF.
    path: String,
    /// Weight tensor name.
    weight_tensor: String,
    /// Input feature width.
    #[arg(long = "in")]
    input_cols: usize,
    /// Output feature width.
    #[arg(long = "out")]
    output_rows: usize,
    /// Repeat count.
    #[arg(long, short = 'n', default_value_t = 1)]
    repeats: usize,
    /// Backend to run.
    #[arg(value_enum, default_value_t = BackendArg::Cpu)]
    backend: BackendArg,
}

impl Cli {
    fn linear_config(&self) -> Result<LinearInferenceConfig, Box<dyn StdError>> {
        Ok(LinearInferenceConfig::builder()
            .in_features(self.input_cols)?
            .out_features(self.output_rows)?
            .build())
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

fn make_input(input_cols: usize) -> Vec<f32> {
    (0..input_cols)
        .map(|index| (index % 11) as f32 * 0.125)
        .collect()
}
