//! Minimal linear inference demo using reusable GGUF-decoded weights.

use clap::{Parser, ValueEnum};
use llama_rs::{
    GgufModel, LinearInferenceConfig, LinearWeights, LlamaBackend,
    run_linear_inference_with_weights_repeats,
};
use std::error::Error as StdError;

fn main() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let parsed = ParsedArgs::from_cli(Cli::parse())?;
    let model = GgufModel::open(&parsed.path)?;
    let weights = LinearWeights::from_model(&model, &parsed.weight_tensor, parsed.config)?;
    let input = make_input(parsed.config.in_features());
    let report = run_linear_inference_with_weights_repeats(
        &weights,
        &input,
        parsed.backend,
        parsed.repeats,
    )?;

    let preview: Vec<f32> = report.output.iter().copied().take(8).collect();
    println!(
        "[{}] linear inference OK: in={} out={} repeats={} preview={preview:?}",
        report.backend_name, report.in_features, report.out_features, parsed.repeats
    );
    Ok(())
}

struct ParsedArgs {
    path: String,
    weight_tensor: String,
    config: LinearInferenceConfig,
    backend: LlamaBackend,
    repeats: usize,
}

impl ParsedArgs {
    fn from_cli(cli: Cli) -> Result<Self, Box<dyn StdError>> {
        Ok(Self {
            path: cli.path,
            weight_tensor: cli.weight_tensor,
            config: LinearInferenceConfig::builder()
                .in_features(cli.input_cols)?
                .out_features(cli.output_rows)?
                .build(),
            backend: cli.backend.into(),
            repeats: cli.repeats,
        })
    }
}

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
