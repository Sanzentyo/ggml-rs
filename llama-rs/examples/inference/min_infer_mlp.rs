//! Minimal MLP-block inference demo on top of `llama-rs` safe APIs.
//!
//! Computes: `down(silu(gate(x)) * up(x))` with deterministic synthetic weights.

use clap::{Parser, ValueEnum};
use llama_rs::{LlamaBackend, MlpInferenceConfig, MlpWeights, mlp_inference_with_weights_repeats};
use std::error::Error as StdError;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    run().map_err(Into::into)
}

fn run() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let cli = Cli::parse();
    let config = MlpInferenceConfig::new(cli.hidden_features, cli.ffn_features)?;
    let weights = MlpWeights::deterministic(config);
    let input: Vec<f32> = (0..cli.hidden_features)
        .map(|index| ((index + 3) % 19) as f32 * 0.125)
        .collect();

    for backend in cli.resolved_backends() {
        let report = mlp_inference_with_weights_repeats(&weights, &input, backend, cli.repeats)?;
        let preview_len = report.output.len().min(8);
        println!(
            "[{}] mlp hidden={} ffn={} repeats={} preview={:?}",
            report.backend_name,
            report.hidden_features,
            report.ffn_features,
            report.repeats,
            &report.output[..preview_len]
        );
    }

    Ok(())
}

#[derive(Debug, Error)]
#[error(transparent)]
struct ExampleError(#[from] Box<dyn StdError>);

#[derive(Debug, Clone, Parser)]
#[command(about = "Minimal deterministic MLP inference", version, long_about = None)]
struct Cli {
    #[arg(long = "hidden", default_value_t = 64)]
    hidden_features: usize,
    #[arg(long = "ffn", default_value_t = 128)]
    ffn_features: usize,
    #[arg(long, short = 'n', default_value_t = 2)]
    repeats: usize,
    #[arg(value_enum)]
    backends: Vec<BackendArg>,
}

impl Cli {
    fn resolved_backends(&self) -> Vec<LlamaBackend> {
        if self.backends.is_empty() {
            vec![LlamaBackend::Cpu, LlamaBackend::Metal]
        } else {
            self.backends.iter().copied().map(Into::into).collect()
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
