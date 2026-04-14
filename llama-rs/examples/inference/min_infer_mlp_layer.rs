//! Resolve-by-layer MLP inference demo with reusable decoded layer weights.

use clap::{Parser, ValueEnum};
use llama_rs::{
    GgufModel, LlamaBackend, mlp_inference_with_weights_repeats, resolve_llama_layer_dimensions,
    resolve_mlp_weights_for_layer, resolve_mlp_weights_for_layer_auto,
};
use std::error::Error as StdError;
use std::path::PathBuf;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    run().map_err(Into::into)
}

fn run() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let cli = Cli::parse();
    let model = GgufModel::open(&cli.model_path)?;
    let dimensions = resolve_llama_layer_dimensions(&model, cli.layer)?;
    let weights = match cli.hidden_features {
        Some(hidden_features) => resolve_mlp_weights_for_layer(&model, cli.layer, hidden_features)?,
        None => resolve_mlp_weights_for_layer_auto(&model, cli.layer)?,
    };
    let input: Vec<f32> = (0..weights.hidden_features)
        .map(|index| ((index + 5) % 19) as f32 * 0.125)
        .collect();

    for backend in cli.resolved_backends() {
        let report = mlp_inference_with_weights_repeats(&weights, &input, backend, cli.repeats)?;
        let preview_len = report.output.len().min(8);
        println!(
            "[{}] layer={} hidden={} ffn={} repeats={} preview={:?}",
            report.backend_name,
            cli.layer,
            report.hidden_features,
            report.ffn_features,
            report.repeats,
            &report.output[..preview_len]
        );
        println!("  resolution_mode={:?}", dimensions.resolution_mode);
    }

    Ok(())
}

#[derive(Debug, Error)]
#[error(transparent)]
struct ExampleError(#[from] Box<dyn StdError>);

#[derive(Debug, Clone, Parser)]
#[command(
    about = "Run MLP layer inference with GGUF-resolved weights",
    version,
    long_about = None
)]
struct Cli {
    model_path: PathBuf,
    #[arg(long, default_value_t = 0)]
    layer: usize,
    #[arg(long = "hidden")]
    hidden_features: Option<usize>,
    #[arg(long, short = 'n', default_value_t = 1)]
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
