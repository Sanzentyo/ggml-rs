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

    let parsed = ParsedArgs::from_cli(Cli::parse());
    let model = GgufModel::open(&parsed.model_path)?;
    let dimensions = resolve_llama_layer_dimensions(&model, parsed.layer)?;
    let weights = match parsed.hidden_features {
        Some(hidden_features) => {
            resolve_mlp_weights_for_layer(&model, parsed.layer, hidden_features)?
        }
        None => resolve_mlp_weights_for_layer_auto(&model, parsed.layer)?,
    };
    let input: Vec<f32> = (0..weights.hidden_features)
        .map(|index| ((index + 5) % 19) as f32 * 0.125)
        .collect();

    for backend in parsed.backends {
        let report = mlp_inference_with_weights_repeats(&weights, &input, backend, parsed.repeats)?;
        let preview_len = report.output.len().min(8);
        println!(
            "[{}] layer={} hidden={} ffn={} repeats={} preview={:?}",
            report.backend_name,
            parsed.layer,
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

#[derive(Debug)]
struct ParsedArgs {
    model_path: PathBuf,
    layer: usize,
    hidden_features: Option<usize>,
    repeats: usize,
    backends: Vec<LlamaBackend>,
}

impl ParsedArgs {
    fn from_cli(cli: Cli) -> Self {
        let backends = if cli.backends.is_empty() {
            vec![BackendArg::Cpu, BackendArg::Metal]
        } else {
            cli.backends
        };
        Self {
            model_path: cli.model_path,
            layer: cli.layer,
            hidden_features: cli.hidden_features,
            repeats: cli.repeats,
            backends: backends.into_iter().map(Into::into).collect(),
        }
    }
}

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
