//! Resolve-by-layer minimal attention inference demo.

use clap::{Parser, ValueEnum};
use llama_rs::{
    AttentionMaskPolicy, GgufModel, LlamaBackend, RotaryEmbedding,
    attention_inference_with_weights_repeats, resolve_attention_weights_for_layer_auto,
    resolve_llama_layer_dimensions,
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
    let mut weights =
        resolve_attention_weights_for_layer_auto(&model, parsed.layer, parsed.sequence_length)?;
    if parsed.no_rope {
        weights.config = weights.config.with_rotary(RotaryEmbedding::Disabled);
    }
    if parsed.causal {
        weights.config = weights
            .config
            .with_mask(AttentionMaskPolicy::Causal { past_tokens: 0 });
    }
    let input: Vec<f32> = (0..(weights.config.hidden_features() * parsed.sequence_length))
        .map(|index| ((index + 3) % 29) as f32 * 0.0625)
        .collect();

    for backend in parsed.backends {
        let report =
            attention_inference_with_weights_repeats(&weights, &input, backend, parsed.repeats)?;
        let preview_len = report.output.len().min(8);
        println!(
            "[{}] attn layer={} hidden={} seq={} repeats={} preview={:?}",
            report.backend_name,
            parsed.layer,
            report.hidden_features,
            report.sequence_length,
            report.repeats,
            &report.output[..preview_len]
        );
        println!(
            "  resolution_mode={:?}, heads={}/{}",
            dimensions.resolution_mode, dimensions.query_head_count, dimensions.kv_head_count
        );
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
    sequence_length: usize,
    repeats: usize,
    causal: bool,
    no_rope: bool,
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
            sequence_length: cli.sequence_length,
            repeats: cli.repeats,
            causal: cli.causal,
            no_rope: cli.no_rope,
            backends: backends.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, Parser)]
#[command(
    about = "Run attention inference by resolved layer index",
    version,
    long_about = None
)]
struct Cli {
    model_path: PathBuf,
    #[arg(long, default_value_t = 0)]
    layer: usize,
    #[arg(long = "seq", default_value_t = 4)]
    sequence_length: usize,
    #[arg(long, short = 'n', default_value_t = 1)]
    repeats: usize,
    #[arg(long)]
    causal: bool,
    #[arg(long = "no-rope")]
    no_rope: bool,
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
