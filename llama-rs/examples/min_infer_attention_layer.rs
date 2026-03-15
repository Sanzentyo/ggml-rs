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

    let cli = Cli::parse();
    let model = GgufModel::open(&cli.model_path)?;
    let dimensions = resolve_llama_layer_dimensions(&model, cli.layer)?;
    let mut weights =
        resolve_attention_weights_for_layer_auto(&model, cli.layer, cli.sequence_length)?;
    if cli.no_rope {
        weights.config = weights.config.with_rotary(RotaryEmbedding::Disabled);
    }
    if cli.causal {
        weights.config = weights
            .config
            .with_mask(AttentionMaskPolicy::Causal { past_tokens: 0 });
    }
    let input: Vec<f32> = (0..(weights.config.hidden_features() * cli.sequence_length))
        .map(|index| ((index + 3) % 29) as f32 * 0.0625)
        .collect();

    for backend in cli.resolved_backends() {
        let report =
            attention_inference_with_weights_repeats(&weights, &input, backend, cli.repeats)?;
        let preview_len = report.output.len().min(8);
        println!(
            "[{}] attn layer={} hidden={} seq={} repeats={} preview={:?}",
            report.backend_name,
            cli.layer,
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
