//! Resolve canonical LLaMA tensor roles against real GGUF tensor names.

use clap::Parser;
use llama_rs::{GgufModel, detect_layer_indices, resolve_llama_tensor_names};
use std::error::Error as StdError;
use std::path::PathBuf;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    run().map_err(Into::into)
}

fn run() -> Result<(), Box<dyn StdError>> {
    let parsed = ParsedArgs::parse();
    let model = GgufModel::open(&parsed.model_path)?;

    let layers = detect_layer_indices(&model);
    println!(
        "model={} tensors={} detected_layers={}",
        model.path().display(),
        model.report().tensors.len(),
        layers.len()
    );

    match resolve_llama_tensor_names(&model) {
        Ok(resolved) => {
            println!("token_embedding={}", resolved.token_embedding);
            println!("output_norm={}", resolved.output_norm);
            println!(
                "output={}",
                resolved.output.as_deref().unwrap_or("<not found>")
            );
            let head = parsed.head.min(resolved.layers.len());
            for layer in resolved.layers.iter().take(head) {
                println!(
                    "layer={} attn_norm={} attn_q={} attn_k={} attn_v={} attn_output={} ffn_norm={} ffn_gate={} ffn_up={} ffn_down={}",
                    layer.layer,
                    layer.attn_norm,
                    layer.attn_q,
                    layer.attn_k,
                    layer.attn_v,
                    layer.attn_output,
                    layer.ffn_norm,
                    layer.ffn_gate,
                    layer.ffn_up,
                    layer.ffn_down
                );
            }
        }
        Err(error) => {
            println!("resolution failed: {error}");
            if parsed.strict {
                return Err(error.into());
            }
            println!("non-strict mode: keeping process successful for inspection workflows");
        }
    }

    Ok(())
}

#[derive(Debug, Error)]
#[error(transparent)]
struct ExampleError(#[from] Box<dyn StdError>);

#[derive(Debug, Parser)]
#[command(about = "Resolve layer tensor role names from GGUF", version, long_about = None)]
struct ParsedArgs {
    model_path: PathBuf,
    #[arg(long, default_value_t = 4)]
    head: usize,
    #[arg(long)]
    strict: bool,
}
