//! Resolve canonical LLaMA tensor roles against real GGUF tensor names.

use llama_rs::{GgufModel, detect_layer_indices, resolve_llama_tensor_names};
use std::error::Error as StdError;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn StdError>> {
    let parsed = parse_args(std::env::args().skip(1))?;
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
            let head = parsed.head_layers.min(resolved.layers.len());
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

#[derive(Debug)]
struct ParsedArgs {
    model_path: PathBuf,
    strict: bool,
    head_layers: usize,
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<ParsedArgs, Box<dyn StdError>> {
    let mut model_path: Option<PathBuf> = None;
    let mut strict = false;
    let mut head_layers = 4usize;
    let mut pending_head = false;

    for arg in args {
        if pending_head {
            head_layers = arg
                .parse::<usize>()
                .map_err(|error| format!("invalid value for --head: {arg} ({error})"))?;
            pending_head = false;
            continue;
        }

        match arg.as_str() {
            "--strict" => strict = true,
            "--head" => pending_head = true,
            token => {
                if model_path.is_some() {
                    return Err(format!("unexpected extra positional argument: {token}").into());
                }
                model_path = Some(PathBuf::from(token));
            }
        }
    }

    if pending_head {
        return Err("missing value after --head".into());
    }

    let model_path =
        model_path.ok_or("usage: resolve_tensor_names <gguf-path> [--head N] [--strict]")?;
    Ok(ParsedArgs {
        model_path,
        strict,
        head_layers,
    })
}
