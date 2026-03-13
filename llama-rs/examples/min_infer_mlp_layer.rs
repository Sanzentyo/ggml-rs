//! Resolve-by-layer MLP inference demo with reusable decoded layer weights.

use llama_rs::{
    GgufModel, LlamaBackend, resolve_llama_layer_dimensions, resolve_mlp_weights_for_layer,
    resolve_mlp_weights_for_layer_auto, run_mlp_inference_with_weights_repeats,
};
use std::error::Error as StdError;
use std::path::PathBuf;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let parsed = parse_args(std::env::args().skip(1))?;
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
        let report =
            run_mlp_inference_with_weights_repeats(&weights, &input, backend, parsed.repeats)?;
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

#[derive(Debug)]
struct ParsedArgs {
    model_path: PathBuf,
    layer: usize,
    hidden_features: Option<usize>,
    repeats: usize,
    backends: Vec<LlamaBackend>,
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<ParsedArgs, Box<dyn StdError>> {
    let mut model_path: Option<PathBuf> = None;
    let mut layer = 0usize;
    let mut hidden_features = None;
    let mut repeats = 1usize;
    let mut backends = Vec::new();

    let mut pending_layer = false;
    let mut pending_hidden = false;
    let mut pending_repeats = false;

    for arg in args {
        if pending_layer {
            layer = parse_usize_arg("--layer", &arg)?;
            pending_layer = false;
            continue;
        }
        if pending_hidden {
            hidden_features = Some(parse_usize_arg("--hidden", &arg)?);
            pending_hidden = false;
            continue;
        }
        if pending_repeats {
            repeats = parse_usize_arg("--repeats", &arg)?;
            pending_repeats = false;
            continue;
        }

        match arg.as_str() {
            "--layer" => pending_layer = true,
            "--hidden" => pending_hidden = true,
            "--repeats" | "-n" => pending_repeats = true,
            token => {
                if model_path.is_none() {
                    model_path = Some(PathBuf::from(token));
                } else {
                    backends.push(LlamaBackend::from_str(token)?);
                }
            }
        }
    }

    if pending_layer {
        return Err("missing value after --layer".into());
    }
    if pending_hidden {
        return Err("missing value after --hidden".into());
    }
    if pending_repeats {
        return Err("missing value after --repeats".into());
    }

    let model_path =
        model_path.ok_or("usage: min_infer_mlp_layer <gguf-path> [--layer N] [--hidden H] [--repeats N] [cpu|metal ...]")?;

    if backends.is_empty() {
        backends.push(LlamaBackend::Cpu);
        backends.push(LlamaBackend::Metal);
    }

    Ok(ParsedArgs {
        model_path,
        layer,
        hidden_features,
        repeats,
        backends,
    })
}

fn parse_usize_arg(flag: &str, value: &str) -> Result<usize, Box<dyn StdError>> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid value for {flag}: {value} ({error})").into())
}
