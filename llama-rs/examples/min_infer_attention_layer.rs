//! Resolve-by-layer minimal attention inference demo.

use llama_rs::{
    AttentionMaskPolicy, GgufModel, LlamaBackend, RotaryEmbedding,
    resolve_attention_weights_for_layer_auto, resolve_llama_layer_dimensions,
    run_attention_inference_with_weights_repeats,
};
use std::error::Error as StdError;
use std::path::PathBuf;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let parsed = parse_args(std::env::args().skip(1))?;
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
        let report = run_attention_inference_with_weights_repeats(
            &weights,
            &input,
            backend,
            parsed.repeats,
        )?;
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

fn parse_args(args: impl Iterator<Item = String>) -> Result<ParsedArgs, Box<dyn StdError>> {
    let mut model_path: Option<PathBuf> = None;
    let mut layer = 0usize;
    let mut sequence_length = 4usize;
    let mut repeats = 1usize;
    let mut causal = false;
    let mut no_rope = false;
    let mut backends = Vec::new();

    let mut pending_layer = false;
    let mut pending_seq = false;
    let mut pending_repeats = false;

    for arg in args {
        if pending_layer {
            layer = parse_usize_arg("--layer", &arg)?;
            pending_layer = false;
            continue;
        }
        if pending_seq {
            sequence_length = parse_usize_arg("--seq", &arg)?;
            pending_seq = false;
            continue;
        }
        if pending_repeats {
            repeats = parse_usize_arg("--repeats", &arg)?;
            pending_repeats = false;
            continue;
        }

        match arg.as_str() {
            "--layer" => pending_layer = true,
            "--seq" => pending_seq = true,
            "--repeats" | "-n" => pending_repeats = true,
            "--causal" => causal = true,
            "--no-rope" => no_rope = true,
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
    if pending_seq {
        return Err("missing value after --seq".into());
    }
    if pending_repeats {
        return Err("missing value after --repeats".into());
    }

    let model_path = model_path.ok_or(
        "usage: min_infer_attention_layer <gguf-path> [--layer N] [--seq S] [--causal] [--no-rope] [--repeats N] [cpu|metal ...]",
    )?;

    if backends.is_empty() {
        backends.push(LlamaBackend::Cpu);
        backends.push(LlamaBackend::Metal);
    }

    Ok(ParsedArgs {
        model_path,
        layer,
        sequence_length,
        repeats,
        causal,
        no_rope,
        backends,
    })
}

fn parse_usize_arg(flag: &str, value: &str) -> Result<usize, Box<dyn StdError>> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid value for {flag}: {value} ({error})").into())
}
