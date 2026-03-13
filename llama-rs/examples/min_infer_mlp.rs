//! Minimal MLP-block inference demo on top of `llama-rs` safe APIs.
//!
//! Computes: `down(silu(gate(x)) * up(x))` with deterministic synthetic weights.

use llama_rs::{
    LlamaBackend, MlpInferenceConfig, MlpWeights, run_mlp_inference_with_weights_repeats,
};
use std::error::Error as StdError;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let parsed = parse_args(std::env::args().skip(1))?;
    let config = MlpInferenceConfig::new(parsed.hidden_features, parsed.ffn_features)?;
    let weights = MlpWeights::deterministic(config);
    let input: Vec<f32> = (0..parsed.hidden_features)
        .map(|index| ((index + 3) % 19) as f32 * 0.125)
        .collect();

    for backend in parsed.backends {
        let report =
            run_mlp_inference_with_weights_repeats(&weights, &input, backend, parsed.repeats)?;
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

#[derive(Debug)]
struct ParsedArgs {
    hidden_features: usize,
    ffn_features: usize,
    repeats: usize,
    backends: Vec<LlamaBackend>,
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<ParsedArgs, Box<dyn StdError>> {
    let mut hidden_features = 64usize;
    let mut ffn_features = 128usize;
    let mut repeats = 2usize;
    let mut backends = Vec::new();

    let mut pending_hidden = false;
    let mut pending_ffn = false;
    let mut pending_repeats = false;

    for arg in args {
        if pending_hidden {
            hidden_features = parse_usize_arg("--hidden", &arg)?;
            pending_hidden = false;
            continue;
        }
        if pending_ffn {
            ffn_features = parse_usize_arg("--ffn", &arg)?;
            pending_ffn = false;
            continue;
        }
        if pending_repeats {
            repeats = parse_usize_arg("--repeats", &arg)?;
            pending_repeats = false;
            continue;
        }

        match arg.as_str() {
            "--hidden" => pending_hidden = true,
            "--ffn" => pending_ffn = true,
            "--repeats" | "-n" => pending_repeats = true,
            token => backends.push(LlamaBackend::from_str(token)?),
        }
    }

    if pending_hidden {
        return Err("missing value after --hidden".into());
    }
    if pending_ffn {
        return Err("missing value after --ffn".into());
    }
    if pending_repeats {
        return Err("missing value after --repeats".into());
    }

    if backends.is_empty() {
        backends.push(LlamaBackend::Cpu);
        backends.push(LlamaBackend::Metal);
    }

    Ok(ParsedArgs {
        hidden_features,
        ffn_features,
        repeats,
        backends,
    })
}

fn parse_usize_arg(flag: &str, value: &str) -> Result<usize, Box<dyn StdError>> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid value for {flag}: {value} ({error})").into())
}
