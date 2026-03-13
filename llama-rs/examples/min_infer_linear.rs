//! Minimal linear inference demo using reusable GGUF-decoded weights.

use llama_rs::{
    GgufModel, LinearInferenceConfig, LinearWeights, LlamaBackend,
    run_linear_inference_with_weights_repeats,
};
use std::error::Error as StdError;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let parsed = parse_args(&args)?;
    let model = GgufModel::open(&parsed.path)?;
    let weights = LinearWeights::from_model(&model, &parsed.weight_tensor, parsed.config)?;
    let input = make_input(parsed.config.in_features());
    let report = run_linear_inference_with_weights_repeats(
        &weights,
        &input,
        parsed.backend,
        parsed.repeats,
    )?;

    let preview: Vec<f32> = report.output.iter().copied().take(8).collect();
    println!(
        "[{}] linear inference OK: in={} out={} repeats={} preview={preview:?}",
        report.backend_name, report.in_features, report.out_features, parsed.repeats
    );
    Ok(())
}

struct ParsedArgs {
    path: String,
    weight_tensor: String,
    config: LinearInferenceConfig,
    backend: LlamaBackend,
    repeats: usize,
}

fn parse_args(args: &[String]) -> Result<ParsedArgs, Box<dyn StdError>> {
    if args.len() < 4 {
        return Err("usage: cargo run -p llama-rs --example min_infer_linear --features link-system -- <model.gguf> <weight_tensor_name> --in N --out M [--repeats N] [cpu|metal]".into());
    }

    let path = args[0].clone();
    let weight_tensor = args[1].clone();
    let mut input_cols = None;
    let mut output_rows = None;
    let mut backend = LlamaBackend::Cpu;
    let mut repeats = 1usize;

    let mut index = 2usize;
    while index < args.len() {
        match args[index].as_str() {
            "--in" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err("missing value after --in".into());
                };
                input_cols = Some(value.parse::<usize>()?);
            }
            "--out" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err("missing value after --out".into());
                };
                output_rows = Some(value.parse::<usize>()?);
            }
            "--repeats" | "-n" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err("missing value after --repeats".into());
                };
                repeats = value.parse::<usize>()?;
            }
            token => {
                backend = LlamaBackend::from_str(token)?;
            }
        }
        index += 1;
    }

    let Some(input_cols) = input_cols else {
        return Err("missing required --in".into());
    };
    let Some(output_rows) = output_rows else {
        return Err("missing required --out".into());
    };

    Ok(ParsedArgs {
        path,
        weight_tensor,
        config: LinearInferenceConfig::builder()
            .in_features(input_cols)?
            .out_features(output_rows)?
            .build(),
        backend,
        repeats,
    })
}

fn make_input(input_cols: usize) -> Vec<f32> {
    (0..input_cols)
        .map(|index| (index % 11) as f32 * 0.125)
        .collect()
}
