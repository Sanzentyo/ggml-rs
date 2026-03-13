//! Benchmark runner for minimal MLP block inference.

use llama_rs::{
    LlamaBackend, MlpInferenceConfig, MlpWeights, run_mlp_inference_with_weights_repeats,
};
use std::error::Error as StdError;
use std::str::FromStr;
use std::time::Instant;

fn main() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();
    let parsed = parse_args(std::env::args().skip(1))?;

    for &(hidden_features, ffn_features) in &parsed.cases {
        let config = MlpInferenceConfig::new(hidden_features, ffn_features)?;
        let weights = MlpWeights::deterministic(config);
        let input: Vec<f32> = (0..hidden_features)
            .map(|index| ((index + 5) % 19) as f32 * 0.125)
            .collect();

        for backend in parsed.backends.iter().copied() {
            run_mlp_inference_with_weights_repeats(&weights, &input, backend, parsed.warmup_iters)?;
            let start = Instant::now();
            let report = run_mlp_inference_with_weights_repeats(
                &weights,
                &input,
                backend,
                parsed.bench_iters,
            )?;
            let elapsed = start.elapsed();
            let avg_ms = elapsed.as_secs_f64() * 1000.0 / parsed.bench_iters as f64;
            let checksum: f64 = report
                .output
                .iter()
                .take(16)
                .map(|value| f64::from(*value))
                .sum();
            println!(
                "[{}] mlp bench hidden={} ffn={} warmup={} iters={} avg={:.3} ms checksum={:.6}",
                report.backend_name,
                report.hidden_features,
                report.ffn_features,
                parsed.warmup_iters,
                parsed.bench_iters,
                avg_ms,
                checksum
            );
        }
    }

    Ok(())
}

#[derive(Debug)]
struct ParsedArgs {
    cases: Vec<(usize, usize)>,
    warmup_iters: usize,
    bench_iters: usize,
    backends: Vec<LlamaBackend>,
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<ParsedArgs, Box<dyn StdError>> {
    let mut hidden_features = 64usize;
    let mut ffn_features = 128usize;
    let mut cases: Vec<(usize, usize)> = Vec::new();
    let mut warmup_iters = 3usize;
    let mut bench_iters = 30usize;
    let mut backends = Vec::new();

    let mut pending_hidden = false;
    let mut pending_ffn = false;
    let mut pending_cases = false;
    let mut pending_warmup = false;
    let mut pending_iters = false;

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
        if pending_cases {
            cases = parse_cases_arg(&arg)?;
            pending_cases = false;
            continue;
        }
        if pending_warmup {
            warmup_iters = parse_usize_arg("--warmup", &arg)?;
            pending_warmup = false;
            continue;
        }
        if pending_iters {
            bench_iters = parse_usize_arg("--iters", &arg)?;
            pending_iters = false;
            continue;
        }

        match arg.as_str() {
            "--hidden" => pending_hidden = true,
            "--ffn" => pending_ffn = true,
            "--cases" => pending_cases = true,
            "--warmup" => pending_warmup = true,
            "--iters" => pending_iters = true,
            token => backends.push(LlamaBackend::from_str(token)?),
        }
    }

    if pending_hidden {
        return Err("missing value after --hidden".into());
    }
    if pending_ffn {
        return Err("missing value after --ffn".into());
    }
    if pending_cases {
        return Err("missing value after --cases".into());
    }
    if pending_warmup {
        return Err("missing value after --warmup".into());
    }
    if pending_iters {
        return Err("missing value after --iters".into());
    }
    if bench_iters == 0 {
        return Err("--iters must be greater than zero".into());
    }

    if backends.is_empty() {
        backends.push(LlamaBackend::Cpu);
        backends.push(LlamaBackend::Metal);
    }

    if cases.is_empty() {
        cases.push((hidden_features, ffn_features));
    }

    Ok(ParsedArgs {
        cases,
        warmup_iters,
        bench_iters,
        backends,
    })
}

fn parse_usize_arg(flag: &str, value: &str) -> Result<usize, Box<dyn StdError>> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid value for {flag}: {value} ({error})").into())
}

fn parse_cases_arg(value: &str) -> Result<Vec<(usize, usize)>, Box<dyn StdError>> {
    let mut cases = Vec::new();
    for token in value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        let (hidden, ffn) = token
            .split_once('x')
            .ok_or_else(|| format!("invalid case `{token}` (expected HxF)"))?;
        let hidden = parse_usize_arg("--cases", hidden)?;
        let ffn = parse_usize_arg("--cases", ffn)?;
        cases.push((hidden, ffn));
    }
    if cases.is_empty() {
        return Err("at least one case must be provided for --cases".into());
    }
    Ok(cases)
}
