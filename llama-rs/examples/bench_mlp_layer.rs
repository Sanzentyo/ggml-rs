//! Benchmark runner for minimal MLP block inference.

use clap::{Parser, ValueEnum};
use llama_rs::{LlamaBackend, MlpInferenceConfig, MlpWeights, mlp_inference_with_weights_repeats};
use std::error::Error as StdError;
use std::time::Instant;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    run().map_err(Into::into)
}

fn run() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();
    let cli = Cli::parse();
    let cases = cli.parse_cases()?;
    let backends = cli.resolved_backends();

    for &(hidden_features, ffn_features) in &cases {
        let config = MlpInferenceConfig::new(hidden_features, ffn_features)?;
        let weights = MlpWeights::deterministic(config);
        let input: Vec<f32> = (0..hidden_features)
            .map(|index| ((index + 5) % 19) as f32 * 0.125)
            .collect();

        for backend in backends.iter().copied() {
            mlp_inference_with_weights_repeats(&weights, &input, backend, cli.warmup_iters)?;
            let start = Instant::now();
            let report =
                mlp_inference_with_weights_repeats(&weights, &input, backend, cli.bench_iters)?;
            let elapsed = start.elapsed();
            let avg_ms = elapsed.as_secs_f64() * 1000.0 / cli.bench_iters as f64;
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
                cli.warmup_iters,
                cli.bench_iters,
                avg_ms,
                checksum
            );
        }
    }

    Ok(())
}

#[derive(Debug, Error)]
#[error(transparent)]
struct ExampleError(#[from] Box<dyn StdError>);

#[derive(Debug, Clone, Parser)]
#[command(about = "Benchmark deterministic MLP layer workload", version, long_about = None)]
struct Cli {
    #[arg(long = "hidden", default_value_t = 64)]
    hidden_features: usize,
    #[arg(long = "ffn", default_value_t = 128)]
    ffn_features: usize,
    #[arg(long = "cases")]
    cases: Option<String>,
    #[arg(long = "warmup", default_value_t = 3)]
    warmup_iters: usize,
    #[arg(long = "iters", default_value_t = 30)]
    bench_iters: usize,
    #[arg(value_enum)]
    backends: Vec<BackendArg>,
}

impl Cli {
    fn parse_cases(&self) -> Result<Vec<(usize, usize)>, Box<dyn StdError>> {
        if self.bench_iters == 0 {
            return Err("--iters must be greater than zero".into());
        }
        match self.cases.as_deref() {
            Some(value) => parse_cases_arg(value),
            None => Ok(vec![(self.hidden_features, self.ffn_features)]),
        }
    }

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
        let hidden = hidden
            .parse::<usize>()
            .map_err(|error| format!("invalid hidden value in --cases `{hidden}` ({error})"))?;
        let ffn = ffn
            .parse::<usize>()
            .map_err(|error| format!("invalid ffn value in --cases `{ffn}` ({error})"))?;
        cases.push((hidden, ffn));
    }
    if cases.is_empty() {
        return Err("at least one case must be provided for --cases".into());
    }
    Ok(cases)
}
