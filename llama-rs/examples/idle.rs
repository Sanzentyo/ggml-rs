use clap::Parser;
use llama_rs::{
    GgufModel, IdleConfig, IdlePauseSchedule, LlamaBackend, PauseScheduleEmpty,
    run_idle_decode_proxy,
};
use std::error::Error as StdError;
use std::str::FromStr;
use thiserror::Error;

const USAGE: &str = "usage: cargo run -p llama-rs --example idle --features link-system -- <model.gguf> [--layer N] [--decode-kv N] [--past N] [--iters N] [--pauses a,b,c] [cpu|metal ...]";

#[derive(Debug, Clone)]
struct ParsedArgs {
    model_path: String,
    layer: usize,
    decode_kv: usize,
    past: usize,
    iters: usize,
    pauses_ms: Vec<u64>,
    backends: Vec<LlamaBackend>,
}

fn main() -> Result<(), ExampleError> {
    ggml_rs::init_timing();
    let args = ParsedArgs::from_cli(Cli::parse());
    let model = GgufModel::open(&args.model_path)?;
    let pauses = IdlePauseSchedule::<PauseScheduleEmpty>::from_vec(args.pauses_ms)?;

    for backend in args.backends {
        let config = IdleConfig::new(
            args.layer,
            args.decode_kv,
            args.past,
            args.iters,
            pauses.clone(),
        )?;
        let report = run_idle_decode_proxy(&model, backend, config)?;
        for pause_report in &report.pauses {
            println!(
                "backend={} requested_layer={} layer={} weights_mode={:?} pause_ms={} avg_decode_ms={:.3} stddev_ms={:.3} iters={} kv={} past={} hidden={} checksum={:.6}",
                report.backend_name,
                report.requested_layer,
                report.layer,
                report.weights_mode,
                pause_report.pause_ms,
                pause_report.average_decode_ms,
                pause_report.stddev_decode_ms,
                report.iterations,
                report.key_value_length,
                report.past_tokens,
                report.hidden_features,
                report.checksum
            );
        }
    }

    Ok(())
}

#[derive(Debug, Error)]
enum ExampleError {
    #[error(transparent)]
    Llama(#[from] llama_rs::LlamaError),
    #[error(transparent)]
    Boxed(#[from] Box<dyn StdError>),
}

impl ParsedArgs {
    fn from_cli(cli: Cli) -> Self {
        let backends = if cli.backends.is_empty() {
            vec![LlamaBackend::Cpu, LlamaBackend::Metal]
        } else {
            cli.backends.into_iter().map(|backend| backend.0).collect()
        };
        Self {
            model_path: cli.model_path,
            layer: cli.layer,
            decode_kv: cli.decode_kv,
            past: cli.past.unwrap_or_else(|| cli.decode_kv.saturating_sub(1)),
            iters: cli.iters,
            pauses_ms: cli.pauses_ms,
            backends,
        }
    }
}

#[derive(Debug, Clone, Parser)]
#[command(about = "Idle decode proxy benchmark", version, after_help = USAGE)]
struct Cli {
    /// Input GGUF model path.
    model_path: String,
    /// Layer index to probe.
    #[arg(long, default_value_t = 0)]
    layer: usize,
    /// Decode KV length.
    #[arg(long = "decode-kv", default_value_t = 128)]
    decode_kv: usize,
    /// Past token count (defaults to decode_kv - 1).
    #[arg(long)]
    past: Option<usize>,
    /// Iteration count per pause bucket.
    #[arg(long = "iters", default_value_t = 3)]
    iters: usize,
    /// Pause buckets in milliseconds.
    #[arg(
        long = "pauses",
        value_delimiter = ',',
        default_values_t = [0_u64, 800_u64, 1_600_u64, 2_400_u64, 3_200_u64, 4_000_u64]
    )]
    pauses_ms: Vec<u64>,
    /// Backend list (e.g. cpu metal). Defaults to cpu+metal.
    backends: Vec<BackendArg>,
}

#[derive(Debug, Clone, Copy)]
struct BackendArg(LlamaBackend);

impl FromStr for BackendArg {
    type Err = <LlamaBackend as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        LlamaBackend::from_str(s).map(Self)
    }
}
