//! Runs token-id based end-to-end generation on a transformer-style GGUF model.

use clap::{Parser, ValueEnum};
use llama_rs::{
    E2eGenerationConfig, GgufModel, LlamaBackend, MixedLayerPolicy, generate_token_ids_from_model,
    resolve_eos_token_id, tokenize_prompt_text,
};
use std::error::Error as StdError;
use std::str::FromStr;
use thiserror::Error;

#[derive(Debug, Error)]
enum ExampleError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    E2e(#[from] llama_rs::E2eError),
    #[error(transparent)]
    Model(#[from] llama_rs::ModelError),
}

impl From<&'static str> for ExampleError {
    fn from(value: &'static str) -> Self {
        Self::Message(value.to_owned())
    }
}

impl From<String> for ExampleError {
    fn from(value: String) -> Self {
        Self::Message(value)
    }
}

#[derive(Debug, Clone, Copy)]
struct BackendArg(LlamaBackend);

impl FromStr for BackendArg {
    type Err = <LlamaBackend as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        LlamaBackend::from_str(s).map(Self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum MixedLayerPolicyArg {
    Strict,
    SkipUnsupportedAttention,
}

impl From<MixedLayerPolicyArg> for MixedLayerPolicy {
    fn from(value: MixedLayerPolicyArg) -> Self {
        match value {
            MixedLayerPolicyArg::Strict => Self::Strict,
            MixedLayerPolicyArg::SkipUnsupportedAttention => Self::SkipUnsupportedAttention,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "e2e_generate_tokens")]
#[command(about = "Token-id based E2E generation with llama-rs")]
struct Cli {
    /// Path to GGUF model.
    #[arg(long)]
    model: String,
    /// Backend (`cpu` or `metal`).
    #[arg(long, default_value = "cpu")]
    backend: BackendArg,
    /// Comma-separated prompt token IDs, e.g. `1,123,456`.
    #[arg(long, conflicts_with = "prompt_text")]
    prompt_tokens: Option<String>,
    /// Raw text prompt tokenized from GGUF tokenizer metadata.
    #[arg(long, conflicts_with = "prompt_tokens")]
    prompt_text: Option<String>,
    /// Maximum number of new tokens to generate.
    #[arg(long, default_value_t = 1)]
    max_new_tokens: usize,
    /// Pad token ID used for fixed-sequence execution.
    #[arg(long, default_value_t = 0)]
    pad_token_id: i32,
    /// Optional EOS token ID (defaults to GGUF metadata when available).
    #[arg(long)]
    eos_token_id: Option<i32>,
    /// Policy for transformer layers missing llama-style attention roles.
    #[arg(long, value_enum, default_value_t = MixedLayerPolicyArg::Strict)]
    mixed_layer_policy: MixedLayerPolicyArg,
}

fn main() -> Result<(), Box<dyn StdError>> {
    let cli = Cli::parse();
    let model = GgufModel::open(&cli.model)?;
    let prompt_token_ids = resolve_prompt_token_ids(&cli, &model)?;
    let eos_token_id = cli.eos_token_id.or_else(|| resolve_eos_token_id(&model));

    let config = E2eGenerationConfig::new(cli.backend.0, prompt_token_ids, cli.max_new_tokens)?
        .with_pad_token_id(cli.pad_token_id)
        .with_eos_token_id(eos_token_id)
        .with_mixed_layer_policy(cli.mixed_layer_policy.into());
    let report = generate_token_ids_from_model(&model, &config)?;

    println!(
        "backend={} prompt_tokens={} generated_tokens={} attention_layers={} mlp_only_layers={} avg_token_ms={:.3} elapsed_ms={:.3}",
        report.backend_name,
        report.prompt_token_count,
        report.generated_token_ids.len(),
        report.attention_layer_count,
        report.mlp_only_layer_count,
        report.avg_generated_token_ms(),
        report.elapsed.as_secs_f64() * 1000.0
    );
    println!("all_token_ids={:?}", report.all_token_ids);
    println!("generated_token_ids={:?}", report.generated_token_ids);

    Ok(())
}

fn resolve_prompt_token_ids(cli: &Cli, model: &GgufModel) -> Result<Vec<i32>, ExampleError> {
    match (&cli.prompt_tokens, &cli.prompt_text) {
        (Some(tokens), None) => parse_prompt_tokens(tokens),
        (None, Some(text)) => tokenize_prompt_text(model, text).map_err(Into::into),
        _ => Err(ExampleError::from(
            "provide exactly one of --prompt-tokens or --prompt-text",
        )),
    }
}

fn parse_prompt_tokens(value: &str) -> Result<Vec<i32>, ExampleError> {
    let tokens: Vec<i32> = value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| {
            token
                .parse::<i32>()
                .map_err(|error| format!("invalid token id `{token}`: {error}"))
        })
        .collect::<Result<_, _>>()?;
    if tokens.is_empty() {
        return Err(ExampleError::from(
            "--prompt-tokens must include at least one token id",
        ));
    }
    Ok(tokens)
}
