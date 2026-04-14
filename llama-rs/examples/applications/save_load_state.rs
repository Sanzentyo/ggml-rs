//! Demonstrates save-load-state with `GenerationSession` + `GenerationCheckpoint`.
//!
//! Usage:
//!   cargo run --example save_load_state --features link-system -- \
//!     --model path/to/model.gguf --prompt-tokens 1,2,3 --max-new-tokens 10 \
//!     --checkpoint-after 3 --checkpoint-file /tmp/checkpoint.bin

use clap::Parser;
use llama_rs::{
    E2eGenerationConfig, GenerationCheckpoint, GenerationSession, GgufModel, LlamaBackend,
    MixedLayerPolicy, resolve_eos_token_id,
};
use std::error::Error as StdError;
use std::fs::File;
use std::io::BufWriter;
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
    #[error(transparent)]
    Io(#[from] std::io::Error),
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

#[derive(Parser, Debug)]
#[command(name = "save_load_state")]
#[command(about = "Demonstrates generation checkpoint save/load/resume")]
struct Cli {
    /// Path to GGUF model.
    #[arg(long)]
    model: String,

    /// Backend (`cpu` or `metal`).
    #[arg(long, default_value = "cpu")]
    backend: BackendArg,

    /// Comma-separated prompt token IDs.
    #[arg(long)]
    prompt_tokens: String,

    /// Total number of new tokens to generate.
    #[arg(long, default_value_t = 10)]
    max_new_tokens: usize,

    /// Checkpoint after generating this many tokens (0 = no checkpoint).
    #[arg(long, default_value_t = 3)]
    checkpoint_after: usize,

    /// File path for checkpoint storage.
    #[arg(long, default_value = "/tmp/llama_rs_checkpoint.bin")]
    checkpoint_file: String,

    /// Also run one-shot session for comparison.
    #[arg(long, default_value_t = false)]
    verify: bool,
}

fn main() -> Result<(), Box<dyn StdError>> {
    let cli = Cli::parse();
    let model = GgufModel::open(&cli.model)?;
    let prompt_ids = parse_prompt_tokens(&cli.prompt_tokens)?;
    let eos_token_id = resolve_eos_token_id(&model);

    let config = E2eGenerationConfig::new(cli.backend.0, prompt_ids.clone(), cli.max_new_tokens)?
        .with_eos_token_id(eos_token_id)
        .with_mixed_layer_policy(MixedLayerPolicy::SkipUnsupportedAttention);

    // Phase 1: Generate tokens up to checkpoint point
    println!("=== Phase 1: Generate {} tokens ===", cli.checkpoint_after);
    let mut session = GenerationSession::new(&model, &config)?;
    for i in 0..cli.checkpoint_after {
        match session.next_token()? {
            Some(token) => println!("  token[{i}] = {token}"),
            None => {
                println!("  generation finished early at token {i}");
                break;
            }
        }
    }
    println!("  generated so far: {:?}", session.generated_tokens());

    // Phase 2: Save checkpoint
    if !session.is_finished() && cli.checkpoint_after > 0 {
        println!(
            "\n=== Phase 2: Save checkpoint to {} ===",
            cli.checkpoint_file
        );
        let checkpoint = session.checkpoint();
        let file = File::create(&cli.checkpoint_file)?;
        checkpoint.save_to(BufWriter::new(file))?;

        let file_size = std::fs::metadata(&cli.checkpoint_file)?.len();
        println!("  checkpoint saved ({file_size} bytes)");
        println!(
            "  prompt: {:?}, generated: {:?}",
            checkpoint.prompt_token_ids(),
            checkpoint.generated_token_ids()
        );

        // Phase 3: Load checkpoint and resume
        println!("\n=== Phase 3: Load checkpoint and resume ===");
        let loaded = GenerationCheckpoint::load_from(File::open(&cli.checkpoint_file)?)?;
        println!(
            "  loaded prompt: {:?}, generated: {:?}",
            loaded.prompt_token_ids(),
            loaded.generated_token_ids()
        );

        let mut resumed = GenerationSession::resume(
            &model,
            cli.backend.0,
            MixedLayerPolicy::SkipUnsupportedAttention,
            loaded,
        )?;
        let remaining = cli.max_new_tokens.saturating_sub(cli.checkpoint_after);
        println!("  generating {remaining} more tokens...");
        for i in 0..remaining {
            match resumed.next_token()? {
                Some(token) => println!("  token[{}] = {token}", cli.checkpoint_after + i),
                None => {
                    println!(
                        "  generation finished at token {}",
                        cli.checkpoint_after + i
                    );
                    break;
                }
            }
        }
        println!("\n=== Result (resumed) ===");
        println!("  all tokens: {:?}", resumed.all_tokens());
        println!("  generated:  {:?}", resumed.generated_tokens());

        // Phase 4: Verify against one-shot generation
        if cli.verify {
            println!("\n=== Verification: one-shot generation ===");
            let mut oneshot = GenerationSession::new(&model, &config)?;
            while oneshot.next_token()?.is_some() {}
            println!("  one-shot generated: {:?}", oneshot.generated_tokens());

            if resumed.generated_tokens() == oneshot.generated_tokens() {
                println!("  ✓ MATCH — resumed session produces identical tokens");
            } else {
                println!("  ✗ MISMATCH — tokens differ!");
                println!("    resumed: {:?}", resumed.generated_tokens());
                println!("    oneshot: {:?}", oneshot.generated_tokens());
            }
        }
    } else {
        println!("\n=== Result (no checkpoint needed) ===");
        println!("  all tokens: {:?}", session.all_tokens());
        println!("  generated:  {:?}", session.generated_tokens());
    }

    Ok(())
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
            "--prompt-tokens must include at least one token id".to_owned(),
        ));
    }
    Ok(tokens)
}
