//! Interactive multi-turn chat with a GGUF model.
//!
//! Formats conversation using ChatML (Qwen3.5, etc.), generates with
//! streaming token output, and loops for follow-up turns.
//!
//! Usage:
//!   cargo run --example simple_chat --features link-system -- \
//!     --model path/to/model.gguf --max-tokens 256
//!
//! Type your message and press Enter. Type `/quit` or Ctrl-D to exit.

use clap::Parser;
use llama_rs::{
    ChatError, ChatFormat, ChatMessage, E2eGenerationConfig, GgufModel, GgufTokenizer,
    LlamaBackend, MixedLayerPolicy, detect_chat_format, format_chat_prompt, read_chat_template,
    resolve_eos_token_id,
};
use std::error::Error as StdError;
use std::io::{self, BufRead, Write};
use std::str::FromStr;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
enum ExampleError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    E2e(#[from] llama_rs::E2eError),
    #[error(transparent)]
    Model(#[from] llama_rs::ModelError),
    #[error(transparent)]
    Tokenizer(#[from] llama_rs::TokenizerError),
    #[error(transparent)]
    Chat(#[from] ChatError),
    #[error(transparent)]
    Io(#[from] io::Error),
}

impl From<&'static str> for ExampleError {
    fn from(value: &'static str) -> Self {
        Self::Message(value.to_owned())
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct BackendArg(LlamaBackend);

impl FromStr for BackendArg {
    type Err = <LlamaBackend as FromStr>::Err;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        LlamaBackend::from_str(s).map(Self)
    }
}

#[derive(Parser, Debug)]
#[command(name = "simple_chat")]
#[command(about = "Interactive multi-turn chat with a GGUF model")]
struct Cli {
    /// Path to GGUF model.
    #[arg(long)]
    model: String,

    /// Backend (`cpu` or `metal`).
    #[arg(long, default_value = "cpu")]
    backend: BackendArg,

    /// Maximum number of tokens to generate per turn.
    #[arg(long, short = 'n', default_value_t = 256)]
    max_tokens: usize,

    /// System prompt to prepend to the conversation.
    #[arg(long, short = 's', default_value = "You are a helpful assistant.")]
    system_prompt: String,
}

// ---------------------------------------------------------------------------
// ChatML special tokens
// ---------------------------------------------------------------------------

/// The special tokens used by ChatML that must bypass BPE.
const CHATML_SPECIAL: &[&str] = &["<|im_start|>", "<|im_end|>"];

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn StdError>> {
    let cli = Cli::parse();

    // Load model and tokenizer
    let model = GgufModel::open(&cli.model)?;
    let tokenizer = GgufTokenizer::from_model(&model)?;

    // Detect chat format
    let template = read_chat_template(&model);
    let format = match &template {
        Some(t) => detect_chat_format(t).ok_or(ExampleError::from(
            "model has a chat template but format is not recognized (expected ChatML)",
        ))?,
        None => {
            eprintln!("warning: no chat template in model metadata, assuming ChatML");
            ChatFormat::ChatMl
        }
    };
    eprintln!("chat format: {format}");

    // Resolve stop tokens
    let eos_token_id = resolve_eos_token_id(&model);
    let im_end_token_id = tokenizer.special_token_id("<|im_end|>");

    eprintln!(
        "eos_token_id={:?}, im_end_token_id={:?}",
        eos_token_id, im_end_token_id
    );
    eprintln!(
        "vocab_size={}, max_tokens_per_turn={}",
        tokenizer.vocab_size(),
        cli.max_tokens
    );
    eprintln!("---");

    // Build conversation history
    let mut history: Vec<ChatMessage> = Vec::new();
    if !cli.system_prompt.is_empty() {
        history.push(ChatMessage::system(&cli.system_prompt));
    }

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        // Prompt
        eprint!("> ");
        io::stderr().flush()?;

        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            // EOF
            eprintln!("\n(end of input)");
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/quit" || input == "/exit" {
            break;
        }

        // Add user message
        history.push(ChatMessage::user(input));

        // Format prompt
        let prompt_text = format_chat_prompt(&history, format)?;

        // Encode with special token handling
        let prompt_token_ids =
            tokenizer.encode_with_special_tokens(&prompt_text, CHATML_SPECIAL)?;
        eprintln!(
            "[{} prompt tokens, generating up to {} tokens]",
            prompt_token_ids.len(),
            cli.max_tokens
        );

        // Configure generation
        let config = E2eGenerationConfig::new(cli.backend.0, prompt_token_ids, cli.max_tokens)?
            .with_eos_token_id(eos_token_id)
            .with_mixed_layer_policy(MixedLayerPolicy::Strict);

        // Generate with streaming output
        let report = llama_rs::generate_token_ids_from_model(&model, &config)?;

        // Decode and print response
        let mut decoder = tokenizer.streaming_decoder();
        let mut response_text = String::new();

        for &token_id in &report.generated_token_ids {
            // Stop on <|im_end|> turn terminator
            if im_end_token_id.is_some_and(|id| id == token_id) {
                break;
            }
            if let Some(text) = decoder.next_token(token_id)? {
                print!("{text}");
                io::stdout().flush()?;
                response_text.push_str(&text);
            }
        }
        // Flush any remaining buffered tokens
        if let Some(text) = decoder.flush()? {
            print!("{text}");
            io::stdout().flush()?;
            response_text.push_str(&text);
        }
        println!();

        eprintln!(
            "[generated {} tokens in {:.1}ms ({:.1} ms/tok)]",
            report.generated_token_ids.len(),
            report.elapsed.as_secs_f64() * 1000.0,
            report.avg_generated_token_ms()
        );

        // Add assistant response to history
        history.push(ChatMessage::assistant(response_text.trim()));
    }

    Ok(())
}
