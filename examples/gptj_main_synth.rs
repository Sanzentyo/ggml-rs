use clap::Parser;
use ggml_rs::{Bytes, Context, Length, Shape2D};
use std::cmp::Ordering;
use std::fmt::Write as _;
use std::time::Instant;
use thiserror::Error;

const GPTJ_VOCAB: usize = 64;
const GPTJ_EMBED: usize = 32;
const CONTEXT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
enum AppError {
    #[error(transparent)]
    Ggml(#[from] ggml_rs::Error),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

type AppResult<T> = Result<T, AppError>;

#[derive(Debug, Clone, Parser)]
#[command(name = "gptj_main_synth")]
struct Args {
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long = "n-predict", default_value_t = 8)]
    n_predict: usize,
    #[arg(long, default_value = "ggml-rs synthetic gpt-j")]
    prompt: String,
    #[arg(long, default_value_t = false)]
    _synthetic: bool,
}

#[derive(Clone)]
struct SynthRng {
    state: u64,
}

impl SynthRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f32_signed(&mut self) -> f32 {
        let unit_bits = (self.next_u64() >> 40) as u32;
        let unit = unit_bits as f32 / ((1u32 << 24) - 1) as f32;
        unit.mul_add(2.0, -1.0)
    }
}

#[derive(Clone)]
struct GptjSynthWeights {
    token_embedding: Vec<f32>,
    proj_weight: Vec<f32>,
    proj_bias: Vec<f32>,
    head_weight: Vec<f32>,
    head_bias: Vec<f32>,
}

impl GptjSynthWeights {
    fn build(seed: u64) -> Self {
        Self {
            token_embedding: synth_values(seed.wrapping_add(11), GPTJ_EMBED * GPTJ_VOCAB, 0.45),
            proj_weight: synth_values(seed.wrapping_add(17), GPTJ_EMBED * GPTJ_EMBED, 0.35),
            proj_bias: synth_values(seed.wrapping_add(23), GPTJ_EMBED, 0.12),
            head_weight: synth_values(seed.wrapping_add(29), GPTJ_VOCAB * GPTJ_EMBED, 0.30),
            head_bias: synth_values(seed.wrapping_add(31), GPTJ_VOCAB, 0.09),
        }
    }
}

fn synth_values(seed: u64, len: usize, scale: f32) -> Vec<f32> {
    let mut rng = SynthRng::new(seed);
    (0..len).map(|_| rng.next_f32_signed() * scale).collect()
}

fn tokenize_prompt(prompt: &str) -> Vec<i32> {
    let mut tokens: Vec<i32> = prompt
        .as_bytes()
        .iter()
        .map(|byte| (usize::from(*byte) % GPTJ_VOCAB) as i32)
        .collect();
    if tokens.is_empty() {
        tokens.extend([1, 2, 3, 4]);
    }
    tokens
}

fn top_k(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|lhs, rhs| {
        values[*rhs]
            .partial_cmp(&values[*lhs])
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs.cmp(rhs))
    });
    order
        .into_iter()
        .take(k)
        .map(|index| (index, values[index]))
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn join_i32(values: &[i32]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn format_logits_top(values: &[(usize, f32)]) -> String {
    let mut out = String::new();
    for (index, (token, value)) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let _ = write!(&mut out, "{token}:{value:.6}");
    }
    out
}

fn logits_checksum(logits: &[f32]) -> f64 {
    logits
        .iter()
        .enumerate()
        .map(|(index, value)| (*value as f64) * ((index + 1) as f64))
        .sum()
}

fn main() -> AppResult<()> {
    ggml_rs::init_timing();
    let args = Args::parse();
    let weights = GptjSynthWeights::build(args.seed);
    let n_predict = args.n_predict.max(1);

    let mut tokens = tokenize_prompt(&args.prompt);
    let prompt_len = tokens.len();
    let max_tokens = prompt_len
        .checked_add(n_predict)
        .ok_or_else(|| AppError::InvalidArgument("token count overflow".into()))?;

    let ctx = Context::new_bytes(Bytes::new(CONTEXT_BYTES))?;
    let token_tensor = ctx.new_tensor_1d::<i32>(Length::new(max_tokens))?;
    let token_embedding = ctx.new_tensor_2d::<f32>(Shape2D::new(GPTJ_EMBED, GPTJ_VOCAB))?;
    let proj_weight = ctx.new_tensor_2d::<f32>(Shape2D::new(GPTJ_EMBED, GPTJ_EMBED))?;
    let proj_bias = ctx.new_tensor_1d::<f32>(Length::new(GPTJ_EMBED))?;
    let head_weight = ctx.new_tensor_2d::<f32>(Shape2D::new(GPTJ_EMBED, GPTJ_VOCAB))?;
    let head_bias = ctx.new_tensor_1d::<f32>(Length::new(GPTJ_VOCAB))?;

    token_embedding.write_data(&weights.token_embedding)?;
    proj_weight.write_data(&weights.proj_weight)?;
    proj_bias.write_data(&weights.proj_bias)?;
    head_weight.write_data(&weights.head_weight)?;
    head_bias.write_data(&weights.head_bias)?;

    let embeddings = ctx.get_rows(&token_embedding, &token_tensor)?;
    let hidden_linear = ctx.mul_mat(&proj_weight, &embeddings)?;
    let hidden_bias = ctx.repeat(&proj_bias, &hidden_linear)?;
    let hidden = ctx.add(&hidden_linear, &hidden_bias)?;
    let hidden = ctx.silu(&hidden)?;

    let logits_linear = ctx.mul_mat(&head_weight, &hidden)?;
    let logits_bias = ctx.repeat(&head_bias, &logits_linear)?;
    let logits = ctx.add(&logits_linear, &logits_bias)?;

    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&logits);

    let mut token_buffer = vec![0_i32; max_tokens];
    token_buffer[..tokens.len()].copy_from_slice(&tokens);
    let started = Instant::now();

    let mut generated = Vec::with_capacity(n_predict);
    let mut final_logits = vec![0.0f32; GPTJ_VOCAB];

    for _ in 0..n_predict {
        token_buffer[..tokens.len()].copy_from_slice(&tokens);
        token_tensor.write_data(&token_buffer)?;
        ctx.compute(&mut graph, 1)?;

        let start = GPTJ_VOCAB
            .checked_mul(tokens.len().saturating_sub(1))
            .ok_or_else(|| AppError::InvalidArgument("logit indexing overflow".into()))?;
        final_logits = graph
            .last_node_typed::<f32>()?
            .read_data_at(start, GPTJ_VOCAB)?;
        let next = argmax(&final_logits) as i32;
        tokens.push(next);
        generated.push(next);
    }

    let top5 = top_k(&final_logits, 5);
    let elapsed_us = started.elapsed().as_micros();

    println!("mode=gptj-main-synth");
    println!("seed={}", args.seed);
    println!("prompt_len={prompt_len}");
    println!("n_predict={n_predict}");
    println!("generated_tokens={}", join_i32(&generated));
    println!("logits_top5={}", format_logits_top(&top5));
    println!("logit_checksum={:.9}", logits_checksum(&final_logits));
    println!("elapsed_us={elapsed_us}");

    Ok(())
}
