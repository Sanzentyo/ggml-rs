use clap::Parser;
use ggml_rs::{Bytes, Context, Length};
use std::time::Instant;
use thiserror::Error;

const DEFAULT_TENSOR_LEN: usize = 4096;
const CONTEXT_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Error)]
enum AppError {
    #[error(transparent)]
    Ggml(#[from] ggml_rs::Error),
}

type AppResult<T> = Result<T, AppError>;

#[derive(Debug, Clone, Parser)]
#[command(name = "gptj_quantize_synth")]
struct Args {
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long = "tensor-len", default_value_t = DEFAULT_TENSOR_LEN)]
    tensor_len: usize,
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

fn synth_values(seed: u64, len: usize, scale: f32) -> Vec<f32> {
    let mut rng = SynthRng::new(seed);
    (0..len).map(|_| rng.next_f32_signed() * scale).collect()
}

fn quantize_q8_0(values: &[f32]) -> (f32, Vec<i8>, Vec<f32>) {
    let max_abs = values
        .iter()
        .fold(0.0f32, |current, value| current.max(value.abs()));
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };

    let quantized = values
        .iter()
        .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    let dequantized = quantized
        .iter()
        .map(|value| (*value as f32) * scale)
        .collect::<Vec<_>>();
    (scale, quantized, dequantized)
}

fn quantized_checksum(values: &[i8]) -> i64 {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| (index as i64 + 1) * (*value as i64))
        .sum()
}

fn main() -> AppResult<()> {
    ggml_rs::init_timing();
    let args = Args::parse();
    let started = Instant::now();
    let tensor_len = args.tensor_len.max(1);

    let tensor_values = synth_values(args.seed.wrapping_add(43), tensor_len, 1.75);

    let ctx = Context::new_bytes(Bytes::new(CONTEXT_BYTES))?;
    let tensor = ctx.new_tensor_1d::<f32>(Length::new(tensor_len))?;
    tensor.write_data(&tensor_values)?;
    let source = tensor.read_data()?;

    let (scale, quantized, dequantized) = quantize_q8_0(&source);

    let mut mse = 0.0f64;
    let mut max_abs_err = 0.0f32;
    for (orig, restored) in source.iter().zip(&dequantized) {
        let diff = *orig - *restored;
        mse += (diff as f64) * (diff as f64);
        max_abs_err = max_abs_err.max(diff.abs());
    }
    mse /= source.len() as f64;

    let elapsed_us = started.elapsed().as_micros();
    let first_q = quantized
        .iter()
        .take(8)
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");

    println!("mode=gptj-quantize-synth");
    println!("seed={}", args.seed);
    println!("tensor_len={tensor_len}");
    println!("scale={scale:.9}");
    println!("quantized_head={first_q}");
    println!("quantized_checksum={}", quantized_checksum(&quantized));
    println!("mse={mse:.12}");
    println!("max_abs_err={max_abs_err:.9}");
    println!("elapsed_us={elapsed_us}");

    Ok(())
}
