use clap::Parser;
use ggml_rs::{Bytes, Context, Length, Shape2D};
use std::cmp::Ordering;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;
use thiserror::Error;

const MAGIKA_VOCAB: usize = 257;
const MAGIKA_HIDDEN: usize = 96;
const MAGIKA_BEG_SIZE: usize = 512;
const MAGIKA_MID_SIZE: usize = 512;
const MAGIKA_END_SIZE: usize = 512;
const MAGIKA_INPUT_SIZE: usize = MAGIKA_BEG_SIZE + MAGIKA_MID_SIZE + MAGIKA_END_SIZE;
const MAGIKA_PADDING_TOKEN: usize = 256;
const CONTEXT_BYTES: usize = 64 * 1024 * 1024;

const MAGIKA_LABELS: [&str; 16] = [
    "ai", "apk", "csv", "elf", "html", "java", "jpeg", "json", "pdf", "png", "python", "rust",
    "sql", "txt", "xml", "zip",
];

#[derive(Debug, Error)]
enum AppError {
    #[error(transparent)]
    Ggml(#[from] ggml_rs::Error),
    #[error("failed to read `{path}`: {source}")]
    ReadFile {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

type AppResult<T> = Result<T, AppError>;

#[derive(Debug, Clone, Parser)]
#[command(name = "magika_main_synth")]
struct Args {
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long = "samples", default_value_t = 3)]
    synthetic_samples: usize,
    #[arg(value_name = "FILE")]
    files: Vec<PathBuf>,
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

fn synthetic_file_bytes(seed: u64, sample_index: usize) -> Vec<u8> {
    let mut rng = SynthRng::new(seed.wrapping_add((sample_index as u64).wrapping_mul(97)));
    let len = 768 + sample_index * 257;
    (0..len)
        .map(|index| {
            let mixed = rng.next_u64() ^ ((index as u64).wrapping_mul(0xA076_1D64_78BD_642F));
            (mixed & 0xFF) as u8
        })
        .collect()
}

fn sampled_tokens(bytes: &[u8]) -> Vec<usize> {
    let mut sampled = vec![MAGIKA_PADDING_TOKEN; MAGIKA_INPUT_SIZE];

    let n_beg = MAGIKA_BEG_SIZE.min(bytes.len());
    for (index, value) in bytes.iter().take(n_beg).enumerate() {
        sampled[index] = usize::from(*value);
    }

    let mid_offs = bytes
        .len()
        .saturating_sub(MAGIKA_MID_SIZE)
        .saturating_div(2);
    let mid_end = (mid_offs + MAGIKA_MID_SIZE).min(bytes.len());
    let mid_slice = &bytes[mid_offs..mid_end];
    let mid_start = MAGIKA_BEG_SIZE + (MAGIKA_MID_SIZE / 2).saturating_sub(mid_slice.len() / 2);
    for (index, value) in mid_slice.iter().enumerate() {
        sampled[mid_start + index] = usize::from(*value);
    }

    let end_offs = bytes.len().saturating_sub(MAGIKA_END_SIZE);
    let end_slice = &bytes[end_offs..];
    let end_start = MAGIKA_BEG_SIZE + MAGIKA_MID_SIZE + MAGIKA_END_SIZE - end_slice.len();
    for (index, value) in end_slice.iter().enumerate() {
        sampled[end_start + index] = usize::from(*value);
    }

    sampled
}

fn histogram_features(tokens: &[usize]) -> Vec<f32> {
    let mut histogram = vec![0.0f32; MAGIKA_VOCAB];
    for token in tokens {
        histogram[*token] += 1.0;
    }
    let normalizer = tokens.len() as f32;
    for value in &mut histogram {
        *value /= normalizer;
    }
    histogram
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

fn format_label_summary(sample_probs: &[f32]) -> String {
    top_k(sample_probs, 3)
        .into_iter()
        .map(|(index, value)| format!("{}@{value:.6}", MAGIKA_LABELS[index]))
        .collect::<Vec<_>>()
        .join(",")
}

fn probability_checksum(probs: &[f32], sample_count: usize) -> f64 {
    let label_count = MAGIKA_LABELS.len();
    let mut checksum = 0.0f64;
    for sample_index in 0..sample_count {
        let row = &probs[sample_index * label_count..(sample_index + 1) * label_count];
        for (label_index, prob) in row.iter().enumerate() {
            checksum += (*prob as f64) * ((sample_index + 1) as f64) * ((label_index + 1) as f64);
        }
    }
    checksum
}

fn main() -> AppResult<()> {
    ggml_rs::init_timing();
    let args = Args::parse();
    let synthetic_samples = args.synthetic_samples.max(1);

    let source_files = if args.files.is_empty() {
        (0..synthetic_samples)
            .map(|index| synthetic_file_bytes(args.seed, index))
            .collect::<Vec<_>>()
    } else {
        args.files
            .iter()
            .map(|path| {
                std::fs::read(path).map_err(|source| AppError::ReadFile {
                    path: path.display().to_string(),
                    source,
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    let sample_count = source_files.len();
    let mut feature_matrix = Vec::with_capacity(sample_count * MAGIKA_VOCAB);
    for bytes in &source_files {
        let sampled = sampled_tokens(bytes);
        feature_matrix.extend(histogram_features(&sampled));
    }

    let dense_w = synth_values(
        args.seed.wrapping_add(101),
        MAGIKA_HIDDEN * MAGIKA_VOCAB,
        0.42,
    );
    let dense_b = synth_values(args.seed.wrapping_add(103), MAGIKA_HIDDEN, 0.08);
    let head_w = synth_values(
        args.seed.wrapping_add(107),
        MAGIKA_LABELS.len() * MAGIKA_HIDDEN,
        0.28,
    );
    let head_b = synth_values(args.seed.wrapping_add(109), MAGIKA_LABELS.len(), 0.07);

    let started = Instant::now();

    let ctx = Context::new_bytes(Bytes::new(CONTEXT_BYTES))?;
    let input = ctx.new_tensor_2d::<f32>(Shape2D::new(MAGIKA_VOCAB, sample_count))?;
    let dense_weight = ctx.new_tensor_2d::<f32>(Shape2D::new(MAGIKA_VOCAB, MAGIKA_HIDDEN))?;
    let dense_bias = ctx.new_tensor_1d::<f32>(Length::new(MAGIKA_HIDDEN))?;
    let head_weight = ctx.new_tensor_2d::<f32>(Shape2D::new(MAGIKA_HIDDEN, MAGIKA_LABELS.len()))?;
    let head_bias = ctx.new_tensor_1d::<f32>(Length::new(MAGIKA_LABELS.len()))?;

    input.write_data(&feature_matrix)?;
    dense_weight.write_data(&dense_w)?;
    dense_bias.write_data(&dense_b)?;
    head_weight.write_data(&head_w)?;
    head_bias.write_data(&head_b)?;

    let hidden_linear = ctx.mul_mat(&dense_weight, &input)?;
    let hidden_bias = ctx.repeat(&dense_bias, &hidden_linear)?;
    let hidden = ctx.add(&hidden_linear, &hidden_bias)?;
    let hidden = ctx.silu(&hidden)?;

    let logits_linear = ctx.mul_mat(&head_weight, &hidden)?;
    let logits_bias = ctx.repeat(&head_bias, &logits_linear)?;
    let logits = ctx.add(&logits_linear, &logits_bias)?;
    let probs = ctx.soft_max(&logits)?;

    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&probs);
    ctx.compute(&mut graph, 1)?;

    let probs = graph.last_node()?.read_data::<f32>()?;
    let elapsed_us = started.elapsed().as_micros();

    let label_count = MAGIKA_LABELS.len();
    let mut labels_top = String::new();
    for sample_index in 0..sample_count {
        if sample_index > 0 {
            labels_top.push(';');
        }
        let row = &probs[sample_index * label_count..(sample_index + 1) * label_count];
        let _ = write!(
            &mut labels_top,
            "sample{sample_index}:{}",
            format_label_summary(row)
        );
    }

    println!("mode=magika-main-synth");
    println!("seed={}", args.seed);
    println!("samples={sample_count}");
    println!("labels_top={labels_top}");
    println!(
        "prob_checksum={:.9}",
        probability_checksum(&probs, sample_count)
    );
    println!("elapsed_us={elapsed_us}");

    Ok(())
}
