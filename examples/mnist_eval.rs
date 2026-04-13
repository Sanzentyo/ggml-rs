//! Synthetic MNIST eval counterpart with safe ggml-rs APIs.

use clap::Parser;
use ggml_rs::{Context, Length, Result, Shape2D, init_timing};
use std::time::Instant;

const SYNTH_NINPUT: usize = 32;
const SYNTH_NHIDDEN: usize = 24;
const SYNTH_NCLASSES: usize = 10;
const SYNTH_NSAMPLES: usize = 64;

#[derive(Debug, Clone, Copy)]
struct EvalSummary {
    loss: f64,
    accuracy: f64,
    checksum: f64,
    pred0: usize,
}

#[derive(Debug, Clone, Copy, Parser)]
#[command(name = "mnist_eval")]
struct Cli {
    #[arg(long = "synthetic", default_value_t = false)]
    synthetic_mode: bool,
    #[arg(long = "synthetic-check", default_value_t = false)]
    synthetic_check: bool,
    #[arg(long = "synthetic-iters", default_value_t = 1)]
    synthetic_iters: usize,
}

fn main() -> Result<()> {
    init_timing();

    let mut cli = Cli::parse();
    cli.synthetic_iters = cli.synthetic_iters.max(1);
    if !cli.synthetic_mode {
        eprintln!(
            "asset-backed MNIST eval is not implemented in safe ggml-rs yet; rerun with --synthetic"
        );
        std::process::exit(2);
    }

    let started = Instant::now();
    let summary = run_synthetic_eval_iters(cli.synthetic_iters)?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    if cli.synthetic_check {
        let check = run_synthetic_eval_iters(1)?;
        let tol = 1e-12;
        let ok = (summary.loss - check.loss).abs() <= tol
            && (summary.accuracy - check.accuracy).abs() <= tol
            && (summary.checksum - check.checksum).abs() <= tol
            && summary.pred0 == check.pred0;
        if !ok {
            eprintln!("mnist_eval synthetic check failed");
            std::process::exit(2);
        }
        println!("mnist_eval: synthetic_check=ok");
    }

    println!(
        "synthetic_result kind=mnist-eval loss={:.8} acc={:.8} checksum={:.8} pred0={} iters={} elapsed_ms={:.3}",
        summary.loss,
        summary.accuracy,
        summary.checksum,
        summary.pred0,
        cli.synthetic_iters,
        elapsed_ms
    );

    Ok(())
}

fn synthetic_unit(a: usize, b: usize, c: usize) -> f32 {
    let mixed = (a * 131 + b * 53 + c * 17 + 19) % 997;
    mixed as f32 / 498.5 - 1.0
}

fn run_synthetic_eval_iters(iters: usize) -> Result<EvalSummary> {
    let images: Vec<f32> = (0..SYNTH_NSAMPLES)
        .flat_map(|sample| {
            (0..SYNTH_NINPUT).map(move |feature| 0.5 + 0.5 * synthetic_unit(sample, feature, 0))
        })
        .collect();
    let labels: Vec<usize> = (0..SYNTH_NSAMPLES)
        .map(|sample| (sample * 7 + 3) % SYNTH_NCLASSES)
        .collect();

    let w1: Vec<f32> = (0..SYNTH_NHIDDEN)
        .flat_map(|hidden| {
            (0..SYNTH_NINPUT).map(move |feature| 0.12 * synthetic_unit(hidden, feature, 1))
        })
        .collect();
    let b1: Vec<f32> = (0..SYNTH_NHIDDEN)
        .map(|hidden| 0.03 * synthetic_unit(hidden, 0, 2))
        .collect();

    let w2: Vec<f32> = (0..SYNTH_NCLASSES)
        .flat_map(|class| {
            (0..SYNTH_NHIDDEN).map(move |hidden| 0.10 * synthetic_unit(class, hidden, 3))
        })
        .collect();
    let b2: Vec<f32> = (0..SYNTH_NCLASSES)
        .map(|class| 0.02 * synthetic_unit(class, 0, 4))
        .collect();

    let ctx = Context::new(16 * 1024 * 1024)?;
    let images_t = ctx.new_tensor_2d::<f32>(Shape2D::new(SYNTH_NINPUT, SYNTH_NSAMPLES))?;
    let w1_t = ctx.new_tensor_2d::<f32>(Shape2D::new(SYNTH_NINPUT, SYNTH_NHIDDEN))?;
    let b1_t = ctx.new_tensor_1d::<f32>(Length::new(SYNTH_NHIDDEN))?;
    let w2_t = ctx.new_tensor_2d::<f32>(Shape2D::new(SYNTH_NHIDDEN, SYNTH_NCLASSES))?;
    let b2_t = ctx.new_tensor_1d::<f32>(Length::new(SYNTH_NCLASSES))?;

    images_t.write_data(&images)?;
    w1_t.write_data(&w1)?;
    b1_t.write_data(&b1)?;
    w2_t.write_data(&w2)?;
    b2_t.write_data(&b2)?;

    let hidden = ctx.add(&ctx.mul_mat(&w1_t, &images_t)?, &b1_t)?;
    let logits = ctx.add(&ctx.mul_mat(&w2_t, &hidden)?, &b2_t)?;

    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&logits);
    let mut summary = EvalSummary {
        loss: 0.0,
        accuracy: 0.0,
        checksum: 0.0,
        pred0: 0,
    };
    for _ in 0..iters {
        ctx.compute(&mut graph, 1)?;
        let values = logits.read_data()?;
        summary = summarize_eval_logits(&values, &labels);
    }
    Ok(summary)
}

fn summarize_eval_logits(values: &[f32], labels: &[usize]) -> EvalSummary {
    let mut loss = 0.0f64;
    let mut correct = 0usize;
    let mut checksum = 0.0f64;
    let mut pred0 = 0usize;

    for sample in 0..SYNTH_NSAMPLES {
        let row_start = sample * SYNTH_NCLASSES;
        let row = &values[row_start..row_start + SYNTH_NCLASSES];

        let (pred, _) = row
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));
        if sample == 0 {
            pred0 = pred;
        }
        if pred == labels[sample] {
            correct += 1;
        }

        for (class, logit) in row.iter().copied().enumerate() {
            checksum += f64::from(logit) * (1.0 + 0.1 * class as f64 + 0.01 * sample as f64);
        }

        let max_logit = row
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |acc, item| acc.max(item));
        let exp_sum: f64 = row
            .iter()
            .copied()
            .map(|logit| f64::from((logit - max_logit).exp()))
            .sum();
        let label_logit = row[labels[sample]];
        let label_exp = f64::from((label_logit - max_logit).exp());
        let label_prob = (label_exp / exp_sum).max(1e-12);
        loss += -label_prob.ln();
    }

    EvalSummary {
        loss: loss / SYNTH_NSAMPLES as f64,
        accuracy: correct as f64 / SYNTH_NSAMPLES as f64,
        checksum,
        pred0,
    }
}
