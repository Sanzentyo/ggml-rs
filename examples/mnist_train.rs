//! Synthetic MNIST train counterpart with safe ggml-rs APIs.

use clap::Parser;
use ggml_rs::{Context, Length, Result, Shape2D, init_timing};
use std::time::Instant;

const SYNTH_NINPUT: usize = 24;
const SYNTH_NCLASSES: usize = 10;
const SYNTH_NSAMPLES: usize = 128;
const SYNTH_EPOCHS: usize = 6;

#[derive(Debug, Clone, Copy)]
struct TrainSummary {
    initial_loss: f64,
    final_loss: f64,
    final_acc: f64,
    checksum: f64,
}

#[derive(Debug, Clone, Copy, Parser)]
#[command(name = "mnist_train")]
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
            "asset-backed MNIST training is not implemented in safe ggml-rs yet; rerun with --synthetic"
        );
        std::process::exit(2);
    }

    let started = Instant::now();
    let summary = run_synthetic_train_iters(cli.synthetic_iters)?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    if cli.synthetic_check {
        let check = run_synthetic_train_iters(1)?;
        let tol = 1e-12;
        let ok = (summary.initial_loss - check.initial_loss).abs() <= tol
            && (summary.final_loss - check.final_loss).abs() <= tol
            && (summary.final_acc - check.final_acc).abs() <= tol
            && (summary.checksum - check.checksum).abs() <= tol;
        if !ok {
            eprintln!("mnist_train synthetic check failed");
            std::process::exit(2);
        }
        println!("mnist_train: synthetic_check=ok");
    }

    println!(
        "synthetic_result kind=mnist-train initial_loss={:.8} final_loss={:.8} final_acc={:.8} checksum={:.8} iters={} elapsed_ms={:.3}",
        summary.initial_loss,
        summary.final_loss,
        summary.final_acc,
        summary.checksum,
        cli.synthetic_iters,
        elapsed_ms
    );

    Ok(())
}

fn synthetic_unit(a: usize, b: usize, c: usize) -> f32 {
    let mixed = (a * 131 + b * 53 + c * 17 + 19) % 997;
    mixed as f32 / 498.5 - 1.0
}

fn run_synthetic_train_iters(iters: usize) -> Result<TrainSummary> {
    let images: Vec<f32> = (0..SYNTH_NSAMPLES)
        .flat_map(|sample| {
            (0..SYNTH_NINPUT).map(move |feature| 0.5 + 0.5 * synthetic_unit(sample, feature, 5))
        })
        .collect();
    let labels: Vec<usize> = (0..SYNTH_NSAMPLES)
        .map(|sample| (sample * 11 + 1) % SYNTH_NCLASSES)
        .collect();

    let initial_weights: Vec<f32> = (0..SYNTH_NCLASSES)
        .flat_map(|class| {
            (0..SYNTH_NINPUT).map(move |feature| 0.05 * synthetic_unit(class, feature, 6))
        })
        .collect();
    let initial_bias: Vec<f32> = (0..SYNTH_NCLASSES)
        .map(|class| 0.01 * synthetic_unit(class, 0, 7))
        .collect();
    let mut weights = initial_weights.clone();
    let mut bias = initial_bias.clone();

    let ctx = Context::new(16 * 1024 * 1024)?;
    let images_t = ctx.new_tensor_2d::<f32>(Shape2D::new(SYNTH_NINPUT, SYNTH_NSAMPLES))?;
    let weights_t = ctx.new_tensor_2d::<f32>(Shape2D::new(SYNTH_NINPUT, SYNTH_NCLASSES))?;
    let bias_t = ctx.new_tensor_1d::<f32>(Length::new(SYNTH_NCLASSES))?;

    images_t.write_data(&images)?;

    let logits_t = ctx.add(&ctx.mul_mat(&weights_t, &images_t)?, &bias_t)?;
    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&logits_t);

    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;
    let mut final_acc = 0.0;

    let mut grad_w = vec![0.0f64; SYNTH_NCLASSES * SYNTH_NINPUT];
    let mut grad_b = [0.0f64; SYNTH_NCLASSES];
    let mut probs = [0.0f64; SYNTH_NCLASSES];

    let mut summary = TrainSummary {
        initial_loss: 0.0,
        final_loss: 0.0,
        final_acc: 0.0,
        checksum: 0.0,
    };
    for _ in 0..iters {
        weights.copy_from_slice(&initial_weights);
        bias.copy_from_slice(&initial_bias);

        for epoch in 0..SYNTH_EPOCHS {
            weights_t.write_data(&weights)?;
            bias_t.write_data(&bias)?;
            ctx.compute(&mut graph, 1)?;

            let logits = logits_t.read_data()?;
            grad_w.fill(0.0);
            grad_b.fill(0.0);

            let mut epoch_loss = 0.0f64;
            let mut epoch_correct = 0usize;

            for sample in 0..SYNTH_NSAMPLES {
                let row_start = sample * SYNTH_NCLASSES;
                let row = &logits[row_start..row_start + SYNTH_NCLASSES];

                let (pred, _) = row
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|(_, lhs), (_, rhs)| {
                        lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or((0, 0.0));
                if pred == labels[sample] {
                    epoch_correct += 1;
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

                for class in 0..SYNTH_NCLASSES {
                    probs[class] = f64::from((row[class] - max_logit).exp()) / exp_sum;
                }

                let label = labels[sample];
                let label_prob = probs[label].max(1e-12);
                epoch_loss += -label_prob.ln();

                let x = &images[sample * SYNTH_NINPUT..(sample + 1) * SYNTH_NINPUT];
                for class in 0..SYNTH_NCLASSES {
                    let target = if class == label { 1.0 } else { 0.0 };
                    let diff = probs[class] - target;
                    grad_b[class] += diff;
                    for (feature, value) in x.iter().copied().enumerate() {
                        grad_w[class * SYNTH_NINPUT + feature] += diff * f64::from(value);
                    }
                }
            }

            let avg_loss = epoch_loss / SYNTH_NSAMPLES as f64;
            if epoch == 0 {
                initial_loss = avg_loss;
            }
            if epoch + 1 == SYNTH_EPOCHS {
                final_loss = avg_loss;
                final_acc = epoch_correct as f64 / SYNTH_NSAMPLES as f64;
            }

            let lr = 0.7 / SYNTH_NSAMPLES as f64;
            for class in 0..SYNTH_NCLASSES {
                bias[class] -= (lr * grad_b[class]) as f32;
                for feature in 0..SYNTH_NINPUT {
                    let idx = class * SYNTH_NINPUT + feature;
                    weights[idx] -= (lr * grad_w[idx]) as f32;
                }
            }
        }

        let mut checksum = 0.0f64;
        for (class, bias_value) in bias.iter().copied().enumerate() {
            checksum += f64::from(bias_value) * (1.0 + class as f64 * 0.2);
            for feature in 0..SYNTH_NINPUT {
                let idx = class * SYNTH_NINPUT + feature;
                checksum += f64::from(weights[idx]) * (1.0 + 0.001 * idx as f64);
            }
        }

        summary = TrainSummary {
            initial_loss,
            final_loss,
            final_acc,
            checksum,
        };
    }

    Ok(summary)
}
