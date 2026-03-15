//! Synthetic YOLOv3-tiny counterpart with safe ggml-rs APIs.

use clap::Parser;
use ggml_rs::{Context, Length, Result, Shape2D, init_timing};
use std::cmp::Ordering;
use std::time::Instant;

const YOLO_SYNTH_GRID: usize = 4;
const YOLO_SYNTH_CELLS: usize = YOLO_SYNTH_GRID * YOLO_SYNTH_GRID;
const YOLO_SYNTH_FEATURES: usize = 6;
const YOLO_SYNTH_CLASSES: usize = 3;
const YOLO_SYNTH_ANCHORS: usize = 3;
const YOLO_SYNTH_CHANNELS: usize = YOLO_SYNTH_ANCHORS * (YOLO_SYNTH_CLASSES + 5);

#[derive(Debug, Clone, Copy, Parser)]
#[command(name = "yolov3_tiny")]
struct Cli {
    #[arg(long = "synthetic", default_value_t = false)]
    synthetic_mode: bool,
    #[arg(long = "synthetic-check", default_value_t = false)]
    synthetic_check: bool,
    #[arg(long = "synthetic-iters", default_value_t = 1)]
    synthetic_iters: usize,
    #[arg(long = "thresh", alias = "th", default_value_t = 0.5, value_parser = parse_threshold)]
    thresh: f32,
}

#[derive(Debug, Clone, Copy)]
struct Detection {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    score: f32,
    class_id: usize,
}

#[derive(Debug, Clone, Copy)]
struct YoloSummary {
    kept: usize,
    best_class: usize,
    best_score: f64,
    checksum: f64,
}

fn main() -> Result<()> {
    init_timing();

    let mut cli = Cli::parse();
    cli.synthetic_iters = cli.synthetic_iters.max(1);
    if !cli.synthetic_mode {
        eprintln!(
            "asset-backed YOLO inference is not implemented in safe ggml-rs yet; rerun with --synthetic"
        );
        std::process::exit(2);
    }

    let started = Instant::now();
    let summary = run_yolo_synthetic_iters(cli)?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    if cli.synthetic_check {
        let mut check_cli = cli;
        check_cli.synthetic_iters = 1;
        let check = run_yolo_synthetic_iters(check_cli)?;
        let tol = 1e-12;
        let ok = summary.kept == check.kept
            && summary.best_class == check.best_class
            && (summary.best_score - check.best_score).abs() <= tol
            && (summary.checksum - check.checksum).abs() <= tol;
        if !ok {
            eprintln!("yolov3_tiny synthetic check failed");
            std::process::exit(2);
        }
        println!("yolov3_tiny: synthetic_check=ok");
    }

    println!(
        "synthetic_result kind=yolo kept={} best_class={} best_score={:.8} checksum={:.8} iters={} elapsed_ms={:.3}",
        summary.kept,
        summary.best_class,
        summary.best_score,
        summary.checksum,
        cli.synthetic_iters,
        elapsed_ms
    );

    Ok(())
}

fn parse_threshold(raw: &str) -> std::result::Result<f32, String> {
    let parsed = raw
        .parse::<f32>()
        .map_err(|_| format!("invalid --thresh value `{raw}`"))?;
    if !(0.0..=1.0).contains(&parsed) {
        return Err(format!("--thresh must be between 0.0 and 1.0, got `{raw}`"));
    }
    Ok(parsed)
}

fn synthetic_unit(a: usize, b: usize, c: usize) -> f32 {
    let mixed = (a * 131 + b * 53 + c * 17 + 19) % 997;
    mixed as f32 / 498.5 - 1.0
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn iou(a: Detection, b: Detection) -> f32 {
    let a_left = a.x - a.w * 0.5;
    let a_right = a.x + a.w * 0.5;
    let b_left = b.x - b.w * 0.5;
    let b_right = b.x + b.w * 0.5;
    let inter_w = (a_right.min(b_right) - a_left.max(b_left)).max(0.0);

    let a_top = a.y - a.h * 0.5;
    let a_bottom = a.y + a.h * 0.5;
    let b_top = b.y - b.h * 0.5;
    let b_bottom = b.y + b.h * 0.5;
    let inter_h = (a_bottom.min(b_bottom) - a_top.max(b_top)).max(0.0);

    let inter_area = inter_w * inter_h;
    let union_area = a.w * a.h + b.w * b.h - inter_area;
    if union_area <= 0.0 {
        return 0.0;
    }
    inter_area / union_area
}

fn run_yolo_synthetic_iters(cli: Cli) -> Result<YoloSummary> {
    let features: Vec<f32> = (0..YOLO_SYNTH_CELLS)
        .flat_map(|cell| {
            (0..YOLO_SYNTH_FEATURES).map(move |feature| 0.7 * synthetic_unit(feature, cell, 20))
        })
        .collect();
    let proj: Vec<f32> = (0..YOLO_SYNTH_CHANNELS)
        .flat_map(|channel| {
            (0..YOLO_SYNTH_FEATURES).map(move |feature| 0.25 * synthetic_unit(channel, feature, 21))
        })
        .collect();
    let bias: Vec<f32> = (0..YOLO_SYNTH_CHANNELS)
        .map(|channel| 0.08 * synthetic_unit(channel, 0, 22))
        .collect();

    let ctx = Context::new(8 * 1024 * 1024)?;
    let features_t =
        ctx.new_tensor_2d::<f32>(Shape2D::new(YOLO_SYNTH_FEATURES, YOLO_SYNTH_CELLS))?;
    let proj_t =
        ctx.new_tensor_2d::<f32>(Shape2D::new(YOLO_SYNTH_FEATURES, YOLO_SYNTH_CHANNELS))?;
    let bias_t = ctx.new_tensor_1d::<f32>(Length::new(YOLO_SYNTH_CHANNELS))?;

    features_t.write_data(&features)?;
    proj_t.write_data(&proj)?;
    bias_t.write_data(&bias)?;

    let raw_t = ctx.add(&ctx.mul_mat(&proj_t, &features_t)?, &bias_t)?;
    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&raw_t);

    let mut summary = YoloSummary {
        kept: 0,
        best_class: 0,
        best_score: 0.0,
        checksum: 0.0,
    };
    for _ in 0..cli.synthetic_iters {
        ctx.compute(&mut graph, 1)?;
        let raw = raw_t.read_data::<f32>()?;
        summary = summarize_yolo_raw(&raw, cli.thresh);
    }

    Ok(summary)
}

fn summarize_yolo_raw(raw: &[f32], threshold: f32) -> YoloSummary {
    let anchors_w = [0.8f32, 1.4, 2.0];
    let anchors_h = [1.0f32, 1.6, 2.4];

    let mut candidates = Vec::<Detection>::new();
    for cell in 0..YOLO_SYNTH_CELLS {
        let row = cell / YOLO_SYNTH_GRID;
        let col = cell % YOLO_SYNTH_GRID;

        for anchor in 0..YOLO_SYNTH_ANCHORS {
            let base = anchor * (YOLO_SYNTH_CLASSES + 5);
            let row_offset = cell * YOLO_SYNTH_CHANNELS;

            let tx = raw[row_offset + base];
            let ty = raw[row_offset + base + 1];
            let tw = raw[row_offset + base + 2];
            let th = raw[row_offset + base + 3];
            let to = raw[row_offset + base + 4] + 0.9;

            let objectness = sigmoid(to);
            let preferred_class = (cell + anchor) % YOLO_SYNTH_CLASSES;

            let mut best_class = 0usize;
            let mut best_class_prob = 0.0f32;
            for class in 0..YOLO_SYNTH_CLASSES {
                let mut class_logit = raw[row_offset + base + 5 + class];
                class_logit += if class == preferred_class { 1.2 } else { -0.4 };
                let class_prob = sigmoid(class_logit);
                if class_prob > best_class_prob {
                    best_class_prob = class_prob;
                    best_class = class;
                }
            }

            let score = objectness * best_class_prob;
            if score < threshold {
                continue;
            }

            let x = (col as f32 + sigmoid(tx)) / YOLO_SYNTH_GRID as f32;
            let y = (row as f32 + sigmoid(ty)) / YOLO_SYNTH_GRID as f32;
            let w = tw.clamp(-2.0, 2.0).exp() * anchors_w[anchor] / YOLO_SYNTH_GRID as f32;
            let h = th.clamp(-2.0, 2.0).exp() * anchors_h[anchor] / YOLO_SYNTH_GRID as f32;

            candidates.push(Detection {
                x,
                y,
                w,
                h,
                score,
                class_id: best_class,
            });
        }
    }

    candidates.sort_by(|lhs, rhs| rhs.score.partial_cmp(&lhs.score).unwrap_or(Ordering::Equal));

    let mut kept = Vec::<Detection>::new();
    for cand in candidates {
        let suppressed = kept
            .iter()
            .copied()
            .any(|selected| cand.class_id == selected.class_id && iou(cand, selected) > 0.45);
        if !suppressed {
            kept.push(cand);
        }
    }

    let mut best_class = 0usize;
    let mut best_score = 0.0f64;
    if let Some(det) = kept.first().copied() {
        best_class = det.class_id;
        best_score = f64::from(det.score);
    }

    let checksum = kept
        .iter()
        .enumerate()
        .map(|(index, det)| {
            f64::from(det.score) * (1.0 + 0.1 * index as f64)
                + 0.01 * f64::from(det.x)
                + 0.02 * f64::from(det.y)
                + 0.03 * f64::from(det.w)
                + 0.04 * f64::from(det.h)
                + 0.001 * det.class_id as f64
        })
        .sum();

    YoloSummary {
        kept: kept.len(),
        best_class,
        best_score,
        checksum,
    }
}
