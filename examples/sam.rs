//! Synthetic SAM counterpart with safe ggml-rs APIs.

use clap::Parser;
use ggml_rs::{Context, Length, Result, Shape2D, init_timing};
use std::time::Instant;

const SAM_SYNTH_GRID_W: usize = 16;
const SAM_SYNTH_GRID_H: usize = 16;
const SAM_SYNTH_CELLS: usize = SAM_SYNTH_GRID_W * SAM_SYNTH_GRID_H;
const SAM_SYNTH_EMBD: usize = 8;
const SAM_SYNTH_MASKS: usize = 3;

#[derive(Debug, Clone, Copy)]
enum Prompt {
    Point { x: f32, y: f32 },
    Box { x1: f32, y1: f32, x2: f32, y2: f32 },
}

#[derive(Debug, Clone, Copy)]
struct PointPromptArg {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy)]
struct BoxPromptArg {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[derive(Debug, Clone, Copy, Parser)]
#[command(name = "sam")]
struct Cli {
    #[arg(long = "synthetic", default_value_t = false)]
    synthetic_mode: bool,
    #[arg(long = "synthetic-check", default_value_t = false)]
    synthetic_check: bool,
    #[arg(long = "synthetic-iters", default_value_t = 1)]
    synthetic_iters: usize,
    #[arg(long = "mask-threshold", alias = "mt", default_value_t = 0.0)]
    mask_threshold: f32,
    #[arg(long = "iou-threshold", alias = "it", default_value_t = 0.88)]
    iou_threshold: f64,
    #[arg(long = "score-threshold", alias = "st", default_value_t = 0.95)]
    score_threshold: f64,
    #[arg(long = "score-offset", alias = "so", default_value_t = 1.0)]
    score_offset: f32,
    #[arg(long = "point-prompt", alias = "p", value_parser = parse_point_prompt, conflicts_with = "box_prompt")]
    point_prompt: Option<PointPromptArg>,
    #[arg(long = "box-prompt", alias = "b", value_parser = parse_box_prompt, conflicts_with = "point_prompt")]
    box_prompt: Option<BoxPromptArg>,
}

impl Cli {
    fn prompt(self) -> Prompt {
        if let Some(prompt) = self.point_prompt {
            Prompt::Point {
                x: prompt.x,
                y: prompt.y,
            }
        } else if let Some(prompt) = self.box_prompt {
            Prompt::Box {
                x1: prompt.x1,
                y1: prompt.y1,
                x2: prompt.x2,
                y2: prompt.y2,
            }
        } else {
            Prompt::Point {
                x: 414.375,
                y: 162.796_88,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SamSummary {
    accepted: usize,
    best_mask: usize,
    best_iou: f64,
    checksum: f64,
}

fn main() -> Result<()> {
    init_timing();

    let mut cli = Cli::parse();
    cli.synthetic_iters = cli.synthetic_iters.max(1);
    if !cli.synthetic_mode {
        eprintln!(
            "asset-backed SAM inference is not implemented in safe ggml-rs yet; rerun with --synthetic"
        );
        std::process::exit(2);
    }

    let started = Instant::now();
    let summary = run_sam_synthetic_iters(cli)?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    if cli.synthetic_check {
        let mut check_cli = cli;
        check_cli.synthetic_iters = 1;
        let check = run_sam_synthetic_iters(check_cli)?;
        let tol = 1e-12;
        let ok = summary.accepted == check.accepted
            && summary.best_mask == check.best_mask
            && (summary.best_iou - check.best_iou).abs() <= tol
            && (summary.checksum - check.checksum).abs() <= tol;
        if !ok {
            eprintln!("sam synthetic check failed");
            std::process::exit(2);
        }
        println!("sam: synthetic_check=ok");
    }

    println!(
        "synthetic_result kind=sam accepted={} best_mask={} best_iou={:.8} checksum={:.8} iters={} elapsed_ms={:.3}",
        summary.accepted,
        summary.best_mask,
        summary.best_iou,
        summary.checksum,
        cli.synthetic_iters,
        elapsed_ms
    );

    Ok(())
}

fn parse_point_prompt(raw: &str) -> std::result::Result<PointPromptArg, String> {
    let vals = parse_csv_f32(raw, 2)?;
    Ok(PointPromptArg {
        x: vals[0],
        y: vals[1],
    })
}

fn parse_box_prompt(raw: &str) -> std::result::Result<BoxPromptArg, String> {
    let vals = parse_csv_f32(raw, 4)?;
    Ok(BoxPromptArg {
        x1: vals[0],
        y1: vals[1],
        x2: vals[2],
        y2: vals[3],
    })
}

fn parse_csv_f32(raw: &str, expected: usize) -> std::result::Result<Vec<f32>, String> {
    let values: Vec<f32> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<f32>()
                .map_err(|_| format!("invalid floating-point value `{s}` in `{raw}`"))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if values.len() != expected {
        return Err(format!(
            "invalid CSV value `{raw}`; expected {expected} numbers"
        ));
    }
    Ok(values)
}

fn synthetic_unit(a: usize, b: usize, c: usize) -> f32 {
    let mixed = (a * 131 + b * 53 + c * 17 + 19) % 997;
    mixed as f32 / 498.5 - 1.0
}

fn run_sam_synthetic_iters(cli: Cli) -> Result<SamSummary> {
    let prompt = cli.prompt();
    let prompt_scale = match prompt {
        Prompt::Point { x, y } => (x + y) / (2.0 * 1024.0),
        Prompt::Box { x1, y1, x2, y2 } => (x1 + y1 + x2 + y2) / (4.0 * 1024.0),
    };

    let image_embd: Vec<f32> = (0..SAM_SYNTH_CELLS)
        .flat_map(|cell| (0..SAM_SYNTH_EMBD).map(move |embd| 0.6 * synthetic_unit(embd, cell, 10)))
        .collect();
    let mask_proj: Vec<f32> = (0..SAM_SYNTH_MASKS)
        .flat_map(|mask| (0..SAM_SYNTH_EMBD).map(move |embd| 0.25 * synthetic_unit(mask, embd, 11)))
        .collect();
    let mask_bias: Vec<f32> = (0..SAM_SYNTH_MASKS)
        .map(|mask| 0.04 * synthetic_unit(mask, 0, 12))
        .collect();

    let ctx = Context::new(8 * 1024 * 1024)?;
    let image_t = ctx.new_tensor_2d::<f32>(Shape2D::new(SAM_SYNTH_EMBD, SAM_SYNTH_CELLS))?;
    let proj_t = ctx.new_tensor_2d::<f32>(Shape2D::new(SAM_SYNTH_EMBD, SAM_SYNTH_MASKS))?;
    let bias_t = ctx.new_tensor_1d::<f32>(Length::new(SAM_SYNTH_MASKS))?;

    image_t.write_data(&image_embd)?;
    proj_t.write_data(&mask_proj)?;
    bias_t.write_data(&mask_bias)?;

    let logits_t = ctx.add(&ctx.mul_mat(&proj_t, &image_t)?, &bias_t)?;
    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&logits_t);

    let mut summary = SamSummary {
        accepted: 0,
        best_mask: 0,
        best_iou: 0.0,
        checksum: 0.0,
    };
    for _ in 0..cli.synthetic_iters {
        ctx.compute(&mut graph, 1)?;
        let logits = logits_t.read_data::<f32>()?;
        summary = summarize_sam_logits(&logits, cli, prompt_scale);
    }

    Ok(summary)
}

fn summarize_sam_logits(logits: &[f32], cli: Cli, prompt_scale: f32) -> SamSummary {
    let threshold = if cli.mask_threshold != 0.0 {
        cli.mask_threshold
    } else {
        0.10
    };
    let intersection_threshold = threshold + cli.score_offset * 0.05;
    let union_threshold = threshold - cli.score_offset * 0.05;

    let mut accepted = 0usize;
    let mut best_mask = 0usize;
    let mut best_iou = -1.0f64;
    let mut checksum = 0.0f64;

    for mask in 0..SAM_SYNTH_MASKS {
        let mut sum_prob = 0.0f64;
        let mut intersections = 0usize;
        let mut unions = 0usize;

        let mut min_x = SAM_SYNTH_GRID_W;
        let mut min_y = SAM_SYNTH_GRID_H;
        let mut max_x = 0usize;
        let mut max_y = 0usize;

        for cell in 0..SAM_SYNTH_CELLS {
            let idx = cell * SAM_SYNTH_MASKS + mask;
            let mut logit = logits[idx];
            logit += 0.2 * prompt_scale * synthetic_unit(mask, cell, 13);

            let prob = 1.0f64 / (1.0 + f64::from((-logit).exp()));
            sum_prob += prob;

            if prob > f64::from(intersection_threshold) {
                intersections += 1;
            }
            if prob > f64::from(union_threshold) {
                unions += 1;
            }
            if prob > f64::from(threshold) {
                let y = cell / SAM_SYNTH_GRID_W;
                let x = cell % SAM_SYNTH_GRID_W;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }

        let mean_prob = sum_prob / SAM_SYNTH_CELLS as f64;
        let iou = 0.8 + 0.2 * mean_prob;
        let stability = if unions > 0 {
            intersections as f64 / unions as f64
        } else {
            0.0
        };

        if iou >= cli.iou_threshold && stability >= cli.score_threshold {
            accepted += 1;
        }

        if iou > best_iou {
            best_iou = iou;
            best_mask = mask;
        }

        let bbox_checksum = 0.001 * (min_x + min_y + max_x + max_y + (mask + 1) * 7) as f64;
        checksum += iou * (mask + 1) as f64 + stability * (0.5 + 0.1 * mask as f64) + bbox_checksum;
    }

    SamSummary {
        accepted,
        best_mask,
        best_iou,
        checksum,
    }
}
