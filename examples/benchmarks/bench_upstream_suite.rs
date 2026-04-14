//! Operational harness for upstream ggml benchmark targets.
//!
//! This example is intentionally configurable so local runs can be scoped and
//! repeated without editing source code.

use clap::{ArgAction, Parser};
use std::collections::HashSet;
use std::fs;
use std::process::Command;
use std::time::Instant;

const DEFAULT_BENCH_TARGETS: &[&str] = &[
    "simple-ctx",
    "simple-backend",
    "perf-metal",
    "gpt-2-ctx",
    "gpt-2-backend",
    "gpt-2-alloc",
    "gpt-2-batched",
    "gpt-2-sched",
    "gpt-2-quantize",
    "gpt-j",
    "gpt-j-quantize",
    "magika",
    "mnist-eval",
    "mnist-train",
    "sam",
    "yolov3-tiny",
];

#[derive(Debug, Parser)]
#[command(name = "bench_upstream_suite")]
struct Cli {
    #[arg(value_name = "TARGET")]
    targets: Vec<String>,
    #[arg(long = "skip-build", action = ArgAction::SetTrue)]
    skip_build: Option<bool>,
    #[arg(long = "list-only", action = ArgAction::SetTrue)]
    list_only: Option<bool>,
    #[arg(long = "keep-going", action = ArgAction::SetTrue, conflicts_with = "fail_fast")]
    keep_going: Option<bool>,
    #[arg(long = "fail-fast", action = ArgAction::SetTrue, conflicts_with = "keep_going")]
    fail_fast: Option<bool>,
}

fn main() {
    let options = BenchOptions::from_cli(Cli::parse());
    assert!(
        !options.targets.is_empty(),
        "no upstream benchmark targets selected"
    );
    let available_targets = discover_targets(&options.build_dir)
        .unwrap_or_else(|error| panic!("failed to discover CMake targets: {error}"));
    let mut runnable_targets = Vec::new();
    let mut skipped_targets = Vec::new();
    for target in &options.targets {
        if available_targets.contains(target) {
            runnable_targets.push(target.clone());
        } else {
            skipped_targets.push(target.clone());
        }
    }
    assert!(
        !runnable_targets.is_empty(),
        "no selected upstream benchmark targets are available in `{}`",
        options.build_dir
    );

    if options.list_only {
        println!("selected upstream benchmark targets:");
        for target in &runnable_targets {
            println!(" - {target}");
        }
        if !skipped_targets.is_empty() {
            println!("skipped unavailable targets:");
            for target in &skipped_targets {
                println!(" - {target}");
            }
        }
        return;
    }

    let suite_started = Instant::now();
    let mut passed = Vec::new();
    let mut failures = Vec::new();
    let mut skipped_runs = Vec::new();

    for target in &runnable_targets {
        let started = Instant::now();
        if !options.skip_build
            && let Err(error) = build_target(&options.build_dir, target)
        {
            failures.push(format!("{target}: build error: {error}"));
            if !options.keep_going {
                break;
            }
            continue;
        }
        let run_args = target_run_args(target);
        if let Some(reason) = default_run_skip_reason(target, run_args.is_some()) {
            skipped_runs.push(format!("{target}: {reason}"));
            continue;
        }

        if let Err(error) = run_target(&options.build_dir, target, run_args.as_deref()) {
            failures.push(format!("{target}: run error: {error}"));
            if !options.keep_going {
                break;
            }
            continue;
        }

        let elapsed = started.elapsed().as_secs_f64();
        passed.push((target.clone(), elapsed));
        println!("[upstream-bench] {target} finished in {elapsed:.2}s");
    }

    let total_elapsed = suite_started.elapsed().as_secs_f64();
    let summary = render_summary(SummaryView {
        build_dir: &options.build_dir,
        requested: &options.targets,
        selected: &runnable_targets,
        skipped: &skipped_targets,
        skipped_runs: &skipped_runs,
        passed: &passed,
        failures: &failures,
        total_elapsed,
    });
    if let Some(path) = options.summary_path.as_deref() {
        fs::write(path, &summary)
            .unwrap_or_else(|error| panic!("failed to write summary file `{path}`: {error}"));
    }
    println!("{summary}");

    assert!(failures.is_empty(), "upstream bench suite failed");
}

#[derive(Debug)]
struct BenchOptions {
    build_dir: String,
    targets: Vec<String>,
    skip_build: bool,
    list_only: bool,
    keep_going: bool,
    summary_path: Option<String>,
}

impl BenchOptions {
    fn from_cli(cli: Cli) -> Self {
        let build_dir = std::env::var("GGML_UPSTREAM_BUILD_DIR")
            .unwrap_or_else(|_| "target/vendor/ggml/build".to_string());
        let skip_build = cli
            .skip_build
            .unwrap_or_else(|| env_flag("GGML_UPSTREAM_SKIP_BUILD"));
        let list_only = cli
            .list_only
            .unwrap_or_else(|| env_flag("GGML_UPSTREAM_LIST_ONLY"));
        let keep_going = match (cli.keep_going, cli.fail_fast) {
            (Some(true), _) => true,
            (_, Some(true)) => false,
            _ => env_flag("GGML_UPSTREAM_KEEP_GOING"),
        };
        let summary_path = std::env::var("GGML_UPSTREAM_SUMMARY_PATH").ok();

        // CLI target list takes precedence. Otherwise, apply include/exclude env
        // selection with stable defaults.
        let targets = if cli.targets.is_empty() {
            selected_targets(
                "GGML_UPSTREAM_BENCH_TARGETS",
                "GGML_UPSTREAM_EXCLUDE_TARGETS",
                DEFAULT_BENCH_TARGETS,
            )
        } else {
            cli.targets
        };

        Self {
            build_dir,
            targets,
            skip_build,
            list_only,
            keep_going,
            summary_path,
        }
    }
}

fn selected_targets(
    include_env_key: &str,
    exclude_env_key: &str,
    default_targets: &[&str],
) -> Vec<String> {
    let mut targets = parse_csv_env(include_env_key).unwrap_or_else(|| {
        default_targets
            .iter()
            .map(|target| (*target).to_string())
            .collect()
    });
    let excludes = parse_csv_env(exclude_env_key).unwrap_or_default();
    targets.retain(|target| !excludes.contains(target));
    targets
}

fn parse_csv_env(key: &str) -> Option<Vec<String>> {
    std::env::var(key).ok().and_then(|raw| {
        let parsed: Vec<String> = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect();
        if parsed.is_empty() {
            None
        } else {
            Some(parsed)
        }
    })
}

fn env_flag(key: &str) -> bool {
    std::env::var(key)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

struct SummaryView<'a> {
    build_dir: &'a str,
    requested: &'a [String],
    selected: &'a [String],
    skipped: &'a [String],
    skipped_runs: &'a [String],
    passed: &'a [(String, f64)],
    failures: &'a [String],
    total_elapsed: f64,
}

fn render_summary(summary: SummaryView<'_>) -> String {
    let mut lines = Vec::new();
    lines.push("[upstream-bench] summary".to_string());
    lines.push(format!("build_dir={}", summary.build_dir));
    lines.push(format!("requested_targets={}", summary.requested.join(",")));
    lines.push(format!("selected_targets={}", summary.selected.join(",")));
    lines.push(format!("skipped_targets={}", summary.skipped.join(",")));
    lines.push(format!(
        "skipped_run_targets={}",
        summary.skipped_runs.len()
    ));
    lines.push(format!("passed={}", summary.passed.len()));
    lines.push(format!("failed={}", summary.failures.len()));
    lines.push(format!("total_elapsed_secs={:.2}", summary.total_elapsed));
    if !summary.skipped.is_empty() {
        lines.push("skipped:".to_string());
        for target in summary.skipped {
            lines.push(format!(" - {target}"));
        }
    }
    if !summary.failures.is_empty() {
        lines.push("failures:".to_string());
        for failure in summary.failures {
            lines.push(format!(" - {failure}"));
        }
    }
    if !summary.skipped_runs.is_empty() {
        lines.push("skipped_runs:".to_string());
        for skipped_run in summary.skipped_runs {
            lines.push(format!(" - {skipped_run}"));
        }
    }
    format!("{}\n", lines.join("\n"))
}

fn build_target(build_dir: &str, target: &str) -> Result<(), String> {
    let mut command = Command::new("cmake");
    command.args(["--build", build_dir, "--target", target, "--parallel"]);
    if let Ok(jobs) = std::env::var("GGML_UPSTREAM_BUILD_JOBS") {
        let jobs = jobs.trim();
        if !jobs.is_empty() {
            command.arg(jobs);
        }
    }

    let status = command
        .status()
        .map_err(|error| format!("failed to spawn cmake for {target}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("cmake build returned non-zero status".to_string())
    }
}

fn discover_targets(build_dir: &str) -> Result<HashSet<String>, String> {
    let output = Command::new("cmake")
        .args(["--build", build_dir, "--target", "help"])
        .output()
        .map_err(|error| format!("failed to spawn cmake target help: {error}"))?;
    if !output.status.success() {
        return Err("cmake target help returned non-zero status".to_string());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut targets = HashSet::new();
    for line in stdout.lines() {
        if let Some(rest) = line.trim().strip_prefix("... ")
            && let Some(target) = rest.split_whitespace().next()
        {
            targets.insert(target.to_string());
        }
    }
    Ok(targets)
}

fn run_target(build_dir: &str, target: &str, args: Option<&[String]>) -> Result<(), String> {
    let bin = format!("{build_dir}/bin/{target}");
    let mut command = Command::new(&bin);
    if let Some(args) = args {
        command.args(args);
    }
    let status = command
        .status()
        .map_err(|error| format!("failed to run upstream benchmark {target} ({bin}): {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("benchmark binary returned non-zero status".to_string())
    }
}

fn default_run_skip_reason(target: &str, has_explicit_args: bool) -> Option<&'static str> {
    if has_explicit_args {
        return None;
    }
    match target {
        "gpt-2-ctx" | "gpt-2-backend" | "gpt-2-alloc" | "gpt-2-batched" | "gpt-2-sched"
        | "gpt-2-quantize" | "gpt-j" | "gpt-j-quantize" | "magika" | "mnist-eval"
        | "mnist-train" | "sam" | "yolov3-tiny" => Some(
            "requires external model/data arguments; set GGML_UPSTREAM_RUN_ARGS_<TARGET> to run",
        ),
        _ => None,
    }
}

fn target_run_args(target: &str) -> Option<Vec<String>> {
    let suffix: String = target
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    let key = format!("GGML_UPSTREAM_RUN_ARGS_{suffix}");
    std::env::var(key).ok().and_then(|raw| {
        let args: Vec<String> = raw.split_whitespace().map(ToOwned::to_owned).collect();
        if args.is_empty() { None } else { Some(args) }
    })
}
