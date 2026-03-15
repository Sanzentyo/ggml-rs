//! Operational harness for upstream ggml benchmark targets.
//!
//! This example is intentionally configurable so local runs can be scoped and
//! repeated without editing source code.

use clap::{ArgAction, Parser};
use std::fs;
use std::process::Command;
use std::time::Instant;

const DEFAULT_BENCH_TARGETS: &[&str] = &["test-backend-ops", "test-quantize-perf"];

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

    if options.list_only {
        println!("selected upstream benchmark targets:");
        for target in &options.targets {
            println!(" - {target}");
        }
        return;
    }

    let suite_started = Instant::now();
    let mut passed = Vec::new();
    let mut failures = Vec::new();

    for target in &options.targets {
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

        if let Err(error) = run_target(&options.build_dir, target) {
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
    let summary = render_summary(
        &options.build_dir,
        &options.targets,
        &passed,
        &failures,
        total_elapsed,
    );
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

fn render_summary(
    build_dir: &str,
    selected: &[String],
    passed: &[(String, f64)],
    failures: &[String],
    total_elapsed: f64,
) -> String {
    let mut lines = Vec::new();
    lines.push("[upstream-bench] summary".to_string());
    lines.push(format!("build_dir={build_dir}"));
    lines.push(format!("selected_targets={}", selected.join(",")));
    lines.push(format!("passed={}", passed.len()));
    lines.push(format!("failed={}", failures.len()));
    lines.push(format!("total_elapsed_secs={total_elapsed:.2}"));
    if !failures.is_empty() {
        lines.push("failures:".to_string());
        for failure in failures {
            lines.push(format!(" - {failure}"));
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

fn run_target(build_dir: &str, target: &str) -> Result<(), String> {
    let bin = format!("{build_dir}/bin/{target}");
    let status = Command::new(&bin)
        .status()
        .map_err(|error| format!("failed to run upstream benchmark {target} ({bin}): {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("benchmark binary returned non-zero status".to_string())
    }
}
