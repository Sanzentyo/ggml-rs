#![cfg(feature = "link-system")]

//! Integration harness for running upstream ggml test binaries.
//!
//! The test is `#[ignore]` because it can be slow and depends on a prepared
//! local upstream build directory.

use std::fs;
use std::process::Command;
use std::time::Instant;

const DEFAULT_TEST_TARGETS: &[&str] = &[
    "test-backend-ops",
    "test-opt",
    "test-quantize-fns",
    "test-quantize-perf",
    "test-pool",
    "test-arange",
    "test-timestep_embedding",
    "test-pad-reflect-1d",
    "test-roll",
    "test-conv-transpose",
    "test-conv-transpose-1d",
    "test-dup",
    "test-rel-pos",
    "test-customop",
    "test-conv1d",
    "test-conv1d-dw-c1",
    "test-conv1d-dw-c2",
    "test-conv2d",
    "test-conv2d-dw",
    "test-cont",
    "test-interpolate",
];

#[test]
#[ignore = "runs full upstream ggml C/C++ test suite from vendor build"]
fn run_all_upstream_ggml_tests() {
    let build_dir = std::env::var("GGML_UPSTREAM_BUILD_DIR")
        .unwrap_or_else(|_| "target/vendor/ggml/build".to_string());
    let targets = selected_targets(
        "GGML_UPSTREAM_TEST_TARGETS",
        "GGML_UPSTREAM_EXCLUDE_TARGETS",
        DEFAULT_TEST_TARGETS,
    );
    assert!(
        !targets.is_empty(),
        "no upstream test targets selected (check GGML_UPSTREAM_TEST_TARGETS / GGML_UPSTREAM_EXCLUDE_TARGETS)"
    );

    if env_flag("GGML_UPSTREAM_LIST_ONLY") {
        eprintln!("selected upstream test targets:");
        for target in &targets {
            eprintln!(" - {target}");
        }
        return;
    }

    let suite_started = Instant::now();
    let skip_build = env_flag("GGML_UPSTREAM_SKIP_BUILD");
    let keep_going = env_flag("GGML_UPSTREAM_KEEP_GOING");
    let mut passed = Vec::new();
    let mut failures = Vec::new();

    for target in &targets {
        let started = Instant::now();

        if !skip_build {
            if let Err(error) = build_target(&build_dir, target) {
                failures.push(format!("{target}: build error: {error}"));
                if !keep_going {
                    break;
                }
                continue;
            }
        }

        if let Err(error) = run_target(&build_dir, target) {
            failures.push(format!("{target}: run error: {error}"));
            if !keep_going {
                break;
            }
            continue;
        }

        let elapsed = started.elapsed().as_secs_f64();
        passed.push((target.clone(), elapsed));
        eprintln!("[upstream-suite] {target} passed in {elapsed:.2}s");
    }

    let total_elapsed = suite_started.elapsed().as_secs_f64();
    let summary = render_summary(&build_dir, &targets, &passed, &failures, total_elapsed);

    if let Ok(path) = std::env::var("GGML_UPSTREAM_SUMMARY_PATH") {
        fs::write(&path, &summary).unwrap_or_else(|error| {
            panic!("failed to write GGML_UPSTREAM_SUMMARY_PATH={path}: {error}")
        });
    }

    eprintln!("{summary}");
    assert!(failures.is_empty(), "upstream suite failed");
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
    lines.push("[upstream-suite] summary".to_string());
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
        .map_err(|error| format!("failed to run upstream test {target} ({bin}): {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("test binary returned non-zero status".to_string())
    }
}
