use clap::Parser;
use llama_rs::{
    AttentionDecodeCacheInput, AttentionDecodePlan, AttentionDecodeStepwiseConfig,
    AttentionInferenceConfig, AttentionLayout, AttentionMaskPolicy, AttentionWeights,
    DecodeStepPlan, LlamaBackend, build_attention_decode_cache,
};
use std::collections::HashMap;
use std::error::Error as StdError;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Clone, Copy)]
struct Case {
    hidden_features: usize,
    query_head_count: usize,
    kv_head_count: usize,
    query_length: usize,
}

impl Case {
    fn build_config(self) -> Result<AttentionInferenceConfig, Box<dyn StdError>> {
        let layout = AttentionLayout::from_hidden_features(
            self.hidden_features,
            self.query_head_count,
            self.kv_head_count,
        )?;
        AttentionInferenceConfig::from_layout(layout, self.query_length).map_err(Into::into)
    }

    fn build_input_with_length(self, sequence_length: usize) -> Vec<f32> {
        (0..(self.hidden_features * sequence_length))
            .map(|index| ((index + 3) % 29) as f32 * 0.0625)
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
struct Args {
    decode_kv_length: usize,
    warmup_iters: usize,
    bench_iters: usize,
    stepwise: Option<StepwiseArgs>,
}

#[derive(Debug, Clone, Copy)]
struct StepwiseArgs {
    key_value_start: usize,
    steps: usize,
    past_start: usize,
}

#[derive(Debug)]
struct CppReport {
    avg_ms: f64,
    checksum: f64,
}

type ParsedArgs = (Args, Vec<Case>, Vec<LlamaBackend>);

fn main() -> Result<(), ExampleError> {
    ggml_rs::init_timing();
    let (args, cases, backends) = parse_args()?;
    let cpp_binary = compile_cpp_reference()?;

    for case in &cases {
        let mut config = case.build_config()?;
        if let Some(stepwise) = args.stepwise {
            config = config.with_mask(AttentionMaskPolicy::Causal {
                past_tokens: stepwise.past_start,
            });
        }
        let weights = AttentionWeights::deterministic(config);
        let query_input = case.build_input_with_length(case.query_length);
        let key_value_input = case.build_input_with_length(args.decode_kv_length);
        let cache =
            build_attention_decode_cache(&weights, &key_value_input, args.decode_kv_length)?;

        for backend in backends.iter().copied() {
            let (rust_avg_ms, rust_checksum) = if let Some(stepwise) = args.stepwise {
                if args.warmup_iters > 0 {
                    let warmup_plan = DecodeStepPlan::builder()
                        .backend(backend)
                        .stepwise(AttentionDecodeStepwiseConfig::new(
                            stepwise.key_value_start,
                            stepwise.steps,
                            stepwise.past_start,
                            args.warmup_iters,
                        ))
                        .build();
                    let _ = warmup_plan.execute_single(&weights, &query_input, &cache, None)?;
                }
                let bench_plan = DecodeStepPlan::builder()
                    .backend(backend)
                    .stepwise(AttentionDecodeStepwiseConfig::new(
                        stepwise.key_value_start,
                        stepwise.steps,
                        stepwise.past_start,
                        args.bench_iters,
                    ))
                    .build();
                let rust_start = Instant::now();
                let rust_report =
                    bench_plan.execute_single(&weights, &query_input, &cache, None)?;
                let rust_elapsed = rust_start.elapsed();
                let total_iters = args
                    .bench_iters
                    .checked_mul(stepwise.steps)
                    .ok_or("stepwise iteration count overflow")?;
                let rust_avg_ms = rust_elapsed.as_secs_f64() * 1000.0 / total_iters as f64;
                let rust_checksum: f64 = rust_report
                    .output
                    .iter()
                    .take(16)
                    .map(|value| f64::from(*value))
                    .sum();
                (rust_avg_ms, rust_checksum)
            } else {
                if args.warmup_iters > 0 {
                    let _ = AttentionDecodePlan::builder()
                        .backend(backend)
                        .repeats(args.warmup_iters)
                        .build()?
                        .execute(AttentionDecodeCacheInput::new(
                            &weights,
                            &query_input,
                            &cache,
                        ))?;
                }
                let rust_start = Instant::now();
                let rust_report = AttentionDecodePlan::builder()
                    .backend(backend)
                    .repeats(args.bench_iters)
                    .build()?
                    .execute(AttentionDecodeCacheInput::new(
                        &weights,
                        &query_input,
                        &cache,
                    ))?;
                let rust_elapsed = rust_start.elapsed();
                let rust_avg_ms = rust_elapsed.as_secs_f64() * 1000.0 / args.bench_iters as f64;
                let rust_checksum: f64 = rust_report
                    .output
                    .iter()
                    .take(16)
                    .map(|value| f64::from(*value))
                    .sum();
                (rust_avg_ms, rust_checksum)
            };

            if args.warmup_iters > 0 {
                let _ = cpp_reference(
                    &cpp_binary,
                    backend,
                    *case,
                    args.decode_kv_length,
                    args.warmup_iters,
                    1,
                    args.stepwise,
                )?;
            }
            let cpp_report = cpp_reference(
                &cpp_binary,
                backend,
                *case,
                args.decode_kv_length,
                0,
                args.bench_iters,
                args.stepwise,
            )?;
            let ratio = rust_avg_ms / cpp_report.avg_ms;
            let checksum_delta = (rust_checksum - cpp_report.checksum).abs();
            let checksum_rel = checksum_delta / rust_checksum.abs().max(1.0);

            if let Some(stepwise) = args.stepwise {
                println!(
                    "[{}] attn stepwise samework hidden={} heads={}/{} q_seq={} kv_start={} steps={} past_start={} warmup={} iters={} rust_avg={:.3} ms cpp_avg={:.3} ms rust/cpp={:.3} rust_checksum={:.6} cpp_checksum={:.6} checksum_delta={:.6} checksum_rel={:.6}",
                    backend_name(backend),
                    case.hidden_features,
                    case.query_head_count,
                    case.kv_head_count,
                    case.query_length,
                    stepwise.key_value_start,
                    stepwise.steps,
                    stepwise.past_start,
                    args.warmup_iters,
                    args.bench_iters,
                    rust_avg_ms,
                    cpp_report.avg_ms,
                    ratio,
                    rust_checksum,
                    cpp_report.checksum,
                    checksum_delta,
                    checksum_rel
                );
            } else {
                println!(
                    "[{}] attn decode samework hidden={} heads={}/{} q_seq={} kv_seq={} warmup={} iters={} rust_avg={:.3} ms cpp_avg={:.3} ms rust/cpp={:.3} rust_checksum={:.6} cpp_checksum={:.6} checksum_delta={:.6} checksum_rel={:.6}",
                    backend_name(backend),
                    case.hidden_features,
                    case.query_head_count,
                    case.kv_head_count,
                    case.query_length,
                    args.decode_kv_length,
                    args.warmup_iters,
                    args.bench_iters,
                    rust_avg_ms,
                    cpp_report.avg_ms,
                    ratio,
                    rust_checksum,
                    cpp_report.checksum,
                    checksum_delta,
                    checksum_rel
                );
            }
        }
    }

    Ok(())
}

#[derive(Debug, Error)]
enum ExampleError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Inference(#[from] llama_rs::InferenceError),
    #[error(transparent)]
    Llama(#[from] llama_rs::LlamaError),
    #[error(transparent)]
    Ggml(#[from] ggml_rs::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error(transparent)]
    Boxed(#[from] Box<dyn StdError>),
}

impl From<String> for ExampleError {
    fn from(value: String) -> Self {
        Self::Message(value)
    }
}

impl From<&'static str> for ExampleError {
    fn from(value: &'static str) -> Self {
        Self::Message(value.to_owned())
    }
}

fn backend_name(backend: LlamaBackend) -> &'static str {
    match backend {
        LlamaBackend::Cpu => "CPU",
        LlamaBackend::Metal => "MTL0",
    }
}

fn compile_cpp_reference() -> Result<PathBuf, Box<dyn StdError>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = manifest_dir.join("tests/cpp/attention_decode_proxy_reference.cpp");
    let ggml_root = ggml_root_dir();
    let ggml_include = ggml_include_dir(&ggml_root);
    let ggml_src = ggml_include
        .parent()
        .map(|parent| parent.join("src"))
        .unwrap_or_else(|| ggml_root.join("src"));
    let output = manifest_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf())
        .join("target/benchmarks/attention_decode_proxy_reference_cpp");
    std::fs::create_dir_all(
        output
            .parent()
            .ok_or("failed to resolve output parent for C++ reference")?,
    )?;

    let lib_dirs: Vec<PathBuf> = match std::env::var_os("GGML_RS_LIB_DIRS") {
        Some(paths) => std::env::split_paths(&paths).collect(),
        None => vec![ggml_root.join("build/src")],
    };
    let mut libs: Vec<String> = std::env::var("GGML_RS_LIBS")
        .unwrap_or_else(|_| "ggml-cpu,ggml-base,ggml,ggml-metal".to_string())
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if libs.iter().all(|lib| lib != "ggml") {
        libs.push("ggml".to_string());
    }
    if libs.iter().all(|lib| lib != "ggml-base") {
        libs.push("ggml-base".to_string());
    }
    if libs.iter().all(|lib| lib != "ggml-cpu") {
        libs.push("ggml-cpu".to_string());
    }

    let mut cmd = Command::new("c++");
    cmd.arg("-O3")
        .arg("-std=c++17")
        .arg(format!("-I{}", ggml_include.display()))
        .arg(format!("-I{}", ggml_src.display()));
    for lib_dir in &lib_dirs {
        cmd.arg(format!("-L{}", lib_dir.display()));
        cmd.arg(format!("-Wl,-rpath,{}", lib_dir.display()));
    }
    for lib in libs {
        cmd.arg(format!("-l{lib}"));
    }
    cmd.arg(&source).arg("-o").arg(&output);

    let compile = cmd.output()?;
    if !compile.status.success() {
        return Err(format!(
            "failed to compile C++ reference: {}",
            String::from_utf8_lossy(&compile.stderr)
        )
        .into());
    }
    Ok(output)
}

fn cpp_reference(
    binary: &Path,
    backend: LlamaBackend,
    case: Case,
    key_value_length: usize,
    warmup_iters: usize,
    bench_iters: usize,
    stepwise: Option<StepwiseArgs>,
) -> Result<CppReport, Box<dyn StdError>> {
    let backend_arg = match backend {
        LlamaBackend::Cpu => "cpu",
        LlamaBackend::Metal => "metal",
    };
    let mut command = Command::new(binary);
    command
        .arg(backend_arg)
        .arg(case.hidden_features.to_string())
        .arg(case.query_head_count.to_string())
        .arg(case.kv_head_count.to_string())
        .arg(case.query_length.to_string())
        .arg(key_value_length.to_string())
        .arg(warmup_iters.to_string())
        .arg(bench_iters.to_string());
    if let Some(stepwise) = stepwise {
        command
            .arg(stepwise.key_value_start.to_string())
            .arg(stepwise.steps.to_string())
            .arg(stepwise.past_start.to_string());
    }
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "cpp reference failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    parse_cpp_report(&String::from_utf8(output.stdout)?)
}

fn parse_cpp_report(line: &str) -> Result<CppReport, Box<dyn StdError>> {
    let mut values = HashMap::new();
    for token in line.split_whitespace() {
        if let Some((key, value)) = token.split_once('=') {
            values.insert(key, value);
        }
    }
    let avg_ms = values
        .get("avg_ms")
        .ok_or_else(|| format!("missing avg_ms in cpp output: {line}"))?
        .parse::<f64>()?;
    let checksum = values
        .get("checksum")
        .ok_or_else(|| format!("missing checksum in cpp output: {line}"))?
        .parse::<f64>()?;
    Ok(CppReport { avg_ms, checksum })
}

fn ggml_root_dir() -> PathBuf {
    std::env::var("GGML_CPP_REFERENCE_GGML_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root().join("target").join("vendor").join("ggml"))
}

fn ggml_include_dir(ggml_root: &Path) -> PathBuf {
    std::env::var_os("GGML_RS_GGML_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| ggml_root.join("include"))
}

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or(manifest_dir)
}

fn parse_args() -> Result<ParsedArgs, Box<dyn StdError>> {
    let cli = Cli::parse();
    let cases = match cli.cases.as_deref() {
        Some(value) => parse_cases_arg(value)?,
        None => default_cases(),
    };

    let backends = if cli.backends.is_empty() {
        vec![LlamaBackend::Cpu, LlamaBackend::Metal]
    } else {
        cli.backends.into_iter().map(|backend| backend.0).collect()
    };

    if cli.decode_kv_length == 0 {
        return Err("--decode-kv must be greater than zero".into());
    }
    if cli.bench_iters == 0 {
        return Err("--iters must be greater than zero".into());
    }
    let stepwise = match (cli.stepwise_start, cli.stepwise_steps, cli.stepwise_past) {
        (None, None, None) => None,
        (Some(key_value_start), Some(steps), Some(past_start)) => {
            if key_value_start == 0 {
                return Err("--stepwise-start must be greater than zero".into());
            }
            if steps == 0 {
                return Err("--stepwise-steps must be greater than zero".into());
            }
            let expected_start = past_start
                .checked_add(1)
                .ok_or("stepwise past/start relation overflow")?;
            if expected_start != key_value_start {
                return Err(format!(
                    "--stepwise-start ({key_value_start}) must equal --past + 1 ({expected_start})"
                )
                .into());
            }
            let expected_decode_kv = key_value_start
                .checked_add(steps)
                .and_then(|value| value.checked_sub(1))
                .ok_or("stepwise key/value range overflow")?;
            if cli.decode_kv_length != expected_decode_kv {
                return Err(format!(
                    "--decode-kv ({}) must equal --stepwise-start + --stepwise-steps - 1 ({expected_decode_kv})",
                    cli.decode_kv_length
                )
                .into());
            }
            if !cases.iter().all(|case| case.query_length == 1) {
                return Err(
                    "stepwise mode requires query length = 1 for all cases (HxQxKxS with S=1)"
                        .into(),
                );
            }
            Some(StepwiseArgs {
                key_value_start,
                steps,
                past_start,
            })
        }
        _ => {
            return Err(
                "stepwise mode requires all of --stepwise-start, --stepwise-steps, and --past"
                    .into(),
            );
        }
    };

    Ok((
        Args {
            decode_kv_length: cli.decode_kv_length,
            warmup_iters: cli.warmup_iters,
            bench_iters: cli.bench_iters,
            stepwise,
        },
        cases,
        backends,
    ))
}

fn default_cases() -> Vec<Case> {
    vec![
        Case {
            hidden_features: 2560,
            query_head_count: 16,
            kv_head_count: 4,
            query_length: 1,
        },
        Case {
            hidden_features: 3072,
            query_head_count: 32,
            kv_head_count: 8,
            query_length: 1,
        },
        Case {
            hidden_features: 3584,
            query_head_count: 28,
            kv_head_count: 4,
            query_length: 1,
        },
        Case {
            hidden_features: 3840,
            query_head_count: 16,
            kv_head_count: 8,
            query_length: 1,
        },
        Case {
            hidden_features: 4096,
            query_head_count: 32,
            kv_head_count: 8,
            query_length: 1,
        },
    ]
}

#[derive(Debug, Clone, Parser)]
#[command(about = "Compare Rust and C++ decode benchmarks", version)]
struct Cli {
    #[arg(long = "cases")]
    cases: Option<String>,
    #[arg(long = "warmup", default_value_t = 2)]
    warmup_iters: usize,
    #[arg(long = "iters", default_value_t = 10)]
    bench_iters: usize,
    #[arg(long = "decode-kv", default_value_t = 128)]
    decode_kv_length: usize,
    #[arg(long = "stepwise-start")]
    stepwise_start: Option<usize>,
    #[arg(long = "stepwise-steps")]
    stepwise_steps: Option<usize>,
    #[arg(long = "past")]
    stepwise_past: Option<usize>,
    backends: Vec<BackendArg>,
}

#[derive(Debug, Clone, Copy)]
struct BackendArg(LlamaBackend);

impl FromStr for BackendArg {
    type Err = <LlamaBackend as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        LlamaBackend::from_str(s).map(Self)
    }
}

fn parse_usize_arg(flag: &str, value: &str) -> Result<usize, Box<dyn StdError>> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid value for {flag}: {value} ({error})").into())
}

fn parse_cases_arg(value: &str) -> Result<Vec<Case>, Box<dyn StdError>> {
    let mut cases = Vec::new();
    for token in value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        let mut parts = token.split('x');
        let hidden_features = parts
            .next()
            .ok_or_else(|| format!("invalid case `{token}` (expected HxQxKxS)"))?;
        let query_head_count = parts
            .next()
            .ok_or_else(|| format!("invalid case `{token}` (expected HxQxKxS)"))?;
        let kv_head_count = parts
            .next()
            .ok_or_else(|| format!("invalid case `{token}` (expected HxQxKxS)"))?;
        let query_length = parts
            .next()
            .ok_or_else(|| format!("invalid case `{token}` (expected HxQxKxS)"))?;
        if parts.next().is_some() {
            return Err(format!("invalid case `{token}` (expected HxQxKxS)").into());
        }
        cases.push(Case {
            hidden_features: parse_usize_arg("--cases", hidden_features)?,
            query_head_count: parse_usize_arg("--cases", query_head_count)?,
            kv_head_count: parse_usize_arg("--cases", kv_head_count)?,
            query_length: parse_usize_arg("--cases", query_length)?,
        });
    }
    if cases.is_empty() {
        return Err("at least one case must be provided for --cases".into());
    }
    Ok(cases)
}
