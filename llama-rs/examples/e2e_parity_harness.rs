//! Compares llama-rs true-E2E generated token IDs with llama.cpp.
//!
//! For robust token-id parity (including prompt-boundary merge cases such as `Hello`),
//! this example uses a tiny helper binary (`llama-simple-token-ids`) that returns sampled
//! token IDs directly from llama.cpp rather than re-tokenizing generated text.

use clap::{Parser, ValueEnum};
use llama_rs::{
    E2eGenerationConfig, E2eGenerationReport, GgufModel, LlamaBackend, MixedLayerPolicy,
    generate_token_ids_from_model, resolve_eos_token_id, tokenize_prompt_text,
};
use std::error::Error as StdError;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use thiserror::Error;

#[derive(Debug, Error)]
enum ExampleError {
    #[error(transparent)]
    E2e(#[from] llama_rs::E2eError),
    #[error(transparent)]
    Model(#[from] llama_rs::ModelError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(
        "llama.cpp token-id helper command failed on backend `{backend}` with status {status:?}: {stderr}"
    )]
    LlamaTokenIdsFailed {
        backend: String,
        status: Option<i32>,
        stderr: String,
    },
    #[error("llama token-id helper binary not found: `{path}`")]
    MissingLlamaTokenIdsBinary { path: PathBuf },
    #[error(
        "llama token-id helper build failed (status {status:?}) for `{source_path}` -> `{output}`: {stderr}"
    )]
    LlamaTokenIdsBuildFailed {
        status: Option<i32>,
        source_path: PathBuf,
        output: PathBuf,
        stderr: String,
    },
    #[error("could not derive llama.cpp source root from llama-simple path: `{path}`")]
    InvalidLlamaSimplePath { path: PathBuf },
    #[error(
        "failed to parse generated token ids from llama.cpp helper output on backend `{backend}`; stdout=`{stdout}`"
    )]
    LlamaTokenIdsParseFailed { backend: String, stdout: String },
    #[error(
        "prompt tokenization mismatch on backend `{backend}`: llama-rs={llama_rs_prompt:?}, llama.cpp={llama_cpp_prompt:?}"
    )]
    PromptTokenizationMismatch {
        backend: String,
        llama_rs_prompt: Vec<i32>,
        llama_cpp_prompt: Vec<i32>,
    },
    #[error(
        "token-id parity mismatch on backend `{backend}`: llama-rs={llama_rs:?}, llama.cpp={llama_cpp:?}"
    )]
    ParityMismatch {
        backend: String,
        llama_rs: Vec<i32>,
        llama_cpp: Vec<i32>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum MixedLayerPolicyArg {
    Strict,
    SkipUnsupportedAttention,
}

impl From<MixedLayerPolicyArg> for MixedLayerPolicy {
    fn from(value: MixedLayerPolicyArg) -> Self {
        match value {
            MixedLayerPolicyArg::Strict => Self::Strict,
            MixedLayerPolicyArg::SkipUnsupportedAttention => Self::SkipUnsupportedAttention,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParityBackend {
    Cpu,
    Metal,
}

impl ParityBackend {
    const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }

    const fn llama_backend(self) -> LlamaBackend {
        match self {
            Self::Cpu => LlamaBackend::Cpu,
            Self::Metal => LlamaBackend::Metal,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "e2e_parity_harness")]
#[command(about = "Compare llama-rs generated token IDs with llama.cpp llama-simple")]
struct Cli {
    /// Path to GGUF model.
    #[arg(long)]
    model: PathBuf,
    /// Comma-separated prompt token IDs, e.g. `1,123,456`.
    #[arg(long, conflicts_with = "prompt_text")]
    prompt_tokens: Option<String>,
    /// Prompt text used by both llama-rs and llama.cpp.
    #[arg(long, conflicts_with = "prompt_tokens")]
    prompt_text: Option<String>,
    /// Maximum number of generated tokens.
    #[arg(long, default_value_t = 1)]
    max_new_tokens: usize,
    /// Path to llama.cpp llama-simple binary.
    #[arg(long, default_value = "target/vendor/llama.cpp/build/bin/llama-simple")]
    llama_simple_bin: PathBuf,
    /// Path to llama.cpp token-id helper binary.
    #[arg(
        long,
        default_value = "target/vendor/llama.cpp/build/bin/llama-simple-token-ids"
    )]
    llama_token_ids_bin: PathBuf,
    /// Path to token-id helper source used for auto-build.
    #[arg(long, default_value = "llama-rs/tools/llama_simple_token_ids.cpp")]
    llama_token_ids_source: PathBuf,
    /// Build token-id helper automatically when missing.
    #[arg(long, default_value_t = true)]
    auto_build_token_ids_helper: bool,
    /// `-ngl` value used for llama-simple CPU run.
    #[arg(long, default_value_t = 0)]
    llama_simple_cpu_ngl: i32,
    /// `-ngl` value used for llama-simple Metal run.
    #[arg(long, default_value_t = 99)]
    llama_simple_metal_ngl: i32,
    /// Skip Metal parity run.
    #[arg(long, default_value_t = false)]
    skip_metal: bool,
    /// Mixed-layer handling policy used by llama-rs E2E.
    #[arg(long, value_enum, default_value_t = MixedLayerPolicyArg::Strict)]
    mixed_layer_policy: MixedLayerPolicyArg,
}

#[derive(Debug)]
struct ParityRunReport {
    backend: ParityBackend,
    llama_rs: E2eGenerationReport,
    llama_rs_prompt_token_ids: Vec<i32>,
    llama_cpp_prompt_token_ids: Vec<i32>,
    llama_cpp_generated_token_ids: Vec<i32>,
    matched: bool,
}

#[derive(Debug, Clone)]
struct LlamaCppTokenIdsReport {
    prompt_token_ids: Vec<i32>,
    generated_token_ids: Vec<i32>,
}

fn main() -> Result<(), Box<dyn StdError>> {
    let cli = Cli::parse();
    run(cli).map_err(Into::into)
}

fn run(cli: Cli) -> Result<(), ExampleError> {
    ensure_llama_token_ids_binary(&cli)?;
    let model = GgufModel::open(&cli.model)?;
    let prompt_token_ids = resolve_prompt_token_ids(&cli, &model)?;
    let eos_token_id = resolve_eos_token_id(&model);
    let mixed_layer_policy: MixedLayerPolicy = cli.mixed_layer_policy.into();

    let mut backends = vec![ParityBackend::Cpu];
    if !cli.skip_metal {
        backends.push(ParityBackend::Metal);
    }

    let mut reports = Vec::with_capacity(backends.len());
    for backend in backends {
        reports.push(run_parity_case(
            &model,
            &cli,
            backend,
            &prompt_token_ids,
            eos_token_id,
            mixed_layer_policy,
        )?);
    }

    for report in &reports {
        println!(
            "backend={} prompt_tokens={} generated_tokens={} llama_rs_prompt_token_ids={:?} llama_cpp_prompt_token_ids={:?} llama_rs_generated_token_ids={:?} llama_cpp_generated_token_ids={:?} match={} attention_layers={} mlp_only_layers={} llama_rs_avg_token_ms={:.3}",
            report.backend.name(),
            report.llama_rs.prompt_token_count,
            report.llama_rs.generated_token_ids.len(),
            report.llama_rs_prompt_token_ids,
            report.llama_cpp_prompt_token_ids,
            report.llama_rs.generated_token_ids,
            report.llama_cpp_generated_token_ids,
            report.matched,
            report.llama_rs.attention_layer_count,
            report.llama_rs.mlp_only_layer_count,
            report.llama_rs.avg_generated_token_ms(),
        );
    }

    if let Some(mismatch) = reports.iter().find(|report| !report.matched) {
        return Err(ExampleError::ParityMismatch {
            backend: mismatch.backend.name().to_string(),
            llama_rs: mismatch.llama_rs.generated_token_ids.clone(),
            llama_cpp: mismatch.llama_cpp_generated_token_ids.clone(),
        });
    }

    Ok(())
}

fn run_parity_case(
    model: &GgufModel,
    cli: &Cli,
    backend: ParityBackend,
    prompt_token_ids: &[i32],
    eos_token_id: Option<i32>,
    mixed_layer_policy: MixedLayerPolicy,
) -> Result<ParityRunReport, ExampleError> {
    let config = E2eGenerationConfig::new(
        backend.llama_backend(),
        prompt_token_ids.to_vec(),
        cli.max_new_tokens,
    )?
    .with_eos_token_id(eos_token_id)
    .with_mixed_layer_policy(mixed_layer_policy);
    let llama_rs_report = generate_token_ids_from_model(model, &config)?;

    let llama_cpp_report = run_llama_token_ids_helper(
        cli,
        prompt_token_ids,
        &cli.model,
        &cli.llama_token_ids_bin,
        cli.max_new_tokens,
        match backend {
            ParityBackend::Cpu => cli.llama_simple_cpu_ngl,
            ParityBackend::Metal => cli.llama_simple_metal_ngl,
        },
        backend,
    )?;

    if prompt_token_ids != llama_cpp_report.prompt_token_ids {
        return Err(ExampleError::PromptTokenizationMismatch {
            backend: backend.name().to_string(),
            llama_rs_prompt: prompt_token_ids.to_vec(),
            llama_cpp_prompt: llama_cpp_report.prompt_token_ids,
        });
    }

    Ok(ParityRunReport {
        backend,
        matched: llama_rs_report.generated_token_ids == llama_cpp_report.generated_token_ids,
        llama_rs: llama_rs_report,
        llama_rs_prompt_token_ids: prompt_token_ids.to_vec(),
        llama_cpp_prompt_token_ids: llama_cpp_report.prompt_token_ids.clone(),
        llama_cpp_generated_token_ids: llama_cpp_report.generated_token_ids,
    })
}

fn ensure_llama_token_ids_binary(cli: &Cli) -> Result<(), ExampleError> {
    let binary_exists = cli.llama_token_ids_bin.is_file();
    let source_is_newer = if binary_exists && cli.llama_token_ids_source.is_file() {
        let source_mtime = fs::metadata(&cli.llama_token_ids_source)
            .and_then(|meta| meta.modified())
            .ok();
        let binary_mtime = fs::metadata(&cli.llama_token_ids_bin)
            .and_then(|meta| meta.modified())
            .ok();
        match (source_mtime, binary_mtime) {
            (Some(source), Some(binary)) => source > binary,
            _ => false,
        }
    } else {
        false
    };

    if binary_exists && !source_is_newer {
        return Ok(());
    }

    if !cli.auto_build_token_ids_helper {
        return Err(ExampleError::MissingLlamaTokenIdsBinary {
            path: cli.llama_token_ids_bin.clone(),
        });
    }

    let build_bin_dir =
        cli.llama_simple_bin
            .parent()
            .ok_or_else(|| ExampleError::InvalidLlamaSimplePath {
                path: cli.llama_simple_bin.clone(),
            })?;
    let build_dir = build_bin_dir
        .parent()
        .ok_or_else(|| ExampleError::InvalidLlamaSimplePath {
            path: cli.llama_simple_bin.clone(),
        })?;
    let llama_cpp_root =
        build_dir
            .parent()
            .ok_or_else(|| ExampleError::InvalidLlamaSimplePath {
                path: cli.llama_simple_bin.clone(),
            })?;

    if let Some(parent) = cli.llama_token_ids_bin.parent() {
        fs::create_dir_all(parent)?;
    }

    let output = Command::new("c++")
        .arg("-std=c++17")
        .arg("-O2")
        .arg(&cli.llama_token_ids_source)
        .arg(format!("-I{}", llama_cpp_root.join("include").display()))
        .arg(format!(
            "-I{}",
            llama_cpp_root.join("ggml").join("include").display()
        ))
        .arg(format!("-L{}", build_bin_dir.display()))
        .arg(format!("-Wl,-rpath,{}", build_bin_dir.display()))
        .arg("-lllama")
        .arg("-lggml")
        .arg("-o")
        .arg(&cli.llama_token_ids_bin)
        .output()?;

    if !output.status.success() {
        return Err(ExampleError::LlamaTokenIdsBuildFailed {
            status: output.status.code(),
            source_path: cli.llama_token_ids_source.clone(),
            output: cli.llama_token_ids_bin.clone(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        });
    }

    if !cli.llama_token_ids_bin.is_file() {
        return Err(ExampleError::MissingLlamaTokenIdsBinary {
            path: cli.llama_token_ids_bin.clone(),
        });
    }

    Ok(())
}

fn run_llama_token_ids_helper(
    cli: &Cli,
    prompt_token_ids: &[i32],
    model_path: &Path,
    llama_token_ids_bin: &Path,
    max_new_tokens: usize,
    n_gpu_layers: i32,
    backend: ParityBackend,
) -> Result<LlamaCppTokenIdsReport, ExampleError> {
    let llama_token_ids_dir = llama_token_ids_bin
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let dylib_fallback = std::env::var("DYLD_FALLBACK_LIBRARY_PATH")
        .ok()
        .filter(|value| !value.is_empty())
        .map_or_else(
            || llama_token_ids_dir.display().to_string(),
            |existing| format!("{}:{existing}", llama_token_ids_dir.display()),
        );

    let output = Command::new(llama_token_ids_bin)
        .arg("-m")
        .arg(model_path)
        .arg("-n")
        .arg(max_new_tokens.to_string())
        .arg("-ngl")
        .arg(n_gpu_layers.to_string())
        .args(llama_cpp_prompt_args(cli, prompt_token_ids))
        .env("DYLD_FALLBACK_LIBRARY_PATH", dylib_fallback)
        .output()?;

    if !output.status.success() {
        return Err(ExampleError::LlamaTokenIdsFailed {
            backend: backend.name().to_string(),
            status: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(LlamaCppTokenIdsReport {
        prompt_token_ids: parse_token_ids_marker(&stdout, backend, "prompt_token_ids=[")?,
        generated_token_ids: parse_token_ids_marker(&stdout, backend, "generated_token_ids=[")?,
    })
}

fn resolve_prompt_token_ids(cli: &Cli, model: &GgufModel) -> Result<Vec<i32>, ExampleError> {
    match (&cli.prompt_tokens, &cli.prompt_text) {
        (Some(tokens), None) => parse_prompt_tokens(tokens).map_err(invalid_input),
        (None, Some(text)) => tokenize_prompt_text(model, text).map_err(Into::into),
        _ => Err(invalid_input(
            "provide exactly one of --prompt-tokens or --prompt-text",
        )),
    }
}

fn parse_prompt_tokens(value: &str) -> Result<Vec<i32>, String> {
    let tokens: Vec<i32> = value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| {
            i32::from_str(token).map_err(|error| format!("invalid token id `{token}`: {error}"))
        })
        .collect::<Result<_, _>>()?;
    if tokens.is_empty() {
        return Err("--prompt-tokens must include at least one token id".to_owned());
    }
    Ok(tokens)
}

fn invalid_input(message: impl Into<String>) -> ExampleError {
    ExampleError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message.into(),
    ))
}

fn llama_cpp_prompt_args(cli: &Cli, prompt_token_ids: &[i32]) -> Vec<String> {
    if let Some(prompt_text) = &cli.prompt_text {
        return vec![prompt_text.clone()];
    }
    vec![
        "--prompt-tokens".to_owned(),
        prompt_token_ids
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(","),
    ]
}

fn parse_token_ids_marker(
    stdout: &str,
    backend: ParityBackend,
    marker: &str,
) -> Result<Vec<i32>, ExampleError> {
    let Some(marker_pos) = stdout.find(marker) else {
        return Err(ExampleError::LlamaTokenIdsParseFailed {
            backend: backend.name().to_string(),
            stdout: stdout.trim().to_string(),
        });
    };

    let ids_start = marker_pos + marker.len();
    let Some(ids_end_rel) = stdout[ids_start..].find(']') else {
        return Err(ExampleError::LlamaTokenIdsParseFailed {
            backend: backend.name().to_string(),
            stdout: stdout.trim().to_string(),
        });
    };

    let ids_text = &stdout[ids_start..ids_start + ids_end_rel];
    let ids_text = ids_text.trim();
    if ids_text.is_empty() {
        return Ok(Vec::new());
    }

    ids_text
        .split(',')
        .map(|id| {
            id.trim()
                .parse::<i32>()
                .map_err(|_| ExampleError::LlamaTokenIdsParseFailed {
                    backend: backend.name().to_string(),
                    stdout: stdout.trim().to_string(),
                })
        })
        .collect()
}
