//! Generate a markdown comparison report from benchmark artifacts.

use clap::Parser;
use llama_rs::{
    parse_attention_bench_output, parse_llama_cpp_jsonl, parse_mlp_bench_output,
    render_markdown_summary,
};
use std::error::Error as StdError;
use std::fs;
use std::path::PathBuf;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    let config = Config::from_cli(Cli::parse());

    let mut cpp_rows = Vec::new();
    for path in &config.llama_cpp_jsonl_paths {
        cpp_rows.extend(parse_llama_cpp_jsonl(path)?);
    }
    let mlp_rows = parse_mlp_bench_output(&config.llama_rs_mlp_output_path)?;
    let attention_rows = parse_attention_bench_output(&config.llama_rs_attention_output_path)?;

    let markdown = render_markdown_summary(&cpp_rows, &mlp_rows, &attention_rows);
    if let Some(path) = &config.output_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &markdown)?;
        eprintln!("wrote report: {}", path.display());
    }

    println!("{markdown}");
    Ok(())
}

#[derive(Debug, Error)]
enum ExampleError {
    #[error(transparent)]
    BenchReport(#[from] llama_rs::BenchReportError),
    #[error(transparent)]
    Llama(#[from] llama_rs::LlamaError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Boxed(#[from] Box<dyn StdError>),
}

#[derive(Debug)]
struct Config {
    llama_cpp_jsonl_paths: Vec<PathBuf>,
    llama_rs_mlp_output_path: PathBuf,
    llama_rs_attention_output_path: PathBuf,
    output_path: Option<PathBuf>,
}

impl Config {
    fn from_cli(cli: Cli) -> Self {
        let llama_cpp_jsonl_paths = if cli.llama_cpp_jsonl_paths.is_empty() {
            vec![
                PathBuf::from("target/benchmarks/llama_cpp_baseline_all.jsonl"),
                PathBuf::from("target/benchmarks/llama_cpp_baseline_extra.jsonl"),
            ]
        } else {
            cli.llama_cpp_jsonl_paths
        };
        Self {
            llama_cpp_jsonl_paths,
            llama_rs_mlp_output_path: cli.llama_rs_mlp_output_path,
            llama_rs_attention_output_path: cli.llama_rs_attention_output_path,
            output_path: cli.output_path,
        }
    }
}

#[derive(Debug, Clone, Parser)]
#[command(about = "Compare llama.cpp and llama-rs benchmark artifacts", version)]
struct Cli {
    /// Path to llama.cpp JSONL output (repeatable).
    #[arg(long = "llama-cpp")]
    llama_cpp_jsonl_paths: Vec<PathBuf>,
    /// Path to bench_mlp_layer output text.
    #[arg(
        long = "llama-rs-mlp",
        default_value = "target/benchmarks/llama_rs_bench_mlp_models.txt"
    )]
    llama_rs_mlp_output_path: PathBuf,
    /// Path to bench_attention_layer output text.
    #[arg(
        long = "llama-rs-attention",
        default_value = "target/benchmarks/llama_rs_bench_attention_models.txt"
    )]
    llama_rs_attention_output_path: PathBuf,
    /// Optional markdown output path.
    #[arg(long = "output")]
    output_path: Option<PathBuf>,
}
