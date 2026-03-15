//! Model catalog demo for GGUF metadata + tensor payload validation.

use clap::Parser;
use llama_rs::GgufModel;
use std::error::Error as StdError;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    run().map_err(Into::into)
}

fn run() -> Result<(), Box<dyn StdError>> {
    let cli = Cli::parse();

    let model = GgufModel::open(&cli.path)?;
    let report = model.report();

    println!("path:         {}", model.path().display());
    println!("file_size:    {}", model.file_size());
    println!("version:      {}", report.version);
    println!("alignment:    {}", report.alignment);
    println!("data_offset:  {}", report.data_offset);
    println!("n_kv:         {}", report.kv_entries.len());
    println!("n_tensors:    {}", report.tensors.len());

    println!("\n[tensors head={}]", cli.head);
    for tensor in report.tensors.iter().take(cli.head) {
        let payload = model.tensor_payload(&tensor.name)?;
        println!(
            "- name={} type={}({}) size={} offset={} payload_len={}",
            tensor.name,
            tensor.ggml_type_name,
            tensor.ggml_type_raw,
            tensor.size,
            tensor.offset,
            payload.len()
        );
    }

    if !cli.check_tensors.is_empty() {
        println!("\n[check tensors]");
        for name in cli.check_tensors {
            let info = model.tensor_info(&name)?;
            let payload = model.tensor_payload(&name)?;
            println!(
                "- {name}: type={}({}) size={} payload_len={}",
                info.ggml_type_name,
                info.ggml_type_raw,
                info.size,
                payload.len()
            );
        }
    }

    Ok(())
}

#[derive(Debug, Error)]
#[error(transparent)]
struct ExampleError(#[from] Box<dyn StdError>);

#[derive(Debug, Parser)]
#[command(about = "Inspect model tensor catalog and payloads", version, long_about = None)]
struct Cli {
    /// Path to GGUF model.
    path: String,
    /// Number of leading tensor rows to print.
    #[arg(long, default_value_t = 10)]
    head: usize,
    /// Tensor names to inspect in detail.
    #[arg(long = "check-tensor")]
    check_tensors: Vec<String>,
}
