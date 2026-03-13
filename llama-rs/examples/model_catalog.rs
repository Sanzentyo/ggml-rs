//! Model catalog demo for GGUF metadata + tensor payload validation.

use llama_rs::GgufModel;
use std::error::Error as StdError;

fn main() -> Result<(), Box<dyn StdError>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (path, head, checks) = parse_args(&args)?;

    let model = GgufModel::open(path)?;
    let report = model.report();

    println!("path:         {}", model.path().display());
    println!("file_size:    {}", model.file_size());
    println!("version:      {}", report.version);
    println!("alignment:    {}", report.alignment);
    println!("data_offset:  {}", report.data_offset);
    println!("n_kv:         {}", report.kv_entries.len());
    println!("n_tensors:    {}", report.tensors.len());

    println!("\n[tensors head={head}]");
    for tensor in report.tensors.iter().take(head) {
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

    if !checks.is_empty() {
        println!("\n[check tensors]");
        for name in checks {
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

fn parse_args(args: &[String]) -> Result<(&str, usize, Vec<String>), Box<dyn StdError>> {
    let mut path = None;
    let mut head = 10usize;
    let mut checks = Vec::new();

    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--head" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err("missing value after --head".into());
                };
                head = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --head value `{value}` ({error})"))?;
            }
            "--check-tensor" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err("missing value after --check-tensor".into());
                };
                checks.push(value.clone());
            }
            token => {
                if path.is_none() {
                    path = Some(token);
                } else {
                    return Err(format!("unexpected argument `{token}`").into());
                }
            }
        }
        index += 1;
    }

    let Some(path) = path else {
        return Err("usage: cargo run -p llama-rs --example model_catalog --features link-system -- <model.gguf> [--head N] [--check-tensor NAME ...]".into());
    };
    Ok((path, head, checks))
}
