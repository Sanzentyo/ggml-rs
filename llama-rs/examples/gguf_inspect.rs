use llama_rs::inspect_gguf;
use std::error::Error as StdError;

fn main() -> Result<(), Box<dyn StdError>> {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!(
            "usage: cargo run -p llama-rs --example gguf_inspect --features link-system -- <file.gguf>"
        );
        std::process::exit(2);
    };

    let report = inspect_gguf(&path)?;
    println!("version:      {}", report.version);
    println!("alignment:    {}", report.alignment);
    println!("data_offset:  {}", report.data_offset);
    println!("n_kv:         {}", report.kv_entries.len());
    println!("n_tensors:    {}", report.tensors.len());

    println!("\n[kv]");
    for entry in &report.kv_entries {
        match &entry.string_value {
            Some(value) => println!("- {} ({}) = {:?}", entry.key, entry.value_type, value),
            None => println!("- {} ({})", entry.key, entry.value_type),
        }
    }

    println!("\n[tensors]");
    for (index, tensor) in report.tensors.iter().enumerate() {
        println!(
            "- [{}] name={} type={}({}) size={} offset={}",
            index,
            tensor.name,
            tensor.ggml_type_name,
            tensor.ggml_type_raw,
            tensor.size,
            tensor.offset
        );
    }

    Ok(())
}
