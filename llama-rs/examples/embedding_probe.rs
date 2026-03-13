//! Embedding probe demo for f32 tensor summary statistics.

use llama_rs::{GgufModel, summarize_embedding_tensor};
use std::error::Error as StdError;

fn main() -> Result<(), Box<dyn StdError>> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        return Err("usage: cargo run -p llama-rs --example embedding_probe --features link-system -- <model.gguf> [tensor_name]".into());
    };
    let tensor_name = args.next().unwrap_or_else(|| "tensor_0".to_string());

    let model = GgufModel::open(path)?;
    let stats = summarize_embedding_tensor(&model, &tensor_name)?;

    println!("tensor:   {tensor_name}");
    println!("len:      {}", stats.len);
    println!("mean:     {:.6}", stats.mean);
    println!("l2_norm:  {:.6}", stats.l2_norm);
    println!("min:      {:.6}", stats.min);
    println!("max:      {:.6}", stats.max);

    Ok(())
}
