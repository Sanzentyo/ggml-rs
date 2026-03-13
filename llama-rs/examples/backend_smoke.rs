use llama_rs::{LlamaBackend, run_backend_smoke};
use std::error::Error as StdError;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    for backend in parse_requested_backends(std::env::args().skip(1))? {
        let report = run_backend_smoke(backend)?;
        println!(
            "[{}] mul mat ({} x {}) OK",
            report.backend_name, report.cols, report.rows
        );
    }

    Ok(())
}

fn parse_requested_backends(
    args: impl Iterator<Item = String>,
) -> Result<Vec<LlamaBackend>, Box<dyn StdError>> {
    let mut parsed = Vec::new();
    for arg in args {
        parsed.push(LlamaBackend::from_str(&arg)?);
    }

    if parsed.is_empty() {
        Ok(vec![LlamaBackend::Cpu, LlamaBackend::Metal])
    } else {
        Ok(parsed)
    }
}
