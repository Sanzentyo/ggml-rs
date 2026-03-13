use llama_rs::{LlamaBackend, MatmulBenchConfig, run_backend_matmul_bench};
use std::error::Error as StdError;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let (config, backends) = parse_args(std::env::args().skip(1))?;
    for backend in backends {
        let report = run_backend_matmul_bench(backend, config)?;
        println!(
            "[{}] llama-rs matmul {}x{} · {}x{} warmup={} bench={} avg={:.3} ms, checksum={:.6}",
            report.backend_name,
            report.rows_a,
            report.cols_a,
            report.rows_b,
            report.cols_b,
            report.warmup_iters,
            report.bench_iters,
            report.avg_ms,
            report.checksum
        );
    }

    Ok(())
}

fn parse_args(
    args: impl Iterator<Item = String>,
) -> Result<(MatmulBenchConfig, Vec<LlamaBackend>), Box<dyn StdError>> {
    let mut config = MatmulBenchConfig::default();
    let mut backends = Vec::new();

    let mut pending_iters = false;
    let mut pending_warmup = false;
    let mut pending_size = false;

    for arg in args {
        if pending_iters {
            config.bench_iters = parse_usize_arg("--iters", &arg)?;
            pending_iters = false;
            continue;
        }
        if pending_warmup {
            config.warmup_iters = parse_usize_arg("--warmup", &arg)?;
            pending_warmup = false;
            continue;
        }
        if pending_size {
            let size = parse_usize_arg("--size", &arg)?;
            config.rows_a = size;
            config.cols_a = size;
            config.rows_b = size;
            config.cols_b = size;
            pending_size = false;
            continue;
        }

        match arg.as_str() {
            "--iters" | "-n" => pending_iters = true,
            "--warmup" | "-w" => pending_warmup = true,
            "--size" | "-s" => pending_size = true,
            token => backends.push(LlamaBackend::from_str(token)?),
        }
    }

    if pending_iters {
        return Err("missing value after --iters".into());
    }
    if pending_warmup {
        return Err("missing value after --warmup".into());
    }
    if pending_size {
        return Err("missing value after --size".into());
    }

    if backends.is_empty() {
        backends.push(LlamaBackend::Cpu);
        backends.push(LlamaBackend::Metal);
    }

    Ok((config, backends))
}

fn parse_usize_arg(flag: &str, value: &str) -> Result<usize, Box<dyn StdError>> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid value for {flag}: {value} ({error})").into())
}
