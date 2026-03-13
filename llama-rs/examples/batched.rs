//! Batched execution demo using `llama-rs` foundation APIs.

use llama_rs::{
    BatchSize, BatchedConfig, BatchedWorkload, LlamaBackend, ReadbackEvery, RepeatCount,
    run_batched_matmul_with_workload,
};
use std::error::Error as StdError;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn StdError>> {
    ggml_rs::init_timing();

    let (config, backends) = parse_args(std::env::args().skip(1))?;
    let workload = BatchedWorkload::deterministic(config)?;
    for backend in backends {
        let report = run_batched_matmul_with_workload(backend, &workload)?;
        println!(
            "[{}] batched matmul {}x{} · {}x{} batch={} repeats={} readback_every={} readbacks={} avg_item={:.3} ms, checksum={:.6}",
            report.backend_name,
            report.rows_a,
            report.cols_a,
            report.rows_b,
            report.cols_b,
            report.batch_size,
            report.repeats,
            report.readback_every,
            report.readback_samples,
            report.avg_item_ms,
            report.checksum
        );
    }

    Ok(())
}

fn parse_args(
    args: impl Iterator<Item = String>,
) -> Result<(BatchedConfig, Vec<LlamaBackend>), Box<dyn StdError>> {
    let mut config = BatchedConfig::default();
    let mut backends = Vec::new();

    let mut pending_batch = false;
    let mut pending_repeats = false;
    let mut pending_readback_every = false;
    let mut pending_size = false;

    for arg in args {
        if pending_batch {
            config.batch_size = BatchSize::new(parse_usize_arg("--batch", &arg)?)?;
            pending_batch = false;
            continue;
        }
        if pending_repeats {
            config.repeats = RepeatCount::new(parse_usize_arg("--repeats", &arg)?)?;
            pending_repeats = false;
            continue;
        }
        if pending_readback_every {
            config.readback_every = ReadbackEvery::new(parse_usize_arg("--readback-every", &arg)?)?;
            pending_readback_every = false;
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
            "--batch" | "-b" => pending_batch = true,
            "--repeats" | "-n" => pending_repeats = true,
            "--readback-every" | "-r" => pending_readback_every = true,
            "--size" | "-s" => pending_size = true,
            token => backends.push(LlamaBackend::from_str(token)?),
        }
    }

    if pending_batch {
        return Err("missing value after --batch".into());
    }
    if pending_repeats {
        return Err("missing value after --repeats".into());
    }
    if pending_readback_every {
        return Err("missing value after --readback-every".into());
    }
    if pending_size {
        return Err("missing value after --size".into());
    }

    if backends.is_empty() {
        backends.push(LlamaBackend::Cpu);
        backends.push(LlamaBackend::Metal);
    }

    Ok((config.validated()?, backends))
}

fn parse_usize_arg(flag: &str, value: &str) -> Result<usize, Box<dyn StdError>> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid value for {flag}: {value} ({error})").into())
}
