use clap::Parser;
use llama_rs::{HashAlgorithm, HashOptions, HashRecord, hash_file};
use std::collections::HashMap;
use std::error::Error as StdError;
use thiserror::Error;

fn main() -> Result<(), ExampleError> {
    let options = CliOptions::from_cli(Cli::parse());
    let records = hash_file(&options.input, &options.hash_options)?;

    if let Some(manifest) = &options.check_manifest {
        verify_manifest(manifest, &records)?;
    } else {
        for record in &records {
            println!(
                "{:<8}  {}  {}",
                record.algorithm.as_label(),
                record.value,
                record.target
            );
        }
    }

    Ok(())
}

#[derive(Debug, Error)]
enum ExampleError {
    #[error(transparent)]
    Hash(#[from] llama_rs::GgufHashError),
    #[error(transparent)]
    Llama(#[from] llama_rs::LlamaError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Boxed(#[from] Box<dyn StdError>),
}

#[derive(Debug, Clone)]
struct CliOptions {
    input: String,
    hash_options: HashOptions,
    check_manifest: Option<String>,
}

impl CliOptions {
    fn from_cli(cli: Cli) -> Self {
        let mut selected = Vec::new();
        if cli.xxh64 {
            selected.push(HashAlgorithm::Xxh64);
        }
        if cli.sha1 {
            selected.push(HashAlgorithm::Sha1);
        }
        if cli.sha256 {
            selected.push(HashAlgorithm::Sha256);
        }
        if cli.uuid {
            selected.push(HashAlgorithm::Uuid);
        }
        if cli.all {
            selected.extend([
                HashAlgorithm::Xxh64,
                HashAlgorithm::Sha1,
                HashAlgorithm::Sha256,
            ]);
        }
        if selected.is_empty() {
            selected.push(HashAlgorithm::Xxh64);
        }
        selected.sort_unstable_by_key(|algo| algo.as_label());
        selected.dedup();

        Self {
            input: cli.input,
            hash_options: HashOptions {
                algorithms: selected,
                include_layers: !cli.no_layer,
            },
            check_manifest: cli.check_manifest,
        }
    }
}

#[derive(Debug, Clone, Parser)]
#[command(about = "Hash GGUF file sections and optional layer ranges", version, long_about = None)]
struct Cli {
    /// Include xxh64.
    #[arg(long)]
    xxh64: bool,
    /// Include sha1.
    #[arg(long)]
    sha1: bool,
    /// Include sha256.
    #[arg(long)]
    sha256: bool,
    /// Include UUIDv5 from hash material.
    #[arg(long)]
    uuid: bool,
    /// Include xxh64 + sha1 + sha256.
    #[arg(long)]
    all: bool,
    /// Exclude per-layer hashes.
    #[arg(long = "no-layer")]
    no_layer: bool,
    /// Verify against manifest.
    #[arg(long = "check", short = 'c')]
    check_manifest: Option<String>,
    /// Input GGUF file.
    input: String,
}

fn verify_manifest(manifest: &str, records: &[HashRecord]) -> Result<(), Box<dyn StdError>> {
    let text = std::fs::read_to_string(manifest)?;
    let mut manifest_map = HashMap::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut parts = trimmed.split_whitespace();
        let Some(algo) = parts.next() else {
            continue;
        };
        let Some(hash) = parts.next() else {
            continue;
        };
        let Some(target) = parts.next() else {
            continue;
        };

        manifest_map.insert((algo.to_string(), target.to_string()), hash.to_string());
    }

    let mut all_ok = true;
    println!("manifest  {manifest}");
    for record in records {
        let key = (
            record.algorithm.as_label().to_string(),
            record.target.to_string(),
        );
        let status = match manifest_map.get(&key) {
            Some(expected) if expected == &record.value => "Ok",
            Some(_) => {
                all_ok = false;
                "Mismatch"
            }
            None => {
                all_ok = false;
                "Missing"
            }
        };

        println!(
            "{:<8}  {}  {}  -  {}",
            record.algorithm.as_label(),
            record.value,
            record.target,
            status
        );
    }

    if all_ok {
        println!("\nVerification results for {manifest} - Success");
        Ok(())
    } else {
        println!("\nVerification results for {manifest} - Failure");
        Err("manifest verification failed".into())
    }
}
