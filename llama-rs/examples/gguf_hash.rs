use llama_rs::{HashAlgorithm, HashOptions, HashRecord, hash_file};
use std::collections::HashMap;
use std::error::Error as StdError;

fn main() -> Result<(), Box<dyn StdError>> {
    let options = parse_options(std::env::args().skip(1))?;
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

#[derive(Debug, Clone)]
struct CliOptions {
    input: String,
    hash_options: HashOptions,
    check_manifest: Option<String>,
}

fn parse_options(args: impl Iterator<Item = String>) -> Result<CliOptions, Box<dyn StdError>> {
    let mut args = args.peekable();
    let mut include_layers = true;
    let mut check_manifest = None;
    let mut selected = Vec::new();
    let mut input = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--xxh64" => selected.push(HashAlgorithm::Xxh64),
            "--sha1" => selected.push(HashAlgorithm::Sha1),
            "--sha256" => selected.push(HashAlgorithm::Sha256),
            "--uuid" => selected.push(HashAlgorithm::Uuid),
            "--all" => selected.extend([
                HashAlgorithm::Xxh64,
                HashAlgorithm::Sha1,
                HashAlgorithm::Sha256,
            ]),
            "--no-layer" => include_layers = false,
            "-c" | "--check" => {
                let Some(path) = args.next() else {
                    return Err("missing manifest path after --check".into());
                };
                check_manifest = Some(path);
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unknown option: {arg}").into());
            }
            _ => {
                input = Some(arg);
                break;
            }
        }
    }

    if input.is_none() {
        input = args.next();
    }

    let Some(input) = input else {
        print_usage();
        return Err("missing input gguf path".into());
    };

    if selected.is_empty() {
        selected.push(HashAlgorithm::Xxh64);
    }
    selected.sort_unstable_by_key(|algo| algo.as_label());
    selected.dedup();

    Ok(CliOptions {
        input,
        hash_options: HashOptions {
            algorithms: selected,
            include_layers,
        },
        check_manifest,
    })
}

fn print_usage() {
    eprintln!(
        "usage: cargo run -p llama-rs --example gguf_hash --features link-system -- [options] <file.gguf>"
    );
    eprintln!("options:");
    eprintln!("  --xxh64             use xxh64 hash");
    eprintln!("  --sha1              use sha1 hash");
    eprintln!("  --sha256            use sha256 hash");
    eprintln!("  --uuid              generate UUIDv5");
    eprintln!("  --all               use xxh64 + sha1 + sha256");
    eprintln!("  --no-layer          exclude per-layer hashes");
    eprintln!("  -c, --check <file>  verify against a manifest");
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
