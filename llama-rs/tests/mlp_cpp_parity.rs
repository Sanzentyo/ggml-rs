#![cfg(feature = "link-system")]

use llama_rs::{LlamaBackend, MlpInferenceConfig, MlpWeights, mlp_inference_with_weights_repeats};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

const CASES: &[(usize, usize)] = &[(8, 16), (16, 32), (32, 64)];

#[test]
fn mlp_cpu_matches_cpp_reference_across_sizes() -> Result<(), Box<dyn Error>> {
    let cpp_binary = compile_cpp_reference()?;
    for &(hidden_features, ffn_features) in CASES {
        let config = MlpInferenceConfig::new(hidden_features, ffn_features)?;
        let weights = MlpWeights::deterministic(config);
        let input = build_input(hidden_features);

        let cpp_output = cpp_reference(&cpp_binary, hidden_features, ffn_features)?;
        let rust_report =
            mlp_inference_with_weights_repeats(&weights, &input, LlamaBackend::Cpu, 1)?;
        assert_eq!(cpp_output.len(), rust_report.output.len());

        let max_delta = max_abs_delta(&cpp_output, &rust_report.output);
        assert!(
            max_delta <= 1e-3,
            "rust/cpp output divergence too high for hidden={hidden_features}, ffn={ffn_features}: max_delta={max_delta}"
        );
    }
    let _ = std::fs::remove_file(&cpp_binary);

    Ok(())
}

#[test]
fn mlp_cpu_matches_metal_across_sizes() -> Result<(), Box<dyn Error>> {
    for &(hidden_features, ffn_features) in CASES {
        let config = MlpInferenceConfig::new(hidden_features, ffn_features)?;
        let weights = MlpWeights::deterministic(config);
        let input = build_input(hidden_features);

        let cpu_report =
            mlp_inference_with_weights_repeats(&weights, &input, LlamaBackend::Cpu, 1)?;
        let metal_report =
            match mlp_inference_with_weights_repeats(&weights, &input, LlamaBackend::Metal, 1) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("metal backend unavailable; skipping parity test: {error}");
                    return Ok(());
                }
            };

        assert_eq!(cpu_report.output.len(), metal_report.output.len());
        let max_delta = max_abs_delta(&cpu_report.output, &metal_report.output);
        assert!(
            max_delta <= 2e-3,
            "cpu/metal output divergence too high for hidden={hidden_features}, ffn={ffn_features}: max_delta={max_delta}"
        );
    }

    Ok(())
}

fn compile_cpp_reference() -> Result<PathBuf, Box<dyn Error>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = manifest_dir.join("tests/cpp/mlp_reference.cpp");
    let ggml_root = ggml_root_dir();
    let ggml_include = ggml_include_dir(&ggml_root);
    let ggml_src = ggml_include
        .parent()
        .map(|parent| parent.join("src"))
        .unwrap_or_else(|| ggml_root.join("src"));
    let ggml_lib = ggml_lib_dir(&ggml_root);
    let output = std::env::temp_dir().join(format!(
        "mlp_reference_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));

    let compile = Command::new("c++")
        .arg("-O2")
        .arg("-std=c++17")
        .arg(format!("-I{}", ggml_include.display()))
        .arg(format!("-I{}", ggml_src.display()))
        .arg(format!("-L{}", ggml_lib.display()))
        .arg(format!("-Wl,-rpath,{}", ggml_lib.display()))
        .arg(&source)
        .arg("-lggml-cpu")
        .arg("-lggml-base")
        .arg("-lggml")
        .arg("-o")
        .arg(&output)
        .output()?;
    if !compile.status.success() {
        return Err(format!(
            "failed to compile C++ reference: {}",
            String::from_utf8_lossy(&compile.stderr)
        )
        .into());
    }

    Ok(output)
}

fn ggml_root_dir() -> PathBuf {
    if let Ok(path) = std::env::var("GGML_CPP_REFERENCE_GGML_DIR") {
        return PathBuf::from(path);
    }

    let workspace = workspace_root();
    [
        workspace.join("target/vendor/ggml"),
        workspace.join("vendor/ggml"),
    ]
    .into_iter()
    .find(|path| path.join("include").exists())
    .unwrap_or_else(|| workspace.join("target/vendor/ggml"))
}

fn ggml_include_dir(ggml_root: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os("GGML_RS_GGML_INCLUDE_DIR") {
        return PathBuf::from(path);
    }

    let include = ggml_root.join("include");
    if include.exists() {
        include
    } else {
        workspace_root().join("vendor/ggml/include")
    }
}

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or(manifest_dir)
}

fn ggml_lib_dir(ggml_root: &Path) -> PathBuf {
    match std::env::var_os("GGML_RS_LIB_DIRS") {
        Some(paths) => std::env::split_paths(&paths)
            .next()
            .map(PathBuf::from)
            .unwrap_or_else(|| ggml_root.join("build/src")),
        None => ggml_root.join("build/src"),
    }
}

fn cpp_reference(
    binary: &Path,
    hidden_features: usize,
    ffn_features: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let output = Command::new(binary)
        .arg(hidden_features.to_string())
        .arg(ffn_features.to_string())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "cpp reference failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let stdout = String::from_utf8(output.stdout)?;
    let values = stdout
        .trim()
        .split(',')
        .filter(|token| !token.is_empty())
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(values)
}

fn build_input(hidden_features: usize) -> Vec<f32> {
    (0..hidden_features)
        .map(|index| ((index + 5) % 19) as f32 * 0.125)
        .collect()
}

fn max_abs_delta(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}
