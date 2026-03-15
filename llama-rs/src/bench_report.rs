use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error as StdError;
use std::fmt;
use std::fs;
use std::num::TryFromIntError;
use std::path::{Path, PathBuf};

type PromptGenPair = (usize, usize);
type MlpCase = (usize, usize);
type AttentionCase = (usize, usize, usize, usize, usize);
type AttentionStepwiseCase = (usize, usize, usize, usize, usize, usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BenchBackend {
    Cpu,
    Metal,
}

impl fmt::Display for BenchBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Metal => write!(f, "metal"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaCppBenchRow {
    pub model_name: String,
    pub n_prompt: usize,
    pub n_gen: usize,
    pub backend: BenchBackend,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlpBenchRow {
    pub hidden_features: usize,
    pub ffn_features: usize,
    pub backend: BenchBackend,
    pub avg_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionBenchRow {
    pub hidden_features: usize,
    pub query_head_count: usize,
    pub kv_head_count: usize,
    pub query_length: usize,
    pub key_value_length: usize,
    pub step_count: usize,
    pub stepwise: bool,
    pub backend: BenchBackend,
    pub avg_ms: f64,
}

#[derive(Debug)]
pub enum BenchReportError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        line: usize,
        source: serde_json::Error,
    },
    MissingField {
        path: PathBuf,
        line: usize,
        field: &'static str,
    },
    InvalidFieldType {
        path: PathBuf,
        line: usize,
        field: &'static str,
    },
    IntegerOverflow {
        path: PathBuf,
        line: usize,
        field: &'static str,
        source: TryFromIntError,
    },
    InvalidBenchLine {
        path: PathBuf,
        line: usize,
        details: String,
    },
    UnsupportedBackendTag {
        path: PathBuf,
        line: usize,
        tag: String,
    },
}

impl fmt::Display for BenchReportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    f,
                    "failed to read benchmark file `{}`: {source}",
                    path.display()
                )
            }
            Self::Json { path, line, source } => write!(
                f,
                "failed to parse JSON in `{}` at line {line}: {source}",
                path.display()
            ),
            Self::MissingField { path, line, field } => write!(
                f,
                "missing JSON field `{field}` in `{}` at line {line}",
                path.display()
            ),
            Self::InvalidFieldType { path, line, field } => write!(
                f,
                "invalid JSON type for field `{field}` in `{}` at line {line}",
                path.display()
            ),
            Self::IntegerOverflow {
                path,
                line,
                field,
                source,
            } => write!(
                f,
                "integer overflow for field `{field}` in `{}` at line {line}: {source}",
                path.display()
            ),
            Self::InvalidBenchLine {
                path,
                line,
                details,
            } => write!(
                f,
                "invalid bench output line in `{}` at line {line}: {details}",
                path.display()
            ),
            Self::UnsupportedBackendTag { path, line, tag } => write!(
                f,
                "unsupported backend tag `[{tag}]` in `{}` at line {line}",
                path.display()
            ),
        }
    }
}

impl StdError for BenchReportError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::IntegerOverflow { source, .. } => Some(source),
            Self::MissingField { .. }
            | Self::InvalidFieldType { .. }
            | Self::InvalidBenchLine { .. }
            | Self::UnsupportedBackendTag { .. } => None,
        }
    }
}

pub fn parse_llama_cpp_jsonl(
    path: impl AsRef<Path>,
) -> Result<Vec<LlamaCppBenchRow>, BenchReportError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|source| BenchReportError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_llama_cpp_jsonl_text(path, &text)
}

pub fn parse_mlp_bench_output(
    path: impl AsRef<Path>,
) -> Result<Vec<MlpBenchRow>, BenchReportError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|source| BenchReportError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_mlp_bench_output_text(path, &text)
}

pub fn parse_attention_bench_output(
    path: impl AsRef<Path>,
) -> Result<Vec<AttentionBenchRow>, BenchReportError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|source| BenchReportError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_attention_bench_output_text(path, &text)
}

pub fn render_markdown_summary(
    llama_cpp_rows: &[LlamaCppBenchRow],
    mlp_rows: &[MlpBenchRow],
    attention_rows: &[AttentionBenchRow],
) -> String {
    let mut markdown = String::new();
    markdown.push_str("# Benchmark comparison summary\n\n");

    markdown.push_str("## llama.cpp (`tok/s`)\n\n");
    markdown.push_str("| Model | Pair (`prompt/gen`) | CPU | Metal | Metal/CPU |\n");
    markdown.push_str("| --- | --- | ---:| ---:| ---:|\n");

    let mut cpp_map: BTreeMap<(String, PromptGenPair, BenchBackend), f64> = BTreeMap::new();
    let mut cpp_models = BTreeSet::new();
    let mut cpp_pairs = BTreeSet::new();
    for row in llama_cpp_rows {
        cpp_models.insert(row.model_name.clone());
        cpp_pairs.insert((row.n_prompt, row.n_gen));
        cpp_map.insert(
            (
                row.model_name.clone(),
                (row.n_prompt, row.n_gen),
                row.backend,
            ),
            row.tokens_per_second,
        );
    }

    for model in cpp_models {
        for pair in &cpp_pairs {
            let cpu = cpp_map.get(&(model.clone(), *pair, BenchBackend::Cpu));
            let metal = cpp_map.get(&(model.clone(), *pair, BenchBackend::Metal));
            let (Some(cpu), Some(metal)) = (cpu, metal) else {
                continue;
            };
            let ratio = if *cpu == 0.0 { f64::NAN } else { *metal / *cpu };
            markdown.push_str(&format!(
                "| {model} | `{}/{}` | {:.3} | {:.3} | {:.3} |\n",
                pair.0, pair.1, cpu, metal, ratio
            ));
        }
    }

    markdown.push('\n');
    markdown.push_str("## llama-rs proxy MLP (`ms/iter`)\n\n");
    markdown.push_str("| Case (`hidden x ffn`) | CPU | Metal | CPU/Metal |\n");
    markdown.push_str("| --- | ---:| ---:| ---:|\n");

    let mut mlp_map: BTreeMap<(MlpCase, BenchBackend), f64> = BTreeMap::new();
    let mut mlp_cases = BTreeSet::new();
    for row in mlp_rows {
        let case = (row.hidden_features, row.ffn_features);
        mlp_cases.insert(case);
        mlp_map.insert((case, row.backend), row.avg_ms);
    }
    for case in mlp_cases {
        let cpu = mlp_map.get(&(case, BenchBackend::Cpu));
        let metal = mlp_map.get(&(case, BenchBackend::Metal));
        let (Some(cpu), Some(metal)) = (cpu, metal) else {
            continue;
        };
        let ratio = if *metal == 0.0 {
            f64::NAN
        } else {
            *cpu / *metal
        };
        markdown.push_str(&format!(
            "| `{}x{}` | {:.3} | {:.3} | {:.3} |\n",
            case.0, case.1, cpu, metal, ratio
        ));
    }

    markdown.push('\n');
    markdown.push_str("## llama-rs proxy attention (`ms/iter`)\n\n");
    markdown.push_str("| Case (`HxQxKxS` or `HxQxKxQ/KV`) | CPU | Metal | CPU/Metal |\n");
    markdown.push_str("| --- | ---:| ---:| ---:|\n");

    let mut attn_map: BTreeMap<(AttentionCase, BenchBackend), f64> = BTreeMap::new();
    let mut attn_cases = BTreeSet::new();
    let mut attn_stepwise_map: BTreeMap<(AttentionStepwiseCase, BenchBackend), f64> =
        BTreeMap::new();
    let mut attn_stepwise_cases = BTreeSet::new();
    for row in attention_rows {
        if row.stepwise {
            let case = (
                row.hidden_features,
                row.query_head_count,
                row.kv_head_count,
                row.query_length,
                row.key_value_length,
                row.step_count,
            );
            attn_stepwise_cases.insert(case);
            attn_stepwise_map.insert((case, row.backend), row.avg_ms);
        } else {
            let case = (
                row.hidden_features,
                row.query_head_count,
                row.kv_head_count,
                row.query_length,
                row.key_value_length,
            );
            attn_cases.insert(case);
            attn_map.insert((case, row.backend), row.avg_ms);
        }
    }
    for case in attn_cases {
        let cpu = attn_map.get(&(case, BenchBackend::Cpu));
        let metal = attn_map.get(&(case, BenchBackend::Metal));
        let (Some(cpu), Some(metal)) = (cpu, metal) else {
            continue;
        };
        let ratio = if *metal == 0.0 {
            f64::NAN
        } else {
            *cpu / *metal
        };
        let case_label = if case.3 == case.4 {
            format!("{}x{}x{}x{}", case.0, case.1, case.2, case.3)
        } else {
            format!("{}x{}x{}x{}/{}", case.0, case.1, case.2, case.3, case.4)
        };
        markdown.push_str(&format!(
            "| `{}` | {:.3} | {:.3} | {:.3} |\n",
            case_label, cpu, metal, ratio
        ));
    }

    markdown.push('\n');
    markdown.push_str("## llama-rs proxy attention stepwise decode (`ms/token`)\n\n");
    markdown.push_str("| Case (`HxQxKxQ/KV+steps`) | CPU | Metal | CPU/Metal |\n");
    markdown.push_str("| --- | ---:| ---:| ---:|\n");

    for case in attn_stepwise_cases {
        let cpu = attn_stepwise_map.get(&(case, BenchBackend::Cpu));
        let metal = attn_stepwise_map.get(&(case, BenchBackend::Metal));
        let (Some(cpu), Some(metal)) = (cpu, metal) else {
            continue;
        };
        let ratio = if *metal == 0.0 {
            f64::NAN
        } else {
            *cpu / *metal
        };
        let case_label = format!(
            "{}x{}x{}x{}/{}+{}",
            case.0, case.1, case.2, case.3, case.4, case.5
        );
        markdown.push_str(&format!(
            "| `{}` | {:.3} | {:.3} | {:.3} |\n",
            case_label, cpu, metal, ratio
        ));
    }

    markdown
}

fn parse_llama_cpp_jsonl_text(
    path: &Path,
    text: &str,
) -> Result<Vec<LlamaCppBenchRow>, BenchReportError> {
    let mut rows = Vec::new();

    for (line_index, line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let value: Value = serde_json::from_str(line).map_err(|source| BenchReportError::Json {
            path: path.to_path_buf(),
            line: line_number,
            source,
        })?;

        let model_filename = required_str(path, line_number, &value, "model_filename")?.to_string();
        let model_name = model_name_from_filename(&model_filename);
        let n_prompt = required_usize(path, line_number, &value, "n_prompt")?;
        let n_gen = required_usize(path, line_number, &value, "n_gen")?;
        let n_gpu_layers = required_usize(path, line_number, &value, "n_gpu_layers")?;
        let tokens_per_second = required_f64(path, line_number, &value, "avg_ts")?;
        let backend = if n_gpu_layers == 0 {
            BenchBackend::Cpu
        } else {
            BenchBackend::Metal
        };

        rows.push(LlamaCppBenchRow {
            model_name,
            n_prompt,
            n_gen,
            backend,
            tokens_per_second,
        });
    }

    Ok(rows)
}

fn parse_mlp_bench_output_text(
    path: &Path,
    text: &str,
) -> Result<Vec<MlpBenchRow>, BenchReportError> {
    let mut rows = Vec::new();
    for (line_index, raw_line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim();
        if !line.contains(" mlp bench ") {
            continue;
        }
        let backend = parse_backend_tag(path, line_number, line)?;
        let hidden_features = token_usize(path, line_number, line, "hidden")?;
        let ffn_features = token_usize(path, line_number, line, "ffn")?;
        let avg_ms = token_f64(path, line_number, line, "avg")?;
        rows.push(MlpBenchRow {
            hidden_features,
            ffn_features,
            backend,
            avg_ms,
        });
    }
    Ok(rows)
}

fn parse_attention_bench_output_text(
    path: &Path,
    text: &str,
) -> Result<Vec<AttentionBenchRow>, BenchReportError> {
    let mut rows = Vec::new();
    for (line_index, raw_line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim();
        let stepwise_decode = line.contains(" attn decode stepwise bench ");
        let decode = line.contains(" attn decode bench ");
        let prefill = line.contains(" attn bench ");
        if !prefill && !decode && !stepwise_decode {
            continue;
        }
        let backend = parse_backend_tag(path, line_number, line)?;
        let hidden_features = token_usize(path, line_number, line, "hidden")?;
        let (query_length, key_value_length, step_count, stepwise, avg_ms) = if stepwise_decode {
            (
                token_usize(path, line_number, line, "q_seq")?,
                token_usize(path, line_number, line, "kv_start")?,
                token_usize(path, line_number, line, "steps")?,
                true,
                token_f64(path, line_number, line, "avg_token")?,
            )
        } else if decode {
            (
                token_usize(path, line_number, line, "q_seq")?,
                token_usize(path, line_number, line, "kv_seq")?,
                1,
                false,
                token_f64(path, line_number, line, "avg")?,
            )
        } else {
            let sequence_length = token_usize(path, line_number, line, "seq")?;
            (
                sequence_length,
                sequence_length,
                1,
                false,
                token_f64(path, line_number, line, "avg")?,
            )
        };

        let heads =
            token_value(line, "heads").ok_or_else(|| BenchReportError::InvalidBenchLine {
                path: path.to_path_buf(),
                line: line_number,
                details: "missing `heads=` token".to_string(),
            })?;
        let (query_head_count, kv_head_count) =
            heads
                .split_once('/')
                .ok_or_else(|| BenchReportError::InvalidBenchLine {
                    path: path.to_path_buf(),
                    line: line_number,
                    details: format!("invalid `heads` token `{heads}`"),
                })?;
        let query_head_count =
            query_head_count
                .parse::<usize>()
                .map_err(|_| BenchReportError::InvalidBenchLine {
                    path: path.to_path_buf(),
                    line: line_number,
                    details: format!("invalid query head count `{query_head_count}`"),
                })?;
        let kv_head_count =
            kv_head_count
                .parse::<usize>()
                .map_err(|_| BenchReportError::InvalidBenchLine {
                    path: path.to_path_buf(),
                    line: line_number,
                    details: format!("invalid kv head count `{kv_head_count}`"),
                })?;

        rows.push(AttentionBenchRow {
            hidden_features,
            query_head_count,
            kv_head_count,
            query_length,
            key_value_length,
            step_count,
            stepwise,
            backend,
            avg_ms,
        });
    }
    Ok(rows)
}

fn model_name_from_filename(model_filename: &str) -> String {
    Path::new(model_filename)
        .file_name()
        .and_then(|name| name.to_str())
        .map_or_else(|| model_filename.to_string(), ToString::to_string)
}

fn required_str<'a>(
    path: &Path,
    line_number: usize,
    value: &'a Value,
    field: &'static str,
) -> Result<&'a str, BenchReportError> {
    let Some(value) = value.get(field) else {
        return Err(BenchReportError::MissingField {
            path: path.to_path_buf(),
            line: line_number,
            field,
        });
    };
    value
        .as_str()
        .ok_or_else(|| BenchReportError::InvalidFieldType {
            path: path.to_path_buf(),
            line: line_number,
            field,
        })
}

fn required_usize(
    path: &Path,
    line_number: usize,
    value: &Value,
    field: &'static str,
) -> Result<usize, BenchReportError> {
    let Some(value) = value.get(field) else {
        return Err(BenchReportError::MissingField {
            path: path.to_path_buf(),
            line: line_number,
            field,
        });
    };
    let Some(value) = value.as_u64() else {
        return Err(BenchReportError::InvalidFieldType {
            path: path.to_path_buf(),
            line: line_number,
            field,
        });
    };
    usize::try_from(value).map_err(|source| BenchReportError::IntegerOverflow {
        path: path.to_path_buf(),
        line: line_number,
        field,
        source,
    })
}

fn required_f64(
    path: &Path,
    line_number: usize,
    value: &Value,
    field: &'static str,
) -> Result<f64, BenchReportError> {
    let Some(value) = value.get(field) else {
        return Err(BenchReportError::MissingField {
            path: path.to_path_buf(),
            line: line_number,
            field,
        });
    };
    value
        .as_f64()
        .ok_or_else(|| BenchReportError::InvalidFieldType {
            path: path.to_path_buf(),
            line: line_number,
            field,
        })
}

fn parse_backend_tag(
    path: &Path,
    line_number: usize,
    line: &str,
) -> Result<BenchBackend, BenchReportError> {
    if !line.starts_with('[') {
        return Err(BenchReportError::InvalidBenchLine {
            path: path.to_path_buf(),
            line: line_number,
            details: "missing backend tag prefix (`[CPU]`/`[MTL0]`)".to_string(),
        });
    }
    let Some(tag_end) = line.find(']') else {
        return Err(BenchReportError::InvalidBenchLine {
            path: path.to_path_buf(),
            line: line_number,
            details: "unterminated backend tag".to_string(),
        });
    };
    let tag = &line[1..tag_end];
    match tag {
        "CPU" => Ok(BenchBackend::Cpu),
        value if value.starts_with("MTL") => Ok(BenchBackend::Metal),
        _ => Err(BenchReportError::UnsupportedBackendTag {
            path: path.to_path_buf(),
            line: line_number,
            tag: tag.to_string(),
        }),
    }
}

fn token_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    for token in line.split_whitespace() {
        if let Some(value) = token.strip_prefix(&prefix) {
            return Some(value);
        }
    }
    None
}

fn token_usize(
    path: &Path,
    line_number: usize,
    line: &str,
    key: &str,
) -> Result<usize, BenchReportError> {
    let Some(value) = token_value(line, key) else {
        return Err(BenchReportError::InvalidBenchLine {
            path: path.to_path_buf(),
            line: line_number,
            details: format!("missing `{key}=` token"),
        });
    };
    value
        .parse::<usize>()
        .map_err(|_| BenchReportError::InvalidBenchLine {
            path: path.to_path_buf(),
            line: line_number,
            details: format!("invalid `{key}` value `{value}`"),
        })
}

fn token_f64(
    path: &Path,
    line_number: usize,
    line: &str,
    key: &str,
) -> Result<f64, BenchReportError> {
    let Some(value) = token_value(line, key) else {
        return Err(BenchReportError::InvalidBenchLine {
            path: path.to_path_buf(),
            line: line_number,
            details: format!("missing `{key}=` token"),
        });
    };
    value
        .parse::<f64>()
        .map_err(|_| BenchReportError::InvalidBenchLine {
            path: path.to_path_buf(),
            line: line_number,
            details: format!("invalid `{key}` value `{value}`"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llama_cpp_jsonl_rows() {
        let text = r#"{"model_filename":"target/models/qwen.gguf","n_prompt":256,"n_gen":0,"n_gpu_layers":0,"avg_ts":100.0}
{"model_filename":"target/models/qwen.gguf","n_prompt":256,"n_gen":0,"n_gpu_layers":99,"avg_ts":350.0}
"#;
        let rows = parse_llama_cpp_jsonl_text(Path::new("<memory>"), text).expect("parse rows");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].model_name, "qwen.gguf");
        assert_eq!(rows[0].backend, BenchBackend::Cpu);
        assert_eq!(rows[1].backend, BenchBackend::Metal);
    }

    #[test]
    fn parse_mlp_rows() {
        let text = "[CPU] mlp bench hidden=4096 ffn=14336 warmup=1 iters=3 avg=58.974 ms checksum=1.0\n[MTL0] mlp bench hidden=4096 ffn=14336 warmup=1 iters=3 avg=57.152 ms checksum=1.0\n";
        let rows =
            parse_mlp_bench_output_text(Path::new("<memory>"), text).expect("parse mlp rows");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].hidden_features, 4096);
        assert_eq!(rows[1].backend, BenchBackend::Metal);
    }

    #[test]
    fn parse_attention_rows() {
        let text = "[CPU] attn bench hidden=4096 heads=32/8 seq=256 causal=true rope=true warmup=1 iters=3 avg=265.973 ms checksum=1.0\n";
        let rows = parse_attention_bench_output_text(Path::new("<memory>"), text)
            .expect("parse attention rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].query_head_count, 32);
        assert_eq!(rows[0].kv_head_count, 8);
        assert_eq!(rows[0].query_length, 256);
        assert_eq!(rows[0].key_value_length, 256);
        assert!(!rows[0].stepwise);
        assert_eq!(rows[0].step_count, 1);
    }

    #[test]
    fn parse_decode_attention_rows() {
        let text = "[MTL0] attn decode bench hidden=4096 heads=32/8 q_seq=1 kv_seq=128 past=127 rope=true warmup=1 iters=3 avg=12.345 ms checksum=1.0\n";
        let rows = parse_attention_bench_output_text(Path::new("<memory>"), text)
            .expect("parse decode attention rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].query_length, 1);
        assert_eq!(rows[0].key_value_length, 128);
        assert!(!rows[0].stepwise);
        assert_eq!(rows[0].step_count, 1);
    }

    #[test]
    fn parse_stepwise_decode_attention_rows() {
        let text = "[CPU] attn decode stepwise bench hidden=4096 heads=32/8 q_seq=1 kv_start=128 steps=8 past_start=127 cache_reuse=true stepwise=true rope=true warmup=1 iters=3 avg_token=52.277 ms checksum=1.0\n";
        let rows = parse_attention_bench_output_text(Path::new("<memory>"), text)
            .expect("parse stepwise decode attention rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].query_length, 1);
        assert_eq!(rows[0].key_value_length, 128);
        assert_eq!(rows[0].step_count, 8);
        assert!(rows[0].stepwise);
    }

    #[test]
    fn render_summary_contains_expected_sections() {
        let markdown = render_markdown_summary(
            &[
                LlamaCppBenchRow {
                    model_name: "model.gguf".to_string(),
                    n_prompt: 256,
                    n_gen: 0,
                    backend: BenchBackend::Cpu,
                    tokens_per_second: 100.0,
                },
                LlamaCppBenchRow {
                    model_name: "model.gguf".to_string(),
                    n_prompt: 256,
                    n_gen: 0,
                    backend: BenchBackend::Metal,
                    tokens_per_second: 300.0,
                },
            ],
            &[
                MlpBenchRow {
                    hidden_features: 4096,
                    ffn_features: 14336,
                    backend: BenchBackend::Cpu,
                    avg_ms: 60.0,
                },
                MlpBenchRow {
                    hidden_features: 4096,
                    ffn_features: 14336,
                    backend: BenchBackend::Metal,
                    avg_ms: 40.0,
                },
            ],
            &[
                AttentionBenchRow {
                    hidden_features: 4096,
                    query_head_count: 32,
                    kv_head_count: 8,
                    query_length: 256,
                    key_value_length: 256,
                    step_count: 1,
                    stepwise: false,
                    backend: BenchBackend::Cpu,
                    avg_ms: 200.0,
                },
                AttentionBenchRow {
                    hidden_features: 4096,
                    query_head_count: 32,
                    kv_head_count: 8,
                    query_length: 256,
                    key_value_length: 256,
                    step_count: 1,
                    stepwise: false,
                    backend: BenchBackend::Metal,
                    avg_ms: 50.0,
                },
            ],
        );
        assert!(markdown.contains("llama.cpp (`tok/s`)"));
        assert!(markdown.contains("`4096x14336`"));
        assert!(markdown.contains("`4096x32x8x256`"));
        assert!(markdown.contains("attention stepwise decode"));
    }
}
