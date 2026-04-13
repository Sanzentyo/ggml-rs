use crate::LlamaBackend;
use ggml_rs::{Backend, Context, Shape2D};
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SyntheticError {
    #[error("{context}: {source}")]
    Ggml {
        context: &'static str,
        #[source]
        source: ggml_rs::Error,
    },
    #[error("n_embd must be greater than zero")]
    InvalidEmbedding,
    #[error("n_vocab must be greater than zero")]
    InvalidVocab,
    #[error("n_batch must be greater than zero")]
    InvalidBatch,
    #[error("n_predict must be greater than zero")]
    InvalidPredict,
    #[error("n_threads must be greater than zero")]
    InvalidThreads,
    #[error("n_parallel must be greater than zero")]
    InvalidParallel,
    #[error("backend list must not be empty")]
    EmptyBackendList,
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl SyntheticError {
    fn ggml(context: &'static str, source: ggml_rs::Error) -> Self {
        Self::Ggml { context, source }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SyntheticConfig {
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_batch: usize,
    pub n_predict: usize,
    pub n_threads: usize,
    pub seed: u64,
}

impl SyntheticConfig {
    pub fn validated(self) -> Result<Self, SyntheticError> {
        if self.n_embd == 0 {
            return Err(SyntheticError::InvalidEmbedding);
        }
        if self.n_vocab == 0 {
            return Err(SyntheticError::InvalidVocab);
        }
        if self.n_batch == 0 {
            return Err(SyntheticError::InvalidBatch);
        }
        if self.n_predict == 0 {
            return Err(SyntheticError::InvalidPredict);
        }
        if NonZeroUsize::new(self.n_threads).is_none() {
            return Err(SyntheticError::InvalidThreads);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone)]
pub struct SyntheticReport {
    pub mode: &'static str,
    pub backend_name: String,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_batch: usize,
    pub n_predict: usize,
    pub compute_buffer_bytes: usize,
    pub total_ms: f64,
    pub avg_step_ms: f64,
    pub avg_item_ms: f64,
    pub checksum: f64,
}

impl SyntheticReport {
    pub fn to_kv_line(&self) -> String {
        format!(
            "mode={} backend={} n_embd={} n_vocab={} n_batch={} n_predict={} compute_buffer_bytes={} total_ms={:.6} avg_step_ms={:.6} avg_item_ms={:.6} checksum={:.9}",
            self.mode,
            self.backend_name,
            self.n_embd,
            self.n_vocab,
            self.n_batch,
            self.n_predict,
            self.compute_buffer_bytes,
            self.total_ms,
            self.avg_step_ms,
            self.avg_item_ms,
            self.checksum
        )
    }
}

#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    pub n_embd: usize,
    pub n_vocab: usize,
    pub seed: u64,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct QuantizeReport {
    pub n_embd: usize,
    pub n_vocab: usize,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub quantize_ms: f64,
    pub rmse: f64,
    pub checksum: f64,
}

impl QuantizeReport {
    pub fn to_kv_line(&self) -> String {
        format!(
            "mode=quantize n_embd={} n_vocab={} input={} output={} input_bytes={} output_bytes={} quantize_ms={:.6} rmse={:.9} checksum={:.9}",
            self.n_embd,
            self.n_vocab,
            self.input_path.display(),
            self.output_path.display(),
            self.input_bytes,
            self.output_bytes,
            self.quantize_ms,
            self.rmse,
            self.checksum
        )
    }
}

pub fn run_ctx(config: SyntheticConfig) -> Result<SyntheticReport, SyntheticError> {
    let config = config.validated()?;
    let active_batch = config.n_batch;
    let shape_rhs = Shape2D::new(config.n_embd, config.n_vocab);
    let shape_lhs = Shape2D::new(config.n_embd, active_batch);
    let ctx_size =
        Context::recommended_matmul_memory::<f32>(shape_rhs, shape_lhs).map_err(|source| {
            SyntheticError::ggml("Context::recommended_matmul_memory::<f32>", source)
        })?;
    let rhs = make_rhs_weights(config.n_embd, config.n_vocab, config.seed);

    let start = Instant::now();
    let mut checksum = 0.0_f64;
    let mut lhs = vec![0.0_f32; config.n_embd * active_batch];

    for step in 0..config.n_predict {
        fill_lhs_batch(&mut lhs, config.seed, step);
        let ctx = Context::new_bytes(ctx_size)
            .map_err(|source| SyntheticError::ggml("Context::new_bytes", source))?;
        let lhs_tensor = ctx
            .new_tensor_2d::<f32>(shape_lhs)
            .map_err(|source| SyntheticError::ggml("Context::new_tensor_2d<f32, A>", source))?;
        let rhs_tensor = ctx
            .new_tensor_2d::<f32>(shape_rhs)
            .map_err(|source| SyntheticError::ggml("Context::new_tensor_2d<f32, B>", source))?;
        lhs_tensor
            .write_data(&lhs)
            .map_err(|source| SyntheticError::ggml("Tensor::write_data<A>", source))?;
        rhs_tensor
            .write_data(&rhs)
            .map_err(|source| SyntheticError::ggml("Tensor::write_data<B>", source))?;
        let logits = ctx
            .mul_mat(&rhs_tensor, &lhs_tensor)
            .map_err(|source| SyntheticError::ggml("Context::mul_mat", source))?;
        let mut graph = ctx
            .new_graph()
            .map_err(|source| SyntheticError::ggml("Context::new_graph", source))?;
        graph.build_forward_expand(&logits);
        ctx.compute(&mut graph, config.n_threads)
            .map_err(|source| SyntheticError::ggml("Context::compute", source))?;

        let values = graph
            .last_node_typed::<f32>()
            .map_err(|source| SyntheticError::ggml("Graph::last_node", source))?
            .read_data()
            .map_err(|source| SyntheticError::ggml("Tensor::read_data", source))?;
        checksum += checksum_from_logits(&values, config.n_vocab, active_batch);
    }

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(SyntheticReport {
        mode: "ctx",
        backend_name: "context".to_string(),
        n_embd: config.n_embd,
        n_vocab: config.n_vocab,
        n_batch: active_batch,
        n_predict: config.n_predict,
        compute_buffer_bytes: ctx_size.get(),
        total_ms,
        avg_step_ms: total_ms / config.n_predict as f64,
        avg_item_ms: total_ms / (config.n_predict * active_batch) as f64,
        checksum,
    })
}

pub fn run_alloc(config: SyntheticConfig) -> Result<SyntheticReport, SyntheticError> {
    run_backend_for_steps(
        "alloc",
        config,
        LlamaBackend::Cpu,
        config.n_batch,
        &(0..config.n_predict).collect::<Vec<_>>(),
    )
}

pub fn run_backend(
    config: SyntheticConfig,
    backend_kind: LlamaBackend,
) -> Result<SyntheticReport, SyntheticError> {
    run_backend_for_steps(
        "backend",
        config,
        backend_kind,
        config.n_batch,
        &(0..config.n_predict).collect::<Vec<_>>(),
    )
}

pub fn run_sched(
    config: SyntheticConfig,
    backends: &[LlamaBackend],
) -> Result<SyntheticReport, SyntheticError> {
    let config = config.validated()?;
    if backends.is_empty() {
        return Err(SyntheticError::EmptyBackendList);
    }

    let mut backend_names = Vec::new();
    let mut total_ms = 0.0_f64;
    let mut total_checksum = 0.0_f64;
    let mut total_buffer_bytes = 0usize;

    for (backend_index, backend_kind) in backends.iter().copied().enumerate() {
        let steps: Vec<usize> = (0..config.n_predict)
            .filter(|step| step % backends.len() == backend_index)
            .collect();
        if steps.is_empty() {
            continue;
        }

        let report = run_backend_for_steps("sched", config, backend_kind, config.n_batch, &steps)?;
        backend_names.push(report.backend_name.clone());
        total_ms += report.total_ms;
        total_checksum += report.checksum;
        total_buffer_bytes = total_buffer_bytes.saturating_add(report.compute_buffer_bytes);
    }

    Ok(SyntheticReport {
        mode: "sched",
        backend_name: backend_names.join("+"),
        n_embd: config.n_embd,
        n_vocab: config.n_vocab,
        n_batch: config.n_batch,
        n_predict: config.n_predict,
        compute_buffer_bytes: total_buffer_bytes,
        total_ms,
        avg_step_ms: total_ms / config.n_predict as f64,
        avg_item_ms: total_ms / (config.n_predict * config.n_batch) as f64,
        checksum: total_checksum,
    })
}

pub fn run_batched(
    config: SyntheticConfig,
    backend_kind: LlamaBackend,
    n_parallel: usize,
) -> Result<SyntheticReport, SyntheticError> {
    let config = config.validated()?;
    if NonZeroUsize::new(n_parallel).is_none() {
        return Err(SyntheticError::InvalidParallel);
    }

    let report = run_backend_for_steps(
        "batched",
        config,
        backend_kind,
        n_parallel,
        &(0..config.n_predict).collect::<Vec<_>>(),
    )?;

    Ok(SyntheticReport {
        mode: "batched",
        backend_name: report.backend_name,
        n_embd: report.n_embd,
        n_vocab: report.n_vocab,
        n_batch: n_parallel,
        n_predict: report.n_predict,
        compute_buffer_bytes: report.compute_buffer_bytes,
        total_ms: report.total_ms,
        avg_step_ms: report.total_ms / config.n_predict as f64,
        avg_item_ms: report.total_ms / (config.n_predict * n_parallel) as f64,
        checksum: report.checksum,
    })
}

fn run_backend_for_steps(
    mode: &'static str,
    config: SyntheticConfig,
    backend_kind: LlamaBackend,
    active_batch: usize,
    step_indices: &[usize],
) -> Result<SyntheticReport, SyntheticError> {
    let config = config.validated()?;
    if NonZeroUsize::new(active_batch).is_none() {
        return Err(SyntheticError::InvalidBatch);
    }
    if step_indices.is_empty() {
        return Err(SyntheticError::Message(
            "step list must not be empty".to_string(),
        ));
    }

    let shape_rhs = Shape2D::new(config.n_embd, config.n_vocab);
    let shape_lhs = Shape2D::new(config.n_embd, active_batch);
    let ctx_size = Context::recommended_backend_matmul_memory::<f32>(shape_rhs, shape_lhs)
        .map_err(|source| {
            SyntheticError::ggml("Context::recommended_backend_matmul_memory::<f32>", source)
        })?;
    let rhs = make_rhs_weights(config.n_embd, config.n_vocab, config.seed);

    let backend = Backend::new(backend_kind.into())
        .map_err(|source| SyntheticError::ggml("Backend::new", source))?;
    let backend_name = backend
        .name()
        .map_err(|source| SyntheticError::ggml("Backend::name", source))?
        .to_string();

    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| SyntheticError::ggml("Context::new_no_alloc_bytes", source))?;
    let lhs_tensor = ctx
        .new_tensor_2d::<f32>(shape_lhs)
        .map_err(|source| SyntheticError::ggml("Context::new_tensor_2d<f32, A>", source))?;
    let rhs_tensor = ctx
        .new_tensor_2d::<f32>(shape_rhs)
        .map_err(|source| SyntheticError::ggml("Context::new_tensor_2d<f32, B>", source))?;
    let logits = ctx
        .mul_mat(&rhs_tensor, &lhs_tensor)
        .map_err(|source| SyntheticError::ggml("Context::mul_mat", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| SyntheticError::ggml("Context::new_graph", source))?;
    graph.build_forward_expand(&logits);

    let buffer = ctx
        .allocate_tensors(&backend)
        .map_err(|source| SyntheticError::ggml("Context::allocate_tensors", source))?;
    let compute_buffer_bytes = buffer.size_bytes();

    rhs_tensor
        .write_data_backend(&rhs)
        .map_err(|source| SyntheticError::ggml("Tensor::write_data_backend<B>", source))?;

    let start = Instant::now();
    let mut checksum = 0.0_f64;
    let mut lhs = vec![0.0_f32; config.n_embd * active_batch];
    let sample_len = (config.n_vocab * active_batch).min(32);

    for step in step_indices.iter().copied() {
        fill_lhs_batch(&mut lhs, config.seed, step);
        lhs_tensor
            .write_data_backend(&lhs)
            .map_err(|source| SyntheticError::ggml("Tensor::write_data_backend<A>", source))?;
        backend
            .compute(&mut graph)
            .map_err(|source| SyntheticError::ggml("Backend::compute", source))?;

        let values = graph
            .last_node_typed::<f32>()
            .map_err(|source| SyntheticError::ggml("Graph::last_node", source))?
            .read_data_backend_at(0, sample_len)
            .map_err(|source| SyntheticError::ggml("Tensor::read_data_backend_at", source))?;
        checksum += checksum_from_logits(&values, config.n_vocab, active_batch);
    }

    backend
        .synchronize()
        .map_err(|source| SyntheticError::ggml("Backend::synchronize", source))?;

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let n_steps = step_indices.len();

    Ok(SyntheticReport {
        mode,
        backend_name,
        n_embd: config.n_embd,
        n_vocab: config.n_vocab,
        n_batch: active_batch,
        n_predict: n_steps,
        compute_buffer_bytes,
        total_ms,
        avg_step_ms: total_ms / n_steps as f64,
        avg_item_ms: total_ms / (n_steps * active_batch) as f64,
        checksum,
    })
}

pub fn run_quantize(config: QuantizeConfig) -> Result<QuantizeReport, SyntheticError> {
    if config.n_embd == 0 {
        return Err(SyntheticError::InvalidEmbedding);
    }
    if config.n_vocab == 0 {
        return Err(SyntheticError::InvalidVocab);
    }

    ensure_parent_dir(&config.input_path)?;
    ensure_parent_dir(&config.output_path)?;

    let values = make_rhs_weights(config.n_embd, config.n_vocab, config.seed);
    write_f32_file(&config.input_path, &values)?;

    let start = Instant::now();
    let max_abs = values
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };

    let quantized: Vec<i8> = values
        .iter()
        .map(|value| {
            let scaled = (*value / scale).round().clamp(-127.0, 127.0);
            scaled as i8
        })
        .collect();

    let mut squared_error_sum = 0.0_f64;
    for (original, quantized_value) in values.iter().zip(quantized.iter().copied()) {
        let restored = f32::from(quantized_value) * scale;
        let delta = f64::from(*original - restored);
        squared_error_sum += delta * delta;
    }
    let rmse = (squared_error_sum / values.len() as f64).sqrt();

    let checksum = quantized
        .iter()
        .take(64)
        .map(|value| f64::from(*value))
        .sum::<f64>()
        + f64::from(scale);

    write_q8_file(&config.output_path, scale, &quantized)?;

    let quantize_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(QuantizeReport {
        n_embd: config.n_embd,
        n_vocab: config.n_vocab,
        input_path: config.input_path,
        output_path: config.output_path,
        input_bytes: values.len() * std::mem::size_of::<f32>(),
        output_bytes: 8 + 4 + quantized.len(),
        quantize_ms,
        rmse,
        checksum,
    })
}

pub fn parse_kv_f64(line: &str, key: &str) -> Option<f64> {
    line.split_whitespace().find_map(|token| {
        let (token_key, token_value) = token.split_once('=')?;
        (token_key == key)
            .then_some(token_value)?
            .parse::<f64>()
            .ok()
    })
}

fn make_rhs_weights(n_embd: usize, n_vocab: usize, seed: u64) -> Vec<f32> {
    (0..(n_embd * n_vocab))
        .map(|index| synth_value(seed, 0, index, 17))
        .collect()
}

fn fill_lhs_batch(values: &mut [f32], seed: u64, step: usize) {
    for (index, value) in values.iter_mut().enumerate() {
        *value = synth_value(seed, step, index, 53);
    }
}

fn synth_value(seed: u64, step: usize, index: usize, salt: u64) -> f32 {
    let mixed = seed
        .wrapping_add(salt)
        .wrapping_add((step as u64).wrapping_mul(131))
        .wrapping_add((index as u64).wrapping_mul(17));
    let raw = (mixed % 251) as f32;
    (raw - 125.0) * 0.01
}

fn checksum_from_logits(values: &[f32], n_vocab: usize, active_batch: usize) -> f64 {
    let sample_len = (n_vocab * active_batch).min(32);
    values
        .iter()
        .take(sample_len)
        .map(|value| f64::from(*value))
        .sum()
}

fn ensure_parent_dir(path: &Path) -> Result<(), SyntheticError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn write_f32_file(path: &Path, values: &[f32]) -> Result<(), SyntheticError> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn write_q8_file(path: &Path, scale: f32, values: &[i8]) -> Result<(), SyntheticError> {
    let mut bytes = Vec::with_capacity(8 + 4 + values.len());
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&scale.to_le_bytes());
    bytes.extend(values.iter().map(|value| *value as u8));
    fs::write(path, bytes)?;
    Ok(())
}
