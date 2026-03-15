//! Batched execution helpers built on top of `ggml-rs` safe APIs.
//!
//! This module intentionally focuses on a reusable matmul workload as a
//! foundation for future token-level batch scheduling.

use crate::backend::{LlamaBackend, ensure_backends_loaded};
use ggml_rs::{Backend, Context, Shape2D};
use std::error::Error as StdError;
use std::fmt;
use std::num::NonZeroUsize;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Strongly typed non-zero batch size.
pub struct BatchSize(NonZeroUsize);

impl BatchSize {
    pub fn new(value: usize) -> Result<Self, BatchedError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(BatchedError::InvalidBatchSize)
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Strongly typed non-zero repeat count.
pub struct RepeatCount(NonZeroUsize);

impl RepeatCount {
    pub fn new(value: usize) -> Result<Self, BatchedError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(BatchedError::InvalidRepeats)
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Strongly typed non-zero readback interval.
pub struct ReadbackEvery(NonZeroUsize);

impl ReadbackEvery {
    pub fn new(value: usize) -> Result<Self, BatchedError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(BatchedError::InvalidReadbackEvery)
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Configuration for the batched matmul runner.
pub struct BatchedConfig {
    /// Left input matrix row count.
    pub rows_a: usize,
    /// Left input matrix column count (inner dimension).
    pub cols_a: usize,
    /// Right input matrix row count.
    pub rows_b: usize,
    /// Right input matrix column count (inner dimension, must match `cols_a`).
    pub cols_b: usize,
    /// Number of items in one batch cycle.
    pub batch_size: BatchSize,
    /// Number of repeated batch cycles.
    pub repeats: RepeatCount,
    /// Host readback cadence for checksum sampling (1 = every item).
    pub readback_every: ReadbackEvery,
}

impl Default for BatchedConfig {
    fn default() -> Self {
        Self::new(256, 256, 256, 256, 8, 5, 1).expect("default config must be valid")
    }
}

impl BatchedConfig {
    /// Creates a config explicitly from matrix dimensions and execution knobs.
    pub fn new(
        rows_a: usize,
        cols_a: usize,
        rows_b: usize,
        cols_b: usize,
        batch_size: usize,
        repeats: usize,
        readback_every: usize,
    ) -> Result<Self, BatchedError> {
        Ok(Self {
            rows_a,
            cols_a,
            rows_b,
            cols_b,
            batch_size: BatchSize::new(batch_size)?,
            repeats: RepeatCount::new(repeats)?,
            readback_every: ReadbackEvery::new(readback_every)?,
        })
    }

    /// Validates shape and scheduling parameters.
    pub fn validated(self) -> Result<Self, BatchedError> {
        if self.cols_a != self.cols_b {
            return Err(BatchedError::InvalidShape {
                cols_a: self.cols_a,
                cols_b: self.cols_b,
            });
        }
        Ok(self)
    }
}

#[derive(Debug, Clone)]
/// Reusable batched workload payload.
pub struct BatchedWorkload {
    pub config: BatchedConfig,
    matrix_b: Vec<f32>,
    // Stored contiguously for better cache locality than `Vec<Vec<f32>>`.
    batch_inputs_flat: Vec<f32>,
    input_stride: usize,
}

impl BatchedWorkload {
    /// Builds a deterministic synthetic workload useful for benchmarks and smoke checks.
    pub fn deterministic(config: BatchedConfig) -> Result<Self, BatchedError> {
        let config = config.validated()?;
        let matrix_b: Vec<f32> = (0..(config.rows_b * config.cols_b))
            .map(|index| (index % 17) as f32 * 0.0625)
            .collect();
        let input_stride = config.rows_a * config.cols_a;
        let mut batch_inputs_flat = Vec::with_capacity(config.batch_size.get() * input_stride);
        for batch in 0..config.batch_size.get() {
            batch_inputs_flat.extend(
                (0..input_stride).map(|index| ((index + batch * 13) % 31) as f32 * 0.03125),
            );
        }
        Ok(Self {
            config,
            matrix_b,
            batch_inputs_flat,
            input_stride,
        })
    }

    /// Builds a workload from caller-provided tensors.
    pub fn from_raw(
        config: BatchedConfig,
        matrix_b: Vec<f32>,
        batch_inputs: Vec<Vec<f32>>,
    ) -> Result<Self, BatchedError> {
        let config = config.validated()?;

        let expected_b = config.rows_b * config.cols_b;
        if matrix_b.len() != expected_b {
            return Err(BatchedError::InvalidMatrixBLength {
                expected: expected_b,
                actual: matrix_b.len(),
            });
        }
        if batch_inputs.len() != config.batch_size.get() {
            return Err(BatchedError::InvalidBatchInputCount {
                expected: config.batch_size.get(),
                actual: batch_inputs.len(),
            });
        }
        let expected_a = config.rows_a * config.cols_a;
        for (index, input) in batch_inputs.iter().enumerate() {
            if input.len() != expected_a {
                return Err(BatchedError::InvalidBatchInputLength {
                    index,
                    expected: expected_a,
                    actual: input.len(),
                });
            }
        }

        Ok(Self {
            config,
            matrix_b,
            batch_inputs_flat: batch_inputs.into_iter().flatten().collect(),
            input_stride: expected_a,
        })
    }

    pub fn matrix_b(&self) -> &[f32] {
        &self.matrix_b
    }

    /// Returns one batch input view by index.
    pub fn batch_input(&self, index: usize) -> &[f32] {
        let start = index * self.input_stride;
        let end = start + self.input_stride;
        &self.batch_inputs_flat[start..end]
    }

    /// Iterates all batch input views in order.
    pub fn batch_inputs(&self) -> impl Iterator<Item = &[f32]> {
        (0..self.config.batch_size.get()).map(|index| self.batch_input(index))
    }
}

#[derive(Debug, Clone)]
/// Runtime metrics produced by [`batched_matmul`].
pub struct BatchedReport {
    pub backend_name: String,
    pub rows_a: usize,
    pub cols_a: usize,
    pub rows_b: usize,
    pub cols_b: usize,
    pub batch_size: usize,
    pub repeats: usize,
    pub readback_every: usize,
    pub readback_samples: usize,
    pub avg_item_ms: f64,
    pub checksum: f64,
}

#[derive(Debug)]
/// Errors surfaced by batched workload execution.
pub enum BatchedError {
    Ggml {
        context: &'static str,
        source: ggml_rs::Error,
    },
    InvalidShape {
        cols_a: usize,
        cols_b: usize,
    },
    InvalidBatchSize,
    InvalidRepeats,
    InvalidReadbackEvery,
    InvalidMatrixBLength {
        expected: usize,
        actual: usize,
    },
    InvalidBatchInputCount {
        expected: usize,
        actual: usize,
    },
    InvalidBatchInputLength {
        index: usize,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for BatchedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ggml { context, source } => write!(f, "{context}: {source}"),
            Self::InvalidShape { cols_a, cols_b } => write!(
                f,
                "incompatible matmul inner dimensions: cols_a={cols_a}, cols_b={cols_b}"
            ),
            Self::InvalidBatchSize => write!(f, "batch_size must be greater than zero"),
            Self::InvalidRepeats => write!(f, "repeats must be greater than zero"),
            Self::InvalidReadbackEvery => write!(f, "readback_every must be greater than zero"),
            Self::InvalidMatrixBLength { expected, actual } => write!(
                f,
                "matrix B length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidBatchInputCount { expected, actual } => write!(
                f,
                "batch input count mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidBatchInputLength {
                index,
                expected,
                actual,
            } => write!(
                f,
                "batch input length mismatch at index {index}: expected {expected}, got {actual}"
            ),
        }
    }
}

impl StdError for BatchedError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Ggml { source, .. } => Some(source),
            Self::InvalidShape { .. }
            | Self::InvalidBatchSize
            | Self::InvalidRepeats
            | Self::InvalidReadbackEvery
            | Self::InvalidMatrixBLength { .. }
            | Self::InvalidBatchInputCount { .. }
            | Self::InvalidBatchInputLength { .. } => None,
        }
    }
}

impl BatchedError {
    fn ggml(context: &'static str, source: ggml_rs::Error) -> Self {
        Self::Ggml { context, source }
    }
}

/// Executes a deterministic batched matmul workload on the requested backend.
///
/// The same graph and backend allocation are reused across all batch items.
/// Input `A` is replaced per item, while `B` stays fixed for the whole run.
pub fn batched_matmul(
    backend_kind: LlamaBackend,
    config: BatchedConfig,
) -> Result<BatchedReport, BatchedError> {
    let workload = BatchedWorkload::deterministic(config)?;
    batched_matmul_with_workload(backend_kind, &workload)
}

/// Executes batched matmul using caller-provided workload tensors.
pub fn batched_matmul_with_workload(
    backend_kind: LlamaBackend,
    workload: &BatchedWorkload,
) -> Result<BatchedReport, BatchedError> {
    let config = workload.config.validated()?;

    ensure_backends_loaded();

    let backend = Backend::new(backend_kind.into())
        .map_err(|source| BatchedError::ggml("Backend::new", source))?;
    let backend_name = backend
        .name()
        .map_err(|source| BatchedError::ggml("Backend::name", source))?
        .to_string();

    let shape_a = Shape2D::new(config.cols_a, config.rows_a);
    let shape_b = Shape2D::new(config.cols_b, config.rows_b);
    let ctx_size =
        Context::recommended_backend_matmul_memory::<f32>(shape_a, shape_b).map_err(|source| {
            BatchedError::ggml("Context::recommended_backend_matmul_memory::<f32>", source)
        })?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| BatchedError::ggml("Context::new_no_alloc_bytes", source))?;

    let a = ctx
        .new_tensor_2d::<f32>(shape_a)
        .map_err(|source| BatchedError::ggml("Context::new_f32_tensor_2d_shape<A>", source))?;
    let b = ctx
        .new_tensor_2d::<f32>(shape_b)
        .map_err(|source| BatchedError::ggml("Context::new_f32_tensor_2d_shape<B>", source))?;
    let result = ctx
        .mul_mat(&a, &b)
        .map_err(|source| BatchedError::ggml("Context::mul_mat", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| BatchedError::ggml("Context::new_graph", source))?;
    graph.build_forward_expand(&result);
    let _buffer = ctx
        .allocate_tensors(&backend)
        .map_err(|source| BatchedError::ggml("Context::allocate_tensors", source))?;

    b.write_data_backend(workload.matrix_b())
        .map_err(|source| BatchedError::ggml("Tensor::write_data_backend<B>", source))?;

    // Warm up with one full batch pass so first-run backend setup does not
    // dominate steady-state measurements.
    for input in workload.batch_inputs() {
        a.write_data_backend(input).map_err(|source| {
            BatchedError::ggml("Tensor::write_data_backend<A>(warmup)", source)
        })?;
        backend
            .compute(&mut graph)
            .map_err(|source| BatchedError::ggml("Backend::compute(warmup)", source))?;
    }

    let total_items = config.batch_size.get() * config.repeats.get();
    let mut checksum = 0.0f64;
    let mut readback_samples = 0usize;
    let mut item_index = 0usize;
    let start = Instant::now();
    for _ in 0..config.repeats.get() {
        for input in workload.batch_inputs() {
            a.write_data_backend(input).map_err(|source| {
                BatchedError::ggml("Tensor::write_data_backend<A>(batch)", source)
            })?;
            backend
                .compute(&mut graph)
                .map_err(|source| BatchedError::ggml("Backend::compute(batch)", source))?;

            // Read back only at the requested cadence to control host/device
            // sync overhead while still keeping a deterministic checksum.
            if item_index.is_multiple_of(config.readback_every.get()) {
                let values = graph
                    .last_node()
                    .map_err(|source| BatchedError::ggml("Graph::last_node", source))?
                    .read_data_backend::<f32>()
                    .map_err(|source| BatchedError::ggml("Tensor::read_data_backend", source))?;
                checksum += values
                    .iter()
                    .take(4)
                    .map(|value| f64::from(*value))
                    .sum::<f64>();
                readback_samples += 1;
            }
            item_index += 1;
        }
    }
    let elapsed = start.elapsed();
    let avg_item_ms = elapsed.as_secs_f64() * 1000.0 / total_items as f64;

    Ok(BatchedReport {
        backend_name,
        rows_a: config.rows_a,
        cols_a: config.cols_a,
        rows_b: config.rows_b,
        cols_b: config.cols_b,
        batch_size: config.batch_size.get(),
        repeats: config.repeats.get(),
        readback_every: config.readback_every.get(),
        readback_samples,
        avg_item_ms,
        checksum,
    })
}
