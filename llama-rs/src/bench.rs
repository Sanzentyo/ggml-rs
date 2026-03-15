use crate::backend::{LlamaBackend, ensure_backends_loaded};
use ggml_rs::{Backend, Context, Shape2D};
use std::error::Error as StdError;
use std::fmt;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulBenchConfig {
    pub rows_a: usize,
    pub cols_a: usize,
    pub rows_b: usize,
    pub cols_b: usize,
    pub warmup_iters: usize,
    pub bench_iters: usize,
}

impl Default for MatmulBenchConfig {
    fn default() -> Self {
        Self {
            rows_a: 256,
            cols_a: 256,
            rows_b: 256,
            cols_b: 256,
            warmup_iters: 3,
            bench_iters: 30,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatmulBenchReport {
    pub backend_name: String,
    pub rows_a: usize,
    pub cols_a: usize,
    pub rows_b: usize,
    pub cols_b: usize,
    pub warmup_iters: usize,
    pub bench_iters: usize,
    pub avg_ms: f64,
    pub checksum: f64,
}

#[derive(Debug)]
pub enum BenchError {
    Ggml {
        context: &'static str,
        source: ggml_rs::Error,
    },
    InvalidShape {
        cols_a: usize,
        cols_b: usize,
    },
    InvalidBenchIters,
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ggml { context, source } => write!(f, "{context}: {source}"),
            Self::InvalidShape { cols_a, cols_b } => write!(
                f,
                "incompatible matmul inner dimensions: cols_a={cols_a}, cols_b={cols_b}"
            ),
            Self::InvalidBenchIters => write!(f, "bench iterations must be greater than zero"),
        }
    }
}

impl StdError for BenchError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Ggml { source, .. } => Some(source),
            Self::InvalidShape { .. } | Self::InvalidBenchIters => None,
        }
    }
}

impl BenchError {
    fn ggml(context: &'static str, source: ggml_rs::Error) -> Self {
        Self::Ggml { context, source }
    }
}

pub fn backend_matmul_bench(
    backend_kind: LlamaBackend,
    config: MatmulBenchConfig,
) -> Result<MatmulBenchReport, BenchError> {
    if config.cols_a != config.cols_b {
        return Err(BenchError::InvalidShape {
            cols_a: config.cols_a,
            cols_b: config.cols_b,
        });
    }
    if config.bench_iters == 0 {
        return Err(BenchError::InvalidBenchIters);
    }

    ensure_backends_loaded();

    let backend = Backend::new(backend_kind.into())
        .map_err(|source| BenchError::ggml("Backend::new", source))?;
    let backend_name = backend
        .name()
        .map_err(|source| BenchError::ggml("Backend::name", source))?
        .to_string();

    let shape_a = Shape2D::new(config.cols_a, config.rows_a);
    let shape_b = Shape2D::new(config.cols_b, config.rows_b);
    let ctx_size =
        Context::recommended_backend_matmul_memory::<f32>(shape_a, shape_b).map_err(|source| {
            BenchError::ggml("Context::recommended_backend_matmul_memory::<f32>", source)
        })?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| BenchError::ggml("Context::new_no_alloc_bytes", source))?;

    let a = ctx
        .new_f32_tensor_2d_shape(shape_a)
        .map_err(|source| BenchError::ggml("Context::new_f32_tensor_2d_shape<A>", source))?;
    let b = ctx
        .new_f32_tensor_2d_shape(shape_b)
        .map_err(|source| BenchError::ggml("Context::new_f32_tensor_2d_shape<B>", source))?;
    let result = ctx
        .mul_mat(&a, &b)
        .map_err(|source| BenchError::ggml("Context::mul_mat", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| BenchError::ggml("Context::new_graph", source))?;
    graph.build_forward_expand(&result);
    let _buffer = ctx
        .allocate_tensors(&backend)
        .map_err(|source| BenchError::ggml("Context::allocate_tensors", source))?;

    let data_a: Vec<f32> = (0..(config.rows_a * config.cols_a))
        .map(|index| (index % 31) as f32 * 0.03125)
        .collect();
    let data_b: Vec<f32> = (0..(config.rows_b * config.cols_b))
        .map(|index| (index % 17) as f32 * 0.0625)
        .collect();

    a.write_data_backend(&data_a)
        .map_err(|source| BenchError::ggml("Tensor::write_data_backend<A>", source))?;
    b.write_data_backend(&data_b)
        .map_err(|source| BenchError::ggml("Tensor::write_data_backend<B>", source))?;

    for _ in 0..config.warmup_iters {
        backend
            .compute(&mut graph)
            .map_err(|source| BenchError::ggml("Backend::compute(warmup)", source))?;
    }
    backend
        .synchronize()
        .map_err(|source| BenchError::ggml("Backend::synchronize(warmup)", source))?;

    let start = Instant::now();
    for _ in 0..config.bench_iters {
        backend
            .compute(&mut graph)
            .map_err(|source| BenchError::ggml("Backend::compute(bench)", source))?;
    }
    backend
        .synchronize()
        .map_err(|source| BenchError::ggml("Backend::synchronize(bench)", source))?;
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / config.bench_iters as f64;

    let values = graph
        .last_node()
        .map_err(|source| BenchError::ggml("Graph::last_node", source))?
        .read_data_backend::<f32>()
        .map_err(|source| BenchError::ggml("Tensor::read_data_backend", source))?;
    let checksum = values.iter().take(16).map(|value| f64::from(*value)).sum();

    Ok(MatmulBenchReport {
        backend_name,
        rows_a: config.rows_a,
        cols_a: config.cols_a,
        rows_b: config.rows_b,
        cols_b: config.cols_b,
        warmup_iters: config.warmup_iters,
        bench_iters: config.bench_iters,
        avg_ms,
        checksum,
    })
}
