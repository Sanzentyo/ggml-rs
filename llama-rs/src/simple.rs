use ggml_rs::{Context, Shape2D, StaticShape2D};
use std::error::Error as StdError;
use std::fmt;

const ROWS_A: usize = 4;
const COLS_A: usize = 2;
const ROWS_B: usize = 3;
const COLS_B: usize = 2;
const SHAPE_A: Shape2D = Shape2D::new(COLS_A, ROWS_A);
const SHAPE_B: Shape2D = Shape2D::new(COLS_B, ROWS_B);
type AShape = StaticShape2D<COLS_A, ROWS_A>;
type BShape = StaticShape2D<COLS_B, ROWS_B>;

const MATRIX_A: [f32; ROWS_A * COLS_A] = [2.0, 8.0, 5.0, 1.0, 4.0, 2.0, 8.0, 6.0];
const MATRIX_B: [f32; ROWS_B * COLS_B] = [10.0, 5.0, 9.0, 9.0, 5.0, 4.0];

const EXPECTED: [f32; ROWS_A * ROWS_B] = [
    60.0, 55.0, 50.0, 110.0, //
    90.0, 54.0, 54.0, 126.0, //
    42.0, 29.0, 28.0, 64.0,
];

const TOLERANCE: f32 = 1e-4;

#[derive(Debug)]
pub enum SimpleError {
    Ggml {
        context: &'static str,
        source: ggml_rs::Error,
    },
    OutputLengthMismatch {
        expected: usize,
        actual: usize,
    },
    OutputMismatch {
        index: usize,
        expected: f32,
        actual: f32,
        tolerance: f32,
    },
}

impl fmt::Display for SimpleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ggml { context, source } => write!(f, "{context}: {source}"),
            Self::OutputLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "output length mismatch: expected {expected} elements but got {actual}"
                )
            }
            Self::OutputMismatch {
                index,
                expected,
                actual,
                tolerance,
            } => write!(
                f,
                "output mismatch at index {index}: expected {expected}, actual {actual}, tolerance {tolerance}"
            ),
        }
    }
}

impl StdError for SimpleError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Ggml { source, .. } => Some(source),
            Self::OutputLengthMismatch { .. } | Self::OutputMismatch { .. } => None,
        }
    }
}

impl SimpleError {
    fn ggml(context: &'static str, source: ggml_rs::Error) -> Self {
        Self::Ggml { context, source }
    }
}

#[derive(Debug, Clone)]
pub struct SimpleReport {
    pub cols: usize,
    pub rows: usize,
    pub values: Vec<f32>,
}

pub fn simple_ctx() -> Result<SimpleReport, SimpleError> {
    let ctx_size = Context::recommended_matmul_memory::<f32>(SHAPE_A, SHAPE_B)
        .map_err(|source| SimpleError::ggml("Context::recommended_matmul_memory::<f32>", source))?;
    let ctx = Context::new_bytes(ctx_size)
        .map_err(|source| SimpleError::ggml("Context::new_bytes", source))?;

    let a = ctx
        .new_tensor_2d_typed::<f32, AShape>()
        .map_err(|source| SimpleError::ggml("Context::new_tensor_2d_typed<f32, A>", source))?;
    let b = ctx
        .new_tensor_2d_typed::<f32, BShape>()
        .map_err(|source| SimpleError::ggml("Context::new_tensor_2d_typed<f32, B>", source))?;
    a.write_data(&MATRIX_A)
        .map_err(|source| SimpleError::ggml("TypedTensor2D::write_data<A>", source))?;
    b.write_data(&MATRIX_B)
        .map_err(|source| SimpleError::ggml("TypedTensor2D::write_data<B>", source))?;

    let result = ctx
        .mul_mat(a.inner(), b.inner())
        .map_err(|source| SimpleError::ggml("Context::mul_mat", source))?;
    let mut graph = ctx
        .new_graph()
        .map_err(|source| SimpleError::ggml("Context::new_graph", source))?;
    graph.build_forward_expand(&result);
    ctx.compute(&mut graph, 1)
        .map_err(|source| SimpleError::ggml("Context::compute", source))?;

    let output = graph
        .last_node()
        .map_err(|source| SimpleError::ggml("Graph::last_node", source))?;
    let values = output
        .read_data::<f32>()
        .map_err(|source| SimpleError::ggml("Tensor::read_data", source))?;
    assert_expected(&values)?;
    let shape = output
        .shape()
        .map_err(|source| SimpleError::ggml("Tensor::shape", source))?;

    Ok(SimpleReport {
        cols: shape.cols.get(),
        rows: shape.rows.get(),
        values,
    })
}

fn assert_expected(values: &[f32]) -> Result<(), SimpleError> {
    if values.len() != EXPECTED.len() {
        return Err(SimpleError::OutputLengthMismatch {
            expected: EXPECTED.len(),
            actual: values.len(),
        });
    }

    for (index, (actual, expected)) in values
        .iter()
        .copied()
        .zip(EXPECTED.iter().copied())
        .enumerate()
    {
        let delta = (actual - expected).abs();
        if delta > TOLERANCE {
            return Err(SimpleError::OutputMismatch {
                index,
                expected,
                actual,
                tolerance: TOLERANCE,
            });
        }
    }

    Ok(())
}
