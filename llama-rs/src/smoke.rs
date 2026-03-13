use crate::backend::LlamaBackend;
use std::error::Error as StdError;
use std::fmt;

const ROWS_A: usize = 4;
const COLS_A: usize = 2;
const ROWS_B: usize = 3;
const COLS_B: usize = 2;
const SHAPE_A: ggml_rs::Shape2D = ggml_rs::Shape2D::new(COLS_A, ROWS_A);
const SHAPE_B: ggml_rs::Shape2D = ggml_rs::Shape2D::new(COLS_B, ROWS_B);
type AShape = ggml_rs::StaticShape2D<COLS_A, ROWS_A>;
type BShape = ggml_rs::StaticShape2D<COLS_B, ROWS_B>;

const MATRIX_A: [f32; ROWS_A * COLS_A] = [2.0, 8.0, 5.0, 1.0, 4.0, 2.0, 8.0, 6.0];
const MATRIX_B: [f32; ROWS_B * COLS_B] = [10.0, 5.0, 9.0, 9.0, 5.0, 4.0];

const EXPECTED: [f32; ROWS_A * ROWS_B] = [
    60.0, 55.0, 50.0, 110.0, //
    90.0, 54.0, 54.0, 126.0, //
    42.0, 29.0, 28.0, 64.0,
];

const TOLERANCE: f32 = 1e-4;

pub type Result<T> = std::result::Result<T, SmokeError>;

#[derive(Debug)]
pub enum SmokeError {
    Ggml(ggml_rs::Error),
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

impl fmt::Display for SmokeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ggml(err) => write!(f, "{err}"),
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

impl StdError for SmokeError {}

impl From<ggml_rs::Error> for SmokeError {
    fn from(value: ggml_rs::Error) -> Self {
        Self::Ggml(value)
    }
}

#[derive(Debug, Clone)]
pub struct SmokeReport {
    pub backend_name: String,
    pub cols: usize,
    pub rows: usize,
    pub values: Vec<f32>,
}

pub fn run_backend_smoke(backend: LlamaBackend) -> Result<SmokeReport> {
    ggml_rs::Backend::load_all();

    let backend = ggml_rs::Backend::new(backend.into())?;
    let backend_name = backend.name()?.to_string();

    let ctx_size =
        ggml_rs::Context::recommended_backend_matmul_memory_f32_shapes_bytes(SHAPE_A, SHAPE_B)?;
    let ctx = ggml_rs::Context::new_no_alloc_bytes(ctx_size)?;

    let a = ctx.new_f32_tensor_2d_typed::<AShape>()?;
    let b = ctx.new_f32_tensor_2d_typed::<BShape>()?;
    let result = ctx.mul_mat(a.inner(), b.inner())?;

    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&result);

    let _buffer = ctx.allocate_tensors(&backend)?;
    a.set_f32_backend(&MATRIX_A)?;
    b.set_f32_backend(&MATRIX_B)?;
    backend.compute(&mut graph)?;

    let output = graph.last_node()?;
    let values = output.to_vec_f32_backend()?;
    assert_expected(&values)?;
    let shape = output.shape()?;

    Ok(SmokeReport {
        backend_name,
        cols: shape.cols.get(),
        rows: shape.rows.get(),
        values,
    })
}

fn assert_expected(values: &[f32]) -> Result<()> {
    if values.len() != EXPECTED.len() {
        return Err(SmokeError::OutputLengthMismatch {
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
            return Err(SmokeError::OutputMismatch {
                index,
                expected,
                actual,
                tolerance: TOLERANCE,
            });
        }
    }

    Ok(())
}
