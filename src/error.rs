use std::ffi::NulError;
use std::num::TryFromIntError;
use std::os::raw::c_int;
use std::str::Utf8Error;
use thiserror::Error;

/// Convenience alias for crate-level fallible operations.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
/// Error type for safe ggml wrapper operations.
pub enum Error {
    #[error("context memory size must be greater than zero")]
    ZeroMemorySize,

    #[error("integer conversion failed ({context})")]
    IntConversion {
        context: String,
        #[source]
        source: TryFromIntError,
    },

    #[error("numeric overflow")]
    Overflow,

    #[error("null pointer returned from ggml API ({context})")]
    NullPointer { context: String },

    #[error("string contains interior NUL byte")]
    CString(#[from] NulError),

    #[error("invalid UTF-8")]
    Utf8(#[from] Utf8Error),

    #[error("length mismatch: expected {expected} elements but got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    #[error("index {index} out of bounds for tensor length {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("thread count must fit in a positive C int, got {0}")]
    InvalidThreadCount(usize),

    #[error("graph index {index} is invalid for node count {node_count}")]
    InvalidGraphIndex { index: i32, node_count: i32 },

    #[error(
        "incompatible matmul shapes: lhs columns ({lhs_cols}) must match rhs columns ({rhs_cols})"
    )]
    IncompatibleMatmulShapes { lhs_cols: usize, rhs_cols: usize },

    #[error("tensor shape is not representable as 2D")]
    UnexpectedShape,

    #[error("tensor rank {0} is unsupported by this operation")]
    UnsupportedRank(usize),

    #[error("unexpected tensor byte size: expected {expected} bytes but got {actual} bytes")]
    UnexpectedTensorByteSize { expected: usize, actual: usize },

    #[error("backend name is not valid UTF-8")]
    InvalidBackendNameUtf8,

    #[error("backend `{0}` is unavailable")]
    BackendUnavailable(&'static str),

    #[error("tensor operations require tensors from the same context")]
    ContextMismatch,

    #[error("unsupported ggml type id: {0}")]
    UnsupportedType(c_int),

    #[error("ggml graph compute failed with status {0}")]
    ComputeFailed(i32),

    #[error("gguf write reported failure")]
    GgufWriteFailed,

    #[error("gguf type mismatch: expected {expected}, got {actual}")]
    GgufTypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },
}

impl Error {
    pub(crate) fn int_conversion(context: &'static str, source: TryFromIntError) -> Self {
        Self::IntConversion {
            context: context.to_owned(),
            source,
        }
    }

    pub(crate) fn null_pointer(context: &'static str) -> Self {
        Self::NullPointer {
            context: context.to_owned(),
        }
    }

    pub fn with_context(self, context: &'static str) -> Self {
        match self {
            Self::IntConversion {
                context: inner,
                source,
            } => Self::IntConversion {
                context: format!("{context} :: {inner}"),
                source,
            },
            Self::NullPointer { context: inner } => Self::NullPointer {
                context: format!("{context} :: {inner}"),
            },
            other => other,
        }
    }
}
