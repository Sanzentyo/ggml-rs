//! Safe Rust wrapper for a focused subset of `ggml`.
//!
//! This crate currently targets matrix multiplication flows equivalent to:
//!
//! - `ggml/examples/simple/simple-ctx.cpp`
//! - `ggml/examples/simple/simple-backend.cpp` (CPU / Metal style backend compute)
//!
//! It also provides trait-based expression helpers for idiomatic arithmetic composition.
//!
//! To link and run, enable the `link-system` feature and make `ggml` libraries
//! discoverable by the linker.
//! You can override library lookup with:
//!
//! - `GGML_RS_LIB_DIR=/path/to/lib`
//! - `GGML_RS_LIB_DIRS=/path/to/lib1:/path/to/lib2`
//! - `GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal`
//!
//! Examples:
//! - `cargo run --example simple_ctx --features link-system`
//! - `cargo run --example backend_matmul --features link-system`
//! - `cargo run --example arithmetic_expr --features link-system`

mod compute;
mod error;
mod ffi;
mod gguf;
mod num_ext;
mod shape;
mod tensor_expr;
mod typed_tensor;
mod types;

pub use compute::{
    Backend, BackendBuffer, Context, Graph, Tensor, graph_overhead_bytes, init_timing,
    tensor_overhead_bytes, type_size,
};
pub use error::{Error, Result};
pub use gguf::{GgufFile, GgufTensorInfo, GgufType};
pub use shape::{
    Bytes, Cols, Length, Rows, Shape2D, Shape2DSpec, StaticShape2D, TensorIndex, ThreadCount,
};
pub use tensor_expr::{BackendElement, TensorExpr};
pub use typed_tensor::{Tensor2D, Tensor2DConst};
pub use types::{BackendKind, RopeExtParams, Type};

pub mod prelude {
    pub use crate::{
        Backend, BackendBuffer, BackendElement, BackendKind, Bytes, Cols, Context, GgufFile, Graph,
        Length, RopeExtParams, Rows, Shape2D, Shape2DSpec, StaticShape2D, Tensor, Tensor2D,
        Tensor2DConst, TensorExpr, TensorIndex, ThreadCount, Type,
    };
}
