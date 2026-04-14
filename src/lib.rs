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
//! - `cargo run --example perf_metal --features link-system`
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
    Backend, BackendBuffer, Context, DynTensor, Graph, GraphAllocator, Tensor,
    decode_tensor_data_to, graph_overhead_bytes, graph_overhead_custom, init_timing,
    tensor_element_count, tensor_overhead_bytes, type_size, with_context, with_no_alloc_context,
};
pub use error::{Error, Result};
pub use gguf::{
    GgufArrayValue, GgufFile, GgufTensorInfo, GgufType, GgufValue, GgufWriter, TryFromGgufValue,
};
pub use shape::{
    Bytes, Cols, Dims, Length, LengthSpec, Rows, Shape2D, Shape2DSpec, Shape3D, Shape3DSpec,
    Shape4D, Shape4DSpec, StaticLength, StaticShape2D, StaticShape3D, StaticShape4D, TensorIndex,
    ThreadCount,
};
pub use tensor_expr::{BackendElement, GgmlElement, TensorExpr};
pub use typed_tensor::{
    Tensor1D, Tensor1DConst, Tensor2D, Tensor2DConst, Tensor3D, Tensor3DConst, Tensor4D,
    Tensor4DConst,
};
pub use types::{BackendDeviceType, BackendKind, ComputeStatus, GgmlType, RopeExtParams, Type};

pub mod prelude {
    pub use crate::{
        Backend, BackendBuffer, BackendDeviceType, BackendElement, BackendKind, Bytes, Cols,
        ComputeStatus, Context, Dims, DynTensor, GgmlElement, GgmlType, GgufArrayValue, GgufFile,
        GgufValue, GgufWriter, Graph, GraphAllocator, Length, LengthSpec, RopeExtParams, Rows,
        Shape2D, Shape2DSpec, Shape3D, Shape3DSpec, Shape4D, Shape4DSpec, StaticLength,
        StaticShape2D, StaticShape3D, StaticShape4D, Tensor, Tensor1D, Tensor1DConst, Tensor2D,
        Tensor2DConst, Tensor3D, Tensor3DConst, Tensor4D, Tensor4DConst, TensorExpr, TensorIndex,
        ThreadCount, TryFromGgufValue, Type, with_context, with_no_alloc_context,
    };
}
