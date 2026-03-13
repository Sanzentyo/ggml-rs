//! Safe Rust wrapper for a focused subset of `ggml`.
//!
//! This crate currently targets matrix multiplication flows equivalent to:
//!
//! - `ggml/examples/simple/simple-ctx.cpp`
//! - `ggml/examples/simple/simple-backend.cpp` (CPU / Metal style backend compute)
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

mod ffi;

use std::error::Error as StdError;
use std::ffi::{CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::ptr::{self, NonNull};

pub type Result<T> = std::result::Result<T, Error>;

const SIMPLE_CONTEXT_SLACK_BYTES: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    ZeroMemorySize,
    IntegerConversion(&'static str),
    NullPointer(&'static str),
    InvalidCString(&'static str),
    InvalidUtf8(&'static str),
    LengthMismatch { expected: usize, actual: usize },
    IndexOutOfBounds { index: usize, len: usize },
    InvalidThreadCount(usize),
    InvalidGraphIndex { index: i32, node_count: i32 },
    IncompatibleMatmulShapes { lhs_cols: usize, rhs_cols: usize },
    UnexpectedShape,
    UnexpectedTensorByteSize { expected: usize, actual: usize },
    InvalidBackendNameUtf8,
    BackendUnavailable(&'static str),
    ComputeFailed(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMemorySize => write!(f, "context memory size must be greater than zero"),
            Self::IntegerConversion(field) => write!(f, "integer conversion failed for {field}"),
            Self::NullPointer(api) => write!(f, "{api} returned a null pointer"),
            Self::InvalidCString(field) => {
                write!(f, "string for {field} contains interior NUL byte")
            }
            Self::InvalidUtf8(field) => write!(f, "invalid UTF-8 in {field}"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "length mismatch: expected {expected} elements but got {actual}"
                )
            }
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds for tensor length {len}")
            }
            Self::InvalidThreadCount(count) => {
                write!(f, "thread count must fit in a positive C int, got {count}")
            }
            Self::InvalidGraphIndex { index, node_count } => {
                write!(
                    f,
                    "graph index {index} is invalid for node count {node_count}"
                )
            }
            Self::IncompatibleMatmulShapes { lhs_cols, rhs_cols } => {
                write!(
                    f,
                    "incompatible matmul shapes: lhs columns ({lhs_cols}) must match rhs columns ({rhs_cols})"
                )
            }
            Self::UnexpectedShape => write!(f, "tensor shape is not representable as 2D"),
            Self::UnexpectedTensorByteSize { expected, actual } => {
                write!(
                    f,
                    "unexpected tensor byte size: expected {expected} bytes but got {actual} bytes"
                )
            }
            Self::InvalidBackendNameUtf8 => write!(f, "backend name is not valid UTF-8"),
            Self::BackendUnavailable(name) => write!(f, "backend `{name}` is unavailable"),
            Self::ComputeFailed(status) => {
                write!(f, "ggml graph compute failed with status {status}")
            }
        }
    }
}

impl StdError for Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    F32,
    I32,
}

impl Type {
    fn as_raw(self) -> c_int {
        match self {
            Self::F32 => ffi::GGML_TYPE_F32,
            Self::I32 => ffi::GGML_TYPE_I32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Metal,
}

impl BackendKind {
    pub const fn as_name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Metal => "Metal",
        }
    }
}

pub fn init_timing() {
    unsafe {
        ffi::ggml_time_init();
    }
}

pub fn type_size(ty: Type) -> usize {
    unsafe { ffi::ggml_type_size(ty.as_raw()) }
}

pub fn tensor_overhead_bytes() -> usize {
    unsafe { ffi::ggml_tensor_overhead() }
}

pub fn graph_overhead_bytes() -> usize {
    unsafe { ffi::ggml_graph_overhead() }
}

pub struct Backend {
    raw: NonNull<ffi::ggml_backend>,
    kind: BackendKind,
    _not_send_sync: PhantomData<*mut ()>,
}

impl Backend {
    pub fn load_all() {
        unsafe {
            ffi::ggml_backend_load_all();
        }
    }

    pub fn new(kind: BackendKind) -> Result<Self> {
        Self::load_all();
        let raw = match kind {
            BackendKind::Cpu => Self::init_cpu_backend()?,
            BackendKind::Metal => Self::init_metal_backend()?,
        };

        Ok(Self {
            raw,
            kind,
            _not_send_sync: PhantomData,
        })
    }

    fn init_by_name(name: *const c_char) -> Option<NonNull<ffi::ggml_backend>> {
        NonNull::new(unsafe { ffi::ggml_backend_init_by_name(name, ptr::null()) })
    }

    fn init_by_type(device_type: c_int) -> Option<NonNull<ffi::ggml_backend>> {
        NonNull::new(unsafe { ffi::ggml_backend_init_by_type(device_type, ptr::null()) })
    }

    fn init_cpu_backend() -> Result<NonNull<ffi::ggml_backend>> {
        Self::init_by_type(ffi::GGML_BACKEND_DEVICE_TYPE_CPU)
            .or_else(|| Self::init_by_name(c"CPU".as_ptr()))
            .ok_or(Error::BackendUnavailable(BackendKind::Cpu.as_name()))
    }

    fn init_metal_backend() -> Result<NonNull<ffi::ggml_backend>> {
        let n_devices = unsafe { ffi::ggml_backend_dev_count() };
        for index in 0..n_devices {
            let device = unsafe { ffi::ggml_backend_dev_get(index) };
            let Some(device) = NonNull::new(device) else {
                continue;
            };

            let device_type = unsafe { ffi::ggml_backend_dev_type(device.as_ptr()) };
            let is_gpu_like = device_type == ffi::GGML_BACKEND_DEVICE_TYPE_GPU
                || device_type == ffi::GGML_BACKEND_DEVICE_TYPE_IGPU;
            if !is_gpu_like {
                continue;
            }

            let device_name = unsafe { ffi::ggml_backend_dev_name(device.as_ptr()) };
            let Some(device_name) = NonNull::new(device_name.cast_mut()) else {
                continue;
            };
            let device_name = unsafe { CStr::from_ptr(device_name.as_ptr()) }
                .to_string_lossy()
                .to_ascii_lowercase();
            if !(device_name.contains("metal") || device_name.contains("mtl")) {
                continue;
            }

            if let Some(raw) =
                NonNull::new(unsafe { ffi::ggml_backend_dev_init(device.as_ptr(), ptr::null()) })
            {
                return Ok(raw);
            }
        }

        for backend_name in [c"Metal".as_ptr(), c"METAL".as_ptr(), c"metal".as_ptr()] {
            if let Some(raw) = Self::init_by_name(backend_name) {
                return Ok(raw);
            }
        }

        for device_type in [
            ffi::GGML_BACKEND_DEVICE_TYPE_IGPU,
            ffi::GGML_BACKEND_DEVICE_TYPE_GPU,
        ] {
            if let Some(raw) = Self::init_by_type(device_type) {
                return Ok(raw);
            }
        }

        Err(Error::BackendUnavailable(BackendKind::Metal.as_name()))
    }

    pub fn kind(&self) -> BackendKind {
        self.kind
    }

    pub fn name(&self) -> Result<&str> {
        let name = unsafe { ffi::ggml_backend_name(self.raw.as_ptr()) };
        let name = NonNull::new(name.cast_mut()).ok_or(Error::NullPointer("ggml_backend_name"))?;
        let cstr = unsafe { CStr::from_ptr(name.as_ptr()) };
        cstr.to_str().map_err(|_| Error::InvalidBackendNameUtf8)
    }

    pub fn compute<'ctx>(&self, graph: &mut Graph<'ctx>) -> Result<()> {
        let status =
            unsafe { ffi::ggml_backend_graph_compute(self.raw.as_ptr(), graph.raw.as_ptr()) };

        if status == ffi::GGML_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(Error::ComputeFailed(status))
        }
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe {
            ffi::ggml_backend_free(self.raw.as_ptr());
        }
    }
}

pub struct BackendBuffer<'ctx> {
    raw: NonNull<ffi::ggml_backend_buffer>,
    _ctx: PhantomData<&'ctx Context>,
}

impl<'ctx> Drop for BackendBuffer<'ctx> {
    fn drop(&mut self) {
        unsafe {
            ffi::ggml_backend_buffer_free(self.raw.as_ptr());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
    Unknown(i32),
}

impl GgufType {
    fn from_raw(raw: c_int) -> Self {
        match raw {
            ffi::GGUF_TYPE_UINT8 => Self::Uint8,
            ffi::GGUF_TYPE_INT8 => Self::Int8,
            ffi::GGUF_TYPE_UINT16 => Self::Uint16,
            ffi::GGUF_TYPE_INT16 => Self::Int16,
            ffi::GGUF_TYPE_UINT32 => Self::Uint32,
            ffi::GGUF_TYPE_INT32 => Self::Int32,
            ffi::GGUF_TYPE_FLOAT32 => Self::Float32,
            ffi::GGUF_TYPE_BOOL => Self::Bool,
            ffi::GGUF_TYPE_STRING => Self::String,
            ffi::GGUF_TYPE_ARRAY => Self::Array,
            ffi::GGUF_TYPE_UINT64 => Self::Uint64,
            ffi::GGUF_TYPE_INT64 => Self::Int64,
            ffi::GGUF_TYPE_FLOAT64 => Self::Float64,
            _ => Self::Unknown(raw),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub ggml_type_raw: i32,
    pub ggml_type_name: String,
}

pub struct GgufFile {
    raw: NonNull<ffi::gguf_context>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl GgufFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path_to_c_string(path.as_ref(), "gguf path")?;
        let params = ffi::gguf_init_params {
            no_alloc: true,
            ctx: ptr::null_mut(),
        };
        let raw = unsafe { ffi::gguf_init_from_file(path.as_ptr(), params) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("gguf_init_from_file"))?;

        Ok(Self {
            raw,
            _not_send_sync: PhantomData,
        })
    }

    pub fn version(&self) -> u32 {
        unsafe { ffi::gguf_get_version(self.raw.as_ptr()) }
    }

    pub fn alignment(&self) -> usize {
        unsafe { ffi::gguf_get_alignment(self.raw.as_ptr()) }
    }

    pub fn data_offset(&self) -> usize {
        unsafe { ffi::gguf_get_data_offset(self.raw.as_ptr()) }
    }

    pub fn kv_count(&self) -> Result<usize> {
        i64_to_usize(
            unsafe { ffi::gguf_get_n_kv(self.raw.as_ptr()) },
            "gguf_get_n_kv",
        )
    }

    pub fn kv_key(&self, index: usize) -> Result<String> {
        let index = usize_to_i64(index, "gguf key index")?;
        let key = unsafe { ffi::gguf_get_key(self.raw.as_ptr(), index) };
        c_string_from_ptr(key, "gguf_get_key")
    }

    pub fn find_key(&self, key: &str) -> Result<Option<usize>> {
        let key = CString::new(key).map_err(|_| Error::InvalidCString("gguf key"))?;
        let idx = unsafe { ffi::gguf_find_key(self.raw.as_ptr(), key.as_ptr()) };
        if idx < 0 {
            Ok(None)
        } else {
            Ok(Some(i64_to_usize(idx, "gguf_find_key")?))
        }
    }

    pub fn kv_type(&self, index: usize) -> Result<GgufType> {
        let index = usize_to_i64(index, "gguf key index")?;
        let raw = unsafe { ffi::gguf_get_kv_type(self.raw.as_ptr(), index) };
        Ok(GgufType::from_raw(raw))
    }

    pub fn kv_type_name(&self, index: usize) -> Result<String> {
        let index = usize_to_i64(index, "gguf key index")?;
        let raw = unsafe { ffi::gguf_get_kv_type(self.raw.as_ptr(), index) };
        let name = unsafe { ffi::gguf_type_name(raw) };
        c_string_from_ptr(name, "gguf_type_name")
    }

    pub fn kv_string_value(&self, index: usize) -> Result<String> {
        let index = usize_to_i64(index, "gguf key index")?;
        let value = unsafe { ffi::gguf_get_val_str(self.raw.as_ptr(), index) };
        c_string_from_ptr(value, "gguf_get_val_str")
    }

    pub fn tensor_count(&self) -> Result<usize> {
        i64_to_usize(
            unsafe { ffi::gguf_get_n_tensors(self.raw.as_ptr()) },
            "gguf_get_n_tensors",
        )
    }

    pub fn tensor_info(&self, index: usize) -> Result<GgufTensorInfo> {
        let index_i64 = usize_to_i64(index, "gguf tensor index")?;
        let name = unsafe { ffi::gguf_get_tensor_name(self.raw.as_ptr(), index_i64) };
        let name = c_string_from_ptr(name, "gguf_get_tensor_name")?;

        let ggml_type_raw = unsafe { ffi::gguf_get_tensor_type(self.raw.as_ptr(), index_i64) };
        let ggml_type_name = unsafe { ffi::ggml_type_name(ggml_type_raw) };
        let ggml_type_name = c_string_from_ptr(ggml_type_name, "ggml_type_name")?;

        let offset = unsafe { ffi::gguf_get_tensor_offset(self.raw.as_ptr(), index_i64) };
        let size = unsafe { ffi::gguf_get_tensor_size(self.raw.as_ptr(), index_i64) };

        Ok(GgufTensorInfo {
            name,
            offset,
            size,
            ggml_type_raw,
            ggml_type_name,
        })
    }
}

impl Drop for GgufFile {
    fn drop(&mut self) {
        unsafe {
            ffi::gguf_free(self.raw.as_ptr());
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RopeExtParams {
    pub n_dims: i32,
    pub mode: i32,
    pub n_ctx_orig: i32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl Default for RopeExtParams {
    fn default() -> Self {
        Self {
            n_dims: 0,
            mode: 0,
            n_ctx_orig: 0,
            freq_base: 10_000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

pub struct Context {
    raw: NonNull<ffi::ggml_context>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl Context {
    pub fn new(mem_size: usize) -> Result<Self> {
        Self::new_with_options(mem_size, false)
    }

    pub fn new_no_alloc(mem_size: usize) -> Result<Self> {
        Self::new_with_options(mem_size, true)
    }

    fn new_with_options(mem_size: usize, no_alloc: bool) -> Result<Self> {
        let mem_size = NonZeroUsize::new(mem_size).ok_or(Error::ZeroMemorySize)?;

        let params = ffi::ggml_init_params {
            mem_size: mem_size.get(),
            mem_buffer: ptr::null_mut(),
            no_alloc,
        };

        let raw = unsafe { ffi::ggml_init(params) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_init"))?;

        Ok(Self {
            raw,
            _not_send_sync: PhantomData,
        })
    }

    pub fn recommended_matmul_memory_f32(
        rows_a: usize,
        cols_a: usize,
        rows_b: usize,
        cols_b: usize,
    ) -> Result<usize> {
        ensure_matmul_compatible(cols_a, cols_b)?;

        let matrix_a_elements = checked_mul(rows_a, cols_a, "A elements")?;
        let matrix_b_elements = checked_mul(rows_b, cols_b, "B elements")?;
        let matrix_result_elements = checked_mul(rows_a, rows_b, "result elements")?;

        let matrix_a_bytes = checked_mul(matrix_a_elements, type_size(Type::F32), "A bytes")?;
        let matrix_b_bytes = checked_mul(matrix_b_elements, type_size(Type::F32), "B bytes")?;
        let matrix_result_bytes =
            checked_mul(matrix_result_elements, type_size(Type::F32), "result bytes")?;

        let tensors_overhead = checked_mul(3, tensor_overhead_bytes(), "tensor overhead")?;
        let graph_and_slack = checked_add(
            graph_overhead_bytes(),
            SIMPLE_CONTEXT_SLACK_BYTES,
            "graph overhead + slack",
        )?;

        checked_add(
            checked_add(
                checked_add(
                    checked_add(matrix_a_bytes, matrix_b_bytes, "A + B bytes")?,
                    matrix_result_bytes,
                    "A+B+result bytes",
                )?,
                tensors_overhead,
                "data + tensor overhead",
            )?,
            graph_and_slack,
            "total context bytes",
        )
    }

    pub fn recommended_backend_matmul_memory_f32(
        rows_a: usize,
        cols_a: usize,
        rows_b: usize,
        cols_b: usize,
    ) -> Result<usize> {
        let _ = rows_a;
        let _ = rows_b;
        ensure_matmul_compatible(cols_a, cols_b)?;

        let tensors_overhead = checked_mul(3, tensor_overhead_bytes(), "tensor overhead")?;
        checked_add(
            tensors_overhead,
            checked_add(
                graph_overhead_bytes(),
                SIMPLE_CONTEXT_SLACK_BYTES,
                "graph overhead + slack",
            )?,
            "backend context bytes",
        )
    }

    pub fn allocate_tensors<'ctx>(&'ctx self, backend: &Backend) -> Result<BackendBuffer<'ctx>> {
        let raw =
            unsafe { ffi::ggml_backend_alloc_ctx_tensors(self.raw.as_ptr(), backend.raw.as_ptr()) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_backend_alloc_ctx_tensors"))?;

        Ok(BackendBuffer {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn new_tensor_2d(&self, ty: Type, cols: usize, rows: usize) -> Result<Tensor<'_>> {
        let cols = usize_to_i64(cols, "cols")?;
        let rows = usize_to_i64(rows, "rows")?;

        let raw = unsafe { ffi::ggml_new_tensor_2d(self.raw.as_ptr(), ty.as_raw(), cols, rows) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_new_tensor_2d"))?;

        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn new_f32_tensor_2d(&self, cols: usize, rows: usize) -> Result<Tensor<'_>> {
        self.new_tensor_2d(Type::F32, cols, rows)
    }

    pub fn new_tensor_1d(&self, ty: Type, len: usize) -> Result<Tensor<'_>> {
        let len = usize_to_i64(len, "len")?;
        let raw = unsafe { ffi::ggml_new_tensor_1d(self.raw.as_ptr(), ty.as_raw(), len) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_new_tensor_1d"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn new_f32_tensor_1d(&self, len: usize) -> Result<Tensor<'_>> {
        self.new_tensor_1d(Type::F32, len)
    }

    pub fn new_i32_tensor_1d(&self, len: usize) -> Result<Tensor<'_>> {
        self.new_tensor_1d(Type::I32, len)
    }

    pub fn new_tensor_3d(
        &self,
        ty: Type,
        ne0: usize,
        ne1: usize,
        ne2: usize,
    ) -> Result<Tensor<'_>> {
        let ne0 = usize_to_i64(ne0, "ne0")?;
        let ne1 = usize_to_i64(ne1, "ne1")?;
        let ne2 = usize_to_i64(ne2, "ne2")?;
        let raw = unsafe { ffi::ggml_new_tensor_3d(self.raw.as_ptr(), ty.as_raw(), ne0, ne1, ne2) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_new_tensor_3d"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn new_tensor_4d(
        &self,
        ty: Type,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
    ) -> Result<Tensor<'_>> {
        let ne0 = usize_to_i64(ne0, "ne0")?;
        let ne1 = usize_to_i64(ne1, "ne1")?;
        let ne2 = usize_to_i64(ne2, "ne2")?;
        let ne3 = usize_to_i64(ne3, "ne3")?;
        let raw =
            unsafe { ffi::ggml_new_tensor_4d(self.raw.as_ptr(), ty.as_raw(), ne0, ne1, ne2, ne3) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_new_tensor_4d"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn mul_mat<'ctx>(&'ctx self, a: &Tensor<'ctx>, b: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let (a_cols, _) = a.shape_2d()?;
        let (b_cols, _) = b.shape_2d()?;
        ensure_matmul_compatible(a_cols, b_cols)?;

        let raw = unsafe { ffi::ggml_mul_mat(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_mul_mat"))?;

        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn add<'ctx>(&'ctx self, a: &Tensor<'ctx>, b: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_add(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_add"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn mul<'ctx>(&'ctx self, a: &Tensor<'ctx>, b: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_mul(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_mul"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn silu<'ctx>(&'ctx self, a: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_silu(self.raw.as_ptr(), a.raw.as_ptr()) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_silu"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn rms_norm<'ctx>(&'ctx self, a: &Tensor<'ctx>, eps: f32) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_rms_norm(self.raw.as_ptr(), a.raw.as_ptr(), eps) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_rms_norm"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn scale<'ctx>(&'ctx self, a: &Tensor<'ctx>, scalar: f32) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_scale(self.raw.as_ptr(), a.raw.as_ptr(), scalar) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_scale"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn get_rows<'ctx>(
        &'ctx self,
        data: &Tensor<'ctx>,
        indices: &Tensor<'ctx>,
    ) -> Result<Tensor<'ctx>> {
        let raw = unsafe {
            ffi::ggml_get_rows(self.raw.as_ptr(), data.raw.as_ptr(), indices.raw.as_ptr())
        };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_get_rows"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn repeat<'ctx>(&'ctx self, a: &Tensor<'ctx>, b: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_repeat(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_repeat"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn cpy<'ctx>(&'ctx self, a: &Tensor<'ctx>, b: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_cpy(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_cpy"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn cont<'ctx>(&'ctx self, a: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_cont(self.raw.as_ptr(), a.raw.as_ptr()) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_cont"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn reshape_2d<'ctx>(
        &'ctx self,
        a: &Tensor<'ctx>,
        ne0: usize,
        ne1: usize,
    ) -> Result<Tensor<'ctx>> {
        let ne0 = usize_to_i64(ne0, "ne0")?;
        let ne1 = usize_to_i64(ne1, "ne1")?;
        let raw = unsafe { ffi::ggml_reshape_2d(self.raw.as_ptr(), a.raw.as_ptr(), ne0, ne1) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_reshape_2d"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn reshape_3d<'ctx>(
        &'ctx self,
        a: &Tensor<'ctx>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
    ) -> Result<Tensor<'ctx>> {
        let ne0 = usize_to_i64(ne0, "ne0")?;
        let ne1 = usize_to_i64(ne1, "ne1")?;
        let ne2 = usize_to_i64(ne2, "ne2")?;
        let raw = unsafe { ffi::ggml_reshape_3d(self.raw.as_ptr(), a.raw.as_ptr(), ne0, ne1, ne2) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_reshape_3d"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn view_1d<'ctx>(
        &'ctx self,
        a: &Tensor<'ctx>,
        ne0: usize,
        offset: usize,
    ) -> Result<Tensor<'ctx>> {
        let ne0 = usize_to_i64(ne0, "ne0")?;
        let raw = unsafe { ffi::ggml_view_1d(self.raw.as_ptr(), a.raw.as_ptr(), ne0, offset) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_view_1d"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn view_2d<'ctx>(
        &'ctx self,
        a: &Tensor<'ctx>,
        ne0: usize,
        ne1: usize,
        row_stride: usize,
        offset: usize,
    ) -> Result<Tensor<'ctx>> {
        let ne0 = usize_to_i64(ne0, "ne0")?;
        let ne1 = usize_to_i64(ne1, "ne1")?;
        let raw = unsafe {
            ffi::ggml_view_2d(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                ne0,
                ne1,
                row_stride,
                offset,
            )
        };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_view_2d"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn permute<'ctx>(
        &'ctx self,
        a: &Tensor<'ctx>,
        axis0: i32,
        axis1: i32,
        axis2: i32,
        axis3: i32,
    ) -> Result<Tensor<'ctx>> {
        let raw = unsafe {
            ffi::ggml_permute(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                axis0,
                axis1,
                axis2,
                axis3,
            )
        };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_permute"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn diag_mask_inf<'ctx>(&'ctx self, a: &Tensor<'ctx>, n_past: i32) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_diag_mask_inf(self.raw.as_ptr(), a.raw.as_ptr(), n_past) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_diag_mask_inf"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn soft_max<'ctx>(&'ctx self, a: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_soft_max(self.raw.as_ptr(), a.raw.as_ptr()) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_soft_max"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn soft_max_ext<'ctx>(
        &'ctx self,
        a: &Tensor<'ctx>,
        mask: Option<&Tensor<'ctx>>,
        scale: f32,
        max_bias: f32,
    ) -> Result<Tensor<'ctx>> {
        let mask_raw = mask.map_or(ptr::null_mut(), |t| t.raw.as_ptr());
        let raw = unsafe {
            ffi::ggml_soft_max_ext(self.raw.as_ptr(), a.raw.as_ptr(), mask_raw, scale, max_bias)
        };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_soft_max_ext"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn rope_ext<'ctx>(
        &'ctx self,
        a: &Tensor<'ctx>,
        positions: &Tensor<'ctx>,
        freq_factors: Option<&Tensor<'ctx>>,
        params: RopeExtParams,
    ) -> Result<Tensor<'ctx>> {
        let freq_factors_raw = freq_factors.map_or(ptr::null_mut(), |t| t.raw.as_ptr());
        let raw = unsafe {
            ffi::ggml_rope_ext(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                positions.raw.as_ptr(),
                freq_factors_raw,
                params.n_dims,
                params.mode,
                params.n_ctx_orig,
                params.freq_base,
                params.freq_scale,
                params.ext_factor,
                params.attn_factor,
                params.beta_fast,
                params.beta_slow,
            )
        };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_rope_ext"))?;
        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn new_graph(&self) -> Result<Graph<'_>> {
        let raw = unsafe { ffi::ggml_new_graph(self.raw.as_ptr()) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_new_graph"))?;

        Ok(Graph {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn compute<'ctx>(&'ctx self, graph: &mut Graph<'ctx>, n_threads: usize) -> Result<()> {
        let n_threads =
            usize_to_positive_c_int(n_threads).ok_or(Error::InvalidThreadCount(n_threads))?;

        let status = unsafe {
            ffi::ggml_graph_compute_with_ctx(self.raw.as_ptr(), graph.raw.as_ptr(), n_threads)
        };

        if status == ffi::GGML_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(Error::ComputeFailed(status))
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            ffi::ggml_free(self.raw.as_ptr());
        }
    }
}

#[derive(Clone, Copy)]
pub struct Tensor<'ctx> {
    raw: NonNull<ffi::ggml_tensor>,
    _ctx: PhantomData<&'ctx Context>,
}

impl<'ctx> Tensor<'ctx> {
    pub fn element_count(&self) -> Result<usize> {
        i64_to_usize(
            unsafe { ffi::ggml_nelements(self.raw.as_ptr()) },
            "ggml_nelements",
        )
    }

    pub fn row_count(&self) -> Result<usize> {
        i64_to_usize(unsafe { ffi::ggml_nrows(self.raw.as_ptr()) }, "ggml_nrows")
    }

    pub fn col_count(&self) -> Result<usize> {
        let rows = self.row_count()?;
        let elements = self.element_count()?;

        if rows == 0 || elements % rows != 0 {
            return Err(Error::UnexpectedShape);
        }

        Ok(elements / rows)
    }

    pub fn shape_2d(&self) -> Result<(usize, usize)> {
        Ok((self.col_count()?, self.row_count()?))
    }

    pub fn nbytes(&self) -> usize {
        unsafe { ffi::ggml_nbytes(self.raw.as_ptr()) }
    }

    fn expected_f32_nbytes(&self) -> Result<usize> {
        let elements = self.element_count()?;
        checked_mul(elements, std::mem::size_of::<f32>(), "f32 tensor bytes")
    }

    fn expected_i32_nbytes(&self) -> Result<usize> {
        let elements = self.element_count()?;
        checked_mul(elements, std::mem::size_of::<i32>(), "i32 tensor bytes")
    }

    pub fn set_name(&self, name: &str) -> Result<()> {
        let name = CString::new(name).map_err(|_| Error::InvalidCString("tensor name"))?;
        let raw = unsafe { ffi::ggml_set_name(self.raw.as_ptr(), name.as_ptr()) };
        let _ = NonNull::new(raw).ok_or(Error::NullPointer("ggml_set_name"))?;
        Ok(())
    }

    pub fn name(&self) -> Result<String> {
        let name = unsafe { ffi::ggml_get_name(self.raw.as_ptr()) };
        c_string_from_ptr(name, "ggml_get_name")
    }

    pub fn set_f32(&self, values: &[f32]) -> Result<()> {
        let expected = self.element_count()?;
        if values.len() != expected {
            return Err(Error::LengthMismatch {
                expected,
                actual: values.len(),
            });
        }

        for (index, value) in values.iter().copied().enumerate() {
            let index = usize_to_c_int(index, "tensor index")?;
            unsafe {
                ffi::ggml_set_f32_1d(self.raw.as_ptr(), index, value);
            }
        }

        Ok(())
    }

    pub fn set_f32_backend(&self, values: &[f32]) -> Result<()> {
        let expected = self.element_count()?;
        if values.len() != expected {
            return Err(Error::LengthMismatch {
                expected,
                actual: values.len(),
            });
        }

        let expected_nbytes = self.expected_f32_nbytes()?;
        let actual_nbytes = self.nbytes();
        if expected_nbytes != actual_nbytes {
            return Err(Error::UnexpectedTensorByteSize {
                expected: expected_nbytes,
                actual: actual_nbytes,
            });
        }

        unsafe {
            ffi::ggml_backend_tensor_set(
                self.raw.as_ptr(),
                values.as_ptr().cast(),
                0,
                expected_nbytes,
            );
        }

        Ok(())
    }

    pub fn set_i32_backend(&self, values: &[i32]) -> Result<()> {
        let expected = self.element_count()?;
        if values.len() != expected {
            return Err(Error::LengthMismatch {
                expected,
                actual: values.len(),
            });
        }

        let expected_nbytes = self.expected_i32_nbytes()?;
        let actual_nbytes = self.nbytes();
        if expected_nbytes != actual_nbytes {
            return Err(Error::UnexpectedTensorByteSize {
                expected: expected_nbytes,
                actual: actual_nbytes,
            });
        }

        unsafe {
            ffi::ggml_backend_tensor_set(
                self.raw.as_ptr(),
                values.as_ptr().cast(),
                0,
                expected_nbytes,
            );
        }

        Ok(())
    }

    pub fn get_f32(&self, index: usize) -> Result<f32> {
        let len = self.element_count()?;
        if index >= len {
            return Err(Error::IndexOutOfBounds { index, len });
        }

        let index = usize_to_c_int(index, "tensor index")?;
        Ok(unsafe { ffi::ggml_get_f32_1d(self.raw.as_ptr(), index) })
    }

    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let len = self.element_count()?;
        let mut out = Vec::with_capacity(len);

        for index in 0..len {
            let index = usize_to_c_int(index, "tensor index")?;
            out.push(unsafe { ffi::ggml_get_f32_1d(self.raw.as_ptr(), index) });
        }

        Ok(out)
    }

    pub fn to_vec_f32_backend(&self) -> Result<Vec<f32>> {
        let len = self.element_count()?;
        let expected_nbytes = self.expected_f32_nbytes()?;
        let actual_nbytes = self.nbytes();
        if expected_nbytes != actual_nbytes {
            return Err(Error::UnexpectedTensorByteSize {
                expected: expected_nbytes,
                actual: actual_nbytes,
            });
        }

        let mut out = vec![0.0_f32; len];
        unsafe {
            ffi::ggml_backend_tensor_get(
                self.raw.as_ptr(),
                out.as_mut_ptr().cast(),
                0,
                expected_nbytes,
            );
        }
        Ok(out)
    }

    pub fn to_vec_i32_backend(&self) -> Result<Vec<i32>> {
        let len = self.element_count()?;
        let expected_nbytes = self.expected_i32_nbytes()?;
        let actual_nbytes = self.nbytes();
        if expected_nbytes != actual_nbytes {
            return Err(Error::UnexpectedTensorByteSize {
                expected: expected_nbytes,
                actual: actual_nbytes,
            });
        }

        let mut out = vec![0_i32; len];
        unsafe {
            ffi::ggml_backend_tensor_get(
                self.raw.as_ptr(),
                out.as_mut_ptr().cast(),
                0,
                expected_nbytes,
            );
        }
        Ok(out)
    }
}

pub struct Graph<'ctx> {
    raw: NonNull<ffi::ggml_cgraph>,
    _ctx: PhantomData<&'ctx Context>,
}

impl<'ctx> Graph<'ctx> {
    pub fn build_forward_expand(&mut self, tensor: &Tensor<'ctx>) {
        unsafe {
            ffi::ggml_build_forward_expand(self.raw.as_ptr(), tensor.raw.as_ptr());
        }
    }

    pub fn node_count(&self) -> i32 {
        unsafe { ffi::ggml_graph_n_nodes(self.raw.as_ptr()) }
    }

    pub fn node(&self, index: i32) -> Result<Tensor<'ctx>> {
        let node_count = self.node_count();
        if node_count <= 0 {
            return Err(Error::InvalidGraphIndex { index, node_count });
        }

        let normalized = if index < 0 { node_count + index } else { index };
        if normalized < 0 || normalized >= node_count {
            return Err(Error::InvalidGraphIndex { index, node_count });
        }

        let raw = unsafe { ffi::ggml_graph_node(self.raw.as_ptr(), normalized) };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer("ggml_graph_node"))?;

        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    pub fn last_node(&self) -> Result<Tensor<'ctx>> {
        let node_count = self.node_count();
        if node_count <= 0 {
            return Err(Error::InvalidGraphIndex {
                index: -1,
                node_count,
            });
        }

        self.node(node_count - 1)
    }
}

fn path_to_c_string(path: &Path, field: &'static str) -> Result<CString> {
    let lossy = path.to_string_lossy();
    CString::new(lossy.as_bytes()).map_err(|_| Error::InvalidCString(field))
}

fn c_string_from_ptr(ptr: *const c_char, field: &'static str) -> Result<String> {
    let ptr = NonNull::new(ptr.cast_mut()).ok_or(Error::NullPointer(field))?;
    let cstr = unsafe { CStr::from_ptr(ptr.as_ptr()) };
    let text = cstr.to_str().map_err(|_| Error::InvalidUtf8(field))?;
    Ok(text.to_string())
}

fn checked_add(lhs: usize, rhs: usize, field: &'static str) -> Result<usize> {
    lhs.checked_add(rhs).ok_or(Error::IntegerConversion(field))
}

fn checked_mul(lhs: usize, rhs: usize, field: &'static str) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or(Error::IntegerConversion(field))
}

fn i64_to_usize(value: i64, field: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::IntegerConversion(field))
}

fn usize_to_i64(value: usize, field: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::IntegerConversion(field))
}

fn usize_to_c_int(value: usize, field: &'static str) -> Result<c_int> {
    c_int::try_from(value).map_err(|_| Error::IntegerConversion(field))
}

fn usize_to_positive_c_int(value: usize) -> Option<c_int> {
    let converted = c_int::try_from(value).ok()?;
    (converted > 0).then_some(converted)
}

fn ensure_matmul_compatible(lhs_cols: usize, rhs_cols: usize) -> Result<()> {
    if lhs_cols == rhs_cols {
        Ok(())
    } else {
        Err(Error::IncompatibleMatmulShapes { lhs_cols, rhs_cols })
    }
}
