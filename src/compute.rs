//! Core safe wrappers for ggml context, backend, tensor, and graph objects.

use crate::ffi;
use crate::num_ext::{CheckedFieldOps, TryIntoChecked};
use crate::{
    BackendElement, BackendKind, Bytes, Cols, Error, Length, Result, RopeExtParams, Rows, Shape2D,
    TensorExpr, TensorIndex, ThreadCount, Type,
};
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::os::raw::c_int;
use std::ptr::{self, NonNull};

const SIMPLE_CONTEXT_SLACK_BYTES: usize = 1024;

/// Initializes ggml global timing infrastructure.
pub fn init_timing() {
    unsafe {
        ffi::ggml_time_init();
    }
}

/// Returns the byte size of a single element for the given type.
pub fn type_size(ty: Type) -> usize {
    unsafe { ffi::ggml_type_size(ty.as_raw()) }
}

/// Returns ggml's internal per-tensor metadata overhead in bytes.
pub fn tensor_overhead_bytes() -> usize {
    unsafe { ffi::ggml_tensor_overhead() }
}

/// Returns ggml's internal per-graph metadata overhead in bytes.
pub fn graph_overhead_bytes() -> usize {
    unsafe { ffi::ggml_graph_overhead() }
}

/// RAII handle for a ggml backend implementation (CPU/Metal).
pub struct Backend {
    raw: NonNull<ffi::ggml_backend>,
    kind: BackendKind,
    _not_send_sync: PhantomData<*mut ()>,
}

impl Backend {
    /// Loads all backend implementations discoverable by ggml.
    pub fn load_all() {
        unsafe {
            ffi::ggml_backend_load_all();
        }
    }

    /// Creates a backend instance for the requested backend family.
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

    fn init_by_name(name: &CStr) -> Option<NonNull<ffi::ggml_backend>> {
        NonNull::new(unsafe { ffi::ggml_backend_init_by_name(name.as_ptr(), ptr::null()) })
    }

    fn init_by_type(device_type: c_int) -> Option<NonNull<ffi::ggml_backend>> {
        NonNull::new(unsafe { ffi::ggml_backend_init_by_type(device_type, ptr::null()) })
    }

    fn init_cpu_backend() -> Result<NonNull<ffi::ggml_backend>> {
        Self::init_by_type(ffi::GGML_BACKEND_DEVICE_TYPE_CPU)
            .or_else(|| Self::init_by_name(c"CPU"))
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

            // On macOS, device names can vary (`Metal`, `MTL0`, etc.), so we
            // filter by common metal-like name fragments before initialization.
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

        for backend_name in [c"Metal", c"METAL", c"metal"] {
            if let Some(raw) = Self::init_by_name(backend_name) {
                return Ok(raw);
            }
        }

        // Final fallback: any GPU-like backend type when explicit metal
        // matching is unavailable in the local ggml build.
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

    /// Returns the backend family used at construction time.
    pub fn kind(&self) -> BackendKind {
        self.kind
    }

    /// Returns the backend runtime name reported by ggml.
    pub fn name(&self) -> Result<&str> {
        let name = unsafe { ffi::ggml_backend_name(self.raw.as_ptr()) };
        let name = NonNull::new(name.cast_mut())
            .ok_or_else(|| Error::null_pointer("ggml_backend_name"))?;
        let cstr = unsafe { CStr::from_ptr(name.as_ptr()) };
        cstr.to_str().map_err(|_| Error::InvalidBackendNameUtf8)
    }

    /// Executes the graph through the backend execution path.
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

/// RAII owner for a ggml context.
pub struct Context {
    raw: NonNull<ffi::ggml_context>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl Context {
    /// Creates a context with ggml-managed memory.
    pub fn new(mem_size: usize) -> Result<Self> {
        Self::new_bytes(Bytes::new(mem_size))
    }

    /// Creates a context in no-allocation mode for backend tensor allocation.
    pub fn new_no_alloc(mem_size: usize) -> Result<Self> {
        Self::new_no_alloc_bytes(Bytes::new(mem_size))
    }

    /// Creates a context with explicit byte-size newtype.
    pub fn new_bytes(mem_size: Bytes) -> Result<Self> {
        Self::new_with_options(mem_size, false)
    }

    /// Creates a no-allocation context with explicit byte-size newtype.
    pub fn new_no_alloc_bytes(mem_size: Bytes) -> Result<Self> {
        Self::new_with_options(mem_size, true)
    }

    fn new_with_options(mem_size: Bytes, no_alloc: bool) -> Result<Self> {
        let mem_size = NonZeroUsize::new(mem_size.get()).ok_or(Error::ZeroMemorySize)?;

        // ggml expects an externally managed buffer only when `mem_buffer` is
        // provided. For safe API simplicity we always let ggml own memory.
        let params = ffi::ggml_init_params {
            mem_size: mem_size.get(),
            mem_buffer: ptr::null_mut(),
            no_alloc,
        };

        let raw = unsafe { ffi::ggml_init(params) };
        let raw = NonNull::new(raw).ok_or_else(|| Error::null_pointer("ggml_init"))?;

        Ok(Self {
            raw,
            _not_send_sync: PhantomData,
        })
    }

    fn wrap_tensor<'ctx>(&'ctx self, raw: *mut ffi::ggml_tensor) -> Result<Tensor<'ctx>> {
        Ok(Tensor {
            raw: NonNull::new(raw)
                .ok_or_else(|| Error::null_pointer("tensor-producing ggml op"))?,
            _ctx: PhantomData,
        })
    }

    fn wrap_graph<'ctx>(&'ctx self, raw: *mut ffi::ggml_cgraph) -> Result<Graph<'ctx>> {
        Ok(Graph {
            raw: NonNull::new(raw).ok_or_else(|| Error::null_pointer("graph-producing ggml op"))?,
            _ctx: PhantomData,
        })
    }

    /// Returns a conservative memory estimate for `f32` matmul on context path.
    pub fn recommended_matmul_memory_f32(
        rows_a: usize,
        cols_a: usize,
        rows_b: usize,
        cols_b: usize,
    ) -> Result<usize> {
        Ok(Self::recommended_matmul_memory_f32_shapes_bytes(
            Shape2D::new(cols_a, rows_a),
            Shape2D::new(cols_b, rows_b),
        )?
        .get())
    }

    /// Returns a conservative memory estimate for `f32` matmul using shape newtypes.
    pub fn recommended_matmul_memory_f32_shapes(lhs: Shape2D, rhs: Shape2D) -> Result<usize> {
        Ok(Self::recommended_matmul_memory_f32_shapes_bytes(lhs, rhs)?.get())
    }

    /// Returns a conservative memory estimate as `Bytes`.
    pub fn recommended_matmul_memory_f32_shapes_bytes(lhs: Shape2D, rhs: Shape2D) -> Result<Bytes> {
        ensure_matmul_compatible(lhs.cols.get(), rhs.cols.get())?;

        // Keep this estimate conservative so examples and tests can reuse the
        // same helper without needing backend-specific tuning.
        let matrix_a_elements = lhs.rows.get().checked_mul_checked(lhs.cols.get())?;
        let matrix_b_elements = rhs.rows.get().checked_mul_checked(rhs.cols.get())?;
        let matrix_result_elements = lhs.rows.get().checked_mul_checked(rhs.rows.get())?;

        let matrix_a_bytes = matrix_a_elements.checked_mul_checked(type_size(Type::F32))?;
        let matrix_b_bytes = matrix_b_elements.checked_mul_checked(type_size(Type::F32))?;
        let matrix_result_bytes =
            matrix_result_elements.checked_mul_checked(type_size(Type::F32))?;

        let tensors_overhead = 3usize.checked_mul_checked(tensor_overhead_bytes())?;
        let graph_and_slack =
            graph_overhead_bytes().checked_add_checked(SIMPLE_CONTEXT_SLACK_BYTES)?;

        let total = matrix_a_bytes
            .checked_add_checked(matrix_b_bytes)?
            .checked_add_checked(matrix_result_bytes)?
            .checked_add_checked(tensors_overhead)?
            .checked_add_checked(graph_and_slack)?;
        Ok(Bytes::new(total))
    }

    /// Returns a conservative memory estimate for backend no-alloc matmul context.
    pub fn recommended_backend_matmul_memory_f32(
        rows_a: usize,
        cols_a: usize,
        rows_b: usize,
        cols_b: usize,
    ) -> Result<usize> {
        Ok(Self::recommended_backend_matmul_memory_f32_shapes_bytes(
            Shape2D::new(cols_a, rows_a),
            Shape2D::new(cols_b, rows_b),
        )?
        .get())
    }

    /// Returns a conservative backend memory estimate using shape newtypes.
    pub fn recommended_backend_matmul_memory_f32_shapes(
        lhs: Shape2D,
        rhs: Shape2D,
    ) -> Result<usize> {
        Ok(Self::recommended_backend_matmul_memory_f32_shapes_bytes(lhs, rhs)?.get())
    }

    /// Returns a conservative backend memory estimate as `Bytes`.
    pub fn recommended_backend_matmul_memory_f32_shapes_bytes(
        lhs: Shape2D,
        rhs: Shape2D,
    ) -> Result<Bytes> {
        ensure_matmul_compatible(lhs.cols.get(), rhs.cols.get())?;

        let tensors_overhead = 3usize.checked_mul_checked(tensor_overhead_bytes())?;
        let graph_and_slack =
            graph_overhead_bytes().checked_add_checked(SIMPLE_CONTEXT_SLACK_BYTES)?;
        let total = tensors_overhead.checked_add_checked(graph_and_slack)?;
        Ok(Bytes::new(total))
    }

    /// Allocates backend storage for all tensors currently owned by this context.
    pub fn allocate_tensors<'ctx>(&'ctx self, backend: &Backend) -> Result<BackendBuffer<'ctx>> {
        let raw =
            unsafe { ffi::ggml_backend_alloc_ctx_tensors(self.raw.as_ptr(), backend.raw.as_ptr()) };
        let raw = NonNull::new(raw)
            .ok_or_else(|| Error::null_pointer("ggml_backend_alloc_ctx_tensors"))?;

        // Buffer lifetime must stay tied to the originating context.
        Ok(BackendBuffer {
            raw,
            _ctx: PhantomData,
        })
    }

    /// Creates a 2D tensor with raw `usize` dimensions.
    pub fn new_tensor_2d(&self, ty: Type, cols: usize, rows: usize) -> Result<Tensor<'_>> {
        self.new_tensor_2d_shape(ty, Shape2D::new(cols, rows))
    }

    /// Creates a 2D tensor using semantic shape newtypes.
    pub fn new_tensor_2d_shape(&self, ty: Type, shape: Shape2D) -> Result<Tensor<'_>> {
        let cols = shape
            .cols
            .get()
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let rows = shape
            .rows
            .get()
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;

        let raw = unsafe { ffi::ggml_new_tensor_2d(self.raw.as_ptr(), ty.as_raw(), cols, rows) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_new_tensor_2d"))
    }

    pub fn new_f32_tensor_2d(&self, cols: usize, rows: usize) -> Result<Tensor<'_>> {
        self.new_tensor_2d(Type::F32, cols, rows)
    }

    pub fn new_f32_tensor_2d_shape(&self, shape: Shape2D) -> Result<Tensor<'_>> {
        self.new_tensor_2d_shape(Type::F32, shape)
    }

    /// Creates a 1D tensor with raw `usize` length.
    pub fn new_tensor_1d(&self, ty: Type, len: usize) -> Result<Tensor<'_>> {
        self.new_tensor_1d_len(ty, Length::new(len))
    }

    /// Creates a 1D tensor with semantic `Length`.
    pub fn new_tensor_1d_len(&self, ty: Type, len: Length) -> Result<Tensor<'_>> {
        let len = len
            .get()
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let raw = unsafe { ffi::ggml_new_tensor_1d(self.raw.as_ptr(), ty.as_raw(), len) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_new_tensor_1d"))
    }

    pub fn new_f32_tensor_1d(&self, len: usize) -> Result<Tensor<'_>> {
        self.new_tensor_1d_len(Type::F32, Length::new(len))
    }

    pub fn new_f32_tensor_1d_len(&self, len: Length) -> Result<Tensor<'_>> {
        self.new_tensor_1d_len(Type::F32, len)
    }

    pub fn new_i32_tensor_1d(&self, len: usize) -> Result<Tensor<'_>> {
        self.new_tensor_1d_len(Type::I32, Length::new(len))
    }

    pub fn new_i32_tensor_1d_len(&self, len: Length) -> Result<Tensor<'_>> {
        self.new_tensor_1d_len(Type::I32, len)
    }

    pub fn new_tensor_3d(
        &self,
        ty: Type,
        ne0: usize,
        ne1: usize,
        ne2: usize,
    ) -> Result<Tensor<'_>> {
        let ne0 = (ne0)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let ne1 = (ne1)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let ne2 = (ne2)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let raw = unsafe { ffi::ggml_new_tensor_3d(self.raw.as_ptr(), ty.as_raw(), ne0, ne1, ne2) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_new_tensor_3d"))
    }

    pub fn new_tensor_4d(
        &self,
        ty: Type,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
    ) -> Result<Tensor<'_>> {
        let ne0 = (ne0)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let ne1 = (ne1)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let ne2 = (ne2)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let ne3 = (ne3)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let raw =
            unsafe { ffi::ggml_new_tensor_4d(self.raw.as_ptr(), ty.as_raw(), ne0, ne1, ne2, ne3) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_new_tensor_4d"))
    }

    pub fn mul_mat<'ctx>(&'ctx self, a: &Tensor<'ctx>, b: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let (a_cols, _) = a.shape_2d()?;
        let (b_cols, _) = b.shape_2d()?;
        ensure_matmul_compatible(a_cols, b_cols)?;

        let raw = unsafe { ffi::ggml_mul_mat(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_mul_mat"))
    }

    pub fn add<'ctx>(&'ctx self, a: &Tensor<'ctx>, b: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_add(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_add"))
    }

    pub fn sub<'ctx>(&'ctx self, a: &Tensor<'ctx>, b: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_sub(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_sub"))
    }

    pub fn mul<'ctx>(&'ctx self, a: &Tensor<'ctx>, b: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_mul(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_mul"))
    }

    pub fn div<'ctx>(&'ctx self, a: &Tensor<'ctx>, b: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_div(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_div"))
    }

    pub fn silu<'ctx>(&'ctx self, a: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_silu(self.raw.as_ptr(), a.raw.as_ptr()) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_silu"))
    }

    pub fn rms_norm<'ctx>(&'ctx self, a: &Tensor<'ctx>, eps: f32) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_rms_norm(self.raw.as_ptr(), a.raw.as_ptr(), eps) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_rms_norm"))
    }

    pub fn scale<'ctx>(&'ctx self, a: &Tensor<'ctx>, scalar: f32) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_scale(self.raw.as_ptr(), a.raw.as_ptr(), scalar) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_scale"))
    }

    pub fn get_rows<'ctx>(
        &'ctx self,
        data: &Tensor<'ctx>,
        indices: &Tensor<'ctx>,
    ) -> Result<Tensor<'ctx>> {
        let raw = unsafe {
            ffi::ggml_get_rows(self.raw.as_ptr(), data.raw.as_ptr(), indices.raw.as_ptr())
        };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_get_rows"))
    }

    pub fn repeat<'ctx>(&'ctx self, a: &Tensor<'ctx>, b: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_repeat(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_repeat"))
    }

    pub fn cpy<'ctx>(&'ctx self, a: &Tensor<'ctx>, b: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_cpy(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_cpy"))
    }

    pub fn cont<'ctx>(&'ctx self, a: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_cont(self.raw.as_ptr(), a.raw.as_ptr()) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_cont"))
    }

    pub fn transpose<'ctx>(&'ctx self, a: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_transpose(self.raw.as_ptr(), a.raw.as_ptr()) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_transpose"))
    }

    pub fn reshape_2d<'ctx>(
        &'ctx self,
        a: &Tensor<'ctx>,
        ne0: usize,
        ne1: usize,
    ) -> Result<Tensor<'ctx>> {
        let ne0 = (ne0)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let ne1 = (ne1)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let raw = unsafe { ffi::ggml_reshape_2d(self.raw.as_ptr(), a.raw.as_ptr(), ne0, ne1) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_reshape_2d"))
    }

    pub fn reshape_3d<'ctx>(
        &'ctx self,
        a: &Tensor<'ctx>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
    ) -> Result<Tensor<'ctx>> {
        let ne0 = (ne0)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let ne1 = (ne1)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let ne2 = (ne2)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let raw = unsafe { ffi::ggml_reshape_3d(self.raw.as_ptr(), a.raw.as_ptr(), ne0, ne1, ne2) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_reshape_3d"))
    }

    pub fn view_1d<'ctx>(
        &'ctx self,
        a: &Tensor<'ctx>,
        ne0: usize,
        offset: usize,
    ) -> Result<Tensor<'ctx>> {
        let ne0 = (ne0)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let raw = unsafe { ffi::ggml_view_1d(self.raw.as_ptr(), a.raw.as_ptr(), ne0, offset) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_view_1d"))
    }

    pub fn view_2d<'ctx>(
        &'ctx self,
        a: &Tensor<'ctx>,
        ne0: usize,
        ne1: usize,
        row_stride: usize,
        offset: usize,
    ) -> Result<Tensor<'ctx>> {
        let ne0 = (ne0)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let ne1 = (ne1)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
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
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_view_2d"))
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
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_permute"))
    }

    pub fn diag_mask_inf<'ctx>(&'ctx self, a: &Tensor<'ctx>, n_past: i32) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_diag_mask_inf(self.raw.as_ptr(), a.raw.as_ptr(), n_past) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_diag_mask_inf"))
    }

    pub fn soft_max<'ctx>(&'ctx self, a: &Tensor<'ctx>) -> Result<Tensor<'ctx>> {
        let raw = unsafe { ffi::ggml_soft_max(self.raw.as_ptr(), a.raw.as_ptr()) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_soft_max"))
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
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_soft_max_ext"))
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
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_rope_ext"))
    }

    pub fn new_graph(&self) -> Result<Graph<'_>> {
        let raw = unsafe { ffi::ggml_new_graph(self.raw.as_ptr()) };
        self.wrap_graph(raw)
            .map_err(|error| error.with_context("ggml_new_graph"))
    }

    /// Wraps a tensor into expression form for operator-based composition.
    pub fn expr<'ctx>(&'ctx self, tensor: Tensor<'ctx>) -> TensorExpr<'ctx> {
        TensorExpr { ctx: self, tensor }
    }

    /// Executes a graph through the legacy context execution path.
    pub fn compute<'ctx>(&'ctx self, graph: &mut Graph<'ctx>, n_threads: usize) -> Result<()> {
        self.compute_with_threads(graph, ThreadCount::new(n_threads))
    }

    /// Executes a graph through the legacy context path with typed thread count.
    pub fn compute_with_threads<'ctx>(
        &'ctx self,
        graph: &mut Graph<'ctx>,
        n_threads: ThreadCount,
    ) -> Result<()> {
        let n_threads = {
            let n_threads: c_int = n_threads
                .get()
                .try_into_checked()
                .map_err(|source| Error::int_conversion("thread count", source))?;
            if n_threads <= 0 {
                return Err(Error::InvalidThreadCount(n_threads as usize));
            }
            n_threads
        };

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

/// Thin safe handle to a ggml tensor allocated from `Context`.
#[derive(Clone, Copy)]
pub struct Tensor<'ctx> {
    raw: NonNull<ffi::ggml_tensor>,
    _ctx: PhantomData<&'ctx Context>,
}

impl<'ctx> Tensor<'ctx> {
    /// Returns total element count (`ggml_nelements`).
    pub fn element_count(&self) -> Result<usize> {
        (unsafe { ffi::ggml_nelements(self.raw.as_ptr()) })
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))
    }

    /// Returns row count (`ggml_nrows`).
    pub fn row_count(&self) -> Result<usize> {
        (unsafe { ffi::ggml_nrows(self.raw.as_ptr()) })
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))
    }

    pub fn col_count(&self) -> Result<usize> {
        let rows = self.row_count()?;
        let elements = self.element_count()?;

        if rows == 0 || elements % rows != 0 {
            return Err(Error::UnexpectedShape);
        }

        Ok(elements / rows)
    }

    /// Returns `(cols, rows)` for tensors representable as 2D.
    pub fn shape_2d(&self) -> Result<(usize, usize)> {
        let shape = self.shape()?;
        Ok((shape.cols.get(), shape.rows.get()))
    }

    /// Returns semantic shape newtypes for 2D-compatible tensors.
    pub fn shape(&self) -> Result<Shape2D> {
        Ok(Shape2D {
            cols: Cols::new(self.col_count()?),
            rows: Rows::new(self.row_count()?),
        })
    }

    pub fn nbytes(&self) -> usize {
        unsafe { ffi::ggml_nbytes(self.raw.as_ptr()) }
    }

    fn expected_nbytes_for<T: BackendElement>(&self) -> Result<usize> {
        let elements = self.element_count()?;
        elements.checked_mul_checked(std::mem::size_of::<T>())
    }

    /// Assigns a debug name to the tensor.
    pub fn set_name(&self, name: &str) -> Result<()> {
        let name = CString::new(name)?;
        let raw = unsafe { ffi::ggml_set_name(self.raw.as_ptr(), name.as_ptr()) };
        let _ = NonNull::new(raw).ok_or_else(|| Error::null_pointer("ggml_set_name"))?;
        Ok(())
    }

    /// Reads a tensor debug name.
    pub fn name(&self) -> Result<String> {
        let ptr = unsafe { ffi::ggml_get_name(self.raw.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::null_pointer("ggml_get_name"));
        }
        let cstr = unsafe { CStr::from_ptr(ptr) };
        Ok(cstr.to_str()?.to_owned())
    }

    /// Writes host `f32` values through context tensor APIs.
    pub fn set_f32(&self, values: &[f32]) -> Result<()> {
        let expected = self.element_count()?;
        if values.len() != expected {
            return Err(Error::LengthMismatch {
                expected,
                actual: values.len(),
            });
        }

        for (index, value) in values.iter().copied().enumerate() {
            let index = (index)
                .try_into_checked()
                .map_err(|source| Error::int_conversion("tensor index", source))?;
            unsafe {
                ffi::ggml_set_f32_1d(self.raw.as_ptr(), index, value);
            }
        }

        Ok(())
    }

    pub fn set_f32_backend(&self, values: &[f32]) -> Result<()> {
        self.set_backend_slice(values)
    }

    /// Writes host values through backend tensor APIs.
    pub fn set_backend_slice<T: BackendElement>(&self, values: &[T]) -> Result<()> {
        let expected = self.element_count()?;
        if values.len() != expected {
            return Err(Error::LengthMismatch {
                expected,
                actual: values.len(),
            });
        }

        // ggml backend APIs are byte-oriented; enforce exact element-size match
        // up front to avoid silent reinterpretation.
        let expected_nbytes = self.expected_nbytes_for::<T>()?;
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
        self.set_backend_slice(values)
    }

    /// Reads a single `f32` element with bounds checking.
    pub fn get_f32(&self, index: usize) -> Result<f32> {
        self.get_f32_at(TensorIndex::new(index))
    }

    pub fn get_f32_at(&self, index: TensorIndex) -> Result<f32> {
        let index = index.get();
        let len = self.element_count()?;
        if index >= len {
            return Err(Error::IndexOutOfBounds { index, len });
        }

        let index = (index)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor index", source))?;
        Ok(unsafe { ffi::ggml_get_f32_1d(self.raw.as_ptr(), index) })
    }

    /// Reads all values through context tensor APIs.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let len = self.element_count()?;
        let mut out = Vec::with_capacity(len);

        for index in 0..len {
            let index = (index)
                .try_into_checked()
                .map_err(|source| Error::int_conversion("tensor index", source))?;
            out.push(unsafe { ffi::ggml_get_f32_1d(self.raw.as_ptr(), index) });
        }

        Ok(out)
    }

    pub fn to_vec_f32_backend(&self) -> Result<Vec<f32>> {
        self.to_vec_backend()
    }

    pub fn to_vec_i32_backend(&self) -> Result<Vec<i32>> {
        self.to_vec_backend()
    }

    /// Reads all values through backend tensor APIs.
    pub fn to_vec_backend<T: BackendElement>(&self) -> Result<Vec<T>> {
        let len = self.element_count()?;
        let expected_nbytes = self.expected_nbytes_for::<T>()?;
        let actual_nbytes = self.nbytes();
        if expected_nbytes != actual_nbytes {
            return Err(Error::UnexpectedTensorByteSize {
                expected: expected_nbytes,
                actual: actual_nbytes,
            });
        }

        let mut out = vec![T::default(); len];
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

/// Thin safe handle to a ggml computation graph.
pub struct Graph<'ctx> {
    raw: NonNull<ffi::ggml_cgraph>,
    _ctx: PhantomData<&'ctx Context>,
}

impl<'ctx> Graph<'ctx> {
    /// Expands the graph forward pass using the provided terminal tensor.
    pub fn build_forward_expand(&mut self, tensor: &Tensor<'ctx>) {
        unsafe {
            ffi::ggml_build_forward_expand(self.raw.as_ptr(), tensor.raw.as_ptr());
        }
    }

    /// Returns number of nodes currently in the graph.
    pub fn node_count(&self) -> i32 {
        unsafe { ffi::ggml_graph_n_nodes(self.raw.as_ptr()) }
    }

    /// Returns a node by index (supports negative indexing from the end).
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
        let raw = NonNull::new(raw).ok_or_else(|| Error::null_pointer("ggml_graph_node"))?;

        Ok(Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    /// Returns the last graph node.
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

fn ensure_matmul_compatible(lhs_cols: usize, rhs_cols: usize) -> Result<()> {
    if lhs_cols == rhs_cols {
        Ok(())
    } else {
        Err(Error::IncompatibleMatmulShapes { lhs_cols, rhs_cols })
    }
}
