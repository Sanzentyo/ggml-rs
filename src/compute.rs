//! Core safe wrappers for ggml context, backend, tensor, and graph objects.

use crate::ffi;
use crate::num_ext::{CheckedFieldOps, TryIntoChecked};
use crate::tensor_expr::HostElement;
use crate::{
    BackendDeviceType, BackendElement, BackendKind, Bytes, Cols, ComputeStatus, Dims, Error,
    GgmlElement, Length, Result, RopeExtParams, Rows, Shape2D, Shape3D, Shape4D, TensorExpr,
    TensorIndex, ThreadCount, Type,
};
use num_traits::NumCast;
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::os::raw::c_int;
use std::ptr::{self, NonNull};

const SIMPLE_CONTEXT_SLACK_BYTES: usize = 1024;
const CONTIGUOUS_BULK_COPY_MIN_ELEMS: usize = 256;

/// Initializes ggml global timing infrastructure.
pub fn init_timing() {
    unsafe {
        ffi::ggml_time_init();
    }
}

/// Returns the byte size of a single element for the given type.
pub fn type_size(ty: Type) -> usize {
    unsafe { ffi::ggml_type_size(ty.as_raw() as _) }
}

fn resolve_ggml_type(ggml_type_raw: c_int) -> Result<ffi::ggml_type> {
    if !(0..ffi::GGML_TYPE_COUNT).contains(&ggml_type_raw) {
        return Err(Error::UnsupportedType(ggml_type_raw));
    }
    Ok(ggml_type_raw as ffi::ggml_type)
}

/// Returns the number of scalar values represented by a tensor payload.
pub fn tensor_element_count(ggml_type_raw: c_int, payload_bytes: usize) -> Result<usize> {
    let ggml_type = resolve_ggml_type(ggml_type_raw)?;
    let block_size = unsafe { ffi::ggml_blck_size(ggml_type) }
        .try_into_checked()
        .map_err(|source: std::num::TryFromIntError| {
            Error::int_conversion("ggml_blck_size", source)
        })?;
    let type_size = unsafe { ffi::ggml_type_size(ggml_type) };
    if block_size == 0 || type_size == 0 {
        return Err(Error::UnsupportedType(ggml_type_raw));
    }
    if !payload_bytes.is_multiple_of(type_size) {
        let expected = (payload_bytes / type_size)
            .checked_mul_checked(type_size)
            .map_err(|source| source.with_context("tensor_element_count"))?;
        return Err(Error::UnexpectedTensorByteSize {
            expected,
            actual: payload_bytes,
        });
    }
    let block_count = payload_bytes
        .checked_div(type_size)
        .ok_or(Error::Overflow)?;
    block_count
        .checked_mul_checked(block_size)
        .map_err(|source| source.with_context("tensor_element_count"))
}

/// Decodes GGML tensor payload bytes into `f32` values via GGML type traits.
fn decode_tensor_data_to_float(ggml_type_raw: c_int, payload: &[u8]) -> Result<Vec<f32>> {
    let ggml_type = resolve_ggml_type(ggml_type_raw)?;
    let element_count = tensor_element_count(ggml_type_raw, payload.len())
        .map_err(|source| source.with_context("decode_tensor_data_to_float"))?;

    let type_traits =
        NonNull::new(unsafe { ffi::ggml_get_type_traits(ggml_type) as *mut ffi::ggml_type_traits })
            .ok_or_else(|| Error::null_pointer("ggml_get_type_traits"))?;
    let to_float =
        unsafe { type_traits.as_ref().to_float }.ok_or(Error::UnsupportedType(ggml_type_raw))?;

    if element_count == 0 {
        return Ok(Vec::new());
    }

    let mut out = Vec::with_capacity(element_count);

    if unsafe { ffi::ggml_is_quantized(ggml_type) } {
        unsafe { ffi::ggml_quantize_init(ggml_type) };
    }

    let k = element_count
        .try_into_checked()
        .map_err(|source| Error::int_conversion("decode_tensor_data_to_float(k)", source))?;
    unsafe {
        // SAFETY:
        // - `out` is newly allocated with capacity `element_count`.
        // - `to_float` writes exactly `k == element_count` `f32` values into `dst`.
        // - We set the vector length only after `to_float` completes.
        let dst = out.spare_capacity_mut().as_mut_ptr().cast::<f32>();
        to_float(payload.as_ptr().cast(), dst, k);
        out.set_len(element_count);
    }
    Ok(out)
}

/// Decodes GGML tensor payload bytes into caller-selected element type `T`.
///
/// For matching plain tensor types (`f32`, `i32`) this performs a direct copy.
/// For other GGML element types (including quantized payloads), values are
/// decoded through GGML's `to_float` conversion and then cast to `T`.
pub fn decode_tensor_data_to<T>(ggml_type_raw: c_int, payload: &[u8]) -> Result<Vec<T>>
where
    T: GgmlElement + NumCast,
{
    let element_count = tensor_element_count(ggml_type_raw, payload.len())?;
    let expected_nbytes = element_count.checked_mul_checked(std::mem::size_of::<T>())?;

    if element_count == 0 {
        return Ok(Vec::new());
    }

    if ggml_type_raw == T::GGML_TYPE as c_int && payload.len() == expected_nbytes {
        let mut out = Vec::with_capacity(element_count);
        unsafe {
            // SAFETY:
            // - `out` has capacity for `element_count` elements.
            // - `payload.len()` matches exactly `element_count * size_of::<T>()`.
            // - Source payload bytes are read-only and copied into owned `out` storage.
            let dst = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
            ptr::copy_nonoverlapping(payload.as_ptr().cast::<T>(), dst, element_count);
            out.set_len(element_count);
        }
        return Ok(out);
    }

    let decoded = decode_tensor_data_to_float(ggml_type_raw, payload)?;
    let mut out = Vec::with_capacity(decoded.len());
    for value in decoded {
        let casted = NumCast::from(value).ok_or(Error::UnsupportedType(ggml_type_raw))?;
        out.push(casted);
    }
    Ok(out)
}

/// Returns ggml's internal per-tensor metadata overhead in bytes.
pub fn tensor_overhead_bytes() -> usize {
    unsafe { ffi::ggml_tensor_overhead() }
}

/// Returns ggml's internal per-graph metadata overhead in bytes.
pub fn graph_overhead_bytes() -> usize {
    unsafe { ffi::ggml_graph_overhead() }
}

/// Creates a scoped context and executes the provided closure.
///
/// The higher-ranked lifetime keeps all `Tensor<'ctx>` values scoped to the
/// closure body.
pub fn with_context<R>(
    mem: Bytes,
    f: impl for<'ctx> FnOnce(&'ctx Context) -> Result<R>,
) -> Result<R> {
    let ctx = Context::new_bytes(mem)?;
    f(&ctx)
}

/// Creates a scoped no-allocation context and executes the provided closure.
///
/// This is useful for backend-only execution paths that allocate tensors via
/// backend buffers.
pub fn with_no_alloc_context<R>(
    mem: Bytes,
    f: impl for<'ctx> FnOnce(&'ctx Context) -> Result<R>,
) -> Result<R> {
    let ctx = Context::new_no_alloc_bytes(mem)?;
    f(&ctx)
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

    fn init_by_type(device_type: BackendDeviceType) -> Option<NonNull<ffi::ggml_backend>> {
        NonNull::new(unsafe {
            ffi::ggml_backend_init_by_type(device_type.as_raw() as _, ptr::null())
        })
    }

    fn init_cpu_backend() -> Result<NonNull<ffi::ggml_backend>> {
        Self::init_by_type(BackendDeviceType::Cpu)
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

            let device_type_raw = unsafe { ffi::ggml_backend_dev_type(device.as_ptr()) };
            let Some(device_type) = BackendDeviceType::from_raw(device_type_raw as c_int) else {
                continue;
            };
            if !device_type.is_gpu_like() {
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
        for device_type in [BackendDeviceType::IntegratedGpu, BackendDeviceType::Gpu] {
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

        if ComputeStatus::is_success(status) {
            Ok(())
        } else {
            Err(Error::ComputeFailed(status))
        }
    }

    /// Synchronizes queued backend work before host-side timing or readback.
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            ffi::ggml_backend_synchronize(self.raw.as_ptr());
        }
        Ok(())
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

    fn recommended_matmul_memory_impl(
        lhs: Shape2D,
        rhs: Shape2D,
        element_type: Type,
        backend_only: bool,
    ) -> Result<Bytes> {
        ensure_matmul_compatible(lhs.cols.get(), rhs.cols.get())?;

        let tensors_overhead = 3usize.checked_mul_checked(tensor_overhead_bytes())?;
        let graph_and_slack =
            graph_overhead_bytes().checked_add_checked(SIMPLE_CONTEXT_SLACK_BYTES)?;

        if backend_only {
            let total = tensors_overhead.checked_add_checked(graph_and_slack)?;
            return Ok(Bytes::new(total));
        }

        // Keep this estimate conservative so examples and tests can reuse the
        // same helper without needing backend-specific tuning.
        let matrix_a_elements = lhs.rows.get().checked_mul_checked(lhs.cols.get())?;
        let matrix_b_elements = rhs.rows.get().checked_mul_checked(rhs.cols.get())?;
        let matrix_result_elements = lhs.rows.get().checked_mul_checked(rhs.rows.get())?;
        let element_size = type_size(element_type);

        let matrix_a_bytes = matrix_a_elements.checked_mul_checked(element_size)?;
        let matrix_b_bytes = matrix_b_elements.checked_mul_checked(element_size)?;
        let matrix_result_bytes = matrix_result_elements.checked_mul_checked(element_size)?;

        let total = matrix_a_bytes
            .checked_add_checked(matrix_b_bytes)?
            .checked_add_checked(matrix_result_bytes)?
            .checked_add_checked(tensors_overhead)?
            .checked_add_checked(graph_and_slack)?;
        Ok(Bytes::new(total))
    }

    /// Returns a conservative memory estimate for typed matmul on context path.
    pub fn recommended_matmul_memory<T: GgmlElement>(lhs: Shape2D, rhs: Shape2D) -> Result<Bytes> {
        Self::recommended_matmul_memory_impl(lhs, rhs, T::GGML_TYPE, false)
    }

    /// Returns a conservative memory estimate for typed matmul on backend path.
    pub fn recommended_backend_matmul_memory<T: GgmlElement>(
        lhs: Shape2D,
        rhs: Shape2D,
    ) -> Result<Bytes> {
        Self::recommended_matmul_memory_impl(lhs, rhs, T::GGML_TYPE, true)
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

    /// Creates a tensor for ranks `1..=4` from generic dimensions.
    pub fn new_tensor<const N: usize>(&self, ty: Type, dims: Dims<N>) -> Result<Tensor<'_>> {
        let dims = *dims.as_array();
        let dim = |index: usize| -> Result<_> {
            dims[index]
                .try_into_checked()
                .map_err(|source| Error::int_conversion("tensor dimension", source))
        };

        let raw = match N {
            1 => unsafe { ffi::ggml_new_tensor_1d(self.raw.as_ptr(), ty.as_raw() as _, dim(0)?) },
            2 => unsafe {
                ffi::ggml_new_tensor_2d(self.raw.as_ptr(), ty.as_raw() as _, dim(0)?, dim(1)?)
            },
            3 => unsafe {
                ffi::ggml_new_tensor_3d(
                    self.raw.as_ptr(),
                    ty.as_raw() as _,
                    dim(0)?,
                    dim(1)?,
                    dim(2)?,
                )
            },
            4 => unsafe {
                ffi::ggml_new_tensor_4d(
                    self.raw.as_ptr(),
                    ty.as_raw() as _,
                    dim(0)?,
                    dim(1)?,
                    dim(2)?,
                    dim(3)?,
                )
            },
            _ => return Err(Error::UnsupportedRank(N)),
        };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_new_tensor"))
    }

    /// Creates a 2D tensor using semantic shape newtypes.
    pub fn new_tensor_2d_shape(&self, ty: Type, shape: Shape2D) -> Result<Tensor<'_>> {
        self.new_tensor(ty, shape.dims())
    }

    pub fn new_f32_tensor_2d_shape(&self, shape: Shape2D) -> Result<Tensor<'_>> {
        self.new_tensor_2d_shape(Type::F32, shape)
    }

    /// Creates a 1D tensor with semantic `Length`.
    pub fn new_tensor_1d_len(&self, ty: Type, len: Length) -> Result<Tensor<'_>> {
        self.new_tensor(ty, Dims::new([len.get()]))
    }

    pub fn new_f32_tensor_1d_len(&self, len: Length) -> Result<Tensor<'_>> {
        self.new_tensor_1d_len(Type::F32, len)
    }

    pub fn new_i32_tensor_1d_len(&self, len: Length) -> Result<Tensor<'_>> {
        self.new_tensor_1d_len(Type::I32, len)
    }

    /// Creates a 3D tensor from semantic dimensions.
    pub fn new_tensor_3d_shape(&self, ty: Type, shape: Shape3D) -> Result<Tensor<'_>> {
        self.new_tensor(ty, shape.dims())
    }

    /// Creates a 4D tensor from semantic dimensions.
    pub fn new_tensor_4d_shape(&self, ty: Type, shape: Shape4D) -> Result<Tensor<'_>> {
        self.new_tensor(ty, shape.dims())
    }

    pub fn new_tensor_3d(
        &self,
        ty: Type,
        ne0: usize,
        ne1: usize,
        ne2: usize,
    ) -> Result<Tensor<'_>> {
        self.new_tensor_3d_shape(ty, Shape3D::new(ne0, ne1, ne2))
    }

    pub fn new_tensor_4d(
        &self,
        ty: Type,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
    ) -> Result<Tensor<'_>> {
        self.new_tensor_4d_shape(ty, Shape4D::new(ne0, ne1, ne2, ne3))
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

    /// Concatenates two 2D tensors along the selected dimension (`0` = cols, `1` = rows).
    pub fn concat<'ctx>(
        &'ctx self,
        a: &Tensor<'ctx>,
        b: &Tensor<'ctx>,
        dim: usize,
    ) -> Result<Tensor<'ctx>> {
        let (a_cols, a_rows) = a.shape_2d()?;
        let (b_cols, b_rows) = b.shape_2d()?;
        match dim {
            0 if a_rows == b_rows => {}
            1 if a_cols == b_cols => {}
            _ => return Err(Error::UnexpectedShape),
        }

        let dim = (dim)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("concat dimension", source))?;
        let raw =
            unsafe { ffi::ggml_concat(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr(), dim) };
        self.wrap_tensor(raw)
            .map_err(|error| error.with_context("ggml_concat"))
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

        if ComputeStatus::is_success(status) {
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
    pub(crate) fn raw_ptr(&self) -> *mut ffi::ggml_tensor {
        self.raw.as_ptr()
    }

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

    /// Returns tensor rank reported by ggml.
    pub fn rank(&self) -> Result<usize> {
        (unsafe { ffi::ggml_n_dims(self.raw.as_ptr()) })
            .try_into_checked()
            .map_err(|source| Error::int_conversion("ggml_n_dims", source))
    }

    /// Returns dynamic tensor dimensions in ggml order (`ne0..ne{rank-1}`).
    pub fn shape_nd(&self) -> Result<Vec<usize>> {
        let rank = self.rank()?;
        let ne = unsafe { self.raw.as_ref().ne };
        (0..rank)
            .map(|index| {
                ne[index]
                    .try_into_checked()
                    .map_err(|source| Error::int_conversion("tensor dimension", source))
            })
            .collect()
    }

    /// Returns fixed-rank dimensions when tensor rank matches `N`.
    pub fn dims<const N: usize>(&self) -> Result<Dims<N>> {
        let rank = self.rank()?;
        if rank != N {
            return Err(Error::UnsupportedRank(rank));
        }
        let ne = unsafe { self.raw.as_ref().ne };
        let mut dims = [0usize; N];
        for (index, dst) in dims.iter_mut().enumerate() {
            *dst = ne[index]
                .try_into_checked()
                .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        }
        Ok(Dims::new(dims))
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

    /// Returns semantic 3D shape when rank is exactly 3.
    pub fn shape_3d(&self) -> Result<Shape3D> {
        Ok(Shape3D::from(self.dims::<3>()?))
    }

    /// Returns semantic 4D shape when rank is exactly 4.
    pub fn shape_4d(&self) -> Result<Shape4D> {
        Ok(Shape4D::from(self.dims::<4>()?))
    }

    pub fn nbytes(&self) -> usize {
        unsafe { ffi::ggml_nbytes(self.raw.as_ptr()) }
    }

    fn is_contiguous(&self) -> bool {
        unsafe { ffi::ggml_is_contiguous(self.raw.as_ptr()) as i32 != 0 }
    }

    fn contiguous_data_ptr<T>(&self) -> Option<NonNull<T>> {
        if !self.is_contiguous() {
            return None;
        }
        let raw = unsafe { ffi::ggml_get_data(self.raw.as_ptr()) }.cast::<T>();
        NonNull::new(raw)
    }

    fn expected_nbytes_for<T: BackendElement>(&self) -> Result<usize> {
        let elements = self.element_count()?;
        elements.checked_mul_checked(std::mem::size_of::<T>())
    }

    fn ensure_backend_slice_compatible<T: BackendElement>(&self) -> Result<usize> {
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
        Ok(expected_nbytes)
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

    /// Writes host values using typed tensor I/O dispatch.
    pub fn write_data<T: GgmlElement>(&self, values: &[T]) -> Result<()> {
        T::write_data(self, values)
    }

    /// Writes backend values with an element type inferred from the slice.
    pub fn write_data_backend<T: BackendElement>(&self, values: &[T]) -> Result<()> {
        self.write_backend_slice(values)
    }

    /// Writes backend values at the provided element offset.
    pub fn write_data_backend_at<T: BackendElement>(
        &self,
        element_offset: usize,
        values: &[T],
    ) -> Result<()> {
        self.write_backend_slice_at(element_offset, values)
    }

    /// Reads all host values using typed tensor I/O dispatch.
    pub fn read_data<T: GgmlElement>(&self) -> Result<Vec<T>> {
        T::read_data(self)
    }

    /// Reads all backend values for the requested element type.
    pub fn read_data_backend<T: BackendElement>(&self) -> Result<Vec<T>> {
        self.read_backend_vec()
    }

    /// Reads one element using typed tensor I/O dispatch.
    pub fn get_data<T: GgmlElement>(&self, index: TensorIndex) -> Result<T> {
        T::get_data(self, index)
    }

    /// Writes host values through context tensor APIs.
    pub(crate) fn write_host_data<T: HostElement>(&self, values: &[T]) -> Result<()> {
        let expected = self.element_count()?;
        if values.len() != expected {
            return Err(Error::LengthMismatch {
                expected,
                actual: values.len(),
            });
        }

        if expected >= CONTIGUOUS_BULK_COPY_MIN_ELEMS {
            let expected_nbytes = expected.checked_mul_checked(std::mem::size_of::<T>())?;
            if expected_nbytes == self.nbytes()
                && let Some(dst) = self.contiguous_data_ptr::<T>()
            {
                unsafe {
                    // SAFETY:
                    // - `dst` points to contiguous tensor storage returned by ggml.
                    // - Byte-size compatibility was validated above, so `expected` elements fit.
                    // - Source and destination do not overlap (`values` is caller-owned host slice).
                    ptr::copy_nonoverlapping(values.as_ptr(), dst.as_ptr(), expected);
                }
                return Ok(());
            }
        }

        for (index, value) in values.iter().copied().enumerate() {
            let index = (index)
                .try_into_checked()
                .map_err(|source| Error::int_conversion("tensor index", source))?;
            T::set_1d_raw(self.raw.as_ptr(), index, value);
        }

        Ok(())
    }

    /// Writes host values through backend tensor APIs.
    fn write_backend_slice<T: BackendElement>(&self, values: &[T]) -> Result<()> {
        let expected = self.element_count()?;
        if values.len() != expected {
            return Err(Error::LengthMismatch {
                expected,
                actual: values.len(),
            });
        }

        let expected_nbytes = self.ensure_backend_slice_compatible::<T>()?;

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

    /// Writes a backend slice into a contiguous tensor region.
    fn write_backend_slice_at<T: BackendElement>(
        &self,
        element_offset: usize,
        values: &[T],
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        let tensor_len = self.element_count()?;
        let end = element_offset
            .checked_add(values.len())
            .ok_or(Error::Overflow)?;
        if end > tensor_len {
            return Err(Error::IndexOutOfBounds {
                index: end.saturating_sub(1),
                len: tensor_len,
            });
        }

        self.ensure_backend_slice_compatible::<T>()?;

        let byte_offset = element_offset.checked_mul_checked(std::mem::size_of::<T>())?;
        let write_nbytes = values.len().checked_mul_checked(std::mem::size_of::<T>())?;
        unsafe {
            ffi::ggml_backend_tensor_set(
                self.raw.as_ptr(),
                values.as_ptr().cast(),
                byte_offset,
                write_nbytes,
            );
        }

        Ok(())
    }

    /// Reads one host element with bounds checking.
    pub(crate) fn read_host_at<T: HostElement>(&self, index: TensorIndex) -> Result<T> {
        let index = index.get();
        let len = self.element_count()?;
        if index >= len {
            return Err(Error::IndexOutOfBounds { index, len });
        }

        let index = (index)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor index", source))?;
        Ok(T::get_1d_raw(self.raw.as_ptr(), index))
    }

    /// Reads all host values through context tensor APIs.
    pub(crate) fn read_host_data<T: HostElement>(&self) -> Result<Vec<T>> {
        let len = self.element_count()?;
        if len >= CONTIGUOUS_BULK_COPY_MIN_ELEMS {
            let expected_nbytes = len.checked_mul_checked(std::mem::size_of::<T>())?;
            if expected_nbytes == self.nbytes()
                && let Some(src) = self.contiguous_data_ptr::<T>()
            {
                let mut out = Vec::with_capacity(len);
                unsafe {
                    // SAFETY:
                    // - `src` points to contiguous tensor storage of `len` elements.
                    // - `out` has capacity for `len` elements; `copy_nonoverlapping` initializes them.
                    // - We set length after initializing all elements.
                    ptr::copy_nonoverlapping(src.as_ptr(), out.as_mut_ptr(), len);
                    out.set_len(len);
                }
                return Ok(out);
            }
        }

        let mut out = Vec::with_capacity(len);

        for index in 0..len {
            let index = (index)
                .try_into_checked()
                .map_err(|source| Error::int_conversion("tensor index", source))?;
            out.push(T::get_1d_raw(self.raw.as_ptr(), index));
        }

        Ok(out)
    }

    /// Reads all values through backend tensor APIs.
    fn read_backend_vec<T: BackendElement>(&self) -> Result<Vec<T>> {
        let len = self.element_count()?;
        let expected_nbytes = self.expected_nbytes_for::<T>()?;
        let actual_nbytes = self.nbytes();
        if expected_nbytes != actual_nbytes {
            return Err(Error::UnexpectedTensorByteSize {
                expected: expected_nbytes,
                actual: actual_nbytes,
            });
        }

        let mut out = Vec::<T>::with_capacity(len);
        unsafe {
            // SAFETY:
            // - `out` has capacity for `len` elements matching `expected_nbytes`.
            // - ggml writes exactly `expected_nbytes` bytes into the destination buffer.
            // - Length is set only after the backend write completes.
            ffi::ggml_backend_tensor_get(
                self.raw.as_ptr(),
                out.as_mut_ptr().cast(),
                0,
                expected_nbytes,
            );
            out.set_len(len);
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
