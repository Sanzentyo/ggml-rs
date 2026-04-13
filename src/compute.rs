//! Core safe wrappers for ggml context, backend, tensor, and graph objects.

use crate::ffi;
use crate::num_ext::{CheckedFieldOps, TryIntoChecked};
use crate::{
    BackendDeviceType, BackendElement, BackendKind, Bytes, Cols, ComputeStatus, Dims, Error,
    GgmlElement, Length, Result, RopeExtParams, Rows, Shape2D, Shape3D, Shape4D, TensorExpr,
    TensorIndex, ThreadCount, Type,
};
use num_traits::NumCast;
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::os::raw::c_int;
use std::ptr::{self, NonNull};

const SIMPLE_CONTEXT_SLACK_BYTES: usize = 1024;
const CONTIGUOUS_BULK_COPY_MIN_ELEMS: usize = 256;

/// Validates that a tensor is contiguous and its element count matches the
/// product of the target dimensions. This prevents the C library from
/// aborting on invalid reshape inputs.
fn validate_reshape_source<T: GgmlElement>(a: &Tensor<'_, T>, target_dims: &[usize]) -> Result<()> {
    let is_contiguous = unsafe { ffi::ggml_is_contiguous(a.raw.as_ptr()) as i32 != 0 };
    if !is_contiguous {
        return Err(Error::NotContiguous);
    }
    let target_count = target_dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(Error::Overflow)?;
    let source_count = a.element_count()?;
    if target_count != source_count {
        return Err(Error::LengthMismatch {
            expected: source_count,
            actual: target_count,
        });
    }
    Ok(())
}

/// Validates that a view's maximum addressed byte stays within the source
/// tensor's byte size.
///
/// `outer_dims` contains `(ne_i, nb_i)` pairs for dimensions 1..N (outer to
/// inner). `inner_row_bytes` is `ne0 * element_size`.
fn validate_view_extent<T: GgmlElement>(
    source: &Tensor<'_, T>,
    offset: usize,
    outer_dims: &[(usize, usize)],
    inner_row_bytes: usize,
) -> Result<()> {
    let mut extent = inner_row_bytes;
    for &(ne, nb) in outer_dims {
        let stride_span = ne
            .saturating_sub(1)
            .checked_mul(nb)
            .ok_or(Error::Overflow)?;
        extent = extent.checked_add(stride_span).ok_or(Error::Overflow)?;
    }
    let max_byte = offset.checked_add(extent).ok_or(Error::Overflow)?;
    let source_size = source.nbytes();
    if max_byte > source_size {
        return Err(Error::ViewOutOfBounds {
            offset,
            extent,
            source_size,
        });
    }
    Ok(())
}

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

fn resolve_ggml_type(ty: Type) -> Result<ffi::ggml_type> {
    let raw = ty.as_raw();
    if ty.is_unknown() || !(0..ffi::GGML_TYPE_COUNT).contains(&raw) {
        return Err(Error::UnsupportedType(raw));
    }
    Ok(raw as ffi::ggml_type)
}

/// Returns the number of scalar values represented by a tensor payload.
pub fn tensor_element_count(ty: Type, payload_bytes: usize) -> Result<usize> {
    let ggml_type = resolve_ggml_type(ty)?;
    let block_size = unsafe { ffi::ggml_blck_size(ggml_type) }
        .try_into_checked()
        .map_err(|source: std::num::TryFromIntError| {
            Error::int_conversion("ggml_blck_size", source)
        })?;
    let type_size = unsafe { ffi::ggml_type_size(ggml_type) };
    if block_size == 0 || type_size == 0 {
        return Err(Error::UnsupportedType(ty.as_raw()));
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
fn decode_tensor_data_to_float(ty: Type, payload: &[u8]) -> Result<Vec<f32>> {
    let ggml_type = resolve_ggml_type(ty)?;
    let element_count = tensor_element_count(ty, payload.len())
        .map_err(|source| source.with_context("decode_tensor_data_to_float"))?;

    let type_traits =
        NonNull::new(unsafe { ffi::ggml_get_type_traits(ggml_type) as *mut ffi::ggml_type_traits })
            .ok_or_else(|| Error::null_pointer("ggml_get_type_traits"))?;
    let to_float =
        unsafe { type_traits.as_ref().to_float }.ok_or(Error::UnsupportedType(ty.as_raw()))?;

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
pub fn decode_tensor_data_to<T>(ty: Type, payload: &[u8]) -> Result<Vec<T>>
where
    T: GgmlElement + NumCast,
{
    let element_count = tensor_element_count(ty, payload.len())?;
    let expected_nbytes = element_count.checked_mul_checked(std::mem::size_of::<T>())?;

    if element_count == 0 {
        return Ok(Vec::new());
    }

    if ty == T::GGML_TYPE && payload.len() == expected_nbytes {
        let mut out = Vec::with_capacity(element_count);
        unsafe {
            // SAFETY:
            // - `out` has capacity for `element_count` elements (= `expected_nbytes` bytes).
            // - `payload.len()` matches exactly `expected_nbytes`.
            // - We copy raw bytes (not typed elements) to avoid alignment requirements
            //   on the source pointer — `payload` is a `&[u8]` that may not be aligned
            //   for `T`. The destination (`spare_capacity_mut`) is a Vec allocation,
            //   which is always properly aligned for `T`.
            let dst = out.spare_capacity_mut().as_mut_ptr().cast::<u8>();
            ptr::copy_nonoverlapping(payload.as_ptr(), dst, expected_nbytes);
            out.set_len(element_count);
        }
        return Ok(out);
    }

    let decoded = decode_tensor_data_to_float(ty, payload)?;
    let len = decoded.len();
    let mut out = Vec::<T>::with_capacity(len);
    unsafe {
        // SAFETY:
        // - `out` has capacity for `len` elements.
        // - We write each cast value directly into spare capacity, then set length.
        // - If any cast fails, the error is returned before `set_len` is called,
        //   so the Vec is dropped with length 0 and capacity `len` (no UB).
        let dst = out.spare_capacity_mut().as_mut_ptr();
        for (i, value) in decoded.iter().enumerate() {
            let casted = NumCast::from(*value).ok_or(Error::UnsupportedType(ty.as_raw()))?;
            dst.add(i).write(MaybeUninit::new(casted));
        }
        out.set_len(len);
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

/// Returns ggml graph metadata overhead for a custom graph capacity.
pub fn graph_overhead_custom(size: usize, grads: bool) -> usize {
    unsafe { ffi::ggml_graph_overhead_custom(size, grads) }
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

impl<'ctx> BackendBuffer<'ctx> {
    /// Returns allocated backend buffer size in bytes.
    pub fn size_bytes(&self) -> usize {
        unsafe { ffi::ggml_backend_buffer_get_size(self.raw.as_ptr()) }
    }
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

    fn wrap_dyn_tensor<'ctx>(&'ctx self, raw: *mut ffi::ggml_tensor) -> Result<DynTensor<'ctx>> {
        Ok(DynTensor {
            raw: NonNull::new(raw)
                .ok_or_else(|| Error::null_pointer("tensor-producing ggml op"))?,
            _ctx: PhantomData,
        })
    }

    fn wrap_typed_tensor<'ctx, T: GgmlElement>(
        &'ctx self,
        raw: *mut ffi::ggml_tensor,
    ) -> Result<Tensor<'ctx, T>> {
        let dyn_tensor = self.wrap_dyn_tensor(raw)?;
        dyn_tensor.as_typed::<T>()
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
    pub fn new_tensor<const N: usize>(&self, ty: Type, dims: Dims<N>) -> Result<DynTensor<'_>> {
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
        self.wrap_dyn_tensor(raw)
            .map_err(|error| error.with_context("ggml_new_tensor"))
    }

    /// Creates a typed tensor for ranks `1..=4` from a generic element type.
    pub fn new_tensor_typed<T: GgmlElement, const N: usize>(
        &self,
        dims: Dims<N>,
    ) -> Result<Tensor<'_, T>> {
        self.new_tensor(Type::of::<T>(), dims)?.as_typed::<T>()
    }

    /// Creates a typed 1D tensor with semantic `Length`.
    pub fn new_tensor_1d<T: GgmlElement>(&self, len: Length) -> Result<Tensor<'_, T>> {
        self.new_tensor_typed::<T, 1>(Dims::new([len.get()]))
    }

    /// Creates a typed 2D tensor using semantic shape newtypes.
    pub fn new_tensor_2d<T: GgmlElement>(&self, shape: Shape2D) -> Result<Tensor<'_, T>> {
        self.new_tensor_typed::<T, 2>(shape.dims())
    }

    /// Creates a typed 3D tensor from semantic dimensions.
    pub fn new_tensor_3d<T: GgmlElement>(&self, shape: Shape3D) -> Result<Tensor<'_, T>> {
        self.new_tensor_typed::<T, 3>(shape.dims())
    }

    /// Creates a typed 4D tensor from semantic dimensions.
    pub fn new_tensor_4d<T: GgmlElement>(&self, shape: Shape4D) -> Result<Tensor<'_, T>> {
        self.new_tensor_typed::<T, 4>(shape.dims())
    }

    pub fn mul_mat<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        b: &Tensor<'ctx, T>,
    ) -> Result<Tensor<'ctx, T>> {
        let (a_cols, _) = a.shape_2d()?;
        let (b_cols, _) = b.shape_2d()?;
        ensure_matmul_compatible(a_cols, b_cols)?;

        let raw = unsafe { ffi::ggml_mul_mat(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_mul_mat"))
    }

    pub fn add<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        b: &Tensor<'ctx, T>,
    ) -> Result<Tensor<'ctx, T>> {
        let raw = unsafe { ffi::ggml_add(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_add"))
    }

    pub fn sub<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        b: &Tensor<'ctx, T>,
    ) -> Result<Tensor<'ctx, T>> {
        let raw = unsafe { ffi::ggml_sub(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_sub"))
    }

    pub fn mul<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        b: &Tensor<'ctx, T>,
    ) -> Result<Tensor<'ctx, T>> {
        let raw = unsafe { ffi::ggml_mul(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_mul"))
    }

    pub fn div<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        b: &Tensor<'ctx, T>,
    ) -> Result<Tensor<'ctx, T>> {
        let raw = unsafe { ffi::ggml_div(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_div"))
    }

    pub fn silu<'ctx, T: GgmlElement>(&'ctx self, a: &Tensor<'ctx, T>) -> Result<Tensor<'ctx, T>> {
        let raw = unsafe { ffi::ggml_silu(self.raw.as_ptr(), a.raw.as_ptr()) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_silu"))
    }

    /// SSM-style depthwise 1D convolution (causal, stride-1).
    ///
    /// - `sx`: pre-padded input `[d_conv - 1 + n_tokens, d_inner]` (3D with
    ///   batch: `[…, d_inner, n_sequences]`). The first `d_conv - 1` positions
    ///   along dimension 0 are the left-padding (past context or zeros).
    /// - `c`: kernel weights `[d_conv, d_inner]`.
    ///
    /// Returns `[d_inner, n_tokens]` (or 3D with batch).
    ///
    /// Only `f32` tensors are supported by the ggml backend kernels.
    pub fn ssm_conv<'ctx>(
        &'ctx self,
        sx: &Tensor<'ctx, f32>,
        c: &Tensor<'ctx, f32>,
    ) -> Result<Tensor<'ctx, f32>> {
        let raw = unsafe { ffi::ggml_ssm_conv(self.raw.as_ptr(), sx.raw.as_ptr(), c.raw.as_ptr()) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_ssm_conv"))
    }

    pub fn rms_norm<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        eps: f32,
    ) -> Result<Tensor<'ctx, T>> {
        let raw = unsafe { ffi::ggml_rms_norm(self.raw.as_ptr(), a.raw.as_ptr(), eps) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_rms_norm"))
    }

    pub fn scale<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        scalar: f32,
    ) -> Result<Tensor<'ctx, T>> {
        let raw = unsafe { ffi::ggml_scale(self.raw.as_ptr(), a.raw.as_ptr(), scalar) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_scale"))
    }

    pub fn get_rows<'ctx, T: GgmlElement>(
        &'ctx self,
        data: &Tensor<'ctx, T>,
        indices: &Tensor<'ctx, i32>,
    ) -> Result<Tensor<'ctx, T>> {
        let raw = unsafe {
            ffi::ggml_get_rows(self.raw.as_ptr(), data.raw.as_ptr(), indices.raw.as_ptr())
        };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_get_rows"))
    }

    pub fn repeat<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        b: &Tensor<'ctx, T>,
    ) -> Result<Tensor<'ctx, T>> {
        let raw = unsafe { ffi::ggml_repeat(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_repeat"))
    }

    /// Concatenates two 2D tensors along the selected dimension (`0` = cols, `1` = rows).
    pub fn concat<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        b: &Tensor<'ctx, T>,
        dim: usize,
    ) -> Result<Tensor<'ctx, T>> {
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
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_concat"))
    }

    pub fn cpy<'ctx, S: GgmlElement, D: GgmlElement>(
        &'ctx self,
        src: &Tensor<'ctx, S>,
        dst: &Tensor<'ctx, D>,
    ) -> Result<Tensor<'ctx, D>> {
        let raw = unsafe { ffi::ggml_cpy(self.raw.as_ptr(), src.raw.as_ptr(), dst.raw.as_ptr()) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_cpy"))
    }

    pub fn cont<'ctx, T: GgmlElement>(&'ctx self, a: &Tensor<'ctx, T>) -> Result<Tensor<'ctx, T>> {
        let raw = unsafe { ffi::ggml_cont(self.raw.as_ptr(), a.raw.as_ptr()) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_cont"))
    }

    pub fn transpose<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
    ) -> Result<Tensor<'ctx, T>> {
        let raw = unsafe { ffi::ggml_transpose(self.raw.as_ptr(), a.raw.as_ptr()) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_transpose"))
    }

    pub fn reshape_1d<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        ne0: usize,
    ) -> Result<Tensor<'ctx, T>> {
        validate_reshape_source(a, &[ne0])?;
        let ne0 = (ne0)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let raw = unsafe { ffi::ggml_reshape_1d(self.raw.as_ptr(), a.raw.as_ptr(), ne0) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_reshape_1d"))
    }

    pub fn reshape_2d<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        ne0: usize,
        ne1: usize,
    ) -> Result<Tensor<'ctx, T>> {
        validate_reshape_source(a, &[ne0, ne1])?;
        let ne0 = (ne0)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let ne1 = (ne1)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let raw = unsafe { ffi::ggml_reshape_2d(self.raw.as_ptr(), a.raw.as_ptr(), ne0, ne1) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_reshape_2d"))
    }

    pub fn reshape_3d<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
    ) -> Result<Tensor<'ctx, T>> {
        validate_reshape_source(a, &[ne0, ne1, ne2])?;
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
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_reshape_3d"))
    }

    pub fn reshape_4d<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
    ) -> Result<Tensor<'ctx, T>> {
        validate_reshape_source(a, &[ne0, ne1, ne2, ne3])?;
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
            unsafe { ffi::ggml_reshape_4d(self.raw.as_ptr(), a.raw.as_ptr(), ne0, ne1, ne2, ne3) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_reshape_4d"))
    }

    pub fn view_1d<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        ne0: usize,
        offset: usize,
    ) -> Result<Tensor<'ctx, T>> {
        let elem_size = std::mem::size_of::<T>();
        let inner_bytes = ne0.checked_mul(elem_size).ok_or(Error::Overflow)?;
        validate_view_extent(a, offset, &[], inner_bytes)?;
        let ne0 = (ne0)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let raw = unsafe { ffi::ggml_view_1d(self.raw.as_ptr(), a.raw.as_ptr(), ne0, offset) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_view_1d"))
    }

    pub fn view_2d<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        ne0: usize,
        ne1: usize,
        nb1: usize,
        offset: usize,
    ) -> Result<Tensor<'ctx, T>> {
        let elem_size = std::mem::size_of::<T>();
        let inner_bytes = ne0.checked_mul(elem_size).ok_or(Error::Overflow)?;
        validate_view_extent(a, offset, &[(ne1, nb1)], inner_bytes)?;
        let ne0 = (ne0)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let ne1 = (ne1)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let raw =
            unsafe { ffi::ggml_view_2d(self.raw.as_ptr(), a.raw.as_ptr(), ne0, ne1, nb1, offset) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_view_2d"))
    }

    #[allow(clippy::too_many_arguments)] // mirrors ggml_view_3d C API
    pub fn view_3d<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> Result<Tensor<'ctx, T>> {
        let elem_size = std::mem::size_of::<T>();
        let inner_bytes = ne0.checked_mul(elem_size).ok_or(Error::Overflow)?;
        validate_view_extent(a, offset, &[(ne2, nb2), (ne1, nb1)], inner_bytes)?;
        let ne0 = (ne0)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let ne1 = (ne1)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let ne2 = (ne2)
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))?;
        let raw = unsafe {
            ffi::ggml_view_3d(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                ne0,
                ne1,
                ne2,
                nb1,
                nb2,
                offset,
            )
        };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_view_3d"))
    }

    #[allow(clippy::too_many_arguments)] // mirrors ggml_view_4d C API
    pub fn view_4d<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> Result<Tensor<'ctx, T>> {
        let elem_size = std::mem::size_of::<T>();
        let inner_bytes = ne0.checked_mul(elem_size).ok_or(Error::Overflow)?;
        validate_view_extent(
            a,
            offset,
            &[(ne3, nb3), (ne2, nb2), (ne1, nb1)],
            inner_bytes,
        )?;
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
        let raw = unsafe {
            ffi::ggml_view_4d(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                ne0,
                ne1,
                ne2,
                ne3,
                nb1,
                nb2,
                nb3,
                offset,
            )
        };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_view_4d"))
    }

    pub fn permute<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        axis0: i32,
        axis1: i32,
        axis2: i32,
        axis3: i32,
    ) -> Result<Tensor<'ctx, T>> {
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
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_permute"))
    }

    pub fn diag_mask_inf<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        n_past: i32,
    ) -> Result<Tensor<'ctx, T>> {
        let raw = unsafe { ffi::ggml_diag_mask_inf(self.raw.as_ptr(), a.raw.as_ptr(), n_past) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_diag_mask_inf"))
    }

    pub fn soft_max<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
    ) -> Result<Tensor<'ctx, T>> {
        let raw = unsafe { ffi::ggml_soft_max(self.raw.as_ptr(), a.raw.as_ptr()) };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_soft_max"))
    }

    pub fn soft_max_ext<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        mask: Option<&Tensor<'ctx, T>>,
        scale: f32,
        max_bias: f32,
    ) -> Result<Tensor<'ctx, T>> {
        let mask_raw = mask.map_or(ptr::null_mut(), |t| t.raw.as_ptr());
        let raw = unsafe {
            ffi::ggml_soft_max_ext(self.raw.as_ptr(), a.raw.as_ptr(), mask_raw, scale, max_bias)
        };
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_soft_max_ext"))
    }

    pub fn rope_ext<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        positions: &Tensor<'ctx, T>,
        freq_factors: Option<&Tensor<'ctx, T>>,
        params: RopeExtParams,
    ) -> Result<Tensor<'ctx, T>> {
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
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_rope_ext"))
    }

    /// Applies RoPE with mixed element types: f32 data tensor and i32 positions.
    ///
    /// This is the common case for LLM inference where position indices are `i32`
    /// while the data tensor is `f32`.
    pub fn rope_ext_with_i32_positions<'ctx, T: GgmlElement>(
        &'ctx self,
        a: &Tensor<'ctx, T>,
        positions: &Tensor<'ctx, i32>,
        freq_factors: Option<&Tensor<'ctx, T>>,
        params: RopeExtParams,
    ) -> Result<Tensor<'ctx, T>> {
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
        self.wrap_typed_tensor(raw)
            .map_err(|error| error.with_context("ggml_rope_ext"))
    }

    pub fn new_graph(&self) -> Result<Graph<'_>> {
        let raw = unsafe { ffi::ggml_new_graph(self.raw.as_ptr()) };
        self.wrap_graph(raw)
            .map_err(|error| error.with_context("ggml_new_graph"))
    }

    /// Creates a graph with explicit node capacity and grad tracking mode.
    pub fn new_graph_custom(&self, size: usize, grads: bool) -> Result<Graph<'_>> {
        let raw = unsafe { ffi::ggml_new_graph_custom(self.raw.as_ptr(), size, grads) };
        self.wrap_graph(raw)
            .map_err(|error| error.with_context("ggml_new_graph_custom"))
    }

    /// Wraps a tensor into expression form for operator-based composition.
    pub fn expr<'ctx, T: GgmlElement>(&'ctx self, tensor: Tensor<'ctx, T>) -> TensorExpr<'ctx, T> {
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

/// Dynamically-typed tensor handle (element type not known at compile time).
///
/// Use this when the element type is only known at runtime, such as when
/// loading tensors from GGUF files or inspecting graph nodes.
#[derive(Clone, Copy)]
pub struct DynTensor<'ctx> {
    pub(crate) raw: NonNull<ffi::ggml_tensor>,
    _ctx: PhantomData<&'ctx Context>,
}

impl<'ctx> DynTensor<'ctx> {
    pub(crate) fn raw_ptr(&self) -> *mut ffi::ggml_tensor {
        self.raw.as_ptr()
    }

    pub fn element_count(&self) -> Result<usize> {
        (unsafe { ffi::ggml_nelements(self.raw.as_ptr()) })
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))
    }

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

    pub fn rank(&self) -> Result<usize> {
        (unsafe { ffi::ggml_n_dims(self.raw.as_ptr()) })
            .try_into_checked()
            .map_err(|source| Error::int_conversion("ggml_n_dims", source))
    }

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

    pub fn shape_2d(&self) -> Result<(usize, usize)> {
        let shape = self.shape()?;
        Ok((shape.cols.get(), shape.rows.get()))
    }

    pub fn shape(&self) -> Result<Shape2D> {
        Ok(Shape2D {
            cols: Cols::new(self.col_count()?),
            rows: Rows::new(self.row_count()?),
        })
    }

    pub fn shape_3d(&self) -> Result<Shape3D> {
        Ok(Shape3D::from(self.dims::<3>()?))
    }

    pub fn shape_4d(&self) -> Result<Shape4D> {
        Ok(Shape4D::from(self.dims::<4>()?))
    }

    pub fn nbytes(&self) -> usize {
        unsafe { ffi::ggml_nbytes(self.raw.as_ptr()) }
    }

    pub fn set_name(&self, name: &str) -> Result<()> {
        let name = CString::new(name)?;
        let raw = unsafe { ffi::ggml_set_name(self.raw.as_ptr(), name.as_ptr()) };
        let _ = NonNull::new(raw).ok_or_else(|| Error::null_pointer("ggml_set_name"))?;
        Ok(())
    }

    pub fn name(&self) -> Result<String> {
        let ptr = unsafe { ffi::ggml_get_name(self.raw.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::null_pointer("ggml_get_name"));
        }
        let cstr = unsafe { CStr::from_ptr(ptr) };
        Ok(cstr.to_str()?.to_owned())
    }

    /// Converts to a typed tensor, checking that the runtime element type matches `T`.
    pub fn as_typed<T: GgmlElement>(&self) -> Result<Tensor<'ctx, T>> {
        let actual_type = unsafe { (*self.raw.as_ptr()).type_ };
        if actual_type != T::GGML_TYPE.as_raw() as ffi::ggml_type {
            return Err(Error::TypeMismatch {
                expected: T::GGML_TYPE.as_raw(),
                actual: actual_type as c_int,
            });
        }
        Ok(Tensor {
            raw: self.raw,
            _ctx: PhantomData,
            _type: PhantomData,
        })
    }
}

/// Compile-time typed tensor handle carrying element type information.
///
/// The type parameter `T` propagates through operations, eliminating the need
/// for explicit type annotations on I/O methods like `write_data` and `read_data`.
pub struct Tensor<'ctx, T: GgmlElement> {
    pub(crate) raw: NonNull<ffi::ggml_tensor>,
    _ctx: PhantomData<&'ctx Context>,
    _type: PhantomData<T>,
}

impl<'ctx, T: GgmlElement> Clone for Tensor<'ctx, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'ctx, T: GgmlElement> Copy for Tensor<'ctx, T> {}

impl<'ctx, T: GgmlElement> Tensor<'ctx, T> {
    pub fn element_count(&self) -> Result<usize> {
        (unsafe { ffi::ggml_nelements(self.raw.as_ptr()) })
            .try_into_checked()
            .map_err(|source| Error::int_conversion("tensor dimension", source))
    }

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

    pub fn rank(&self) -> Result<usize> {
        (unsafe { ffi::ggml_n_dims(self.raw.as_ptr()) })
            .try_into_checked()
            .map_err(|source| Error::int_conversion("ggml_n_dims", source))
    }

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

    pub fn shape_2d(&self) -> Result<(usize, usize)> {
        let shape = self.shape()?;
        Ok((shape.cols.get(), shape.rows.get()))
    }

    pub fn shape(&self) -> Result<Shape2D> {
        Ok(Shape2D {
            cols: Cols::new(self.col_count()?),
            rows: Rows::new(self.row_count()?),
        })
    }

    pub fn shape_3d(&self) -> Result<Shape3D> {
        Ok(Shape3D::from(self.dims::<3>()?))
    }

    pub fn shape_4d(&self) -> Result<Shape4D> {
        Ok(Shape4D::from(self.dims::<4>()?))
    }

    pub fn nbytes(&self) -> usize {
        unsafe { ffi::ggml_nbytes(self.raw.as_ptr()) }
    }

    fn is_contiguous(&self) -> bool {
        unsafe { ffi::ggml_is_contiguous(self.raw.as_ptr()) as i32 != 0 }
    }

    fn contiguous_data_ptr<U>(&self) -> Option<NonNull<U>> {
        if !self.is_contiguous() {
            return None;
        }
        let raw = unsafe { ffi::ggml_get_data(self.raw.as_ptr()) }.cast::<U>();
        NonNull::new(raw)
    }

    fn expected_nbytes_for<U: BackendElement>(&self) -> Result<usize> {
        let elements = self.element_count()?;
        elements.checked_mul_checked(std::mem::size_of::<U>())
    }

    fn ensure_backend_slice_compatible<U: BackendElement>(&self) -> Result<usize> {
        let expected_nbytes = self.expected_nbytes_for::<U>()?;
        let actual_nbytes = self.nbytes();
        if expected_nbytes != actual_nbytes {
            return Err(Error::UnexpectedTensorByteSize {
                expected: expected_nbytes,
                actual: actual_nbytes,
            });
        }
        Ok(expected_nbytes)
    }

    pub fn set_name(&self, name: &str) -> Result<()> {
        let name = CString::new(name)?;
        let raw = unsafe { ffi::ggml_set_name(self.raw.as_ptr(), name.as_ptr()) };
        let _ = NonNull::new(raw).ok_or_else(|| Error::null_pointer("ggml_set_name"))?;
        Ok(())
    }

    pub fn name(&self) -> Result<String> {
        let ptr = unsafe { ffi::ggml_get_name(self.raw.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::null_pointer("ggml_get_name"));
        }
        let cstr = unsafe { CStr::from_ptr(ptr) };
        Ok(cstr.to_str()?.to_owned())
    }

    /// Converts to a dynamically-typed tensor, discarding compile-time type information.
    pub fn into_dyn(self) -> DynTensor<'ctx> {
        DynTensor {
            raw: self.raw,
            _ctx: self._ctx,
        }
    }

    /// Writes host values into the tensor.
    pub fn write_data(&self, values: &[T]) -> Result<()> {
        T::write_data(self, values)
    }

    /// Writes host values at the provided element offset.
    pub fn write_data_at(&self, element_offset: usize, values: &[T]) -> Result<()> {
        T::write_data_at(self, element_offset, values)
    }

    /// Writes backend values with an element type inferred from the slice.
    pub fn write_data_backend(&self, values: &[T]) -> Result<()>
    where
        T: BackendElement,
    {
        self.write_backend_slice(values)
    }

    /// Writes backend values at the provided element offset.
    pub fn write_data_backend_at(&self, element_offset: usize, values: &[T]) -> Result<()>
    where
        T: BackendElement,
    {
        self.write_backend_slice_at(element_offset, values)
    }

    /// Reads all host values.
    pub fn read_data(&self) -> Result<Vec<T>> {
        T::read_data(self)
    }

    /// Reads a host tensor slice.
    pub fn read_data_at(&self, element_offset: usize, element_count: usize) -> Result<Vec<T>> {
        T::read_data_at(self, element_offset, element_count)
    }

    /// Reads all backend values.
    pub fn read_data_backend(&self) -> Result<Vec<T>>
    where
        T: BackendElement,
    {
        self.read_backend_vec()
    }

    /// Reads a backend tensor slice.
    pub fn read_data_backend_at(
        &self,
        element_offset: usize,
        element_count: usize,
    ) -> Result<Vec<T>>
    where
        T: BackendElement,
    {
        self.read_backend_vec_at(element_offset, element_count)
    }

    /// Reads one element with bounds checking.
    pub fn get_data(&self, index: TensorIndex) -> Result<T> {
        T::get_data(self, index)
    }

    /// Writes host values through context tensor APIs.
    pub(crate) fn write_host_data(&self, values: &[T]) -> Result<()> {
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

    /// Writes a host slice with bounds checking.
    pub(crate) fn write_host_data_at(&self, element_offset: usize, values: &[T]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        let len = self.element_count()?;
        let end = element_offset
            .checked_add(values.len())
            .ok_or(Error::Overflow)?;
        if end > len {
            return Err(Error::IndexOutOfBounds {
                index: end.saturating_sub(1),
                len,
            });
        }

        let element_size = std::mem::size_of::<T>();
        let expected_nbytes = len.checked_mul_checked(element_size)?;
        if expected_nbytes == self.nbytes()
            && let Some(dst) = self.contiguous_data_ptr::<T>()
        {
            unsafe {
                // SAFETY:
                // - `dst` points to contiguous tensor storage with at least `len` elements.
                // - `element_offset..element_offset + values.len()` was bounds-checked.
                // - Source and destination do not overlap (`values` is caller-owned slice).
                ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    dst.as_ptr().add(element_offset),
                    values.len(),
                );
            }
            return Ok(());
        }

        for (index, value) in values.iter().copied().enumerate() {
            let index = element_offset
                .checked_add(index)
                .ok_or(Error::Overflow)?
                .try_into_checked()
                .map_err(|source| Error::int_conversion("tensor index", source))?;
            T::set_1d_raw(self.raw.as_ptr(), index, value);
        }

        Ok(())
    }

    /// Writes backend values through backend tensor APIs.
    fn write_backend_slice(&self, values: &[T]) -> Result<()>
    where
        T: BackendElement,
    {
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
    fn write_backend_slice_at(&self, element_offset: usize, values: &[T]) -> Result<()>
    where
        T: BackendElement,
    {
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
    pub(crate) fn read_host_at(&self, index: TensorIndex) -> Result<T> {
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
    pub(crate) fn read_host_data(&self) -> Result<Vec<T>> {
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

    /// Reads a host slice with bounds checking.
    pub(crate) fn read_host_data_at(
        &self,
        element_offset: usize,
        element_count: usize,
    ) -> Result<Vec<T>> {
        if element_count == 0 {
            return Ok(Vec::new());
        }

        let len = self.element_count()?;
        let end = element_offset
            .checked_add(element_count)
            .ok_or(Error::Overflow)?;
        if end > len {
            return Err(Error::IndexOutOfBounds {
                index: end.saturating_sub(1),
                len,
            });
        }

        let element_size = std::mem::size_of::<T>();
        let expected_nbytes = len.checked_mul_checked(element_size)?;
        if expected_nbytes == self.nbytes()
            && let Some(src) = self.contiguous_data_ptr::<T>()
        {
            let mut out = Vec::with_capacity(element_count);
            unsafe {
                // SAFETY:
                // - `src` points to contiguous tensor storage with at least `len` elements.
                // - `element_offset..element_offset + element_count` was bounds-checked.
                // - `out` has capacity for `element_count`; set_len is called after init.
                ptr::copy_nonoverlapping(
                    src.as_ptr().add(element_offset),
                    out.as_mut_ptr(),
                    element_count,
                );
                out.set_len(element_count);
            }
            return Ok(out);
        }

        let mut out = Vec::with_capacity(element_count);
        for index in element_offset..end {
            let index = (index)
                .try_into_checked()
                .map_err(|source| Error::int_conversion("tensor index", source))?;
            out.push(T::get_1d_raw(self.raw.as_ptr(), index));
        }
        Ok(out)
    }

    /// Reads all values through backend tensor APIs.
    fn read_backend_vec(&self) -> Result<Vec<T>>
    where
        T: BackendElement,
    {
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

    /// Reads a contiguous value range through backend tensor APIs.
    fn read_backend_vec_at(&self, element_offset: usize, element_count: usize) -> Result<Vec<T>>
    where
        T: BackendElement,
    {
        if element_count == 0 {
            return Ok(Vec::new());
        }

        let tensor_len = self.element_count()?;
        let end = element_offset
            .checked_add(element_count)
            .ok_or(Error::Overflow)?;
        if end > tensor_len {
            return Err(Error::IndexOutOfBounds {
                index: end.saturating_sub(1),
                len: tensor_len,
            });
        }

        self.ensure_backend_slice_compatible::<T>()?;

        let byte_offset = element_offset.checked_mul_checked(std::mem::size_of::<T>())?;
        let read_nbytes = element_count.checked_mul_checked(std::mem::size_of::<T>())?;
        let mut out = Vec::<T>::with_capacity(element_count);
        unsafe {
            // SAFETY:
            // - `out` has capacity for `element_count` elements.
            // - Requested byte range was bounds-checked against tensor element count.
            // - ggml writes exactly `read_nbytes` bytes into the destination buffer.
            ffi::ggml_backend_tensor_get(
                self.raw.as_ptr(),
                out.as_mut_ptr().cast(),
                byte_offset,
                read_nbytes,
            );
            out.set_len(element_count);
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
    pub fn build_forward_expand<T: GgmlElement>(&mut self, tensor: &Tensor<'ctx, T>) {
        unsafe {
            ffi::ggml_build_forward_expand(self.raw.as_ptr(), tensor.raw.as_ptr());
        }
    }

    /// Expands the graph forward pass using a dynamically-typed tensor.
    pub fn build_forward_expand_dyn(&mut self, tensor: &DynTensor<'ctx>) {
        unsafe {
            ffi::ggml_build_forward_expand(self.raw.as_ptr(), tensor.raw.as_ptr());
        }
    }

    /// Returns number of nodes currently in the graph.
    pub fn node_count(&self) -> i32 {
        unsafe { ffi::ggml_graph_n_nodes(self.raw.as_ptr()) }
    }

    /// Returns a dynamically-typed node by index (supports negative indexing from the end).
    pub fn node(&self, index: i32) -> Result<DynTensor<'ctx>> {
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

        Ok(DynTensor {
            raw,
            _ctx: PhantomData,
        })
    }

    /// Returns a typed node by index with runtime type checking.
    pub fn node_typed<T: GgmlElement>(&self, index: i32) -> Result<Tensor<'ctx, T>> {
        self.node(index)?.as_typed::<T>()
    }

    /// Returns the last graph node as a dynamically-typed tensor.
    pub fn last_node(&self) -> Result<DynTensor<'ctx>> {
        let node_count = self.node_count();
        if node_count <= 0 {
            return Err(Error::InvalidGraphIndex {
                index: -1,
                node_count,
            });
        }

        self.node(node_count - 1)
    }

    /// Returns the last graph node as a typed tensor with runtime type checking.
    pub fn last_node_typed<T: GgmlElement>(&self) -> Result<Tensor<'ctx, T>> {
        self.last_node()?.as_typed::<T>()
    }
}

fn ensure_matmul_compatible(lhs_cols: usize, rhs_cols: usize) -> Result<()> {
    if lhs_cols == rhs_cols {
        Ok(())
    } else {
        Err(Error::IncompatibleMatmulShapes { lhs_cols, rhs_cols })
    }
}
