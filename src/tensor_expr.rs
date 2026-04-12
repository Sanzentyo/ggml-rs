use crate::{Context, DynTensor, Error, GgmlType, Result, Tensor, TensorIndex, ffi};
use std::ops::{Add, Div, Mul, Sub};

/// Element types allowed for backend tensor transfer helpers.
pub trait BackendElement: Copy + Default {}

impl BackendElement for f32 {}

impl BackendElement for i32 {}

/// Low-level host accessor contract for 1D element read/write.
///
/// This trait provides the raw FFI accessors that underpin `GgmlElement` I/O.
/// It is sealed in practice because only `f32` and `i32` implement it within this crate.
pub(crate) trait HostElement: Copy + Default {
    fn set_1d_raw(raw: *mut ffi::ggml_tensor, index: i32, value: Self);
    fn get_1d_raw(raw: *mut ffi::ggml_tensor, index: i32) -> Self;
}

impl HostElement for f32 {
    fn set_1d_raw(raw: *mut ffi::ggml_tensor, index: i32, value: Self) {
        unsafe {
            ffi::ggml_set_f32_1d(raw, index, value);
        }
    }

    fn get_1d_raw(raw: *mut ffi::ggml_tensor, index: i32) -> Self {
        unsafe { ffi::ggml_get_f32_1d(raw, index) }
    }
}

impl HostElement for i32 {
    fn set_1d_raw(raw: *mut ffi::ggml_tensor, index: i32, value: Self) {
        unsafe {
            ffi::ggml_set_i32_1d(raw, index, value);
        }
    }

    fn get_1d_raw(raw: *mut ffi::ggml_tensor, index: i32) -> Self {
        unsafe { ffi::ggml_get_i32_1d(raw, index) }
    }
}

/// Element types that map to ggml tensor types and typed tensor I/O helpers.
///
/// This trait serves as a bounds marker combining `BackendElement`, `GgmlType`,
/// and `HostElement`. The I/O methods that were previously on this trait are now
/// inherent methods on `Tensor<'ctx, T>`, where `T: GgmlElement`.
pub trait GgmlElement: BackendElement + GgmlType + HostElement {
    /// Writes host values into the tensor through the most appropriate path.
    fn write_data(tensor: &Tensor<'_, Self>, values: &[Self]) -> Result<()>;

    /// Writes a host tensor slice.
    fn write_data_at(
        tensor: &Tensor<'_, Self>,
        element_offset: usize,
        values: &[Self],
    ) -> Result<()>;

    /// Reads tensor values into host memory through the most appropriate path.
    fn read_data(tensor: &Tensor<'_, Self>) -> Result<Vec<Self>>;

    /// Reads a host tensor slice into host memory.
    fn read_data_at(
        tensor: &Tensor<'_, Self>,
        element_offset: usize,
        element_count: usize,
    ) -> Result<Vec<Self>>;

    /// Reads one element with bounds checking.
    fn get_data(tensor: &Tensor<'_, Self>, index: TensorIndex) -> Result<Self>;
}

impl GgmlElement for f32 {
    fn write_data(tensor: &Tensor<'_, Self>, values: &[Self]) -> Result<()> {
        tensor.write_host_data(values)
    }

    fn write_data_at(
        tensor: &Tensor<'_, Self>,
        element_offset: usize,
        values: &[Self],
    ) -> Result<()> {
        tensor.write_host_data_at(element_offset, values)
    }

    fn read_data(tensor: &Tensor<'_, Self>) -> Result<Vec<Self>> {
        tensor.read_host_data()
    }

    fn read_data_at(
        tensor: &Tensor<'_, Self>,
        element_offset: usize,
        element_count: usize,
    ) -> Result<Vec<Self>> {
        tensor.read_host_data_at(element_offset, element_count)
    }

    fn get_data(tensor: &Tensor<'_, Self>, index: TensorIndex) -> Result<Self> {
        tensor.read_host_at(index)
    }
}

impl GgmlElement for i32 {
    fn write_data(tensor: &Tensor<'_, Self>, values: &[Self]) -> Result<()> {
        tensor.write_host_data(values)
    }

    fn write_data_at(
        tensor: &Tensor<'_, Self>,
        element_offset: usize,
        values: &[Self],
    ) -> Result<()> {
        tensor.write_host_data_at(element_offset, values)
    }

    fn read_data(tensor: &Tensor<'_, Self>) -> Result<Vec<Self>> {
        tensor.read_host_data()
    }

    fn read_data_at(
        tensor: &Tensor<'_, Self>,
        element_offset: usize,
        element_count: usize,
    ) -> Result<Vec<Self>> {
        tensor.read_host_data_at(element_offset, element_count)
    }

    fn get_data(tensor: &Tensor<'_, Self>, index: TensorIndex) -> Result<Self> {
        tensor.read_host_at(index)
    }
}

#[derive(Clone, Copy)]
/// Lightweight expression wrapper that keeps the originating `Context`.
///
/// Arithmetic operators return `Result<TensorExpr>` so context mismatch and
/// ggml allocation errors stay explicit.
pub struct TensorExpr<'ctx, T: GgmlElement> {
    pub(crate) ctx: &'ctx Context,
    pub(crate) tensor: Tensor<'ctx, T>,
}

impl<'ctx, T: GgmlElement> TensorExpr<'ctx, T> {
    pub fn into_tensor(self) -> Tensor<'ctx, T> {
        self.tensor
    }

    pub fn tensor(self) -> Tensor<'ctx, T> {
        self.tensor
    }

    pub fn scale(self, scalar: f32) -> Result<Self> {
        self.ctx.scale(&self.tensor, scalar).map(|tensor| Self {
            ctx: self.ctx,
            tensor,
        })
    }

    pub fn rms_norm(self, eps: f32) -> Result<Self> {
        self.ctx.rms_norm(&self.tensor, eps).map(|tensor| Self {
            ctx: self.ctx,
            tensor,
        })
    }
}

impl<'ctx, T: GgmlElement> From<TensorExpr<'ctx, T>> for Tensor<'ctx, T> {
    fn from(value: TensorExpr<'ctx, T>) -> Self {
        value.tensor
    }
}

impl<'ctx, T: GgmlElement> From<TensorExpr<'ctx, T>> for DynTensor<'ctx> {
    fn from(value: TensorExpr<'ctx, T>) -> Self {
        value.tensor.into_dyn()
    }
}

impl<'ctx, T: GgmlElement> Add for TensorExpr<'ctx, T> {
    type Output = Result<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        if !std::ptr::eq(self.ctx, rhs.ctx) {
            return Err(Error::ContextMismatch);
        }
        self.ctx.add(&self.tensor, &rhs.tensor).map(|tensor| Self {
            ctx: self.ctx,
            tensor,
        })
    }
}

impl<'ctx, T: GgmlElement> Sub for TensorExpr<'ctx, T> {
    type Output = Result<Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        if !std::ptr::eq(self.ctx, rhs.ctx) {
            return Err(Error::ContextMismatch);
        }
        self.ctx.sub(&self.tensor, &rhs.tensor).map(|tensor| Self {
            ctx: self.ctx,
            tensor,
        })
    }
}

impl<'ctx, T: GgmlElement> Mul for TensorExpr<'ctx, T> {
    type Output = Result<Self>;

    fn mul(self, rhs: Self) -> Self::Output {
        if !std::ptr::eq(self.ctx, rhs.ctx) {
            return Err(Error::ContextMismatch);
        }
        self.ctx.mul(&self.tensor, &rhs.tensor).map(|tensor| Self {
            ctx: self.ctx,
            tensor,
        })
    }
}

impl<'ctx, T: GgmlElement> Div for TensorExpr<'ctx, T> {
    type Output = Result<Self>;

    fn div(self, rhs: Self) -> Self::Output {
        if !std::ptr::eq(self.ctx, rhs.ctx) {
            return Err(Error::ContextMismatch);
        }
        self.ctx.div(&self.tensor, &rhs.tensor).map(|tensor| Self {
            ctx: self.ctx,
            tensor,
        })
    }
}
