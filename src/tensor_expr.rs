use crate::{Context, Error, Result, Tensor, TensorIndex, Type, ffi};
use std::ops::{Add, Div, Mul, Sub};

/// Element types allowed for backend tensor transfer helpers.
pub trait BackendElement: Copy + Default {}

impl BackendElement for f32 {}

impl BackendElement for i32 {}

/// Internal host accessor contract used by fast generic tensor host I/O.
///
/// Kept crate-private on purpose so `GgmlElement` remains a high-level public
/// trait and does not leak raw-pointer FFI requirements to external users.
pub(crate) trait HostElement: GgmlElement {
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
pub trait GgmlElement: BackendElement {
    /// ggml element kind represented by this Rust type.
    const GGML_TYPE: Type;

    /// Writes host values into the tensor through the most appropriate path.
    fn write_data(tensor: &Tensor<'_>, values: &[Self]) -> Result<()>;

    /// Reads tensor values into host memory through the most appropriate path.
    fn read_data(tensor: &Tensor<'_>) -> Result<Vec<Self>>;

    /// Reads one element with bounds checking.
    fn get_data(tensor: &Tensor<'_>, index: TensorIndex) -> Result<Self>;
}

impl GgmlElement for f32 {
    const GGML_TYPE: Type = Type::F32;

    fn write_data(tensor: &Tensor<'_>, values: &[Self]) -> Result<()> {
        tensor.write_host_data(values)
    }

    fn read_data(tensor: &Tensor<'_>) -> Result<Vec<Self>> {
        tensor.read_host_data()
    }

    fn get_data(tensor: &Tensor<'_>, index: TensorIndex) -> Result<Self> {
        tensor.read_host_at(index)
    }
}

impl GgmlElement for i32 {
    const GGML_TYPE: Type = Type::I32;

    fn write_data(tensor: &Tensor<'_>, values: &[Self]) -> Result<()> {
        tensor.write_host_data(values)
    }

    fn read_data(tensor: &Tensor<'_>) -> Result<Vec<Self>> {
        tensor.read_host_data()
    }

    fn get_data(tensor: &Tensor<'_>, index: TensorIndex) -> Result<Self> {
        tensor.read_host_at(index)
    }
}

#[derive(Clone, Copy)]
/// Lightweight expression wrapper that keeps the originating `Context`.
///
/// Arithmetic operators return `Result<TensorExpr>` so context mismatch and
/// ggml allocation errors stay explicit.
pub struct TensorExpr<'ctx> {
    pub(crate) ctx: &'ctx Context,
    pub(crate) tensor: Tensor<'ctx>,
}

impl<'ctx> TensorExpr<'ctx> {
    pub fn into_tensor(self) -> Tensor<'ctx> {
        self.tensor
    }

    pub fn tensor(self) -> Tensor<'ctx> {
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

impl<'ctx> From<TensorExpr<'ctx>> for Tensor<'ctx> {
    fn from(value: TensorExpr<'ctx>) -> Self {
        value.tensor
    }
}

impl<'ctx> Add for TensorExpr<'ctx> {
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

impl<'ctx> Sub for TensorExpr<'ctx> {
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

impl<'ctx> Mul for TensorExpr<'ctx> {
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

impl<'ctx> Div for TensorExpr<'ctx> {
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
