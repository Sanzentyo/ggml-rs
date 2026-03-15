use crate::{Context, Error, Result, Tensor, TensorIndex, Type};
use std::ops::{Add, Div, Mul, Sub};

/// Element types allowed for backend tensor transfer helpers.
pub trait BackendElement: Copy + Default {}

impl BackendElement for f32 {}

impl BackendElement for i32 {}

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
        tensor.set_f32(values)
    }

    fn read_data(tensor: &Tensor<'_>) -> Result<Vec<Self>> {
        tensor.to_vec_f32()
    }

    fn get_data(tensor: &Tensor<'_>, index: TensorIndex) -> Result<Self> {
        tensor.get_f32_at(index)
    }
}

impl GgmlElement for i32 {
    const GGML_TYPE: Type = Type::I32;

    fn write_data(tensor: &Tensor<'_>, values: &[Self]) -> Result<()> {
        tensor.set_i32_backend(values)
    }

    fn read_data(tensor: &Tensor<'_>) -> Result<Vec<Self>> {
        tensor.to_vec_i32_backend()
    }

    fn get_data(tensor: &Tensor<'_>, index: TensorIndex) -> Result<Self> {
        let index = index.get();
        let values = tensor.to_vec_i32_backend()?;
        let len = values.len();
        if index >= len {
            return Err(Error::IndexOutOfBounds { index, len });
        }
        Ok(values[index])
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
