use crate::{Context, Error, Result, Tensor};
use std::ops::{Add, Div, Mul, Sub};

/// Element types allowed for backend tensor transfer helpers.
pub trait BackendElement: Copy + Default {}

impl BackendElement for f32 {}

impl BackendElement for i32 {}

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
