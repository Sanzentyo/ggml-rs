use crate::{Context, Result, Shape2D, Shape2DSpec, StaticShape2D, Tensor, Type};
use std::marker::PhantomData;

#[derive(Clone, Copy)]
/// 2D tensor wrapper carrying shape information at the type level.
///
/// This is a zero-cost abstraction over `Tensor` and helps encode expected
/// dimensions in APIs and local aliases.
pub struct Tensor2D<'ctx, S: Shape2DSpec> {
    inner: Tensor<'ctx>,
    _shape: PhantomData<S>,
}

impl<'ctx, S: Shape2DSpec> Tensor2D<'ctx, S> {
    pub(crate) fn new(inner: Tensor<'ctx>) -> Self {
        Self {
            inner,
            _shape: PhantomData,
        }
    }

    pub const fn shape() -> Shape2D {
        S::SHAPE
    }

    pub fn inner(&self) -> &Tensor<'ctx> {
        &self.inner
    }

    pub fn into_inner(self) -> Tensor<'ctx> {
        self.inner
    }

    pub fn set_f32(&self, values: &[f32]) -> Result<()> {
        self.inner.set_f32(values)
    }

    pub fn set_f32_backend(&self, values: &[f32]) -> Result<()> {
        self.inner.set_f32_backend(values)
    }

    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.inner.to_vec_f32()
    }

    pub fn to_vec_f32_backend(&self) -> Result<Vec<f32>> {
        self.inner.to_vec_f32_backend()
    }
}

pub type Tensor2DConst<'ctx, const COLS: usize, const ROWS: usize> =
    Tensor2D<'ctx, StaticShape2D<COLS, ROWS>>;

impl Context {
    /// Creates a typed 2D tensor from compile-time shape information.
    pub fn new_tensor_2d_typed<'ctx, S: Shape2DSpec>(
        &'ctx self,
        ty: Type,
    ) -> Result<Tensor2D<'ctx, S>> {
        self.new_tensor_2d_shape(ty, S::SHAPE).map(Tensor2D::new)
    }

    /// Creates a typed `f32` 2D tensor.
    pub fn new_f32_tensor_2d_typed<'ctx, S: Shape2DSpec>(&'ctx self) -> Result<Tensor2D<'ctx, S>> {
        self.new_tensor_2d_typed::<S>(Type::F32)
    }

    /// Creates a typed `i32` 2D tensor.
    pub fn new_i32_tensor_2d_typed<'ctx, S: Shape2DSpec>(&'ctx self) -> Result<Tensor2D<'ctx, S>> {
        self.new_tensor_2d_typed::<S>(Type::I32)
    }
}
