use crate::{
    Context, GgmlElement, Length, LengthSpec, Result, Shape2D, Shape2DSpec, Shape3D, Shape3DSpec,
    Shape4D, Shape4DSpec, StaticLength, StaticShape2D, StaticShape3D, StaticShape4D, Tensor,
    TensorIndex,
};
use std::marker::PhantomData;

#[derive(Clone, Copy)]
/// 1D tensor wrapper carrying compile-time length information.
pub struct Tensor1D<'ctx, S: LengthSpec> {
    inner: Tensor<'ctx>,
    _shape: PhantomData<S>,
}

impl<'ctx, S: LengthSpec> Tensor1D<'ctx, S> {
    pub(crate) fn new(inner: Tensor<'ctx>) -> Self {
        Self {
            inner,
            _shape: PhantomData,
        }
    }

    pub const fn shape() -> Length {
        S::LENGTH
    }

    pub fn inner(&self) -> &Tensor<'ctx> {
        &self.inner
    }

    pub fn into_inner(self) -> Tensor<'ctx> {
        self.inner
    }

    pub fn write_data<T: GgmlElement>(&self, values: &[T]) -> Result<()> {
        self.inner.write_data(values)
    }

    pub fn write_data_backend<T: GgmlElement>(&self, values: &[T]) -> Result<()> {
        self.inner.write_data_backend(values)
    }

    pub fn read_data<T: GgmlElement>(&self) -> Result<Vec<T>> {
        self.inner.read_data()
    }

    pub fn get_data<T: GgmlElement>(&self, index: TensorIndex) -> Result<T> {
        self.inner.get_data(index)
    }
}

pub type Tensor1DConst<'ctx, const LEN: usize> = Tensor1D<'ctx, StaticLength<LEN>>;

#[derive(Clone, Copy)]
/// 2D tensor wrapper carrying compile-time shape information.
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

    pub fn write_data<T: GgmlElement>(&self, values: &[T]) -> Result<()> {
        self.inner.write_data(values)
    }

    pub fn write_data_backend<T: GgmlElement>(&self, values: &[T]) -> Result<()> {
        self.inner.write_data_backend(values)
    }

    pub fn read_data<T: GgmlElement>(&self) -> Result<Vec<T>> {
        self.inner.read_data()
    }

    pub fn get_data<T: GgmlElement>(&self, index: TensorIndex) -> Result<T> {
        self.inner.get_data(index)
    }
}

pub type Tensor2DConst<'ctx, const COLS: usize, const ROWS: usize> =
    Tensor2D<'ctx, StaticShape2D<COLS, ROWS>>;

#[derive(Clone, Copy)]
/// 3D tensor wrapper carrying compile-time shape information.
pub struct Tensor3D<'ctx, S: Shape3DSpec> {
    inner: Tensor<'ctx>,
    _shape: PhantomData<S>,
}

impl<'ctx, S: Shape3DSpec> Tensor3D<'ctx, S> {
    pub(crate) fn new(inner: Tensor<'ctx>) -> Self {
        Self {
            inner,
            _shape: PhantomData,
        }
    }

    pub const fn shape() -> Shape3D {
        S::SHAPE
    }

    pub fn inner(&self) -> &Tensor<'ctx> {
        &self.inner
    }

    pub fn into_inner(self) -> Tensor<'ctx> {
        self.inner
    }

    pub fn write_data<T: GgmlElement>(&self, values: &[T]) -> Result<()> {
        self.inner.write_data(values)
    }

    pub fn write_data_backend<T: GgmlElement>(&self, values: &[T]) -> Result<()> {
        self.inner.write_data_backend(values)
    }

    pub fn read_data<T: GgmlElement>(&self) -> Result<Vec<T>> {
        self.inner.read_data()
    }

    pub fn get_data<T: GgmlElement>(&self, index: TensorIndex) -> Result<T> {
        self.inner.get_data(index)
    }
}

pub type Tensor3DConst<'ctx, const NE0: usize, const NE1: usize, const NE2: usize> =
    Tensor3D<'ctx, StaticShape3D<NE0, NE1, NE2>>;

#[derive(Clone, Copy)]
/// 4D tensor wrapper carrying compile-time shape information.
pub struct Tensor4D<'ctx, S: Shape4DSpec> {
    inner: Tensor<'ctx>,
    _shape: PhantomData<S>,
}

impl<'ctx, S: Shape4DSpec> Tensor4D<'ctx, S> {
    pub(crate) fn new(inner: Tensor<'ctx>) -> Self {
        Self {
            inner,
            _shape: PhantomData,
        }
    }

    pub const fn shape() -> Shape4D {
        S::SHAPE
    }

    pub fn inner(&self) -> &Tensor<'ctx> {
        &self.inner
    }

    pub fn into_inner(self) -> Tensor<'ctx> {
        self.inner
    }

    pub fn write_data<T: GgmlElement>(&self, values: &[T]) -> Result<()> {
        self.inner.write_data(values)
    }

    pub fn write_data_backend<T: GgmlElement>(&self, values: &[T]) -> Result<()> {
        self.inner.write_data_backend(values)
    }

    pub fn read_data<T: GgmlElement>(&self) -> Result<Vec<T>> {
        self.inner.read_data()
    }

    pub fn get_data<T: GgmlElement>(&self, index: TensorIndex) -> Result<T> {
        self.inner.get_data(index)
    }
}

pub type Tensor4DConst<
    'ctx,
    const NE0: usize,
    const NE1: usize,
    const NE2: usize,
    const NE3: usize,
> = Tensor4D<'ctx, StaticShape4D<NE0, NE1, NE2, NE3>>;

impl Context {
    /// Creates a typed 1D tensor from compile-time length information.
    pub fn new_tensor_1d_typed<'ctx, T: GgmlElement, S: LengthSpec>(
        &'ctx self,
    ) -> Result<Tensor1D<'ctx, S>> {
        self.new_tensor_1d::<T>(S::LENGTH).map(Tensor1D::new)
    }

    /// Creates a typed 2D tensor from compile-time shape information.
    pub fn new_tensor_2d_typed<'ctx, T: GgmlElement, S: Shape2DSpec>(
        &'ctx self,
    ) -> Result<Tensor2D<'ctx, S>> {
        self.new_tensor_2d::<T>(S::SHAPE).map(Tensor2D::new)
    }

    /// Creates a typed 3D tensor from compile-time shape information.
    pub fn new_tensor_3d_typed<'ctx, T: GgmlElement, S: Shape3DSpec>(
        &'ctx self,
    ) -> Result<Tensor3D<'ctx, S>> {
        self.new_tensor_3d::<T>(S::SHAPE).map(Tensor3D::new)
    }

    /// Creates a typed 4D tensor from compile-time shape information.
    pub fn new_tensor_4d_typed<'ctx, T: GgmlElement, S: Shape4DSpec>(
        &'ctx self,
    ) -> Result<Tensor4D<'ctx, S>> {
        self.new_tensor_4d::<T>(S::SHAPE).map(Tensor4D::new)
    }
}
