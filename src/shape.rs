//! Zero-cost semantic wrappers for dimensions, lengths, and byte counts.
//!
//! These newtypes make call sites self-descriptive while compiling down to
//! plain integers (`#[repr(transparent)]`).

macro_rules! define_semantic_usize {
    ($(#[$meta:meta])* $name:ident) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        $(#[$meta])*
        pub struct $name(usize);

        impl $name {
            #[inline]
            pub const fn new(value: usize) -> Self {
                Self(value)
            }

            #[inline]
            pub const fn get(self) -> usize {
                self.0
            }
        }
    };
}

define_semantic_usize!(
    /// Number of columns in a 2D tensor view (`ne0` in ggml terms).
    Cols
);
define_semantic_usize!(
    /// Number of rows in a 2D tensor view (`ne1` in ggml terms).
    Rows
);
define_semantic_usize!(
    /// 1D tensor length wrapper.
    Length
);
define_semantic_usize!(
    /// Tensor index wrapper used by bounds-checked accessors.
    TensorIndex
);
define_semantic_usize!(
    /// Number of compute threads used by legacy graph execution.
    ThreadCount
);
define_semantic_usize!(
    /// Byte-size wrapper used for context memory sizing APIs.
    Bytes
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Generic fixed-rank tensor dimensions.
pub struct Dims<const N: usize> {
    values: [usize; N],
}

impl<const N: usize> Dims<N> {
    #[inline]
    pub const fn new(values: [usize; N]) -> Self {
        Self { values }
    }

    #[inline]
    pub const fn rank(&self) -> usize {
        N
    }

    #[inline]
    pub const fn as_array(&self) -> &[usize; N] {
        &self.values
    }

    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.values
    }
}

impl<const N: usize> Default for Dims<N> {
    fn default() -> Self {
        Self { values: [0; N] }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
/// Semantic pair of `(cols, rows)` used by 2D tensor constructors.
pub struct Shape2D {
    pub cols: Cols,
    pub rows: Rows,
}

impl Shape2D {
    #[inline]
    pub const fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols: Cols::new(cols),
            rows: Rows::new(rows),
        }
    }

    #[inline]
    pub const fn dims(self) -> Dims<2> {
        Dims::new([self.cols.get(), self.rows.get()])
    }

    #[inline]
    pub const fn from_dims(dims: Dims<2>) -> Self {
        let dims = *dims.as_array();
        Self::new(dims[0], dims[1])
    }
}

impl From<Shape2D> for Dims<2> {
    fn from(value: Shape2D) -> Self {
        value.dims()
    }
}

impl From<Dims<2>> for Shape2D {
    fn from(value: Dims<2>) -> Self {
        Self::from_dims(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
/// Semantic 3D tensor dimensions (`ne0`, `ne1`, `ne2`).
pub struct Shape3D {
    pub ne0: usize,
    pub ne1: usize,
    pub ne2: usize,
}

impl Shape3D {
    #[inline]
    pub const fn new(ne0: usize, ne1: usize, ne2: usize) -> Self {
        Self { ne0, ne1, ne2 }
    }

    #[inline]
    pub const fn dims(self) -> Dims<3> {
        Dims::new([self.ne0, self.ne1, self.ne2])
    }

    #[inline]
    pub const fn from_dims(dims: Dims<3>) -> Self {
        let dims = *dims.as_array();
        Self::new(dims[0], dims[1], dims[2])
    }
}

impl From<Shape3D> for Dims<3> {
    fn from(value: Shape3D) -> Self {
        value.dims()
    }
}

impl From<Dims<3>> for Shape3D {
    fn from(value: Dims<3>) -> Self {
        Self::from_dims(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
/// Semantic 4D tensor dimensions (`ne0`, `ne1`, `ne2`, `ne3`).
pub struct Shape4D {
    pub ne0: usize,
    pub ne1: usize,
    pub ne2: usize,
    pub ne3: usize,
}

impl Shape4D {
    #[inline]
    pub const fn new(ne0: usize, ne1: usize, ne2: usize, ne3: usize) -> Self {
        Self { ne0, ne1, ne2, ne3 }
    }

    #[inline]
    pub const fn dims(self) -> Dims<4> {
        Dims::new([self.ne0, self.ne1, self.ne2, self.ne3])
    }

    #[inline]
    pub const fn from_dims(dims: Dims<4>) -> Self {
        let dims = *dims.as_array();
        Self::new(dims[0], dims[1], dims[2], dims[3])
    }
}

impl From<Shape4D> for Dims<4> {
    fn from(value: Shape4D) -> Self {
        value.dims()
    }
}

impl From<Dims<4>> for Shape4D {
    fn from(value: Dims<4>) -> Self {
        Self::from_dims(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
/// Compile-time shape marker used by typed tensor wrappers.
pub struct StaticShape2D<const COLS: usize, const ROWS: usize>;

/// Trait implemented by static shape markers.
///
/// It provides a `const` value that can be consumed by APIs expecting
/// runtime `Shape2D`.
pub trait Shape2DSpec {
    const SHAPE: Shape2D;
}

impl<const COLS: usize, const ROWS: usize> Shape2DSpec for StaticShape2D<COLS, ROWS> {
    const SHAPE: Shape2D = Shape2D::new(COLS, ROWS);
}
