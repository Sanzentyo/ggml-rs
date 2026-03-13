//! Zero-cost semantic wrappers for dimensions, lengths, and byte counts.
//!
//! These newtypes make call sites self-descriptive while compiling down to
//! plain integers (`#[repr(transparent)]`).

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
/// Number of columns in a 2D tensor view (`ne0` in ggml terms).
pub struct Cols(usize);

impl Cols {
    #[inline]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
/// Number of rows in a 2D tensor view (`ne1` in ggml terms).
pub struct Rows(usize);

impl Rows {
    #[inline]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
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

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
/// 1D tensor length wrapper.
pub struct Length(usize);

impl Length {
    #[inline]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
/// Tensor index wrapper used by bounds-checked accessors.
pub struct TensorIndex(usize);

impl TensorIndex {
    #[inline]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
/// Number of compute threads used by legacy graph execution.
pub struct ThreadCount(usize);

impl ThreadCount {
    #[inline]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
/// Byte-size wrapper used for context memory sizing APIs.
pub struct Bytes(usize);

impl Bytes {
    #[inline]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}
