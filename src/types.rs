use crate::{Error, ffi};
use std::os::raw::c_int;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Subset of ggml tensor element types supported by this safe wrapper.
pub enum Type {
    F32,
    I32,
}

impl Type {
    pub(crate) fn as_raw(self) -> c_int {
        match self {
            Self::F32 => ffi::GGML_TYPE_F32,
            Self::I32 => ffi::GGML_TYPE_I32,
        }
    }
}

impl TryFrom<c_int> for Type {
    type Error = Error;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        match value {
            ffi::GGML_TYPE_F32 => Ok(Self::F32),
            ffi::GGML_TYPE_I32 => Ok(Self::I32),
            other => Err(Error::UnsupportedType(other)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Backend family requested by safe backend initialization.
pub enum BackendKind {
    Cpu,
    Metal,
}

impl BackendKind {
    pub const fn as_name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Metal => "Metal",
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Parameters for `ggml_rope_ext`.
///
/// Defaults match common ggml usage and can be overridden per call.
pub struct RopeExtParams {
    pub n_dims: i32,
    pub mode: i32,
    pub n_ctx_orig: i32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl Default for RopeExtParams {
    fn default() -> Self {
        Self {
            n_dims: 0,
            mode: 0,
            n_ctx_orig: 0,
            freq_base: 10_000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}
