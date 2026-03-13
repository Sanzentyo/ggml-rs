use crate::{Error, ffi};
use std::os::raw::c_int;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Subset of ggml tensor element types supported by this safe wrapper.
#[repr(i32)]
pub enum Type {
    F32 = ffi::GGML_TYPE_F32,
    I32 = ffi::GGML_TYPE_I32,
}

impl Type {
    pub(crate) const fn as_raw(self) -> c_int {
        self as c_int
    }
}

impl TryFrom<c_int> for Type {
    type Error = Error;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        match value {
            raw if raw == Self::F32.as_raw() => Ok(Self::F32),
            raw if raw == Self::I32.as_raw() => Ok(Self::I32),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Raw ggml backend device type (`ggml_backend_dev_type`).
#[repr(i32)]
pub enum BackendDeviceType {
    Cpu = ffi::GGML_BACKEND_DEVICE_TYPE_CPU,
    Gpu = ffi::GGML_BACKEND_DEVICE_TYPE_GPU,
    IntegratedGpu = ffi::GGML_BACKEND_DEVICE_TYPE_IGPU,
}

impl BackendDeviceType {
    pub(crate) const fn as_raw(self) -> c_int {
        self as c_int
    }

    pub(crate) const fn from_raw(raw: c_int) -> Option<Self> {
        match raw {
            ffi::GGML_BACKEND_DEVICE_TYPE_CPU => Some(Self::Cpu),
            ffi::GGML_BACKEND_DEVICE_TYPE_GPU => Some(Self::Gpu),
            ffi::GGML_BACKEND_DEVICE_TYPE_IGPU => Some(Self::IntegratedGpu),
            _ => None,
        }
    }

    pub(crate) const fn is_gpu_like(self) -> bool {
        matches!(self, Self::Gpu | Self::IntegratedGpu)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Raw ggml compute status (`ggml_graph_compute_with_ctx`, `ggml_backend_graph_compute`).
#[repr(i32)]
pub enum ComputeStatus {
    Success = ffi::GGML_STATUS_SUCCESS,
}

impl ComputeStatus {
    pub(crate) const fn is_success(raw: c_int) -> bool {
        raw == Self::Success as c_int
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendDeviceType, ComputeStatus, Type};

    #[test]
    fn converts_tensor_type_from_raw() {
        assert_eq!(Type::try_from(Type::F32 as i32).ok(), Some(Type::F32));
        assert_eq!(Type::try_from(Type::I32 as i32).ok(), Some(Type::I32));
    }

    #[test]
    fn backend_device_type_flags_gpu_like() {
        assert!(BackendDeviceType::Gpu.is_gpu_like());
        assert!(BackendDeviceType::IntegratedGpu.is_gpu_like());
        assert!(!BackendDeviceType::Cpu.is_gpu_like());
    }

    #[test]
    fn compute_status_success_check() {
        assert!(ComputeStatus::is_success(ComputeStatus::Success as i32));
        assert!(!ComputeStatus::is_success(-1));
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
