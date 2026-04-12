use crate::ffi;
use std::fmt;
use std::os::raw::c_int;

/// All ggml tensor element/storage types.
///
/// Covers native floats, integers, and quantized block formats.  An `Unknown`
/// variant preserves forward-compatibility with future ggml versions.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1M,
    BF16,
    TQ1_0,
    TQ2_0,
    MXFP4,
    /// Forward-compatibility: an unrecognised ggml type id.
    Unknown(i32),
}

impl Type {
    /// Convert from a raw ggml type integer.
    pub const fn from_raw(raw: c_int) -> Self {
        match raw {
            ffi::GGML_TYPE_F32 => Self::F32,
            ffi::GGML_TYPE_F16 => Self::F16,
            ffi::GGML_TYPE_Q4_0 => Self::Q4_0,
            ffi::GGML_TYPE_Q4_1 => Self::Q4_1,
            ffi::GGML_TYPE_Q5_0 => Self::Q5_0,
            ffi::GGML_TYPE_Q5_1 => Self::Q5_1,
            ffi::GGML_TYPE_Q8_0 => Self::Q8_0,
            ffi::GGML_TYPE_Q8_1 => Self::Q8_1,
            ffi::GGML_TYPE_Q2_K => Self::Q2K,
            ffi::GGML_TYPE_Q3_K => Self::Q3K,
            ffi::GGML_TYPE_Q4_K => Self::Q4K,
            ffi::GGML_TYPE_Q5_K => Self::Q5K,
            ffi::GGML_TYPE_Q6_K => Self::Q6K,
            ffi::GGML_TYPE_Q8_K => Self::Q8K,
            ffi::GGML_TYPE_IQ2_XXS => Self::IQ2XXS,
            ffi::GGML_TYPE_IQ2_XS => Self::IQ2XS,
            ffi::GGML_TYPE_IQ3_XXS => Self::IQ3XXS,
            ffi::GGML_TYPE_IQ1_S => Self::IQ1S,
            ffi::GGML_TYPE_IQ4_NL => Self::IQ4NL,
            ffi::GGML_TYPE_IQ3_S => Self::IQ3S,
            ffi::GGML_TYPE_IQ2_S => Self::IQ2S,
            ffi::GGML_TYPE_IQ4_XS => Self::IQ4XS,
            ffi::GGML_TYPE_I8 => Self::I8,
            ffi::GGML_TYPE_I16 => Self::I16,
            ffi::GGML_TYPE_I32 => Self::I32,
            ffi::GGML_TYPE_I64 => Self::I64,
            ffi::GGML_TYPE_F64 => Self::F64,
            ffi::GGML_TYPE_IQ1_M => Self::IQ1M,
            ffi::GGML_TYPE_BF16 => Self::BF16,
            ffi::GGML_TYPE_TQ1_0 => Self::TQ1_0,
            ffi::GGML_TYPE_TQ2_0 => Self::TQ2_0,
            ffi::GGML_TYPE_MXFP4 => Self::MXFP4,
            other => Self::Unknown(other),
        }
    }

    /// Convert to the raw ggml type integer.
    pub const fn as_raw(self) -> c_int {
        match self {
            Self::F32 => ffi::GGML_TYPE_F32,
            Self::F16 => ffi::GGML_TYPE_F16,
            Self::Q4_0 => ffi::GGML_TYPE_Q4_0,
            Self::Q4_1 => ffi::GGML_TYPE_Q4_1,
            Self::Q5_0 => ffi::GGML_TYPE_Q5_0,
            Self::Q5_1 => ffi::GGML_TYPE_Q5_1,
            Self::Q8_0 => ffi::GGML_TYPE_Q8_0,
            Self::Q8_1 => ffi::GGML_TYPE_Q8_1,
            Self::Q2K => ffi::GGML_TYPE_Q2_K,
            Self::Q3K => ffi::GGML_TYPE_Q3_K,
            Self::Q4K => ffi::GGML_TYPE_Q4_K,
            Self::Q5K => ffi::GGML_TYPE_Q5_K,
            Self::Q6K => ffi::GGML_TYPE_Q6_K,
            Self::Q8K => ffi::GGML_TYPE_Q8_K,
            Self::IQ2XXS => ffi::GGML_TYPE_IQ2_XXS,
            Self::IQ2XS => ffi::GGML_TYPE_IQ2_XS,
            Self::IQ3XXS => ffi::GGML_TYPE_IQ3_XXS,
            Self::IQ1S => ffi::GGML_TYPE_IQ1_S,
            Self::IQ4NL => ffi::GGML_TYPE_IQ4_NL,
            Self::IQ3S => ffi::GGML_TYPE_IQ3_S,
            Self::IQ2S => ffi::GGML_TYPE_IQ2_S,
            Self::IQ4XS => ffi::GGML_TYPE_IQ4_XS,
            Self::I8 => ffi::GGML_TYPE_I8,
            Self::I16 => ffi::GGML_TYPE_I16,
            Self::I32 => ffi::GGML_TYPE_I32,
            Self::I64 => ffi::GGML_TYPE_I64,
            Self::F64 => ffi::GGML_TYPE_F64,
            Self::IQ1M => ffi::GGML_TYPE_IQ1_M,
            Self::BF16 => ffi::GGML_TYPE_BF16,
            Self::TQ1_0 => ffi::GGML_TYPE_TQ1_0,
            Self::TQ2_0 => ffi::GGML_TYPE_TQ2_0,
            Self::MXFP4 => ffi::GGML_TYPE_MXFP4,
            Self::Unknown(raw) => raw,
        }
    }

    /// Human-readable name (matches ggml convention, e.g. `"f32"`, `"q4_K"`).
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2K => "q2_K",
            Self::Q3K => "q3_K",
            Self::Q4K => "q4_K",
            Self::Q5K => "q5_K",
            Self::Q6K => "q6_K",
            Self::Q8K => "q8_K",
            Self::IQ2XXS => "iq2_xxs",
            Self::IQ2XS => "iq2_xs",
            Self::IQ3XXS => "iq3_xxs",
            Self::IQ1S => "iq1_s",
            Self::IQ4NL => "iq4_nl",
            Self::IQ3S => "iq3_s",
            Self::IQ2S => "iq2_s",
            Self::IQ4XS => "iq4_xs",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F64 => "f64",
            Self::IQ1M => "iq1_m",
            Self::BF16 => "bf16",
            Self::TQ1_0 => "tq1_0",
            Self::TQ2_0 => "tq2_0",
            Self::MXFP4 => "mxfp4",
            Self::Unknown(_) => "unknown",
        }
    }

    /// Whether this is a quantized block format (as opposed to a native scalar type).
    pub const fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Q4_0
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q8_1
                | Self::Q2K
                | Self::Q3K
                | Self::Q4K
                | Self::Q5K
                | Self::Q6K
                | Self::Q8K
                | Self::IQ2XXS
                | Self::IQ2XS
                | Self::IQ3XXS
                | Self::IQ1S
                | Self::IQ4NL
                | Self::IQ3S
                | Self::IQ2S
                | Self::IQ4XS
                | Self::IQ1M
                | Self::TQ1_0
                | Self::TQ2_0
                | Self::MXFP4
        )
    }

    /// Whether this is a native float type (`f32`, `f16`, `bf16`, `f64`).
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::F64)
    }

    /// Whether this is a native integer type (`i8`, `i16`, `i32`, `i64`).
    pub const fn is_integer(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    /// Convenience for callers that need to ensure the type is recognised.
    pub const fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown(_))
    }

    /// Shorthand used by the safe wrapper to map a Rust host type.
    pub const fn of<T: GgmlType>() -> Self {
        T::GGML_TYPE
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(raw) => write!(f, "Type::Unknown({raw})"),
            _ => write!(
                f,
                "Type::{}",
                match self {
                    Self::F32 => "F32",
                    Self::F16 => "F16",
                    Self::Q4_0 => "Q4_0",
                    Self::Q4_1 => "Q4_1",
                    Self::Q5_0 => "Q5_0",
                    Self::Q5_1 => "Q5_1",
                    Self::Q8_0 => "Q8_0",
                    Self::Q8_1 => "Q8_1",
                    Self::Q2K => "Q2K",
                    Self::Q3K => "Q3K",
                    Self::Q4K => "Q4K",
                    Self::Q5K => "Q5K",
                    Self::Q6K => "Q6K",
                    Self::Q8K => "Q8K",
                    Self::IQ2XXS => "IQ2XXS",
                    Self::IQ2XS => "IQ2XS",
                    Self::IQ3XXS => "IQ3XXS",
                    Self::IQ1S => "IQ1S",
                    Self::IQ4NL => "IQ4NL",
                    Self::IQ3S => "IQ3S",
                    Self::IQ2S => "IQ2S",
                    Self::IQ4XS => "IQ4XS",
                    Self::I8 => "I8",
                    Self::I16 => "I16",
                    Self::I32 => "I32",
                    Self::I64 => "I64",
                    Self::F64 => "F64",
                    Self::IQ1M => "IQ1M",
                    Self::BF16 => "BF16",
                    Self::TQ1_0 => "TQ1_0",
                    Self::TQ2_0 => "TQ2_0",
                    Self::MXFP4 => "MXFP4",
                    Self::Unknown(_) => unreachable!(),
                }
            ),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl From<c_int> for Type {
    fn from(raw: c_int) -> Self {
        Self::from_raw(raw)
    }
}

/// Type-level mapping between Rust host element types and ggml tensor kinds.
///
/// Only implemented for types that can live element-by-element in host memory
/// (`f32`, `i32`).  This is distinct from the storage `Type` enum which also
/// covers quantized block formats.
pub trait GgmlType {
    const GGML_TYPE: Type;
}

impl GgmlType for f32 {
    const GGML_TYPE: Type = Type::F32;
}

impl GgmlType for i32 {
    const GGML_TYPE: Type = Type::I32;
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
    fn from_raw_roundtrips_known_types() {
        for raw in 0..ffi::GGML_TYPE_COUNT {
            let ty = Type::from_raw(raw);
            assert_eq!(ty.as_raw(), raw, "roundtrip failed for raw={raw}");
        }
    }

    #[test]
    fn from_raw_known_types_are_not_unknown() {
        let known_count = (0..ffi::GGML_TYPE_COUNT)
            .filter(|&raw| !Type::from_raw(raw).is_unknown())
            .count();
        // We support at least the 32 known types (some raw ids are gaps)
        assert!(
            known_count >= 30,
            "expected >=30 known types, got {known_count}"
        );
    }

    #[test]
    fn unknown_type_preserved() {
        let ty = Type::from_raw(9999);
        assert!(ty.is_unknown());
        assert_eq!(ty.as_raw(), 9999);
        assert_eq!(ty.name(), "unknown");
    }

    #[test]
    fn classification_flags() {
        assert!(Type::F32.is_float());
        assert!(!Type::F32.is_quantized());
        assert!(!Type::F32.is_integer());

        assert!(Type::Q4K.is_quantized());
        assert!(!Type::Q4K.is_float());

        assert!(Type::I32.is_integer());
        assert!(!Type::I32.is_float());
    }

    #[test]
    fn display_matches_name() {
        assert_eq!(format!("{}", Type::Q4K), "q4_K");
        assert_eq!(format!("{}", Type::F32), "f32");
        assert_eq!(format!("{}", Type::BF16), "bf16");
    }

    #[test]
    fn from_c_int_trait() {
        let ty: Type = (0i32).into();
        assert_eq!(ty, Type::F32);
    }

    use crate::ffi;

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

    #[test]
    fn unknown_negative_raw_roundtrips() {
        let ty = Type::from_raw(-1);
        assert!(ty.is_unknown());
        assert_eq!(ty.as_raw(), -1);

        let ty2 = Type::from_raw(i32::MIN);
        assert!(ty2.is_unknown());
        assert_eq!(ty2.as_raw(), i32::MIN);
    }

    #[test]
    fn unknown_equality_by_raw_value() {
        assert_eq!(Type::Unknown(9999), Type::Unknown(9999));
        assert_ne!(Type::Unknown(9999), Type::Unknown(9998));
        assert_ne!(Type::Unknown(0), Type::F32); // raw 0 is F32, but Unknown(0) ≠ F32
    }

    #[test]
    fn all_float_types_classified() {
        let floats = [Type::F32, Type::F16, Type::BF16, Type::F64];
        for ty in floats {
            assert!(ty.is_float(), "{ty:?} should be float");
            assert!(!ty.is_integer(), "{ty:?} should not be integer");
            assert!(!ty.is_quantized(), "{ty:?} should not be quantized");
            assert!(!ty.is_unknown(), "{ty:?} should not be unknown");
        }
    }

    #[test]
    fn all_integer_types_classified() {
        let ints = [Type::I8, Type::I16, Type::I32, Type::I64];
        for ty in ints {
            assert!(ty.is_integer(), "{ty:?} should be integer");
            assert!(!ty.is_float(), "{ty:?} should not be float");
            assert!(!ty.is_quantized(), "{ty:?} should not be quantized");
            assert!(!ty.is_unknown(), "{ty:?} should not be unknown");
        }
    }

    #[test]
    fn all_quantized_types_classified() {
        let quants = [
            Type::Q4_0,
            Type::Q4_1,
            Type::Q5_0,
            Type::Q5_1,
            Type::Q8_0,
            Type::Q8_1,
            Type::Q2K,
            Type::Q3K,
            Type::Q4K,
            Type::Q5K,
            Type::Q6K,
            Type::Q8K,
            Type::IQ2XXS,
            Type::IQ2XS,
            Type::IQ3XXS,
            Type::IQ1S,
            Type::IQ4NL,
            Type::IQ3S,
            Type::IQ2S,
            Type::IQ4XS,
            Type::IQ1M,
            Type::TQ1_0,
            Type::TQ2_0,
            Type::MXFP4,
        ];
        for ty in quants {
            assert!(ty.is_quantized(), "{ty:?} should be quantized");
            assert!(!ty.is_float(), "{ty:?} should not be float");
            assert!(!ty.is_integer(), "{ty:?} should not be integer");
            assert!(!ty.is_unknown(), "{ty:?} should not be unknown");
        }
    }

    #[test]
    fn unknown_classification_all_false() {
        let ty = Type::Unknown(12345);
        assert!(!ty.is_float());
        assert!(!ty.is_integer());
        assert!(!ty.is_quantized());
        assert!(ty.is_unknown());
    }

    #[test]
    fn every_known_variant_is_exactly_one_category() {
        for raw in 0..ffi::GGML_TYPE_COUNT {
            let ty = Type::from_raw(raw);
            if ty.is_unknown() {
                continue;
            }
            let categories = [ty.is_float(), ty.is_integer(), ty.is_quantized()];
            let active = categories.iter().filter(|&&b| b).count();
            assert_eq!(
                active, 1,
                "{ty:?} (raw={raw}) should belong to exactly 1 category, got {active}"
            );
        }
    }

    #[test]
    fn debug_format_known_vs_unknown() {
        assert_eq!(format!("{:?}", Type::F32), "Type::F32");
        assert_eq!(format!("{:?}", Type::Q4K), "Type::Q4K");
        assert_eq!(format!("{:?}", Type::Unknown(42)), "Type::Unknown(42)");
    }

    #[test]
    fn type_of_maps_rust_scalars() {
        assert_eq!(Type::of::<f32>(), Type::F32);
        assert_eq!(Type::of::<i32>(), Type::I32);
    }

    #[test]
    fn hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Type::F32);
        set.insert(Type::F32);
        assert_eq!(set.len(), 1);

        set.insert(Type::Unknown(999));
        set.insert(Type::Unknown(999));
        assert_eq!(set.len(), 2);
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
