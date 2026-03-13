//! C FFI surface for ggml.
//!
//! Primary path: bindgen-generated bindings (`build.rs`, `ggml_rs_bindgen` cfg).
//! Fallback path: checked-in manual bindings (`ffi_manual.rs`) when bindgen is
//! disabled or ggml headers are unavailable.

#[cfg(ggml_rs_bindgen)]
mod bindings {
    #![allow(
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        improper_ctypes
    )]
    use std::os::raw::c_int;

    include!(concat!(env!("OUT_DIR"), "/ffi_bindings.rs"));

    pub const GGML_TYPE_F32: c_int = ggml_type_GGML_TYPE_F32 as c_int;
    pub const GGML_TYPE_I32: c_int = ggml_type_GGML_TYPE_I32 as c_int;
    pub const GGML_STATUS_SUCCESS: c_int = ggml_status_GGML_STATUS_SUCCESS as c_int;
    pub const GGML_BACKEND_DEVICE_TYPE_CPU: c_int =
        ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU as c_int;
    pub const GGML_BACKEND_DEVICE_TYPE_GPU: c_int =
        ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU as c_int;
    pub const GGML_BACKEND_DEVICE_TYPE_IGPU: c_int =
        ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU as c_int;
    pub const GGUF_TYPE_UINT8: c_int = gguf_type_GGUF_TYPE_UINT8 as c_int;
    pub const GGUF_TYPE_INT8: c_int = gguf_type_GGUF_TYPE_INT8 as c_int;
    pub const GGUF_TYPE_UINT16: c_int = gguf_type_GGUF_TYPE_UINT16 as c_int;
    pub const GGUF_TYPE_INT16: c_int = gguf_type_GGUF_TYPE_INT16 as c_int;
    pub const GGUF_TYPE_UINT32: c_int = gguf_type_GGUF_TYPE_UINT32 as c_int;
    pub const GGUF_TYPE_INT32: c_int = gguf_type_GGUF_TYPE_INT32 as c_int;
    pub const GGUF_TYPE_FLOAT32: c_int = gguf_type_GGUF_TYPE_FLOAT32 as c_int;
    pub const GGUF_TYPE_BOOL: c_int = gguf_type_GGUF_TYPE_BOOL as c_int;
    pub const GGUF_TYPE_STRING: c_int = gguf_type_GGUF_TYPE_STRING as c_int;
    pub const GGUF_TYPE_ARRAY: c_int = gguf_type_GGUF_TYPE_ARRAY as c_int;
    pub const GGUF_TYPE_UINT64: c_int = gguf_type_GGUF_TYPE_UINT64 as c_int;
    pub const GGUF_TYPE_INT64: c_int = gguf_type_GGUF_TYPE_INT64 as c_int;
    pub const GGUF_TYPE_FLOAT64: c_int = gguf_type_GGUF_TYPE_FLOAT64 as c_int;
}

#[cfg(not(ggml_rs_bindgen))]
mod bindings {
    #![allow(non_camel_case_types)]
    include!("ffi_manual.rs");
}

pub use bindings::*;
