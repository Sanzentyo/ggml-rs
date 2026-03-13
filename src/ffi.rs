#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_void};

pub const GGML_TYPE_F32: c_int = 0;
pub const GGML_TYPE_I32: c_int = 26;
pub const GGML_STATUS_SUCCESS: c_int = 0;
pub const GGML_BACKEND_DEVICE_TYPE_CPU: c_int = 0;
pub const GGML_BACKEND_DEVICE_TYPE_GPU: c_int = 1;
pub const GGML_BACKEND_DEVICE_TYPE_IGPU: c_int = 2;
pub const GGUF_TYPE_UINT8: c_int = 0;
pub const GGUF_TYPE_INT8: c_int = 1;
pub const GGUF_TYPE_UINT16: c_int = 2;
pub const GGUF_TYPE_INT16: c_int = 3;
pub const GGUF_TYPE_UINT32: c_int = 4;
pub const GGUF_TYPE_INT32: c_int = 5;
pub const GGUF_TYPE_FLOAT32: c_int = 6;
pub const GGUF_TYPE_BOOL: c_int = 7;
pub const GGUF_TYPE_STRING: c_int = 8;
pub const GGUF_TYPE_ARRAY: c_int = 9;
pub const GGUF_TYPE_UINT64: c_int = 10;
pub const GGUF_TYPE_INT64: c_int = 11;
pub const GGUF_TYPE_FLOAT64: c_int = 12;

#[repr(C)]
pub struct ggml_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_tensor {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_cgraph {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_backend {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_backend_buffer {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_backend_device {
    _private: [u8; 0],
}

#[repr(C)]
pub struct gguf_context {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ggml_init_params {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct gguf_init_params {
    pub no_alloc: bool,
    pub ctx: *mut *mut ggml_context,
}

unsafe extern "C" {
    pub fn ggml_time_init();
    pub fn ggml_type_size(type_: c_int) -> usize;
    pub fn ggml_type_name(type_: c_int) -> *const c_char;
    pub fn ggml_tensor_overhead() -> usize;
    pub fn ggml_graph_overhead() -> usize;

    pub fn ggml_init(params: ggml_init_params) -> *mut ggml_context;
    pub fn ggml_free(ctx: *mut ggml_context);

    pub fn ggml_new_tensor_2d(
        ctx: *mut ggml_context,
        type_: c_int,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;
    pub fn ggml_new_tensor_1d(ctx: *mut ggml_context, type_: c_int, ne0: i64) -> *mut ggml_tensor;
    pub fn ggml_new_tensor_3d(
        ctx: *mut ggml_context,
        type_: c_int,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut ggml_tensor;
    pub fn ggml_new_tensor_4d(
        ctx: *mut ggml_context,
        type_: c_int,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> *mut ggml_tensor;

    pub fn ggml_nelements(tensor: *const ggml_tensor) -> i64;
    pub fn ggml_nrows(tensor: *const ggml_tensor) -> i64;
    pub fn ggml_nbytes(tensor: *const ggml_tensor) -> usize;

    pub fn ggml_set_f32_1d(tensor: *const ggml_tensor, i: c_int, value: f32);
    pub fn ggml_get_f32_1d(tensor: *const ggml_tensor, i: c_int) -> f32;
    pub fn ggml_set_name(tensor: *mut ggml_tensor, name: *const c_char) -> *mut ggml_tensor;
    pub fn ggml_get_name(tensor: *const ggml_tensor) -> *const c_char;

    pub fn ggml_mul_mat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    pub fn ggml_add(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    pub fn ggml_mul(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    pub fn ggml_scale(ctx: *mut ggml_context, a: *mut ggml_tensor, s: f32) -> *mut ggml_tensor;
    pub fn ggml_repeat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    pub fn ggml_silu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_rms_norm(ctx: *mut ggml_context, a: *mut ggml_tensor, eps: f32)
    -> *mut ggml_tensor;
    pub fn ggml_get_rows(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    pub fn ggml_cpy(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    pub fn ggml_cont(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_reshape_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;
    pub fn ggml_reshape_3d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut ggml_tensor;
    pub fn ggml_view_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        offset: usize,
    ) -> *mut ggml_tensor;
    pub fn ggml_view_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        nb1: usize,
        offset: usize,
    ) -> *mut ggml_tensor;
    pub fn ggml_permute(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        axis0: c_int,
        axis1: c_int,
        axis2: c_int,
        axis3: c_int,
    ) -> *mut ggml_tensor;
    pub fn ggml_diag_mask_inf(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: c_int,
    ) -> *mut ggml_tensor;
    pub fn ggml_soft_max(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_soft_max_ext(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        mask: *mut ggml_tensor,
        scale: f32,
        max_bias: f32,
    ) -> *mut ggml_tensor;
    pub fn ggml_rope_ext(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        n_dims: c_int,
        mode: c_int,
        n_ctx_orig: c_int,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> *mut ggml_tensor;

    pub fn ggml_new_graph(ctx: *mut ggml_context) -> *mut ggml_cgraph;
    pub fn ggml_build_forward_expand(cgraph: *mut ggml_cgraph, tensor: *mut ggml_tensor);
    pub fn ggml_graph_n_nodes(cgraph: *mut ggml_cgraph) -> c_int;
    pub fn ggml_graph_node(cgraph: *mut ggml_cgraph, i: c_int) -> *mut ggml_tensor;
    pub fn ggml_graph_compute_with_ctx(
        ctx: *mut ggml_context,
        cgraph: *mut ggml_cgraph,
        n_threads: c_int,
    ) -> c_int;

    pub fn ggml_backend_load_all();
    pub fn ggml_backend_init_by_name(
        name: *const c_char,
        params: *const c_char,
    ) -> *mut ggml_backend;
    pub fn ggml_backend_init_by_type(type_: c_int, params: *const c_char) -> *mut ggml_backend;
    pub fn ggml_backend_name(backend: *mut ggml_backend) -> *const c_char;
    pub fn ggml_backend_free(backend: *mut ggml_backend);
    pub fn ggml_backend_dev_count() -> usize;
    pub fn ggml_backend_dev_get(index: usize) -> *mut ggml_backend_device;
    pub fn ggml_backend_dev_name(device: *mut ggml_backend_device) -> *const c_char;
    pub fn ggml_backend_dev_type(device: *mut ggml_backend_device) -> c_int;
    pub fn ggml_backend_dev_init(
        device: *mut ggml_backend_device,
        params: *const c_char,
    ) -> *mut ggml_backend;

    pub fn ggml_backend_graph_compute(
        backend: *mut ggml_backend,
        cgraph: *mut ggml_cgraph,
    ) -> c_int;

    pub fn ggml_backend_tensor_set(
        tensor: *mut ggml_tensor,
        data: *const c_void,
        offset: usize,
        size: usize,
    );
    pub fn ggml_backend_tensor_get(
        tensor: *const ggml_tensor,
        data: *mut c_void,
        offset: usize,
        size: usize,
    );

    pub fn ggml_backend_alloc_ctx_tensors(
        ctx: *mut ggml_context,
        backend: *mut ggml_backend,
    ) -> *mut ggml_backend_buffer;
    pub fn ggml_backend_buffer_free(buffer: *mut ggml_backend_buffer);

    pub fn gguf_init_from_file(fname: *const c_char, params: gguf_init_params)
    -> *mut gguf_context;
    pub fn gguf_free(ctx: *mut gguf_context);

    pub fn gguf_get_version(ctx: *const gguf_context) -> u32;
    pub fn gguf_get_alignment(ctx: *const gguf_context) -> usize;
    pub fn gguf_get_data_offset(ctx: *const gguf_context) -> usize;

    pub fn gguf_get_n_kv(ctx: *const gguf_context) -> i64;
    pub fn gguf_find_key(ctx: *const gguf_context, key: *const c_char) -> i64;
    pub fn gguf_get_key(ctx: *const gguf_context, key_id: i64) -> *const c_char;
    pub fn gguf_get_kv_type(ctx: *const gguf_context, key_id: i64) -> c_int;
    pub fn gguf_get_val_str(ctx: *const gguf_context, key_id: i64) -> *const c_char;
    pub fn gguf_type_name(type_: c_int) -> *const c_char;

    pub fn gguf_get_n_tensors(ctx: *const gguf_context) -> i64;
    pub fn gguf_get_tensor_name(ctx: *const gguf_context, tensor_id: i64) -> *const c_char;
    pub fn gguf_get_tensor_type(ctx: *const gguf_context, tensor_id: i64) -> c_int;
    pub fn gguf_get_tensor_size(ctx: *const gguf_context, tensor_id: i64) -> usize;
    pub fn gguf_get_tensor_offset(ctx: *const gguf_context, tensor_id: i64) -> usize;
}
