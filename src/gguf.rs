//! Safe GGUF inspection helpers built on top of ggml's GGUF APIs.

use crate::ffi;
use crate::num_ext::TryIntoChecked;
use crate::{Error, Result};
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::os::raw::c_int;
use std::path::Path;
use std::ptr::{self, NonNull};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// GGUF key-value type.
pub enum GgufType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
    Unknown(i32),
}

impl GgufType {
    fn from_raw(raw: c_int) -> Self {
        match raw {
            ffi::GGUF_TYPE_UINT8 => Self::Uint8,
            ffi::GGUF_TYPE_INT8 => Self::Int8,
            ffi::GGUF_TYPE_UINT16 => Self::Uint16,
            ffi::GGUF_TYPE_INT16 => Self::Int16,
            ffi::GGUF_TYPE_UINT32 => Self::Uint32,
            ffi::GGUF_TYPE_INT32 => Self::Int32,
            ffi::GGUF_TYPE_FLOAT32 => Self::Float32,
            ffi::GGUF_TYPE_BOOL => Self::Bool,
            ffi::GGUF_TYPE_STRING => Self::String,
            ffi::GGUF_TYPE_ARRAY => Self::Array,
            ffi::GGUF_TYPE_UINT64 => Self::Uint64,
            ffi::GGUF_TYPE_INT64 => Self::Int64,
            ffi::GGUF_TYPE_FLOAT64 => Self::Float64,
            _ => Self::Unknown(raw),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Metadata for a tensor entry in a GGUF file.
pub struct GgufTensorInfo {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub ggml_type_raw: i32,
    pub ggml_type_name: String,
}

/// RAII owner for a GGUF context.
pub struct GgufFile {
    raw: NonNull<ffi::gguf_context>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl GgufFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path_to_c_string(path.as_ref())?;
        let params = ffi::gguf_init_params {
            no_alloc: true,
            ctx: ptr::null_mut(),
        };
        let raw = unsafe { ffi::gguf_init_from_file(path.as_ptr(), params) };
        let raw = NonNull::new(raw).ok_or_else(|| Error::null_pointer("gguf_init_from_file"))?;

        Ok(Self {
            raw,
            _not_send_sync: PhantomData,
        })
    }

    pub fn version(&self) -> u32 {
        unsafe { ffi::gguf_get_version(self.raw.as_ptr()) }
    }

    pub fn alignment(&self) -> usize {
        unsafe { ffi::gguf_get_alignment(self.raw.as_ptr()) }
    }

    pub fn data_offset(&self) -> usize {
        unsafe { ffi::gguf_get_data_offset(self.raw.as_ptr()) }
    }

    pub fn kv_count(&self) -> Result<usize> {
        (unsafe { ffi::gguf_get_n_kv(self.raw.as_ptr()) })
            .try_into_checked()
            .map_err(|source| Error::int_conversion("gguf_get_n_kv", source))
    }

    pub fn kv_key(&self, index: usize) -> Result<String> {
        let index = index
            .try_into_checked()
            .map_err(|source| Error::int_conversion("gguf_get_key index", source))?;
        let ptr = unsafe { ffi::gguf_get_key(self.raw.as_ptr(), index) };
        if ptr.is_null() {
            return Err(Error::null_pointer("gguf_get_key"));
        }
        let cstr = unsafe { CStr::from_ptr(ptr) };
        Ok(cstr.to_str()?.to_owned())
    }

    pub fn find_key(&self, key: &str) -> Result<Option<usize>> {
        let key = CString::new(key)?;
        let idx = unsafe { ffi::gguf_find_key(self.raw.as_ptr(), key.as_ptr()) };
        if idx < 0 {
            Ok(None)
        } else {
            Ok(Some(idx.try_into_checked().map_err(|source| {
                Error::int_conversion("gguf_find_key index", source)
            })?))
        }
    }

    pub fn kv_type(&self, index: usize) -> Result<GgufType> {
        let index = index
            .try_into_checked()
            .map_err(|source| Error::int_conversion("gguf_get_kv_type index", source))?;
        let raw = unsafe { ffi::gguf_get_kv_type(self.raw.as_ptr(), index) };
        Ok(GgufType::from_raw(raw))
    }

    pub fn kv_type_name(&self, index: usize) -> Result<String> {
        let index = index
            .try_into_checked()
            .map_err(|source| Error::int_conversion("gguf_get_kv_type index", source))?;
        let raw = unsafe { ffi::gguf_get_kv_type(self.raw.as_ptr(), index) };
        let ptr = unsafe { ffi::gguf_type_name(raw) };
        if ptr.is_null() {
            return Err(Error::null_pointer("gguf_type_name"));
        }
        let cstr = unsafe { CStr::from_ptr(ptr) };
        Ok(cstr.to_str()?.to_owned())
    }

    pub fn kv_string_value(&self, index: usize) -> Result<String> {
        let index = index
            .try_into_checked()
            .map_err(|source| Error::int_conversion("gguf_get_val_str index", source))?;
        let ptr = unsafe { ffi::gguf_get_val_str(self.raw.as_ptr(), index) };
        if ptr.is_null() {
            return Err(Error::null_pointer("gguf_get_val_str"));
        }
        let cstr = unsafe { CStr::from_ptr(ptr) };
        Ok(cstr.to_str()?.to_owned())
    }

    pub fn tensor_count(&self) -> Result<usize> {
        (unsafe { ffi::gguf_get_n_tensors(self.raw.as_ptr()) })
            .try_into_checked()
            .map_err(|source| Error::int_conversion("gguf_get_n_tensors", source))
    }

    pub fn tensor_info(&self, index: usize) -> Result<GgufTensorInfo> {
        let index_i64 = index
            .try_into_checked()
            .map_err(|source| Error::int_conversion("gguf_get_tensor_name index", source))?;

        let name_ptr = unsafe { ffi::gguf_get_tensor_name(self.raw.as_ptr(), index_i64) };
        if name_ptr.is_null() {
            return Err(Error::null_pointer("gguf_get_tensor_name"));
        }
        let name = unsafe { CStr::from_ptr(name_ptr) }.to_str()?.to_owned();

        let ggml_type_raw = unsafe { ffi::gguf_get_tensor_type(self.raw.as_ptr(), index_i64) };
        let type_ptr = unsafe { ffi::ggml_type_name(ggml_type_raw) };
        if type_ptr.is_null() {
            return Err(Error::null_pointer("ggml_type_name"));
        }
        let ggml_type_name = unsafe { CStr::from_ptr(type_ptr) }.to_str()?.to_owned();

        let offset = unsafe { ffi::gguf_get_tensor_offset(self.raw.as_ptr(), index_i64) };
        let size = unsafe { ffi::gguf_get_tensor_size(self.raw.as_ptr(), index_i64) };

        Ok(GgufTensorInfo {
            name,
            offset,
            size,
            ggml_type_raw,
            ggml_type_name,
        })
    }
}

impl Drop for GgufFile {
    fn drop(&mut self) {
        unsafe {
            ffi::gguf_free(self.raw.as_ptr());
        }
    }
}

fn path_to_c_string(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        CString::new(path.as_os_str().as_bytes()).map_err(Error::from)
    }

    #[cfg(not(unix))]
    {
        let lossy = path.to_string_lossy();
        CString::new(lossy.as_bytes()).map_err(Error::from)
    }
}
