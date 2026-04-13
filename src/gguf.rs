//! Safe GGUF inspection helpers built on top of ggml's GGUF APIs.

use crate::ffi;
use crate::num_ext::TryIntoChecked;
use crate::types::Type;
use crate::{DynTensor, Error, Result, Tensor};
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::os::raw::{c_char, c_int};
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

#[derive(Debug, Clone, PartialEq)]
/// Typed GGUF array payload.
pub enum GgufArrayValue {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    String(Vec<String>),
}

impl GgufArrayValue {
    pub fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::I8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::Bool(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::String(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn element_type_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F64(_) => "f64",
            Self::String(_) => "string",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Typed GGUF key-value payload.
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(GgufArrayValue),
}

impl GgufValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F64(_) => "f64",
            Self::Array(_) => "array",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Metadata for a tensor entry in a GGUF file.
pub struct GgufTensorInfo {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub ggml_type: Type,
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

    pub fn find_key(&self, key: impl AsRef<str>) -> Result<Option<usize>> {
        let key = CString::new(key.as_ref())?;
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
        Ok(GgufType::from_raw(raw as c_int))
    }

    pub fn kv_value(&self, index: usize) -> Result<GgufValue> {
        let value_type = self.kv_type(index)?;
        let key_id = index
            .try_into_checked()
            .map_err(|source| Error::int_conversion("gguf_get_val index", source))?;
        self.kv_value_from_type(key_id, value_type)
    }

    pub fn kv_value_by_key(&self, key: impl AsRef<str>) -> Result<Option<GgufValue>> {
        let Some(index) = self.find_key(key.as_ref())? else {
            return Ok(None);
        };
        self.kv_value(index).map(Some)
    }

    pub fn kv_value_as<T: TryFromGgufValue>(&self, key: impl AsRef<str>) -> Result<Option<T>> {
        let Some(value) = self.kv_value_by_key(key)? else {
            return Ok(None);
        };
        T::try_from_gguf_value(&value).map(Some)
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
        c_string_from_ptr(ptr, "gguf_get_val_str")
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
        let ggml_type = Type::from_raw(ggml_type_raw as c_int);

        let offset = unsafe { ffi::gguf_get_tensor_offset(self.raw.as_ptr(), index_i64) };
        let size = unsafe { ffi::gguf_get_tensor_size(self.raw.as_ptr(), index_i64) };

        Ok(GgufTensorInfo {
            name,
            offset,
            size,
            ggml_type,
        })
    }

    fn kv_value_from_type(&self, index: i64, value_type: GgufType) -> Result<GgufValue> {
        let value = match value_type {
            GgufType::Uint8 => {
                GgufValue::U8(unsafe { ffi::gguf_get_val_u8(self.raw.as_ptr(), index) })
            }
            GgufType::Int8 => {
                GgufValue::I8(unsafe { ffi::gguf_get_val_i8(self.raw.as_ptr(), index) })
            }
            GgufType::Uint16 => {
                GgufValue::U16(unsafe { ffi::gguf_get_val_u16(self.raw.as_ptr(), index) })
            }
            GgufType::Int16 => {
                GgufValue::I16(unsafe { ffi::gguf_get_val_i16(self.raw.as_ptr(), index) })
            }
            GgufType::Uint32 => {
                GgufValue::U32(unsafe { ffi::gguf_get_val_u32(self.raw.as_ptr(), index) })
            }
            GgufType::Int32 => {
                GgufValue::I32(unsafe { ffi::gguf_get_val_i32(self.raw.as_ptr(), index) })
            }
            GgufType::Float32 => {
                GgufValue::F32(unsafe { ffi::gguf_get_val_f32(self.raw.as_ptr(), index) })
            }
            GgufType::Bool => {
                GgufValue::Bool(unsafe { ffi::gguf_get_val_bool(self.raw.as_ptr(), index) })
            }
            GgufType::String => {
                let ptr = unsafe { ffi::gguf_get_val_str(self.raw.as_ptr(), index) };
                GgufValue::String(c_string_from_ptr(ptr, "gguf_get_val_str")?)
            }
            GgufType::Uint64 => {
                GgufValue::U64(unsafe { ffi::gguf_get_val_u64(self.raw.as_ptr(), index) })
            }
            GgufType::Int64 => {
                GgufValue::I64(unsafe { ffi::gguf_get_val_i64(self.raw.as_ptr(), index) })
            }
            GgufType::Float64 => {
                GgufValue::F64(unsafe { ffi::gguf_get_val_f64(self.raw.as_ptr(), index) })
            }
            GgufType::Array => GgufValue::Array(self.read_array(index)?),
            GgufType::Unknown(raw) => return Err(Error::UnsupportedType(raw)),
        };
        Ok(value)
    }

    fn read_array(&self, index: i64) -> Result<GgufArrayValue> {
        let arr_type_raw = unsafe { ffi::gguf_get_arr_type(self.raw.as_ptr(), index) };
        let arr_type = GgufType::from_raw(arr_type_raw as c_int);
        let len = unsafe { ffi::gguf_get_arr_n(self.raw.as_ptr(), index) };
        match arr_type {
            GgufType::Uint8 => self
                .read_array_data::<u8>(index, len, "gguf_get_arr_data<u8>")
                .map(GgufArrayValue::U8),
            GgufType::Int8 => self
                .read_array_data::<i8>(index, len, "gguf_get_arr_data<i8>")
                .map(GgufArrayValue::I8),
            GgufType::Uint16 => self
                .read_array_data::<u16>(index, len, "gguf_get_arr_data<u16>")
                .map(GgufArrayValue::U16),
            GgufType::Int16 => self
                .read_array_data::<i16>(index, len, "gguf_get_arr_data<i16>")
                .map(GgufArrayValue::I16),
            GgufType::Uint32 => self
                .read_array_data::<u32>(index, len, "gguf_get_arr_data<u32>")
                .map(GgufArrayValue::U32),
            GgufType::Int32 => self
                .read_array_data::<i32>(index, len, "gguf_get_arr_data<i32>")
                .map(GgufArrayValue::I32),
            GgufType::Float32 => self
                .read_array_data::<f32>(index, len, "gguf_get_arr_data<f32>")
                .map(GgufArrayValue::F32),
            GgufType::Uint64 => self
                .read_array_data::<u64>(index, len, "gguf_get_arr_data<u64>")
                .map(GgufArrayValue::U64),
            GgufType::Int64 => self
                .read_array_data::<i64>(index, len, "gguf_get_arr_data<i64>")
                .map(GgufArrayValue::I64),
            GgufType::Float64 => self
                .read_array_data::<f64>(index, len, "gguf_get_arr_data<f64>")
                .map(GgufArrayValue::F64),
            GgufType::Bool => self
                .read_array_data::<i8>(index, len, "gguf_get_arr_data<bool>")
                .map(|values| values.into_iter().map(|value| value != 0).collect())
                .map(GgufArrayValue::Bool),
            GgufType::String => {
                let mut values = Vec::with_capacity(len);
                for arr_index in 0..len {
                    let ptr = unsafe { ffi::gguf_get_arr_str(self.raw.as_ptr(), index, arr_index) };
                    values.push(c_string_from_ptr(ptr, "gguf_get_arr_str")?);
                }
                Ok(GgufArrayValue::String(values))
            }
            GgufType::Array => Err(Error::UnsupportedType(ffi::GGUF_TYPE_ARRAY)),
            GgufType::Unknown(raw) => Err(Error::UnsupportedType(raw)),
        }
    }

    fn read_array_data<T: Copy>(
        &self,
        index: i64,
        len: usize,
        context: &'static str,
    ) -> Result<Vec<T>> {
        let ptr = unsafe { ffi::gguf_get_arr_data(self.raw.as_ptr(), index) };
        if ptr.is_null() && len != 0 {
            return Err(Error::null_pointer(context));
        }
        let data = if len == 0 {
            Vec::new()
        } else {
            let slice = unsafe { std::slice::from_raw_parts(ptr.cast::<T>(), len) };
            slice.to_vec()
        };
        Ok(data)
    }
}

impl Drop for GgufFile {
    fn drop(&mut self) {
        unsafe {
            ffi::gguf_free(self.raw.as_ptr());
        }
    }
}

/// RAII owner for a writable GGUF context.
pub struct GgufWriter {
    raw: NonNull<ffi::gguf_context>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl GgufWriter {
    pub fn new() -> Result<Self> {
        let raw = unsafe { ffi::gguf_init_empty() };
        let raw = NonNull::new(raw).ok_or_else(|| Error::null_pointer("gguf_init_empty"))?;
        Ok(Self {
            raw,
            _not_send_sync: PhantomData,
        })
    }

    pub fn set_value(&mut self, key: impl AsRef<str>, value: &GgufValue) -> Result<()> {
        let key = CString::new(key.as_ref())?;
        let key_ptr = key.as_ptr();

        match value {
            GgufValue::U8(value) => unsafe {
                ffi::gguf_set_val_u8(self.raw.as_ptr(), key_ptr, *value);
            },
            GgufValue::I8(value) => unsafe {
                ffi::gguf_set_val_i8(self.raw.as_ptr(), key_ptr, *value);
            },
            GgufValue::U16(value) => unsafe {
                ffi::gguf_set_val_u16(self.raw.as_ptr(), key_ptr, *value);
            },
            GgufValue::I16(value) => unsafe {
                ffi::gguf_set_val_i16(self.raw.as_ptr(), key_ptr, *value);
            },
            GgufValue::U32(value) => unsafe {
                ffi::gguf_set_val_u32(self.raw.as_ptr(), key_ptr, *value);
            },
            GgufValue::I32(value) => unsafe {
                ffi::gguf_set_val_i32(self.raw.as_ptr(), key_ptr, *value);
            },
            GgufValue::F32(value) => unsafe {
                ffi::gguf_set_val_f32(self.raw.as_ptr(), key_ptr, *value);
            },
            GgufValue::Bool(value) => unsafe {
                ffi::gguf_set_val_bool(self.raw.as_ptr(), key_ptr, *value);
            },
            GgufValue::String(value) => {
                let value = CString::new(value.as_str())?;
                unsafe {
                    ffi::gguf_set_val_str(self.raw.as_ptr(), key_ptr, value.as_ptr());
                }
            }
            GgufValue::U64(value) => unsafe {
                ffi::gguf_set_val_u64(self.raw.as_ptr(), key_ptr, *value);
            },
            GgufValue::I64(value) => unsafe {
                ffi::gguf_set_val_i64(self.raw.as_ptr(), key_ptr, *value);
            },
            GgufValue::F64(value) => unsafe {
                ffi::gguf_set_val_f64(self.raw.as_ptr(), key_ptr, *value);
            },
            GgufValue::Array(value) => self.set_array_value(key_ptr, value)?,
        }

        Ok(())
    }

    pub fn set_values<'a, I>(&mut self, values: I) -> Result<()>
    where
        I: IntoIterator<Item = (&'a str, &'a GgufValue)>,
    {
        for (key, value) in values {
            self.set_value(key, value)?;
        }
        Ok(())
    }

    pub fn remove_key(&mut self, key: impl AsRef<str>) -> Result<Option<usize>> {
        let key = CString::new(key.as_ref())?;
        let removed = unsafe { ffi::gguf_remove_key(self.raw.as_ptr(), key.as_ptr()) };
        if removed < 0 {
            Ok(None)
        } else {
            Ok(Some(removed.try_into_checked().map_err(|source| {
                Error::int_conversion("gguf_remove_key", source)
            })?))
        }
    }

    pub fn merge_kv_from(&mut self, source: &GgufFile) {
        unsafe {
            ffi::gguf_set_kv(self.raw.as_ptr(), source.raw.as_ptr());
        }
    }

    pub fn add_tensor(&mut self, tensor: &DynTensor<'_>) {
        unsafe {
            ffi::gguf_add_tensor(self.raw.as_ptr(), tensor.raw_ptr().cast_const());
        }
    }

    /// Adds a typed tensor to the GGUF context, converting to dynamic form.
    pub fn add_typed_tensor<T: crate::GgmlElement>(&mut self, tensor: &Tensor<'_, T>) {
        self.add_tensor(&tensor.into_dyn())
    }

    pub fn write_to_file<P: AsRef<Path>>(&self, path: P, only_meta: bool) -> Result<()> {
        let path = path_to_c_string(path.as_ref())?;
        let written =
            unsafe { ffi::gguf_write_to_file(self.raw.as_ptr(), path.as_ptr(), only_meta) };
        if written {
            Ok(())
        } else {
            Err(Error::GgufWriteFailed)
        }
    }

    pub fn write_data_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.write_to_file(path, false)
    }

    pub fn write_metadata_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.write_to_file(path, true)
    }

    fn set_array_value(&mut self, key: *const c_char, value: &GgufArrayValue) -> Result<()> {
        match value {
            GgufArrayValue::U8(values) => {
                self.set_array_data(key, ffi::GGUF_TYPE_UINT8 as ffi::gguf_type, values);
            }
            GgufArrayValue::I8(values) => {
                self.set_array_data(key, ffi::GGUF_TYPE_INT8 as ffi::gguf_type, values);
            }
            GgufArrayValue::U16(values) => {
                self.set_array_data(key, ffi::GGUF_TYPE_UINT16 as ffi::gguf_type, values);
            }
            GgufArrayValue::I16(values) => {
                self.set_array_data(key, ffi::GGUF_TYPE_INT16 as ffi::gguf_type, values);
            }
            GgufArrayValue::U32(values) => {
                self.set_array_data(key, ffi::GGUF_TYPE_UINT32 as ffi::gguf_type, values);
            }
            GgufArrayValue::I32(values) => {
                self.set_array_data(key, ffi::GGUF_TYPE_INT32 as ffi::gguf_type, values);
            }
            GgufArrayValue::F32(values) => {
                self.set_array_data(key, ffi::GGUF_TYPE_FLOAT32 as ffi::gguf_type, values);
            }
            GgufArrayValue::Bool(values) => {
                self.set_array_data(key, ffi::GGUF_TYPE_BOOL as ffi::gguf_type, values);
            }
            GgufArrayValue::U64(values) => {
                self.set_array_data(key, ffi::GGUF_TYPE_UINT64 as ffi::gguf_type, values);
            }
            GgufArrayValue::I64(values) => {
                self.set_array_data(key, ffi::GGUF_TYPE_INT64 as ffi::gguf_type, values);
            }
            GgufArrayValue::F64(values) => {
                self.set_array_data(key, ffi::GGUF_TYPE_FLOAT64 as ffi::gguf_type, values);
            }
            GgufArrayValue::String(values) => {
                let c_values = values
                    .iter()
                    .map(|value| CString::new(value.as_str()))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let mut ptrs = c_values
                    .iter()
                    .map(|value| value.as_ptr())
                    .collect::<Vec<*const c_char>>();
                let data_ptr = if ptrs.is_empty() {
                    ptr::null_mut()
                } else {
                    ptrs.as_mut_ptr()
                };
                unsafe {
                    ffi::gguf_set_arr_str(self.raw.as_ptr(), key, data_ptr, ptrs.len());
                }
            }
        }
        Ok(())
    }

    fn set_array_data<T>(&mut self, key: *const c_char, ty: ffi::gguf_type, values: &[T]) {
        let data_ptr = if values.is_empty() {
            ptr::null()
        } else {
            values.as_ptr().cast()
        };
        unsafe {
            ffi::gguf_set_arr_data(self.raw.as_ptr(), key, ty, data_ptr, values.len());
        }
    }
}

impl Drop for GgufWriter {
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

fn c_string_from_ptr(ptr: *const c_char, context: &'static str) -> Result<String> {
    if ptr.is_null() {
        return Err(Error::null_pointer(context));
    }
    let cstr = unsafe { CStr::from_ptr(ptr) };
    Ok(cstr.to_str()?.to_owned())
}

/// Trait for extracting typed values from `GgufValue`.
///
/// Provides a fallible conversion from the `GgufValue` enum to Rust primitive
/// types, returning an error when the variant does not match the requested type.
/// This enables the `kv_value_as::<T>()` convenience method on `GgufFile`.
pub trait TryFromGgufValue: Sized {
    fn try_from_gguf_value(value: &GgufValue) -> Result<Self>;
}

impl TryFromGgufValue for u8 {
    fn try_from_gguf_value(value: &GgufValue) -> Result<Self> {
        match value {
            GgufValue::U8(v) => Ok(*v),
            other => Err(Error::GgufTypeMismatch {
                expected: "u8",
                actual: other.type_name(),
            }),
        }
    }
}

impl TryFromGgufValue for i8 {
    fn try_from_gguf_value(value: &GgufValue) -> Result<Self> {
        match value {
            GgufValue::I8(v) => Ok(*v),
            other => Err(Error::GgufTypeMismatch {
                expected: "i8",
                actual: other.type_name(),
            }),
        }
    }
}

impl TryFromGgufValue for u16 {
    fn try_from_gguf_value(value: &GgufValue) -> Result<Self> {
        match value {
            GgufValue::U16(v) => Ok(*v),
            other => Err(Error::GgufTypeMismatch {
                expected: "u16",
                actual: other.type_name(),
            }),
        }
    }
}

impl TryFromGgufValue for i16 {
    fn try_from_gguf_value(value: &GgufValue) -> Result<Self> {
        match value {
            GgufValue::I16(v) => Ok(*v),
            other => Err(Error::GgufTypeMismatch {
                expected: "i16",
                actual: other.type_name(),
            }),
        }
    }
}

impl TryFromGgufValue for u32 {
    fn try_from_gguf_value(value: &GgufValue) -> Result<Self> {
        match value {
            GgufValue::U32(v) => Ok(*v),
            other => Err(Error::GgufTypeMismatch {
                expected: "u32",
                actual: other.type_name(),
            }),
        }
    }
}

impl TryFromGgufValue for i32 {
    fn try_from_gguf_value(value: &GgufValue) -> Result<Self> {
        match value {
            GgufValue::I32(v) => Ok(*v),
            other => Err(Error::GgufTypeMismatch {
                expected: "i32",
                actual: other.type_name(),
            }),
        }
    }
}

impl TryFromGgufValue for f32 {
    fn try_from_gguf_value(value: &GgufValue) -> Result<Self> {
        match value {
            GgufValue::F32(v) => Ok(*v),
            other => Err(Error::GgufTypeMismatch {
                expected: "f32",
                actual: other.type_name(),
            }),
        }
    }
}

impl TryFromGgufValue for bool {
    fn try_from_gguf_value(value: &GgufValue) -> Result<Self> {
        match value {
            GgufValue::Bool(v) => Ok(*v),
            other => Err(Error::GgufTypeMismatch {
                expected: "bool",
                actual: other.type_name(),
            }),
        }
    }
}

impl TryFromGgufValue for String {
    fn try_from_gguf_value(value: &GgufValue) -> Result<Self> {
        match value {
            GgufValue::String(v) => Ok(v.clone()),
            other => Err(Error::GgufTypeMismatch {
                expected: "String",
                actual: other.type_name(),
            }),
        }
    }
}

impl TryFromGgufValue for u64 {
    fn try_from_gguf_value(value: &GgufValue) -> Result<Self> {
        match value {
            GgufValue::U64(v) => Ok(*v),
            other => Err(Error::GgufTypeMismatch {
                expected: "u64",
                actual: other.type_name(),
            }),
        }
    }
}

impl TryFromGgufValue for i64 {
    fn try_from_gguf_value(value: &GgufValue) -> Result<Self> {
        match value {
            GgufValue::I64(v) => Ok(*v),
            other => Err(Error::GgufTypeMismatch {
                expected: "i64",
                actual: other.type_name(),
            }),
        }
    }
}

impl TryFromGgufValue for f64 {
    fn try_from_gguf_value(value: &GgufValue) -> Result<Self> {
        match value {
            GgufValue::F64(v) => Ok(*v),
            other => Err(Error::GgufTypeMismatch {
                expected: "f64",
                actual: other.type_name(),
            }),
        }
    }
}
