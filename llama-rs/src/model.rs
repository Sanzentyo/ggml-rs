//! GGUF-backed model loading and tensor access helpers.
//!
//! This module provides a safe base layer for future inference pipelines.

use crate::gguf::{GgufKvEntry, GgufReport, inspect_gguf};
use ggml_rs::{GgmlElement, GgufTensorInfo, GgufValue};
use num_traits::NumCast;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Opaque tensor identifier returned by [`GgufModel::find_tensor`].
pub struct TensorHandle(usize);

#[derive(Debug, Error)]
/// Errors surfaced by [`GgufModel`] loading and tensor access.
pub enum ModelError {
    #[error("{context}: {source}")]
    Io {
        context: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("{context}: {source}")]
    Ggml {
        context: &'static str,
        #[source]
        source: ggml_rs::Error,
    },
    #[error("duplicate tensor name in GGUF tensor table: {name}")]
    DuplicateTensorName { name: String },
    #[error("duplicate key in GGUF KV table: {key}")]
    DuplicateKvKey { key: String },
    #[error("tensor not found: {name}")]
    MissingTensor { name: String },
    #[error(
        "tensor `{tensor_name}` points outside file range: [{start}, {end}) with file size {file_size}"
    )]
    InvalidTensorRange {
        tensor_name: String,
        start: usize,
        end: usize,
        file_size: usize,
    },
    #[error("tensor `{tensor_name}` is not decodable (type={ggml_type_name} raw={ggml_type_raw})")]
    UnsupportedTensorType {
        tensor_name: String,
        ggml_type_name: String,
        ggml_type_raw: i32,
    },
    #[error("tensor `{tensor_name}` payload length is not element-aligned: {bytes} bytes")]
    InvalidTensorByteLength { tensor_name: String, bytes: usize },
}

impl ModelError {
    fn io(context: &'static str, source: std::io::Error) -> Self {
        Self::Io { context, source }
    }

    fn ggml(context: &'static str, source: ggml_rs::Error) -> Self {
        Self::Ggml { context, source }
    }
}

/// Conversion trait used by generic GGUF metadata accessors.
pub trait TryFromGgufValue: Sized {
    fn try_from_gguf(value: &GgufValue) -> Option<Self>;
}

impl TryFromGgufValue for usize {
    fn try_from_gguf(value: &GgufValue) -> Option<Self> {
        match value {
            GgufValue::U8(value) => Some(*value as usize),
            GgufValue::I8(value) if *value >= 0 => Some(*value as usize),
            GgufValue::U16(value) => Some(*value as usize),
            GgufValue::I16(value) if *value >= 0 => Some(*value as usize),
            GgufValue::U32(value) => Some(*value as usize),
            GgufValue::I32(value) if *value >= 0 => Some(*value as usize),
            GgufValue::U64(value) => usize::try_from(*value).ok(),
            GgufValue::I64(value) if *value >= 0 => usize::try_from(*value as u64).ok(),
            GgufValue::F32(value) if *value >= 0.0 && value.fract() == 0.0 => Some(*value as usize),
            GgufValue::F64(value) if *value >= 0.0 && value.fract() == 0.0 => Some(*value as usize),
            _ => None,
        }
    }
}

impl TryFromGgufValue for f32 {
    fn try_from_gguf(value: &GgufValue) -> Option<Self> {
        match value {
            GgufValue::F32(value) => Some(*value),
            GgufValue::F64(value) => Some(*value as f32),
            GgufValue::U8(value) => Some(*value as f32),
            GgufValue::I8(value) => Some(*value as f32),
            GgufValue::U16(value) => Some(*value as f32),
            GgufValue::I16(value) => Some(*value as f32),
            GgufValue::U32(value) => Some(*value as f32),
            GgufValue::I32(value) => Some(*value as f32),
            GgufValue::U64(value) => Some(*value as f32),
            GgufValue::I64(value) => Some(*value as f32),
            _ => None,
        }
    }
}

impl TryFromGgufValue for String {
    fn try_from_gguf(value: &GgufValue) -> Option<Self> {
        match value {
            GgufValue::String(value) => Some(value.clone()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
/// In-memory GGUF model view with validated tensor lookup.
pub struct GgufModel {
    path: PathBuf,
    bytes: Vec<u8>,
    report: GgufReport,
    tensor_index: HashMap<String, usize>,
    kv_index: HashMap<String, usize>,
}

impl GgufModel {
    /// Opens a GGUF file, parses metadata, loads bytes, and validates tensor ranges.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, ModelError> {
        let path = path.as_ref().to_path_buf();
        let report =
            inspect_gguf(&path).map_err(|source| ModelError::ggml("inspect_gguf", source))?;
        let bytes =
            std::fs::read(&path).map_err(|source| ModelError::io("std::fs::read", source))?;

        let mut tensor_index = HashMap::with_capacity(report.tensors.len());
        for (index, tensor) in report.tensors.iter().enumerate() {
            if tensor_index.insert(tensor.name.clone(), index).is_some() {
                return Err(ModelError::DuplicateTensorName {
                    name: tensor.name.clone(),
                });
            }
        }
        let mut kv_index = HashMap::with_capacity(report.kv_entries.len());
        for (index, entry) in report.kv_entries.iter().enumerate() {
            if kv_index.insert(entry.key.clone(), index).is_some() {
                return Err(ModelError::DuplicateKvKey {
                    key: entry.key.clone(),
                });
            }
        }

        let model = Self {
            path,
            bytes,
            report,
            tensor_index,
            kv_index,
        };

        // Validate all tensor ranges once at load time so future lookups are
        // predictable and fail-fast.
        for tensor in &model.report.tensors {
            model.tensor_payload_internal(tensor)?;
        }

        Ok(model)
    }

    /// Returns the source GGUF path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the total file size in bytes.
    pub fn file_size(&self) -> usize {
        self.bytes.len()
    }

    /// Returns parsed GGUF metadata and tensor table.
    pub fn report(&self) -> &GgufReport {
        &self.report
    }

    /// Returns tensor metadata by exact tensor name.
    pub fn tensor_info(&self, name: impl AsRef<str>) -> Result<&GgufTensorInfo, ModelError> {
        let name = name.as_ref();
        let handle = self
            .find_tensor(name)
            .ok_or_else(|| ModelError::MissingTensor {
                name: name.to_owned(),
            })?;
        Ok(self.tensor_info_by_handle(handle))
    }

    /// Resolves a tensor name to an opaque handle.
    pub fn find_tensor(&self, name: impl AsRef<str>) -> Option<TensorHandle> {
        self.tensor_index
            .get(name.as_ref())
            .copied()
            .map(TensorHandle)
    }

    /// Returns tensor metadata from a previously resolved handle.
    pub fn tensor_info_by_handle(&self, handle: TensorHandle) -> &GgufTensorInfo {
        &self.report.tensors[handle.0]
    }

    /// Returns raw tensor payload bytes by exact tensor name.
    pub fn tensor_payload(&self, name: impl AsRef<str>) -> Result<&[u8], ModelError> {
        let name = name.as_ref();
        let handle = self
            .find_tensor(name)
            .ok_or_else(|| ModelError::MissingTensor {
                name: name.to_owned(),
            })?;
        self.tensor_payload_by_handle(handle)
    }

    /// Returns raw tensor payload bytes by tensor handle.
    pub fn tensor_payload_by_handle(&self, handle: TensorHandle) -> Result<&[u8], ModelError> {
        let tensor = self.tensor_info_by_handle(handle);
        self.tensor_payload_internal(tensor)
    }

    /// Decodes a tensor into caller-selected element type.
    pub fn tensor_values<T>(&self, name: impl AsRef<str>) -> Result<Vec<T>, ModelError>
    where
        T: GgmlElement + NumCast,
    {
        self.decode_tensor(name)
    }

    /// Decodes a tensor into caller-selected element type.
    pub fn decode_tensor<T>(&self, name: impl AsRef<str>) -> Result<Vec<T>, ModelError>
    where
        T: GgmlElement + NumCast,
    {
        let name = name.as_ref();
        let handle = self
            .find_tensor(name)
            .ok_or_else(|| ModelError::MissingTensor {
                name: name.to_owned(),
            })?;
        self.decode_tensor_by_handle(handle)
    }

    /// Decodes a tensor by handle into caller-selected element type.
    pub fn decode_tensor_by_handle<T>(&self, handle: TensorHandle) -> Result<Vec<T>, ModelError>
    where
        T: GgmlElement + NumCast,
    {
        let tensor = self.tensor_info_by_handle(handle);
        let payload = self.tensor_payload_internal(tensor)?;
        ggml_rs::decode_tensor_data_to::<T>(tensor.ggml_type_raw, payload).map_err(|source| {
            match source {
                ggml_rs::Error::UnsupportedType(_) => ModelError::UnsupportedTensorType {
                    tensor_name: tensor.name.clone(),
                    ggml_type_name: tensor.ggml_type_name.clone(),
                    ggml_type_raw: tensor.ggml_type_raw,
                },
                other => ModelError::ggml("decode_tensor_data_to", other),
            }
        })
    }

    /// Iterates all tensor names in file order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.report
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
    }

    /// Returns a GGUF KV entry by exact key.
    pub fn kv_entry(&self, key: impl AsRef<str>) -> Option<&GgufKvEntry> {
        self.kv_index
            .get(key.as_ref())
            .and_then(|&index| self.report.kv_entries.get(index))
    }

    /// Returns a GGUF KV value by exact key.
    pub fn kv_value(&self, key: impl AsRef<str>) -> Option<&GgufValue> {
        self.kv_entry(key).map(|entry| &entry.value)
    }

    /// Returns metadata value converted by [`TryFromGgufValue`].
    pub fn kv_value_as<T>(&self, key: impl AsRef<str>) -> Option<T>
    where
        T: TryFromGgufValue,
    {
        self.kv_value(key).and_then(T::try_from_gguf)
    }

    /// Returns a metadata string value by key when the key exists and is a string.
    pub fn kv_string(&self, key: impl AsRef<str>) -> Option<&str> {
        match self.kv_value(key) {
            Some(GgufValue::String(value)) => Some(value.as_str()),
            _ => None,
        }
    }

    /// Returns a metadata numeric value converted to `usize` when representable.
    pub fn kv_usize(&self, key: impl AsRef<str>) -> Option<usize> {
        self.kv_value_as(key)
    }

    /// Returns a metadata numeric value converted to `f32` when possible.
    pub fn kv_f32(&self, key: impl AsRef<str>) -> Option<f32> {
        self.kv_value_as(key)
    }

    /// Returns the scalar element count for a tensor payload.
    pub fn tensor_len(&self, name: impl AsRef<str>) -> Result<usize, ModelError> {
        let name = name.as_ref();
        let handle = self
            .find_tensor(name)
            .ok_or_else(|| ModelError::MissingTensor {
                name: name.to_owned(),
            })?;
        self.tensor_len_by_handle(handle)
    }

    /// Returns the scalar element count by tensor handle.
    pub fn tensor_len_by_handle(&self, handle: TensorHandle) -> Result<usize, ModelError> {
        let tensor = self.tensor_info_by_handle(handle);
        let payload = self.tensor_payload_internal(tensor)?;
        ggml_rs::tensor_element_count(tensor.ggml_type_raw, payload.len()).map_err(|source| {
            match source {
                ggml_rs::Error::UnsupportedType(_) => ModelError::UnsupportedTensorType {
                    tensor_name: tensor.name.clone(),
                    ggml_type_name: tensor.ggml_type_name.clone(),
                    ggml_type_raw: tensor.ggml_type_raw,
                },
                other => ModelError::ggml("tensor_element_count", other),
            }
        })
    }

    fn tensor_payload_internal<'a>(
        &'a self,
        tensor: &GgufTensorInfo,
    ) -> Result<&'a [u8], ModelError> {
        let start = self.report.data_offset.saturating_add(tensor.offset);
        let end = start.saturating_add(tensor.size);
        if end > self.bytes.len() || start > end {
            return Err(ModelError::InvalidTensorRange {
                tensor_name: tensor.name.clone(),
                start,
                end,
                file_size: self.bytes.len(),
            });
        }

        Ok(&self.bytes[start..end])
    }
}
