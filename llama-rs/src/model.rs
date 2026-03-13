//! GGUF-backed model loading and tensor access helpers.
//!
//! This module provides a safe base layer for future inference pipelines.

use crate::gguf::{GgufKvEntry, GgufReport, inspect_gguf};
use ggml_rs::{GgufTensorInfo, GgufValue};
use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Opaque tensor identifier returned by [`GgufModel::find_tensor`].
pub struct TensorHandle(usize);

#[derive(Debug)]
/// Errors surfaced by [`GgufModel`] loading and tensor access.
pub enum ModelError {
    Io {
        context: &'static str,
        source: std::io::Error,
    },
    Ggml {
        context: &'static str,
        source: ggml_rs::Error,
    },
    DuplicateTensorName {
        name: String,
    },
    DuplicateKvKey {
        key: String,
    },
    MissingTensor {
        name: String,
    },
    InvalidTensorRange {
        tensor_name: String,
        start: usize,
        end: usize,
        file_size: usize,
    },
    UnsupportedTensorType {
        tensor_name: String,
        ggml_type_name: String,
        ggml_type_raw: i32,
    },
    InvalidTensorByteLength {
        tensor_name: String,
        bytes: usize,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { context, source } => write!(f, "{context}: {source}"),
            Self::Ggml { context, source } => write!(f, "{context}: {source}"),
            Self::DuplicateTensorName { name } => {
                write!(f, "duplicate tensor name in GGUF tensor table: {name}")
            }
            Self::DuplicateKvKey { key } => {
                write!(f, "duplicate key in GGUF KV table: {key}")
            }
            Self::MissingTensor { name } => write!(f, "tensor not found: {name}"),
            Self::InvalidTensorRange {
                tensor_name,
                start,
                end,
                file_size,
            } => write!(
                f,
                "tensor `{tensor_name}` points outside file range: [{start}, {end}) with file size {file_size}"
            ),
            Self::UnsupportedTensorType {
                tensor_name,
                ggml_type_name,
                ggml_type_raw,
            } => write!(
                f,
                "tensor `{tensor_name}` is not f32 (type={ggml_type_name} raw={ggml_type_raw})"
            ),
            Self::InvalidTensorByteLength { tensor_name, bytes } => write!(
                f,
                "tensor `{tensor_name}` payload length is not f32-aligned: {bytes} bytes"
            ),
        }
    }
}

impl StdError for ModelError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Ggml { source, .. } => Some(source),
            Self::DuplicateTensorName { .. }
            | Self::DuplicateKvKey { .. }
            | Self::MissingTensor { .. }
            | Self::InvalidTensorRange { .. }
            | Self::UnsupportedTensorType { .. }
            | Self::InvalidTensorByteLength { .. } => None,
        }
    }
}

impl ModelError {
    fn io(context: &'static str, source: std::io::Error) -> Self {
        Self::Io { context, source }
    }

    fn ggml(context: &'static str, source: ggml_rs::Error) -> Self {
        Self::Ggml { context, source }
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
    pub fn tensor_info(&self, name: &str) -> Result<&GgufTensorInfo, ModelError> {
        let handle = self
            .find_tensor(name)
            .ok_or_else(|| ModelError::MissingTensor {
                name: name.to_string(),
            })?;
        Ok(self.tensor_info_by_handle(handle))
    }

    /// Resolves a tensor name to an opaque handle.
    pub fn find_tensor(&self, name: &str) -> Option<TensorHandle> {
        self.tensor_index.get(name).copied().map(TensorHandle)
    }

    /// Returns tensor metadata from a previously resolved handle.
    pub fn tensor_info_by_handle(&self, handle: TensorHandle) -> &GgufTensorInfo {
        &self.report.tensors[handle.0]
    }

    /// Returns raw tensor payload bytes by exact tensor name.
    pub fn tensor_payload(&self, name: &str) -> Result<&[u8], ModelError> {
        let handle = self
            .find_tensor(name)
            .ok_or_else(|| ModelError::MissingTensor {
                name: name.to_string(),
            })?;
        self.tensor_payload_by_handle(handle)
    }

    /// Returns raw tensor payload bytes by tensor handle.
    pub fn tensor_payload_by_handle(&self, handle: TensorHandle) -> Result<&[u8], ModelError> {
        let tensor = self.tensor_info_by_handle(handle);
        self.tensor_payload_internal(tensor)
    }

    /// Decodes a tensor as little-endian `f32` values into a new vector.
    pub fn tensor_f32_values(&self, name: &str) -> Result<Vec<f32>, ModelError> {
        let mut values = Vec::new();
        self.decode_tensor_f32_into(name, &mut values)?;
        Ok(values)
    }

    /// Decodes a tensor as little-endian `f32` values into caller-owned storage.
    ///
    /// This is useful for hot paths that want to reuse buffers and avoid
    /// repeated allocations.
    pub fn decode_tensor_f32_into(&self, name: &str, out: &mut Vec<f32>) -> Result<(), ModelError> {
        let handle = self
            .find_tensor(name)
            .ok_or_else(|| ModelError::MissingTensor {
                name: name.to_string(),
            })?;
        self.decode_tensor_f32_into_by_handle(handle, out)
    }

    /// Decodes a tensor handle as little-endian `f32` values into caller-owned storage.
    pub fn decode_tensor_f32_into_by_handle(
        &self,
        handle: TensorHandle,
        out: &mut Vec<f32>,
    ) -> Result<(), ModelError> {
        let tensor = self.tensor_info_by_handle(handle);
        if tensor.ggml_type_name != "f32" && tensor.ggml_type_raw != 0 {
            return Err(ModelError::UnsupportedTensorType {
                tensor_name: tensor.name.clone(),
                ggml_type_name: tensor.ggml_type_name.clone(),
                ggml_type_raw: tensor.ggml_type_raw,
            });
        }

        let payload = self.tensor_payload_internal(tensor)?;
        if payload.len() % std::mem::size_of::<f32>() != 0 {
            return Err(ModelError::InvalidTensorByteLength {
                tensor_name: tensor.name.clone(),
                bytes: payload.len(),
            });
        }

        out.clear();
        out.reserve(payload.len() / std::mem::size_of::<f32>());
        out.extend(
            payload
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
        );
        Ok(())
    }

    /// Iterates all tensor names in file order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.report
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
    }

    /// Returns a GGUF KV entry by exact key.
    pub fn kv_entry(&self, key: &str) -> Option<&GgufKvEntry> {
        self.kv_index
            .get(key)
            .and_then(|&index| self.report.kv_entries.get(index))
    }

    /// Returns a GGUF KV value by exact key.
    pub fn kv_value(&self, key: &str) -> Option<&GgufValue> {
        self.kv_entry(key).map(|entry| &entry.value)
    }

    /// Returns a metadata string value by key when the key exists and is a string.
    pub fn kv_string(&self, key: &str) -> Option<&str> {
        match self.kv_value(key) {
            Some(GgufValue::String(value)) => Some(value.as_str()),
            _ => None,
        }
    }

    /// Returns a metadata numeric value converted to `usize` when representable.
    pub fn kv_usize(&self, key: &str) -> Option<usize> {
        match self.kv_value(key)? {
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

    /// Returns a metadata numeric value converted to `f32` when possible.
    pub fn kv_f32(&self, key: &str) -> Option<f32> {
        match self.kv_value(key)? {
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

    /// Returns the element count for an `f32` tensor without decoding payload.
    pub fn tensor_f32_len(&self, name: &str) -> Result<usize, ModelError> {
        let handle = self
            .find_tensor(name)
            .ok_or_else(|| ModelError::MissingTensor {
                name: name.to_string(),
            })?;
        self.tensor_f32_len_by_handle(handle)
    }

    /// Returns the element count for an `f32` tensor by handle without decoding payload.
    pub fn tensor_f32_len_by_handle(&self, handle: TensorHandle) -> Result<usize, ModelError> {
        let tensor = self.tensor_info_by_handle(handle);
        if tensor.ggml_type_name != "f32" && tensor.ggml_type_raw != 0 {
            return Err(ModelError::UnsupportedTensorType {
                tensor_name: tensor.name.clone(),
                ggml_type_name: tensor.ggml_type_name.clone(),
                ggml_type_raw: tensor.ggml_type_raw,
            });
        }

        let payload = self.tensor_payload_internal(tensor)?;
        if payload.len() % std::mem::size_of::<f32>() != 0 {
            return Err(ModelError::InvalidTensorByteLength {
                tensor_name: tensor.name.clone(),
                bytes: payload.len(),
            });
        }
        Ok(payload.len() / std::mem::size_of::<f32>())
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
