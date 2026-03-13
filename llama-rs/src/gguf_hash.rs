use crate::inspect_gguf;
use std::error::Error as StdError;
use std::fmt;
use std::path::Path;
use uuid::Uuid;
use xxhash_rust::xxh64::xxh64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashAlgorithm {
    Xxh64,
    Sha1,
    Sha256,
    Uuid,
}

impl HashAlgorithm {
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::Xxh64 => "xxh64",
            Self::Sha1 => "sha1",
            Self::Sha256 => "sha256",
            Self::Uuid => "uuid",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashRecord {
    pub algorithm: HashAlgorithm,
    pub value: String,
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashOptions {
    pub algorithms: Vec<HashAlgorithm>,
    pub include_layers: bool,
}

impl Default for HashOptions {
    fn default() -> Self {
        Self {
            algorithms: vec![HashAlgorithm::Xxh64],
            include_layers: true,
        }
    }
}

#[derive(Debug)]
pub enum GgufHashError {
    Io {
        context: &'static str,
        source: std::io::Error,
    },
    Ggml {
        context: &'static str,
        source: ggml_rs::Error,
    },
    InvalidTensorRange {
        tensor_name: String,
        start: usize,
        end: usize,
        file_size: usize,
    },
}

impl fmt::Display for GgufHashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { context, source } => write!(f, "{context}: {source}"),
            Self::Ggml { context, source } => write!(f, "{context}: {source}"),
            Self::InvalidTensorRange {
                tensor_name,
                start,
                end,
                file_size,
            } => write!(
                f,
                "tensor `{tensor_name}` points outside file range: [{start}, {end}) with file size {file_size}"
            ),
        }
    }
}

impl StdError for GgufHashError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Ggml { source, .. } => Some(source),
            Self::InvalidTensorRange { .. } => None,
        }
    }
}

impl GgufHashError {
    fn io(context: &'static str, source: std::io::Error) -> Self {
        Self::Io { context, source }
    }

    fn ggml(context: &'static str, source: ggml_rs::Error) -> Self {
        Self::Ggml { context, source }
    }
}

pub fn hash_file<P: AsRef<Path>>(
    path: P,
    options: &HashOptions,
) -> Result<Vec<HashRecord>, GgufHashError> {
    let path = path.as_ref();
    let report =
        inspect_gguf(path).map_err(|source| GgufHashError::ggml("inspect_gguf", source))?;
    let bytes = std::fs::read(path).map_err(|source| GgufHashError::io("std::fs::read", source))?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_string())
        .unwrap_or_else(|| path.to_string_lossy().into_owned());

    let mut records = Vec::new();

    for &algorithm in &options.algorithms {
        if options.include_layers && algorithm != HashAlgorithm::Uuid {
            for tensor in &report.tensors {
                let payload = tensor_payload(&bytes, report.data_offset, tensor)?;
                records.push(HashRecord {
                    algorithm,
                    value: compute_hash(algorithm, payload),
                    target: format!("{filename}:{}", tensor.name),
                });
            }
        }

        let mut full_payload = Vec::new();
        for tensor in &report.tensors {
            let payload = tensor_payload(&bytes, report.data_offset, tensor)?;
            full_payload.extend_from_slice(payload);
        }

        records.push(HashRecord {
            algorithm,
            value: compute_hash(algorithm, &full_payload),
            target: filename.clone(),
        });
    }

    Ok(records)
}

fn tensor_payload<'a>(
    bytes: &'a [u8],
    data_offset: usize,
    tensor: &ggml_rs::GgufTensorInfo,
) -> Result<&'a [u8], GgufHashError> {
    let start = data_offset.saturating_add(tensor.offset);
    let end = start.saturating_add(tensor.size);
    if end > bytes.len() || start > end {
        return Err(GgufHashError::InvalidTensorRange {
            tensor_name: tensor.name.clone(),
            start,
            end,
            file_size: bytes.len(),
        });
    }

    Ok(&bytes[start..end])
}

fn compute_hash(algorithm: HashAlgorithm, payload: &[u8]) -> String {
    match algorithm {
        HashAlgorithm::Xxh64 => format!("{:016x}", xxh64(payload, 0)),
        HashAlgorithm::Sha1 => {
            use sha1::{Digest as _, Sha1};
            let mut hasher = Sha1::new();
            hasher.update(payload);
            format!("{:x}", hasher.finalize())
        }
        HashAlgorithm::Sha256 => {
            use sha2::{Digest as _, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(payload);
            format!("{:x}", hasher.finalize())
        }
        HashAlgorithm::Uuid => {
            const NAMESPACE_LLAMA_CPP: u128 = 0xef00_1206_dadc_5f6d_a15f_3359_e577_d4e5;
            let namespace = Uuid::from_u128(NAMESPACE_LLAMA_CPP);
            Uuid::new_v5(&namespace, payload).to_string()
        }
    }
}
