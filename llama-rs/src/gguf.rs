use ggml_rs::{GgufFile, GgufTensorInfo, GgufValue};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct GgufKvEntry {
    pub key: String,
    pub value: GgufValue,
}

impl GgufKvEntry {
    pub fn value_type_name(&self) -> &'static str {
        self.value.type_name()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufReport {
    pub version: u32,
    pub alignment: usize,
    pub data_offset: usize,
    pub kv_entries: Vec<GgufKvEntry>,
    pub tensors: Vec<GgufTensorInfo>,
}

pub fn inspect_gguf<P: AsRef<Path>>(path: P) -> ggml_rs::Result<GgufReport> {
    let file = GgufFile::open(path).map_err(|error| error.with_context("GgufFile::open"))?;

    let kv_count = file
        .kv_count()
        .map_err(|error| error.with_context("GgufFile::kv_count"))?;
    let kv_entries = (0..kv_count)
        .map(|index| {
            let key = file
                .kv_key(index)
                .map_err(|error| error.with_context("GgufFile::kv_key"))?;
            let value = file
                .kv_value(index)
                .map_err(|error| error.with_context("GgufFile::kv_value"))?;

            Ok(GgufKvEntry { key, value })
        })
        .collect::<ggml_rs::Result<Vec<_>>>()?;

    let tensor_count = file
        .tensor_count()
        .map_err(|error| error.with_context("GgufFile::tensor_count"))?;
    let tensors = (0..tensor_count)
        .map(|index| {
            file.tensor_info(index)
                .map_err(|error| error.with_context("GgufFile::tensor_info"))
        })
        .collect::<ggml_rs::Result<Vec<_>>>()?;

    Ok(GgufReport {
        version: file.version(),
        alignment: file.alignment(),
        data_offset: file.data_offset(),
        kv_entries,
        tensors,
    })
}
