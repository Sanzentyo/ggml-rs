use ggml_rs::{GgufFile, GgufTensorInfo, GgufType};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufKvEntry {
    pub key: String,
    pub value_type: String,
    pub string_value: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
            let value_type = file
                .kv_type_name(index)
                .map_err(|error| error.with_context("GgufFile::kv_type_name"))?;
            let string_value = match file
                .kv_type(index)
                .map_err(|error| error.with_context("GgufFile::kv_type"))?
            {
                GgufType::String => Some(
                    file.kv_string_value(index)
                        .map_err(|error| error.with_context("GgufFile::kv_string_value"))?,
                ),
                _ => None,
            };

            Ok(GgufKvEntry {
                key,
                value_type,
                string_value,
            })
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
