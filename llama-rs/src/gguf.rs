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
    let file = GgufFile::open(path)?;

    let kv_entries = (0..file.kv_count()?)
        .map(|index| {
            let key = file.kv_key(index)?;
            let value_type = file.kv_type_name(index)?;
            let string_value = match file.kv_type(index)? {
                GgufType::String => Some(file.kv_string_value(index)?),
                _ => None,
            };

            Ok(GgufKvEntry {
                key,
                value_type,
                string_value,
            })
        })
        .collect::<ggml_rs::Result<Vec<_>>>()?;

    let tensors = (0..file.tensor_count()?)
        .map(|index| file.tensor_info(index))
        .collect::<ggml_rs::Result<Vec<_>>>()?;

    Ok(GgufReport {
        version: file.version(),
        alignment: file.alignment(),
        data_offset: file.data_offset(),
        kv_entries,
        tensors,
    })
}
