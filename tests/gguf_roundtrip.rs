#![cfg(feature = "link-system")]

use ggml_rs::{Context, GgufArrayValue, GgufFile, GgufValue, GgufWriter, Length};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn gguf_writer_roundtrip_kv_and_tensor_metadata() -> Result<(), ggml_rs::Error> {
    let output_path = unique_tmp_path("gguf_writer_roundtrip");

    {
        let ctx = Context::new(2 * 1024 * 1024)?;
        let tensor = ctx.new_tensor_1d::<f32>(Length::new(4))?;
        tensor.set_name("tensor_roundtrip")?;
        tensor.write_data(&[1.0, 2.0, 3.0, 4.0])?;

        let mut writer = GgufWriter::new()?;
        let kv_entries = vec![
            (
                "general.architecture".to_owned(),
                GgufValue::String("llama".to_owned()),
            ),
            ("test.flag".to_owned(), GgufValue::Bool(true)),
            (
                "test.array_i16".to_owned(),
                GgufValue::Array(GgufArrayValue::I16(vec![1, 2, 3])),
            ),
        ];
        writer.set_values(kv_entries.iter().map(|(key, value)| (key.as_str(), value)))?;
        writer.set_value("test.remove_me", &GgufValue::I32(-1))?;
        assert!(writer.remove_key("test.remove_me")?.is_some());
        writer.add_tensor(&tensor);
        writer.write_data_to_file(&output_path)?;
    }

    let file = GgufFile::open(&output_path)?;
    assert_eq!(
        file.kv_value_by_key("general.architecture")?,
        Some(GgufValue::String("llama".to_owned()))
    );
    assert_eq!(
        file.kv_value_by_key("test.flag")?,
        Some(GgufValue::Bool(true))
    );
    assert_eq!(
        file.kv_value_by_key("test.array_i16")?,
        Some(GgufValue::Array(GgufArrayValue::I16(vec![1, 2, 3])))
    );
    assert_eq!(file.kv_value_by_key("test.remove_me")?, None);
    assert_eq!(file.tensor_count()?, 1);
    assert_eq!(file.tensor_info(0)?.name, "tensor_roundtrip");

    let _ = std::fs::remove_file(output_path);
    Ok(())
}

fn unique_tmp_path(prefix: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}_{pid}_{nanos}.gguf"))
}
