#![cfg(feature = "link-system")]

use ggml_rs::{Context, GgufArrayValue, GgufFile, GgufValue, GgufWriter, Length, TryFromGgufValue};
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
        writer.add_typed_tensor(&tensor);
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

#[test]
fn gguf_kv_value_as_extracts_typed_values() -> Result<(), ggml_rs::Error> {
    let output_path = unique_tmp_path("gguf_kv_value_as");

    {
        let ctx = Context::new(2 * 1024 * 1024)?;
        let tensor = ctx.new_tensor_1d::<f32>(Length::new(4))?;
        tensor.set_name("test_tensor")?;
        tensor.write_data(&[0.0; 4])?;

        let mut writer = GgufWriter::new()?;
        writer.set_value("test.u32", &GgufValue::U32(42))?;
        writer.set_value("test.i32", &GgufValue::I32(-7))?;
        writer.set_value("test.f32", &GgufValue::F32(3.14))?;
        writer.set_value("test.bool", &GgufValue::Bool(true))?;
        writer.set_value("test.string", &GgufValue::String("hello".to_owned()))?;
        writer.set_value("test.u64", &GgufValue::U64(100))?;
        writer.set_value("test.i64", &GgufValue::I64(-200))?;
        writer.set_value("test.f64", &GgufValue::F64(2.718))?;
        writer.add_typed_tensor(&tensor);
        writer.write_data_to_file(&output_path)?;
    }

    let file = GgufFile::open(&output_path)?;

    // Type-matched extractions succeed
    assert_eq!(file.kv_value_as::<u32>("test.u32")?, Some(42u32));
    assert_eq!(file.kv_value_as::<i32>("test.i32")?, Some(-7i32));
    assert_eq!(file.kv_value_as::<f32>("test.f32")?, Some(3.14f32));
    assert_eq!(file.kv_value_as::<bool>("test.bool")?, Some(true));
    assert_eq!(
        file.kv_value_as::<String>("test.string")?,
        Some("hello".to_owned())
    );
    assert_eq!(file.kv_value_as::<u64>("test.u64")?, Some(100u64));
    assert_eq!(file.kv_value_as::<i64>("test.i64")?, Some(-200i64));

    // f64 comparison with tolerance
    let f64_val = file.kv_value_as::<f64>("test.f64")?;
    assert!(f64_val.is_some());
    assert!((f64_val.unwrap() - 2.718).abs() < 1e-10);

    // Missing key returns None
    assert_eq!(file.kv_value_as::<i32>("nonexistent.key")?, None);

    // Type mismatch returns error
    let err = file.kv_value_as::<i32>("test.f32").unwrap_err();
    assert!(matches!(err, ggml_rs::Error::GgufTypeMismatch { .. }));

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
