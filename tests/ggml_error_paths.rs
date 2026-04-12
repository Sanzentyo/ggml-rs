#![cfg(feature = "link-system")]

//! Error path and boundary tests: validates that all Error variants are
//! reachable through the public API under the expected conditions.

use ggml_rs::{
    Bytes, Context, Dims, Error, GgufFile, GgufValue, GgufWriter, Length, Shape2D, ThreadCount,
    Type, decode_tensor_data_to, with_context,
};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn zero_memory_size_context_new() {
    let result = Context::new(0);
    assert!(matches!(result, Err(Error::ZeroMemorySize)));
}

#[test]
fn zero_memory_size_context_new_bytes() {
    let result = Context::new_bytes(Bytes::new(0));
    assert!(matches!(result, Err(Error::ZeroMemorySize)));
}

#[test]
fn zero_memory_size_context_new_no_alloc() {
    let result = Context::new_no_alloc(0);
    assert!(matches!(result, Err(Error::ZeroMemorySize)));
}

#[test]
fn zero_memory_size_context_new_no_alloc_bytes() {
    let result = Context::new_no_alloc_bytes(Bytes::new(0));
    assert!(matches!(result, Err(Error::ZeroMemorySize)));
}

#[test]
fn invalid_thread_count_zero() {
    let mem =
        Context::recommended_matmul_memory::<f32>(Shape2D::new(2, 2), Shape2D::new(2, 2)).unwrap();
    with_context(mem, |ctx| {
        let a = ctx.new_tensor_2d::<f32>(Shape2D::new(2, 2))?;
        let b = ctx.new_tensor_2d::<f32>(Shape2D::new(2, 2))?;
        a.write_data(&[1.0, 0.0, 0.0, 1.0])?;
        b.write_data(&[1.0, 0.0, 0.0, 1.0])?;
        let result = ctx.mul_mat(&a, &b)?;
        let mut graph = ctx.new_graph()?;
        graph.build_forward_expand(&result);
        let err = ctx
            .compute_with_threads(&mut graph, ThreadCount::new(0))
            .expect_err("thread count 0 should error");
        assert!(matches!(err, Error::InvalidThreadCount(0)));
        Ok(())
    })
    .unwrap();
}

#[test]
fn index_out_of_bounds_host_read() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let tensor = ctx.new_tensor_1d::<f32>(Length::new(4))?;
        tensor.write_data(&[1.0, 2.0, 3.0, 4.0])?;
        let err = tensor
            .read_data_at(4, 1)
            .expect_err("read past end should error");
        assert!(matches!(err, Error::IndexOutOfBounds { .. }));
        Ok(())
    })
    .unwrap();
}

#[test]
fn index_out_of_bounds_host_write() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let tensor = ctx.new_tensor_1d::<f32>(Length::new(4))?;
        let err = tensor
            .write_data_at(4, &[1.0])
            .expect_err("write past end should error");
        assert!(matches!(err, Error::IndexOutOfBounds { .. }));
        Ok(())
    })
    .unwrap();
}

#[test]
fn index_out_of_bounds_get_data() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let tensor = ctx.new_tensor_1d::<f32>(Length::new(4))?;
        tensor.write_data(&[1.0, 2.0, 3.0, 4.0])?;
        let err = tensor
            .get_data(ggml_rs::TensorIndex::new(4))
            .expect_err("get_data past end should error");
        assert!(matches!(err, Error::IndexOutOfBounds { .. }));
        Ok(())
    })
    .unwrap();
}

#[test]
fn length_mismatch_write_data() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let tensor = ctx.new_tensor_1d::<f32>(Length::new(4))?;
        let err = tensor
            .write_data(&[1.0, 2.0])
            .expect_err("wrong-length write should error");
        assert!(matches!(
            err,
            Error::LengthMismatch {
                expected: 4,
                actual: 2
            }
        ));
        Ok(())
    })
    .unwrap();
}

#[test]
fn unsupported_type_decode_tensor_data_to() {
    // GGML_TYPE_COUNT or any large invalid type ID should fail
    let payload = vec![0u8; 16];
    let err = decode_tensor_data_to::<f32>(9999, &payload).expect_err("invalid type should error");
    assert!(matches!(err, Error::UnsupportedType(9999)));
}

#[test]
fn overflow_in_recommended_matmul_memory() {
    // Use extremely large dimensions that overflow usize multiplication
    let huge = Shape2D::new(usize::MAX, usize::MAX);
    let result = Context::recommended_matmul_memory::<f32>(huge, huge);
    assert!(matches!(result, Err(Error::Overflow)));
}

#[test]
fn unexpected_shape_on_3d_tensor_shape_2d() {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let t3 = ctx.new_tensor(Type::F32, Dims::new([4, 3, 2]))?;
        let err = t3
            .dims::<2>()
            .expect_err("3D DynTensor should not support dims<2>");
        assert!(matches!(err, Error::UnsupportedRank(3)));
        Ok(())
    })
    .unwrap();
}

#[test]
fn unexpected_shape_on_1d_tensor_shape_2d() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let t1 = ctx.new_tensor_1d::<f32>(Length::new(8))?;
        let err = t1
            .dims::<2>()
            .expect_err("1D tensor should not support dims<2>");
        assert!(matches!(err, Error::UnsupportedRank(1)));
        Ok(())
    })
    .unwrap();
}

#[test]
fn gguf_type_mismatch() {
    let output_path = unique_tmp_path("gguf_type_mismatch");
    {
        let ctx = Context::new(2 * 1024 * 1024).unwrap();
        let tensor = ctx.new_tensor_1d::<f32>(Length::new(2)).unwrap();
        tensor.set_name("t").unwrap();
        tensor.write_data(&[0.0, 0.0]).unwrap();

        let mut writer = GgufWriter::new().unwrap();
        writer.set_value("test.f32", &GgufValue::F32(3.14)).unwrap();
        writer.add_typed_tensor(&tensor);
        writer.write_data_to_file(&output_path).unwrap();
    }

    let file = GgufFile::open(&output_path).unwrap();
    let err = file.kv_value_as::<i32>("test.f32").unwrap_err();
    assert!(matches!(err, Error::GgufTypeMismatch { .. }));

    let _ = std::fs::remove_file(output_path);
}

#[test]
fn incompatible_matmul_shapes() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let a = ctx.new_tensor_2d::<f32>(Shape2D::new(3, 2))?;
        let b = ctx.new_tensor_2d::<f32>(Shape2D::new(2, 2))?;
        let result = ctx.mul_mat(&a, &b);
        assert!(matches!(
            result,
            Err(Error::IncompatibleMatmulShapes { .. })
        ));
        Ok(())
    })
    .unwrap();
}

#[test]
fn invalid_graph_index() {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let t = ctx.new_tensor_1d::<f32>(Length::new(4))?;
        let mut graph = ctx.new_graph()?;
        graph.build_forward_expand(&t);
        let result = graph.node(5);
        assert!(matches!(result, Err(Error::InvalidGraphIndex { .. })));
        Ok(())
    })
    .unwrap();
}

#[test]
fn type_mismatch_dyn_tensor_as_typed() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let t = ctx.new_tensor_1d::<f32>(Length::new(4))?;
        let dyn_t = t.into_dyn();
        let result = dyn_t.as_typed::<i32>();
        assert!(matches!(result, Err(Error::TypeMismatch { .. })));
        Ok(())
    })
    .unwrap();
}

#[test]
fn unexpected_tensor_byte_size_decode() {
    // decode_tensor_data_to with a payload whose byte size is not a multiple
    // of the type size triggers UnexpectedTensorByteSize
    let payload = vec![0u8; 3]; // 3 bytes is not a multiple of f32's 4 bytes
    let err = decode_tensor_data_to::<f32>(Type::F32 as i32, &payload)
        .expect_err("misaligned payload should error");
    assert!(matches!(err, Error::UnexpectedTensorByteSize { .. }));
}

fn unique_tmp_path(prefix: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}_{pid}_{nanos}.gguf"))
}
