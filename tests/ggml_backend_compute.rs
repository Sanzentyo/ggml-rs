#![cfg(feature = "link-system")]

//! Backend compute path tests: CPU backend creation, buffer allocation,
//! backend data round-trips, backend matmul, and Metal backend on macOS.

use ggml_rs::{
    Backend, BackendKind, Bytes, Context, Error, Length, Shape2D, ThreadCount, with_context,
    with_no_alloc_context,
};

#[test]
fn cpu_backend_creation_succeeds() -> Result<(), Error> {
    let backend = Backend::new(BackendKind::Cpu)?;
    assert_eq!(backend.kind(), BackendKind::Cpu);
    Ok(())
}

#[test]
fn cpu_backend_name_returns_valid_string() -> Result<(), Error> {
    let backend = Backend::new(BackendKind::Cpu)?;
    let name = backend.name()?;
    assert!(!name.is_empty());
    Ok(())
}

#[test]
fn cpu_backend_synchronize_completes() -> Result<(), Error> {
    let backend = Backend::new(BackendKind::Cpu)?;
    backend.synchronize()?;
    Ok(())
}

#[test]
fn backend_buffer_allocation_f32() -> Result<(), Error> {
    let mem =
        Context::recommended_backend_matmul_memory::<f32>(Shape2D::new(2, 2), Shape2D::new(2, 2))?;
    with_no_alloc_context(mem, |ctx| {
        let backend = Backend::new(BackendKind::Cpu)?;
        let tensor = ctx.new_tensor_2d::<f32>(Shape2D::new(2, 2))?;
        let buffer = ctx.allocate_tensors(&backend)?;
        assert!(buffer.size_bytes() > 0);
        let _ = tensor;
        Ok(())
    })
}

#[test]
fn backend_buffer_allocation_i32() -> Result<(), Error> {
    let mem = Bytes::new(128 * 1024);
    with_no_alloc_context(mem, |ctx| {
        let backend = Backend::new(BackendKind::Cpu)?;
        let tensor = ctx.new_tensor_1d::<i32>(Length::new(8))?;
        let buffer = ctx.allocate_tensors(&backend)?;
        assert!(buffer.size_bytes() > 0);
        let _ = tensor;
        Ok(())
    })
}

#[test]
fn backend_matmul_f32_via_backend_path() -> Result<(), Error> {
    let lhs = Shape2D::new(2, 2);
    let rhs = Shape2D::new(2, 2);
    let mem = Context::recommended_backend_matmul_memory::<f32>(lhs, rhs)?;

    with_no_alloc_context(mem, |ctx| {
        let backend = Backend::new(BackendKind::Cpu)?;

        let a = ctx.new_tensor_2d::<f32>(lhs)?;
        let b = ctx.new_tensor_2d::<f32>(rhs)?;
        let result = ctx.mul_mat(&a, &b)?;

        // Build graph first, then allocate backend memory
        let mut graph = ctx.new_graph()?;
        graph.build_forward_expand(&result);
        let _buffer = ctx.allocate_tensors(&backend)?;

        // Write data to backend after allocation
        // Identity * B = B
        a.write_data_backend(&[1.0, 0.0, 0.0, 1.0])?;
        b.write_data_backend(&[5.0, 6.0, 7.0, 8.0])?;

        backend.compute(&mut graph)?;
        backend.synchronize()?;

        let output = graph.last_node_typed::<f32>()?.read_data_backend()?;
        assert_eq!(output.len(), 4);
        assert!((output[0] - 5.0).abs() < 1e-4);
        assert!((output[1] - 6.0).abs() < 1e-4);
        assert!((output[2] - 7.0).abs() < 1e-4);
        assert!((output[3] - 8.0).abs() < 1e-4);

        Ok(())
    })
}

#[test]
fn backend_write_read_roundtrip_f32() -> Result<(), Error> {
    let mem = Bytes::new(128 * 1024);
    with_no_alloc_context(mem, |ctx| {
        let backend = Backend::new(BackendKind::Cpu)?;
        let tensor = ctx.new_tensor_1d::<f32>(Length::new(6))?;
        let _buffer = ctx.allocate_tensors(&backend)?;

        let values = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        tensor.write_data_backend(&values)?;
        let read_back = tensor.read_data_backend()?;
        assert_eq!(read_back.len(), 6);
        for (actual, expected) in read_back.iter().zip(values.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        Ok(())
    })
}

#[test]
fn backend_write_read_roundtrip_i32() -> Result<(), Error> {
    let mem = Bytes::new(128 * 1024);
    with_no_alloc_context(mem, |ctx| {
        let backend = Backend::new(BackendKind::Cpu)?;
        let tensor = ctx.new_tensor_1d::<i32>(Length::new(4))?;
        let _buffer = ctx.allocate_tensors(&backend)?;

        let values = [100, 200, 300, 400];
        tensor.write_data_backend(&values)?;
        let read_back = tensor.read_data_backend()?;
        assert_eq!(read_back, values.to_vec());

        Ok(())
    })
}

#[test]
fn backend_buffer_size_is_positive() -> Result<(), Error> {
    let mem = Bytes::new(128 * 1024);
    with_no_alloc_context(mem, |ctx| {
        let backend = Backend::new(BackendKind::Cpu)?;
        let _t1 = ctx.new_tensor_1d::<f32>(Length::new(16))?;
        let _t2 = ctx.new_tensor_1d::<i32>(Length::new(16))?;
        let buffer = ctx.allocate_tensors(&backend)?;
        // Buffer covers both f32 and i32 tensors
        assert!(buffer.size_bytes() >= 16 * 4 + 16 * 4);
        Ok(())
    })
}

#[test]
fn cpu_backend_compute_with_context_path() -> Result<(), Error> {
    let lhs = Shape2D::new(2, 2);
    let rhs = Shape2D::new(2, 2);
    let mem = Context::recommended_matmul_memory::<f32>(lhs, rhs)?;

    with_context(mem, |ctx| {
        let a = ctx.new_tensor_2d::<f32>(lhs)?;
        let b = ctx.new_tensor_2d::<f32>(rhs)?;
        a.write_data(&[1.0, 0.0, 0.0, 1.0])?;
        b.write_data(&[5.0, 6.0, 7.0, 8.0])?;

        let result = ctx.mul_mat(&a, &b)?;
        let mut graph = ctx.new_graph()?;
        graph.build_forward_expand(&result);
        ctx.compute_with_threads(&mut graph, ThreadCount::new(1))?;

        let output = graph.last_node_typed::<f32>()?.read_data()?;
        assert!((output[0] - 5.0).abs() < 1e-4);
        assert!((output[1] - 6.0).abs() < 1e-4);
        assert!((output[2] - 7.0).abs() < 1e-4);
        assert!((output[3] - 8.0).abs() < 1e-4);

        Ok(())
    })
}

#[test]
#[cfg(target_os = "macos")]
fn metal_backend_creation_succeeds() -> Result<(), Error> {
    let backend = Backend::new(BackendKind::Metal)?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    let name = backend.name()?;
    let lower = name.to_ascii_lowercase();
    assert!(
        lower.contains("metal") || lower.contains("gpu") || lower.contains("mtl"),
        "Metal backend name should reference metal/gpu/mtl, got: {name}"
    );
    backend.synchronize()?;
    Ok(())
}
