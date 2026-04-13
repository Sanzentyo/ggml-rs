#![cfg(feature = "link-system")]

use ggml_rs::{
    Backend, BackendKind, Bytes, Context, Dims, Error, Length, Shape2D, Shape3D, Shape4D,
    TensorIndex, Type, with_context, with_no_alloc_context,
};

#[test]
fn scoped_context_helpers_work() -> Result<(), Error> {
    let lhs = Shape2D::new(2, 2);
    let rhs = Shape2D::new(2, 2);
    let mem = Context::recommended_matmul_memory::<f32>(lhs, rhs)?;

    let element_count = with_context(mem, |ctx| {
        let tensor = ctx.new_tensor_2d::<f32>(lhs)?;
        tensor.write_data(&[1.0, 2.0, 3.0, 4.0])?;
        assert_eq!(tensor.read_data_at(1, 2)?, vec![2.0, 3.0]);
        tensor.write_data_at(2, &[9.0, 8.0])?;
        assert_eq!(tensor.read_data()?, vec![1.0, 2.0, 9.0, 8.0]);
        let err = tensor
            .read_data_at(4, 1)
            .expect_err("host read range past end should error");
        assert!(matches!(err, Error::IndexOutOfBounds { .. }));
        let err = tensor
            .write_data_at(4, &[1.0])
            .expect_err("host write range past end should error");
        assert!(matches!(err, Error::IndexOutOfBounds { .. }));
        tensor.element_count()
    })?;
    assert_eq!(element_count, 4);

    let no_alloc_mem = Context::recommended_backend_matmul_memory::<f32>(lhs, rhs)?;
    with_no_alloc_context(no_alloc_mem, |ctx| {
        let backend = Backend::new(BackendKind::Cpu)?;
        let tensor = ctx.new_tensor_2d::<f32>(lhs)?;
        let _buffer = ctx.allocate_tensors(&backend)?;
        tensor.write_data_backend(&[1.0, 2.0, 3.0, 4.0])?;
        let out = tensor.read_data_backend()?;
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    })?;

    Ok(())
}

#[test]
fn nd_tensor_constructor_and_introspection() -> Result<(), Error> {
    let mem = Bytes::new(128 * 1024);
    with_context(mem, |ctx| {
        let t3 = ctx.new_tensor(Type::F32, Dims::new([4, 3, 2]))?;
        assert_eq!(t3.rank()?, 3);
        assert_eq!(*t3.dims::<3>()?.as_array(), [4, 3, 2]);
        assert_eq!(t3.shape_3d()?, Shape3D::new(4, 3, 2));
        assert_eq!(t3.shape_nd()?, vec![4, 3, 2]);

        let t4 = ctx.new_tensor_4d::<i32>(Shape4D::new(5, 4, 3, 2))?;
        assert_eq!(t4.rank()?, 4);
        assert_eq!(*t4.dims::<4>()?.as_array(), [5, 4, 3, 2]);
        assert_eq!(t4.shape_4d()?, Shape4D::new(5, 4, 3, 2));
        Ok(())
    })
}

#[test]
fn reshape_view_permute_smoke() -> Result<(), Error> {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let base = ctx.new_tensor_2d::<f32>(Shape2D::new(6, 2))?;
        base.write_data(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])?;

        let reshaped = ctx.reshape_3d(&base, 3, 2, 2)?;
        assert_eq!(*reshaped.dims::<3>()?.as_array(), [3, 2, 2]);

        let viewed = ctx.view_1d(&base, 4, 0)?;
        assert_eq!(viewed.element_count()?, 4);
        assert_eq!(viewed.get_data(TensorIndex::new(3))?, 3.0);

        let tensor4 = ctx.new_tensor_4d::<f32>(Shape4D::new(3, 2, 2, 2))?;
        let permuted = ctx.permute(&tensor4, 1, 0, 2, 3)?;
        assert_eq!(*permuted.dims::<4>()?.as_array(), [2, 3, 2, 2]);
        Ok(())
    })
}

#[test]
fn backend_roundtrip_and_bounds_checks() -> Result<(), Error> {
    let mem = Bytes::new(128 * 1024);
    with_no_alloc_context(mem, |ctx| {
        let backend = Backend::new(BackendKind::Cpu)?;
        let tensor = ctx.new_tensor_1d::<i32>(Length::new(8))?;
        let _buffer = ctx.allocate_tensors(&backend)?;

        let values = [10, 20, 30, 40, 50, 60, 70, 80];
        tensor.write_data_backend(&values)?;
        assert_eq!(tensor.read_data_backend()?, values.to_vec());

        tensor.write_data_backend_at(2, &[111, 222])?;
        assert_eq!(
            tensor.read_data_backend()?,
            vec![10, 20, 111, 222, 50, 60, 70, 80]
        );
        assert_eq!(tensor.read_data_backend_at(2, 3)?, vec![111, 222, 50]);

        let err = tensor
            .write_data_backend_at(8, &[1])
            .expect_err("offset past end should error");
        assert!(matches!(err, Error::IndexOutOfBounds { .. }));

        let err = tensor
            .read_data_backend_at(7, 2)
            .expect_err("read range past end should error");
        assert!(matches!(err, Error::IndexOutOfBounds { .. }));

        Ok(())
    })
}

// -- reshape_1d / reshape_4d --

#[test]
fn reshape_1d_flattens_2d() -> Result<(), Error> {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let base = ctx.new_tensor_2d::<f32>(Shape2D::new(3, 4))?;
        let flat = ctx.reshape_1d(&base, 12)?;
        assert_eq!(flat.element_count()?, 12);
        Ok(())
    })
}

#[test]
fn reshape_4d_smoke() -> Result<(), Error> {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        // 2×3×4×5 = 120 elements
        let base = ctx.new_tensor_2d::<f32>(Shape2D::new(12, 10))?; // 120
        let r4 = ctx.reshape_4d(&base, 2, 3, 4, 5)?;
        assert_eq!(*r4.dims::<4>()?.as_array(), [2, 3, 4, 5]);
        assert_eq!(r4.element_count()?, 120);
        Ok(())
    })
}

#[test]
fn reshape_element_count_mismatch_returns_error() -> Result<(), Error> {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let base = ctx.new_tensor_2d::<f32>(Shape2D::new(3, 4))?; // 12 elements
        assert!(matches!(
            ctx.reshape_1d(&base, 10),
            Err(Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            ctx.reshape_4d(&base, 2, 2, 2, 2),
            Err(Error::LengthMismatch { .. })
        ));
        Ok(())
    })
}

#[test]
fn reshape_non_contiguous_returns_error() -> Result<(), Error> {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let base = ctx.new_tensor_2d::<f32>(Shape2D::new(4, 3))?; // 12 elements
        let transposed = ctx.transpose(&base)?; // non-contiguous
        assert!(matches!(
            ctx.reshape_1d(&transposed, 12),
            Err(Error::NotContiguous)
        ));
        Ok(())
    })
}

// -- view_2d with stride and offset --

#[test]
fn view_2d_strided_with_offset() -> Result<(), Error> {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        // Base: 12 f32 values = [0..11]
        let base = ctx.new_tensor_1d::<f32>(Length::new(12))?;
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        base.write_data(&data)?;

        let elem_size = std::mem::size_of::<f32>();
        // View 2 elements per row, 3 rows, stride = 4 * sizeof(f32) (skip 2 elements between rows)
        let viewed = ctx.view_2d(&base, 2, 3, 4 * elem_size, 0)?;
        assert_eq!(*viewed.dims::<2>()?.as_array(), [2, 3]);
        // Row 0: [0,1], Row 1: [4,5], Row 2: [8,9]
        assert_eq!(viewed.get_data(TensorIndex::new(0))?, 0.0);
        assert_eq!(viewed.get_data(TensorIndex::new(1))?, 1.0);

        // View with offset: start at element 1
        let viewed_off = ctx.view_2d(&base, 2, 2, 4 * elem_size, 1 * elem_size)?;
        assert_eq!(viewed_off.get_data(TensorIndex::new(0))?, 1.0);
        Ok(())
    })
}

// -- view_3d --

#[test]
fn view_3d_smoke() -> Result<(), Error> {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        // 24 f32 elements
        let base = ctx.new_tensor_1d::<f32>(Length::new(24))?;
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        base.write_data(&data)?;

        let elem = std::mem::size_of::<f32>();
        // 3D view: ne0=2, ne1=3, ne2=4
        // Contiguous strides: nb1 = 2*elem, nb2 = 2*3*elem
        let v3 = ctx.view_3d(&base, 2, 3, 4, 2 * elem, 6 * elem, 0)?;
        assert_eq!(*v3.dims::<3>()?.as_array(), [2, 3, 4]);
        assert_eq!(v3.element_count()?, 24);
        // First element
        assert_eq!(v3.get_data(TensorIndex::new(0))?, 0.0);
        Ok(())
    })
}

// -- view_4d --

#[test]
fn view_4d_smoke() -> Result<(), Error> {
    let mem = Bytes::new(512 * 1024);
    with_context(mem, |ctx| {
        // 2*3*4*5 = 120 elements
        let base = ctx.new_tensor_1d::<f32>(Length::new(120))?;
        let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
        base.write_data(&data)?;

        let elem = std::mem::size_of::<f32>();
        let v4 = ctx.view_4d(&base, 2, 3, 4, 5, 2 * elem, 6 * elem, 24 * elem, 0)?;
        assert_eq!(*v4.dims::<4>()?.as_array(), [2, 3, 4, 5]);
        assert_eq!(v4.element_count()?, 120);
        assert_eq!(v4.get_data(TensorIndex::new(0))?, 0.0);
        Ok(())
    })
}

// -- aliasing: write through view, verify base changed --

#[test]
fn view_aliases_base_tensor() -> Result<(), Error> {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let base = ctx.new_tensor_1d::<f32>(Length::new(8))?;
        base.write_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;

        let elem = std::mem::size_of::<f32>();
        // View of last 4 elements
        let tail = ctx.view_1d(&base, 4, 4 * elem)?;
        assert_eq!(tail.get_data(TensorIndex::new(0))?, 5.0);

        // Write through the view
        tail.write_data(&[50.0, 60.0, 70.0, 80.0])?;

        // Verify base tensor was modified
        let all = base.read_data()?;
        assert_eq!(all, vec![1.0, 2.0, 3.0, 4.0, 50.0, 60.0, 70.0, 80.0]);
        Ok(())
    })
}

#[test]
fn reshape_aliases_base_tensor() -> Result<(), Error> {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let base = ctx.new_tensor_2d::<f32>(Shape2D::new(3, 2))?;
        base.write_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

        let flat = ctx.reshape_1d(&base, 6)?;
        flat.write_data(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])?;

        let base_data = base.read_data()?;
        assert_eq!(base_data, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        Ok(())
    })
}

// -- view bounds checks --

#[test]
fn view_1d_out_of_bounds_returns_error() -> Result<(), Error> {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let base = ctx.new_tensor_1d::<f32>(Length::new(4))?; // 16 bytes
        // Try to view 5 elements (20 bytes) from a 16-byte tensor
        assert!(matches!(
            ctx.view_1d(&base, 5, 0),
            Err(Error::ViewOutOfBounds { .. })
        ));

        // Offset pushes view past end
        assert!(matches!(
            ctx.view_1d(&base, 2, 3 * std::mem::size_of::<f32>()),
            Err(Error::ViewOutOfBounds { .. })
        ));
        Ok(())
    })
}

#[test]
fn view_3d_out_of_bounds_returns_error() -> Result<(), Error> {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let base = ctx.new_tensor_1d::<f32>(Length::new(8))?; // 32 bytes
        let elem = std::mem::size_of::<f32>();
        // ne0=2, ne1=2, ne2=3 → contiguous would need 12 elements = 48 bytes
        assert!(matches!(
            ctx.view_3d(&base, 2, 2, 3, 2 * elem, 4 * elem, 0),
            Err(Error::ViewOutOfBounds { .. })
        ));
        Ok(())
    })
}

// -- cross-context view tests --

/// Verify that `view_1d_of` can create a view from a different (longer-lived) context.
#[test]
fn cross_context_view_1d_smoke() -> Result<(), Error> {
    let src_ctx = Context::new(256 * 1024)?;
    let base = src_ctx.new_tensor_1d::<f32>(Length::new(8))?;
    base.write_data(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])?;

    let view_ctx = Context::new(256 * 1024)?;
    let v = view_ctx.view_1d_of(&base, 4, 0)?;
    assert_eq!(v.element_count()?, 4);
    assert_eq!(v.get_data(TensorIndex::new(0))?, 10.0);
    assert_eq!(v.get_data(TensorIndex::new(3))?, 40.0);

    // View with offset
    let elem = std::mem::size_of::<f32>();
    let v2 = view_ctx.view_1d_of(&base, 3, 2 * elem)?;
    assert_eq!(v2.element_count()?, 3);
    assert_eq!(v2.get_data(TensorIndex::new(0))?, 30.0);
    assert_eq!(v2.get_data(TensorIndex::new(2))?, 50.0);

    Ok(())
}

/// Verify that `view_2d_of` works across contexts.
#[test]
fn cross_context_view_2d_smoke() -> Result<(), Error> {
    let src_ctx = Context::new(256 * 1024)?;
    let elem = std::mem::size_of::<f32>();
    let base = src_ctx.new_tensor_2d::<f32>(Shape2D::new(4, 3))?;
    #[rustfmt::skip]
    base.write_data(&[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ])?;

    let view_ctx = Context::new(256 * 1024)?;
    let v = view_ctx.view_2d_of(&base, 2, 3, 4 * elem, 0)?;
    assert_eq!(*v.dims::<2>()?.as_array(), [2, 3]);
    // First element of each row: 1.0, 5.0, 9.0
    assert_eq!(v.get_data(TensorIndex::new(0))?, 1.0);
    assert_eq!(v.get_data(TensorIndex::new(2))?, 5.0);

    Ok(())
}

/// Verify that `view_4d_of` works across contexts with backend tensors.
#[test]
fn cross_context_view_4d_backend() -> Result<(), Error> {
    let backend = Backend::new(BackendKind::Cpu)?;

    // Source context with backend-allocated tensor
    let src_ctx = Context::new_no_alloc(256 * 1024)?;
    let base = src_ctx.new_tensor_4d::<f32>(Shape4D::new(2, 3, 4, 1))?;
    let _src_buf = src_ctx.allocate_tensors(&backend)?;
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    base.write_data_backend(&data)?;

    // View context referencing the source tensor
    let view_ctx = Context::new_no_alloc(256 * 1024)?;
    let elem = std::mem::size_of::<f32>();
    let v = view_ctx.view_4d_of(&base, 2, 3, 2, 1, 2 * elem, 6 * elem, 12 * elem, 0)?;
    // ggml may report ndims=3 when last dim is 1, so check element count instead
    assert_eq!(v.element_count()?, 12);

    Ok(())
}

/// Verify that cross-context OOB checks still work.
#[test]
fn cross_context_view_oob_rejected() -> Result<(), Error> {
    let src_ctx = Context::new(256 * 1024)?;
    let base = src_ctx.new_tensor_1d::<f32>(Length::new(4))?;
    base.write_data(&[1.0, 2.0, 3.0, 4.0])?;

    let view_ctx = Context::new(256 * 1024)?;
    // Try to view 5 elements from a 4-element tensor
    assert!(matches!(
        view_ctx.view_1d_of(&base, 5, 0),
        Err(Error::ViewOutOfBounds { .. })
    ));
    // Try to view with offset past end
    let elem = std::mem::size_of::<f32>();
    assert!(matches!(
        view_ctx.view_1d_of(&base, 2, 3 * elem),
        Err(Error::ViewOutOfBounds { .. })
    ));

    Ok(())
}

/// Cross-context view used in a compute graph (the main use case for persistent KV cache).
#[test]
fn cross_context_view_in_graph() -> Result<(), Error> {
    let backend = Backend::new(BackendKind::Cpu)?;

    // Persistent context: holds the source data tensor
    let persistent_ctx = Context::new_no_alloc(256 * 1024)?;
    let source = persistent_ctx.new_tensor_1d::<f32>(Length::new(8))?;

    // Ephemeral context: builds compute graph referencing the persistent tensor
    let graph_ctx = Context::new_no_alloc(256 * 1024)?;
    let view = graph_ctx.view_1d_of(&source, 4, 0)?;
    let local = graph_ctx.new_tensor_1d::<f32>(Length::new(4))?;
    let result = graph_ctx.add(&view, &local)?;

    let mut graph = graph_ctx.new_graph()?;
    graph.build_forward_expand(&result);

    // Allocate both contexts on the same backend
    let _persistent_buf = persistent_ctx.allocate_tensors(&backend)?;
    let _graph_buf = graph_ctx.allocate_tensors(&backend)?;

    source.write_data_backend(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
    local.write_data_backend(&[10.0, 20.0, 30.0, 40.0])?;

    backend.compute(&mut graph)?;
    let out = result.read_data_backend()?;
    assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);

    Ok(())
}
