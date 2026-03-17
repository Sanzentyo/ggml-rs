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
        assert_eq!(tensor.read_data_at::<f32>(1, 2)?, vec![2.0, 3.0]);
        tensor.write_data_at(2, &[9.0, 8.0])?;
        assert_eq!(tensor.read_data::<f32>()?, vec![1.0, 2.0, 9.0, 8.0]);
        let err = tensor
            .read_data_at::<f32>(4, 1)
            .expect_err("host read range past end should error");
        assert!(matches!(err, Error::IndexOutOfBounds { .. }));
        let err = tensor
            .write_data_at(4, &[1.0])
            .expect_err("host write range past end should error");
        assert!(matches!(err, Error::IndexOutOfBounds { .. }));
        Ok(tensor.element_count()?)
    })?;
    assert_eq!(element_count, 4);

    let no_alloc_mem = Context::recommended_backend_matmul_memory::<f32>(lhs, rhs)?;
    with_no_alloc_context(no_alloc_mem, |ctx| {
        let backend = Backend::new(BackendKind::Cpu)?;
        let tensor = ctx.new_tensor_2d::<f32>(lhs)?;
        let _buffer = ctx.allocate_tensors(&backend)?;
        tensor.write_data_backend(&[1.0, 2.0, 3.0, 4.0])?;
        let out = tensor.read_data_backend::<f32>()?;
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
        assert_eq!(viewed.get_data::<f32>(TensorIndex::new(3))?, 3.0);

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
        assert_eq!(tensor.read_data_backend::<i32>()?, values.to_vec());

        tensor.write_data_backend_at(2, &[111, 222])?;
        assert_eq!(
            tensor.read_data_backend::<i32>()?,
            vec![10, 20, 111, 222, 50, 60, 70, 80]
        );
        assert_eq!(
            tensor.read_data_backend_at::<i32>(2, 3)?,
            vec![111, 222, 50]
        );

        let err = tensor
            .write_data_backend_at(8, &[1])
            .expect_err("offset past end should error");
        assert!(matches!(err, Error::IndexOutOfBounds { .. }));

        let err = tensor
            .read_data_backend_at::<i32>(7, 2)
            .expect_err("read range past end should error");
        assert!(matches!(err, Error::IndexOutOfBounds { .. }));

        Ok(())
    })
}
