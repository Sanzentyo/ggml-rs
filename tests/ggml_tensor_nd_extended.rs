#![cfg(feature = "link-system")]

//! Extended ND tensor tests: 1D through 4D creation, data round-trips,
//! shape introspection, DynTensor conversions, and rank/shape error cases.

use ggml_rs::{
    Bytes, Dims, DynTensor, Error, Length, Shape2D, Shape3D, Shape4D, Type, with_context,
};

#[test]
fn tensor_1d_creation_and_data_roundtrip() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let t = ctx.new_tensor_1d::<f32>(Length::new(6))?;
        assert_eq!(t.rank()?, 1);
        assert_eq!(t.element_count()?, 6);
        assert_eq!(t.shape_nd()?, vec![6]);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        t.write_data(&data)?;
        let read_back = t.read_data()?;
        assert_eq!(read_back, data);
        Ok(())
    })
    .unwrap();
}

#[test]
fn tensor_2d_creation_and_data_roundtrip() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let t = ctx.new_tensor_2d::<f32>(Shape2D::new(3, 2))?;
        assert_eq!(t.rank()?, 2);
        assert_eq!(t.element_count()?, 6);
        assert_eq!(t.shape_nd()?, vec![3, 2]);
        assert_eq!(t.shape_2d()?, (3, 2));

        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        t.write_data(&data)?;
        let read_back = t.read_data()?;
        assert_eq!(read_back, data);
        Ok(())
    })
    .unwrap();
}

#[test]
fn tensor_3d_creation_reshape_and_data_roundtrip() {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let t = ctx.new_tensor_3d::<f32>(Shape3D::new(4, 3, 2))?;
        assert_eq!(t.rank()?, 3);
        assert_eq!(t.element_count()?, 24);
        assert_eq!(t.shape_nd()?, vec![4, 3, 2]);
        assert_eq!(t.shape_3d()?, Shape3D::new(4, 3, 2));

        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        t.write_data(&data)?;
        let read_back = t.read_data()?;
        assert_eq!(read_back, data);

        let reshaped = ctx.reshape_2d(&t, 6, 4)?;
        assert_eq!(reshaped.rank()?, 2);
        assert_eq!(reshaped.element_count()?, 24);
        assert_eq!(reshaped.shape_2d()?, (6, 4));
        Ok(())
    })
    .unwrap();
}

#[test]
fn tensor_4d_creation_and_shape_introspection() {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let t = ctx.new_tensor_4d::<f32>(Shape4D::new(5, 4, 3, 2))?;
        assert_eq!(t.rank()?, 4);
        assert_eq!(t.element_count()?, 5 * 4 * 3 * 2);
        assert_eq!(t.shape_nd()?, vec![5, 4, 3, 2]);
        assert_eq!(t.shape_4d()?, Shape4D::new(5, 4, 3, 2));
        Ok(())
    })
    .unwrap();
}

#[test]
fn tensor_rank_on_various_ranks() {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let t1 = ctx.new_tensor_1d::<f32>(Length::new(8))?;
        assert_eq!(t1.rank()?, 1);

        let t2 = ctx.new_tensor_2d::<f32>(Shape2D::new(4, 2))?;
        assert_eq!(t2.rank()?, 2);

        let t3 = ctx.new_tensor_3d::<f32>(Shape3D::new(4, 3, 2))?;
        assert_eq!(t3.rank()?, 3);

        let t4 = ctx.new_tensor_4d::<f32>(Shape4D::new(5, 4, 3, 2))?;
        assert_eq!(t4.rank()?, 4);
        Ok(())
    })
    .unwrap();
}

#[test]
fn tensor_shape_nd_returns_correct_dimensions() {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let t1 = ctx.new_tensor_1d::<f32>(Length::new(10))?;
        assert_eq!(t1.shape_nd()?, vec![10]);

        let t2 = ctx.new_tensor_2d::<f32>(Shape2D::new(7, 3))?;
        assert_eq!(t2.shape_nd()?, vec![7, 3]);

        let t3 = ctx.new_tensor_3d::<f32>(Shape3D::new(5, 4, 3))?;
        assert_eq!(t3.shape_nd()?, vec![5, 4, 3]);

        let t4 = ctx.new_tensor_4d::<f32>(Shape4D::new(6, 5, 4, 3))?;
        assert_eq!(t4.shape_nd()?, vec![6, 5, 4, 3]);
        Ok(())
    })
    .unwrap();
}

#[test]
fn tensor_element_count_on_multidimensional_tensors() {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let t2 = ctx.new_tensor_2d::<f32>(Shape2D::new(5, 3))?;
        assert_eq!(t2.element_count()?, 15);

        let t3 = ctx.new_tensor_3d::<f32>(Shape3D::new(4, 3, 2))?;
        assert_eq!(t3.element_count()?, 24);

        let t4 = ctx.new_tensor_4d::<f32>(Shape4D::new(3, 4, 5, 6))?;
        assert_eq!(t4.element_count()?, 3 * 4 * 5 * 6);
        Ok(())
    })
    .unwrap();
}

#[test]
fn unsupported_rank_on_shape_2d_for_3d_tensor() {
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
fn unexpected_shape_on_non_2d_tensor() {
    // In ggml's model, a 1D tensor has row_count=1, col_count=N, so shape()
    // succeeds with (N, 1). Test the actual error path on a 3D tensor instead.
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let t3 = ctx.new_tensor_3d::<f32>(Shape3D::new(4, 3, 2))?;
        let err = t3
            .dims::<2>()
            .expect_err("3D tensor should not support dims<2>");
        assert!(matches!(err, Error::UnsupportedRank(3)));
        Ok(())
    })
    .unwrap();
}

#[test]
fn i32_tensor_creation_and_data_roundtrip() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let t = ctx.new_tensor_1d::<i32>(Length::new(5))?;
        assert_eq!(t.rank()?, 1);
        assert_eq!(t.element_count()?, 5);

        let data = vec![10, 20, 30, 40, 50];
        t.write_data(&data)?;
        let read_back = t.read_data()?;
        assert_eq!(read_back, data);
        Ok(())
    })
    .unwrap();
}

#[test]
fn i32_tensor_2d_creation_and_data_roundtrip() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let t = ctx.new_tensor_2d::<i32>(Shape2D::new(3, 2))?;
        assert_eq!(t.rank()?, 2);
        assert_eq!(t.element_count()?, 6);

        let data = vec![1, 2, 3, 4, 5, 6];
        t.write_data(&data)?;
        let read_back = t.read_data()?;
        assert_eq!(read_back, data);
        Ok(())
    })
    .unwrap();
}

#[test]
fn dyn_tensor_creation_via_new_tensor() {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let dt = ctx.new_tensor(Type::F32, Dims::new([4, 3, 2]))?;
        assert_eq!(dt.rank()?, 3);
        assert_eq!(dt.element_count()?, 24);
        assert_eq!(dt.shape_nd()?, vec![4, 3, 2]);
        Ok(())
    })
    .unwrap();
}

#[test]
fn dyn_tensor_as_typed_conversion() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let dt = ctx.new_tensor(Type::F32, Dims::new([4]))?;
        let typed = dt.as_typed::<f32>()?;
        assert_eq!(typed.element_count()?, 4);

        let data = vec![1.0, 2.0, 3.0, 4.0];
        typed.write_data(&data)?;
        let read_back = typed.read_data()?;
        assert_eq!(read_back, data);
        Ok(())
    })
    .unwrap();
}

#[test]
fn dyn_tensor_as_typed_type_mismatch() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let dt = ctx.new_tensor(Type::F32, Dims::new([4]))?;
        let result = dt.as_typed::<i32>();
        assert!(matches!(result, Err(Error::TypeMismatch { .. })));
        Ok(())
    })
    .unwrap();
}

#[test]
fn tensor_into_dyn_conversion() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let t = ctx.new_tensor_1d::<f32>(Length::new(4))?;
        let data = vec![10.0, 20.0, 30.0, 40.0];
        t.write_data(&data)?;

        let dyn_t: DynTensor = t.into_dyn();
        assert_eq!(dyn_t.rank()?, 1);
        assert_eq!(dyn_t.element_count()?, 4);
        assert_eq!(dyn_t.shape_nd()?, vec![4]);
        Ok(())
    })
    .unwrap();
}

#[test]
fn dyn_tensor_roundtrip_typed_to_dyn_and_back() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let t = ctx.new_tensor_2d::<f32>(Shape2D::new(3, 2))?;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        t.write_data(&data)?;

        let dyn_t = t.into_dyn();
        assert_eq!(dyn_t.rank()?, 2);
        assert_eq!(dyn_t.shape_nd()?, vec![3, 2]);

        let typed_back = dyn_t.as_typed::<f32>()?;
        assert_eq!(typed_back.element_count()?, 6);
        let read_back = typed_back.read_data()?;
        assert_eq!(read_back, data);
        Ok(())
    })
    .unwrap();
}

#[test]
fn dyn_tensor_dims_method() {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let dt = ctx.new_tensor(Type::F32, Dims::new([5, 4, 3]))?;
        let dims = dt.dims::<3>()?;
        assert_eq!(*dims.as_array(), [5, 4, 3]);

        let err = dt
            .dims::<2>()
            .expect_err("3D tensor should not support dims<2>");
        assert!(matches!(err, Error::UnsupportedRank(3)));
        Ok(())
    })
    .unwrap();
}

#[test]
fn tensor_dims_method() {
    let mem = Bytes::new(256 * 1024);
    with_context(mem, |ctx| {
        let t = ctx.new_tensor_2d::<f32>(Shape2D::new(7, 3))?;
        let dims = t.dims::<2>()?;
        assert_eq!(*dims.as_array(), [7, 3]);

        let err = t
            .dims::<3>()
            .expect_err("2D tensor should not support dims<3>");
        assert!(matches!(err, Error::UnsupportedRank(2)));
        Ok(())
    })
    .unwrap();
}

#[test]
fn dyn_tensor_name_set_and_get() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let dt = ctx.new_tensor(Type::I32, Dims::new([8]))?;
        dt.set_name("dyn_tensor")?;
        assert_eq!(dt.name()?, "dyn_tensor");
        Ok(())
    })
    .unwrap();
}

#[test]
fn tensor_nbytes() {
    let mem = Bytes::new(64 * 1024);
    with_context(mem, |ctx| {
        let t = ctx.new_tensor_1d::<f32>(Length::new(8))?;
        assert_eq!(t.nbytes(), 8 * 4);

        let dt = ctx.new_tensor(Type::I32, Dims::new([16]))?;
        assert_eq!(dt.nbytes(), 16 * 4);
        Ok(())
    })
    .unwrap();
}
