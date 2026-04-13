#![cfg(feature = "link-system")]

//! Integration tests for `GraphAllocator` — pre-reserved graph-level memory allocator.

use ggml_rs::{Backend, BackendKind, Context, Error, GraphAllocator, Shape2D};

/// Basic lifecycle: new → reserve → alloc_graph → compute.
#[test]
fn graph_allocator_basic_lifecycle() -> Result<(), Error> {
    Backend::load_all();
    let backend = Backend::new(BackendKind::Cpu)?;
    let mut gallocr = GraphAllocator::new(&backend)?;

    // Build a simple add graph for reservation.
    let ctx = Context::new_no_alloc(512 * 1024)?;
    let a = ctx.new_tensor_2d::<f32>(Shape2D::new(4, 4))?;
    let b = ctx.new_tensor_2d::<f32>(Shape2D::new(4, 4))?;
    let c = ctx.add(&a, &b)?;
    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&c);

    gallocr.reserve(&graph)?;
    assert!(gallocr.buffer_size() > 0);

    gallocr.alloc_graph(&mut graph)?;

    a.write_data_backend(&[1.0_f32; 16])?;
    b.write_data_backend(&[2.0_f32; 16])?;
    backend.compute(&mut graph)?;

    let result: Vec<f32> = c.read_data_backend()?;
    assert_eq!(result.len(), 16);
    for v in &result {
        assert!((v - 3.0).abs() < 1e-5);
    }

    Ok(())
}

/// Reserve once, reuse for multiple graph computations.
#[test]
fn graph_allocator_reuse_across_steps() -> Result<(), Error> {
    Backend::load_all();
    let backend = Backend::new(BackendKind::Cpu)?;
    let mut gallocr = GraphAllocator::new(&backend)?;

    // Reserve with a "max-size" graph.
    {
        let ctx = Context::new_no_alloc(512 * 1024)?;
        let a = ctx.new_tensor_2d::<f32>(Shape2D::new(8, 8))?;
        let b = ctx.new_tensor_2d::<f32>(Shape2D::new(8, 8))?;
        let c = ctx.add(&a, &b)?;
        let mut graph = ctx.new_graph()?;
        graph.build_forward_expand(&c);
        gallocr.reserve(&graph)?;
    }

    let reserved_size = gallocr.buffer_size();

    // Step 1: smaller graph reuses the buffer.
    {
        let ctx = Context::new_no_alloc(512 * 1024)?;
        let a = ctx.new_tensor_2d::<f32>(Shape2D::new(4, 4))?;
        let b = ctx.new_tensor_2d::<f32>(Shape2D::new(4, 4))?;
        let c = ctx.add(&a, &b)?;
        let mut graph = ctx.new_graph()?;
        graph.build_forward_expand(&c);
        gallocr.alloc_graph(&mut graph)?;

        a.write_data_backend(&[1.0_f32; 16])?;
        b.write_data_backend(&[2.0_f32; 16])?;
        backend.compute(&mut graph)?;

        let result: Vec<f32> = c.read_data_backend()?;
        assert!(result.iter().all(|v| (v - 3.0).abs() < 1e-5));
    }

    // Step 2: same-size graph also works.
    {
        let ctx = Context::new_no_alloc(512 * 1024)?;
        let a = ctx.new_tensor_2d::<f32>(Shape2D::new(4, 4))?;
        let b = ctx.new_tensor_2d::<f32>(Shape2D::new(4, 4))?;
        let c = ctx.add(&a, &b)?;
        let mut graph = ctx.new_graph()?;
        graph.build_forward_expand(&c);
        gallocr.alloc_graph(&mut graph)?;

        a.write_data_backend(&[10.0_f32; 16])?;
        b.write_data_backend(&[5.0_f32; 16])?;
        backend.compute(&mut graph)?;

        let result: Vec<f32> = c.read_data_backend()?;
        assert!(result.iter().all(|v| (v - 15.0).abs() < 1e-5));
    }

    // Buffer size should not have changed (reuse, no reallocation).
    assert_eq!(gallocr.buffer_size(), reserved_size);

    Ok(())
}

/// GraphAllocator skips tensors that already have backend buffers.
#[test]
fn graph_allocator_skips_pre_allocated_tensors() -> Result<(), Error> {
    Backend::load_all();
    let backend = Backend::new(BackendKind::Cpu)?;
    let mut gallocr = GraphAllocator::new(&backend)?;

    // Create a "persistent" tensor with its own buffer.
    let persistent_ctx = Context::new_no_alloc(512 * 1024)?;
    let persistent = persistent_ctx.new_tensor_2d::<f32>(Shape2D::new(4, 4))?;
    let _persistent_buffer = persistent_ctx.allocate_tensors(&backend)?;
    persistent.write_data_backend(&[100.0_f32; 16])?;

    // Build a graph that uses a cross-context view of the persistent tensor.
    let ctx = Context::new_no_alloc(512 * 1024)?;
    let view = ctx.view_2d_of(&persistent, 4, 4, 4 * 4, 0)?;
    let b = ctx.new_tensor_2d::<f32>(Shape2D::new(4, 4))?;
    let c = ctx.add(&view, &b)?;
    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&c);

    gallocr.reserve(&graph)?;
    gallocr.alloc_graph(&mut graph)?;

    // The persistent tensor should still have its original data.
    b.write_data_backend(&[1.0_f32; 16])?;
    backend.compute(&mut graph)?;

    let result: Vec<f32> = c.read_data_backend()?;
    assert!(result.iter().all(|v| (v - 101.0).abs() < 1e-5));

    Ok(())
}

/// GraphAllocator works with matmul graphs.
#[test]
fn graph_allocator_matmul() -> Result<(), Error> {
    Backend::load_all();
    let backend = Backend::new(BackendKind::Cpu)?;
    let mut gallocr = GraphAllocator::new(&backend)?;

    let ctx = Context::new_no_alloc(512 * 1024)?;
    let a = ctx.new_tensor_2d::<f32>(Shape2D::new(4, 2))?;
    let b = ctx.new_tensor_2d::<f32>(Shape2D::new(4, 3))?;
    let c = ctx.mul_mat(&a, &b)?;
    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&c);

    gallocr.reserve(&graph)?;
    gallocr.alloc_graph(&mut graph)?;

    // Identity-like: a = [[1,0,0,0],[0,1,0,0]], b = ones(4,3)
    let a_data = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
    let b_data = [1.0_f32; 12];
    a.write_data_backend(&a_data)?;
    b.write_data_backend(&b_data)?;
    backend.compute(&mut graph)?;

    let result: Vec<f32> = c.read_data_backend()?;
    assert_eq!(result.len(), 6); // 2×3
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 1.0).abs() < 1e-5);

    Ok(())
}

/// Buffer size returns zero before reservation.
#[test]
fn graph_allocator_buffer_size_zero_before_reserve() -> Result<(), Error> {
    Backend::load_all();
    let backend = Backend::new(BackendKind::Cpu)?;
    let gallocr = GraphAllocator::new(&backend)?;
    assert_eq!(gallocr.buffer_size(), 0);
    Ok(())
}
