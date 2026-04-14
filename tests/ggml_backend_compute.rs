#![cfg(feature = "link-system")]

//! Backend compute path tests: CPU backend creation, buffer allocation,
//! backend data round-trips, backend matmul, and Metal backend on macOS.

use ggml_rs::{
    Backend, BackendKind, Bytes, Context, Dims, Error, Length, Shape2D, Shape4D, ThreadCount, Type,
    with_context, with_no_alloc_context,
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

/// Multi-op graph: matmul + add (bias).
/// Verifies that a backend can execute graphs with more than one operation.
fn multi_op_matmul_add_on_backend(kind: BackendKind) -> Result<(), Error> {
    let mem = Bytes::new(64 * 1024 * 1024);
    with_no_alloc_context(mem, |ctx| {
        let backend = Backend::new(kind)?;

        // W: 3×2, x: 2×1, b: 3-element bias (broadcasts to mul_mat result [3,1])
        let w = ctx.new_tensor_2d::<f32>(Shape2D::new(2, 3))?;
        let x = ctx.new_tensor_2d::<f32>(Shape2D::new(2, 1))?;
        let b = ctx.new_tensor_1d::<f32>(Length::new(3))?;

        let wx = ctx.mul_mat(&w, &x)?;
        let result = ctx.add(&wx, &b)?;

        let mut graph = ctx.new_graph()?;
        graph.build_forward_expand(&result);
        let _buffer = ctx.allocate_tensors(&backend)?;

        w.write_data_backend(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        x.write_data_backend(&[10.0_f32, 20.0])?;
        b.write_data_backend(&[100.0_f32, 200.0, 300.0])?;

        backend.compute(&mut graph)?;
        backend.synchronize()?;

        let output = result.read_data_backend()?;
        // W*x = [50, 110, 170], + b = [150, 310, 470]
        let expected = [150.0_f32, 310.0, 470.0];
        assert_eq!(output.len(), expected.len());
        for (i, (&actual, &expected)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-3,
                "multi_op index {i}: expected {expected}, got {actual}"
            );
        }
        Ok(())
    })
}

#[test]
fn cpu_multi_op_matmul_add() -> Result<(), Error> {
    multi_op_matmul_add_on_backend(BackendKind::Cpu)
}

#[test]
#[cfg(target_os = "macos")]
fn metal_multi_op_matmul_add() -> Result<(), Error> {
    multi_op_matmul_add_on_backend(BackendKind::Metal)
}

/// Backend matmul parity: verify CPU and Metal produce the same result.
fn backend_matmul_on(kind: BackendKind) -> Result<Vec<f32>, Error> {
    let lhs = Shape2D::new(3, 2);
    let rhs = Shape2D::new(3, 4);
    let mem = Context::recommended_backend_matmul_memory::<f32>(lhs, rhs)?;

    with_no_alloc_context(mem, |ctx| {
        let backend = Backend::new(kind)?;
        let a = ctx.new_tensor_2d::<f32>(lhs)?;
        let b = ctx.new_tensor_2d::<f32>(rhs)?;
        let result = ctx.mul_mat(&a, &b)?;

        let mut graph = ctx.new_graph()?;
        graph.build_forward_expand(&result);
        let _buffer = ctx.allocate_tensors(&backend)?;

        let a_data: Vec<f32> = (1..=6).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (1..=12).map(|i| i as f32 * 0.5).collect();
        a.write_data_backend(&a_data)?;
        b.write_data_backend(&b_data)?;

        backend.compute(&mut graph)?;
        backend.synchronize()?;
        result.read_data_backend()
    })
}

#[test]
fn cpu_backend_matmul_nontrivial() -> Result<(), Error> {
    let output = backend_matmul_on(BackendKind::Cpu)?;
    assert_eq!(output.len(), 2 * 4); // 2 rows × 4 rows (matmul transposes)
    // Verify non-zero
    assert!(output.iter().any(|&v| v.abs() > 0.1));
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn metal_cpu_matmul_parity() -> Result<(), Error> {
    let cpu_output = backend_matmul_on(BackendKind::Cpu)?;
    let metal_output = backend_matmul_on(BackendKind::Metal)?;
    assert_eq!(cpu_output.len(), metal_output.len());
    for (i, (&cpu, &metal)) in cpu_output.iter().zip(metal_output.iter()).enumerate() {
        assert!(
            (cpu - metal).abs() < 1e-3,
            "CPU/Metal mismatch at {i}: CPU={cpu}, Metal={metal}"
        );
    }
    Ok(())
}

// ── Sigmoid parity ──────────────────────────────────────────────────────

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_on_backend(kind: BackendKind) -> Result<Vec<f32>, Error> {
    let input: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0, -10.0];
    let n = input.len();
    let mem = Bytes::new(n * 4 * 16 + 131072);
    let ctx = Context::new_no_alloc_bytes(mem)?;
    let a = ctx.new_tensor_1d::<f32>(Length::new(n))?;
    let out = ctx.sigmoid(&a)?;
    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&out);
    let backend = Backend::new(kind)?;
    let _buf = ctx.allocate_tensors(&backend)?;
    a.write_data_backend(&input)?;
    backend.compute(&mut graph)?;
    out.read_data_backend()
}

#[test]
fn sigmoid_cpu_matches_reference() -> Result<(), Error> {
    let input: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0, -10.0];
    let expected: Vec<f32> = input.iter().map(|&x| sigmoid_scalar(x)).collect();
    let result = sigmoid_on_backend(BackendKind::Cpu)?;
    assert_eq!(result.len(), expected.len());
    for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "sigmoid mismatch at {i}: got={got}, want={want}"
        );
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn sigmoid_metal_cpu_parity() -> Result<(), Error> {
    let cpu = sigmoid_on_backend(BackendKind::Cpu)?;
    let metal = sigmoid_on_backend(BackendKind::Metal)?;
    assert_eq!(cpu.len(), metal.len());
    for (i, (&c, &m)) in cpu.iter().zip(metal.iter()).enumerate() {
        assert!(
            (c - m).abs() < 1e-4,
            "sigmoid CPU/Metal mismatch at {i}: cpu={c}, metal={m}"
        );
    }
    Ok(())
}

// ── flash_attn_ext parity ───────────────────────────────────────────────

/// Reference causal attention (no flash) via manual scoring.
/// Data layout: `[D, T, H]` — column-major per head as ggml stores it.
#[allow(clippy::too_many_arguments)] // mathematical params map to Q/K/V dimensions
fn reference_causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: usize,
    t: usize,
    h: usize,
    hkv: usize,
    scale: f32,
) -> Vec<f32> {
    let groups = h / hkv;
    let mut output = vec![0.0_f32; d * t * h];
    for head in 0..h {
        let kv_head = head / groups;
        for tok in 0..t {
            let mut scores = vec![f32::NEG_INFINITY; t];
            for src in 0..=tok {
                let mut s = 0.0_f32;
                for dim in 0..d {
                    let qi = q[dim + tok * d + head * t * d];
                    let ki = k[dim + src * d + kv_head * t * d];
                    s += qi * ki;
                }
                scores[src] = s * scale;
            }
            let max_s = scores[..=tok]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            let mut exp_scores = vec![0.0_f32; t];
            for src in 0..=tok {
                exp_scores[src] = (scores[src] - max_s).exp();
                sum += exp_scores[src];
            }
            for dim in 0..d {
                let mut val = 0.0_f32;
                for src in 0..=tok {
                    let vi = v[dim + src * d + kv_head * t * d];
                    val += vi * exp_scores[src] / sum;
                }
                output[dim + tok * d + head * t * d] = val;
            }
        }
    }
    output
}

/// Convert a single f32 to IEEE 754 half-precision (f16) bits.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;

    if exponent == 0xFF {
        // Inf / NaN
        if mantissa == 0 {
            return sign | 0x7C00; // Inf
        } else {
            return sign | 0x7E00; // NaN (quiet)
        }
    }

    let unbiased = exponent - 127;
    if unbiased > 15 {
        // Overflow → Inf
        return sign | 0x7C00;
    }
    if unbiased < -24 {
        // Too small → zero
        return sign;
    }
    if unbiased < -14 {
        // Subnormal
        let shift = (-14 - unbiased) as u32;
        let m = (mantissa | 0x80_0000) >> (shift + 13);
        return sign | m as u16;
    }
    let exp16 = ((unbiased + 15) as u16) << 10;
    let man16 = (mantissa >> 13) as u16;
    sign | exp16 | man16
}

/// Convert f32 values to f16 bytes (little-endian).
fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut buf = vec![0u8; values.len() * 2];
    for (i, &v) in values.iter().enumerate() {
        let fp16 = f32_to_f16_bits(v);
        buf[i * 2..i * 2 + 2].copy_from_slice(&fp16.to_le_bytes());
    }
    buf
}

/// Run flash_attn_ext on backend with causal mask, returning `[D, T, H]`
/// after permute+cont.
#[allow(clippy::too_many_arguments)] // mathematical params map to Q/K/V dimensions + backend
fn flash_attn_ext_on_backend(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    d: usize,
    t: usize,
    h: usize,
    hkv: usize,
    scale: f32,
    kind: BackendKind,
) -> Result<Vec<f32>, Error> {
    let total_elems = d * t * h + 2 * d * t * hkv + t * t + d * h * t;
    let mem = Bytes::new(total_elems * 4 * 8 + 262144);
    let ctx = Context::new_no_alloc_bytes(mem)?;

    let q = ctx.new_tensor_4d::<f32>(Shape4D::new(d, t, h, 1))?;
    let k = ctx.new_tensor_4d::<f32>(Shape4D::new(d, t, hkv, 1))?;
    let v = ctx.new_tensor_4d::<f32>(Shape4D::new(d, t, hkv, 1))?;

    // Causal mask: [Tkv, T, 1, 1] as f16 — ggml CPU kernel reads mask as fp16
    let mask = ctx.new_tensor(Type::F16, Dims::new([t, t, 1, 1]))?;
    let mut mask_f32 = vec![0.0_f32; t * t];
    for row in 0..t {
        for col in 0..t {
            if col > row {
                mask_f32[col + row * t] = f32::NEG_INFINITY;
            }
        }
    }
    let mask_f16_bytes = f32_to_f16_bytes(&mask_f32);

    let attn = ctx.flash_attn_ext(&q, &k, &v, Some(&mask), scale, 0.0, 0.0)?;
    let permuted = ctx.permute(&attn, 0, 2, 1, 3)?;
    let contig = ctx.cont(&permuted)?;

    let mut graph = ctx.new_graph()?;
    graph.build_forward_expand(&contig);

    let backend = Backend::new(kind)?;
    let _buf = ctx.allocate_tensors(&backend)?;

    q.write_data_backend(q_data)?;
    k.write_data_backend(k_data)?;
    v.write_data_backend(v_data)?;
    mask.write_bytes_backend(&mask_f16_bytes)?;

    backend.compute(&mut graph)?;
    contig.read_data_backend()
}

#[test]
fn flash_attn_ext_cpu_matches_reference_mha() -> Result<(), Error> {
    let d = 4;
    let t = 3;
    let h = 2;
    let hkv = 2;
    let scale = 1.0 / (d as f32).sqrt();

    let q: Vec<f32> = (0..d * t * h)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..d * t * hkv)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..d * t * hkv)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();

    let expected = reference_causal_attention(&q, &k, &v, d, t, h, hkv, scale);
    let result = flash_attn_ext_on_backend(&q, &k, &v, d, t, h, hkv, scale, BackendKind::Cpu)?;

    assert_eq!(result.len(), expected.len());
    for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "flash_attn MHA mismatch at {i}: got={got:.6}, want={want:.6}, diff={}",
            (got - want).abs()
        );
    }
    Ok(())
}

#[test]
fn flash_attn_ext_cpu_matches_reference_gqa() -> Result<(), Error> {
    let d = 4;
    let t = 3;
    let h = 4;
    let hkv = 2;
    let scale = 1.0 / (d as f32).sqrt();

    let q: Vec<f32> = (0..d * t * h)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
        .collect();
    let k: Vec<f32> = (0..d * t * hkv)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.08)
        .collect();
    let v: Vec<f32> = (0..d * t * hkv)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.06)
        .collect();

    let expected = reference_causal_attention(&q, &k, &v, d, t, h, hkv, scale);
    let result = flash_attn_ext_on_backend(&q, &k, &v, d, t, h, hkv, scale, BackendKind::Cpu)?;

    assert_eq!(result.len(), expected.len());
    for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "flash_attn GQA mismatch at {i}: got={got:.6}, want={want:.6}, diff={}",
            (got - want).abs()
        );
    }
    Ok(())
}

#[test]
fn flash_attn_ext_output_shape_is_permuted() -> Result<(), Error> {
    let d = 4;
    let t = 3;
    let h = 2;

    let mem = Bytes::new(262144);
    let ctx = Context::new_no_alloc_bytes(mem)?;

    let q = ctx.new_tensor_4d::<f32>(Shape4D::new(d, t, h, 1))?;
    let k = ctx.new_tensor_4d::<f32>(Shape4D::new(d, t, h, 1))?;
    let v = ctx.new_tensor_4d::<f32>(Shape4D::new(d, t, h, 1))?;
    let mask = ctx.new_tensor(Type::F16, Dims::new([t, t, 1, 1]))?;

    let attn = ctx.flash_attn_ext(&q, &k, &v, Some(&mask), 0.5, 0.0, 0.0)?;

    // ggml reports rank as 3 when batch=1 (trailing dim squeezed).
    // Use shape_nd() which returns whatever rank ggml reports.
    let shape = attn.shape_nd()?;
    assert_eq!(shape[0], d, "dim0 should be D");
    assert_eq!(shape[1], h, "dim1 should be H (permuted)");
    assert_eq!(shape[2], t, "dim2 should be T (permuted)");

    Ok(())
}
