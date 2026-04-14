# Fully Fused Single-Graph Full Attention

**Date**: 2026-04-20
**Branch**: `exp/oh-my`
**File**: `llama-rs/src/e2e/attention.rs`

## Summary

Merged the previous two-graph pipeline into a single ggml compute graph for
the full attention prefill path. This eliminates the host↔device round-trip
between QKV projection and attention scoring.

## Before (two-graph pipeline)

```
Graph 1: mul_mat(W_q, X), mul_mat(W_k, X), mul_mat(W_v, X) → read Q_full, K, V to host
Host:    deinterleave Q/gate, per-head RMS norm, NeoX RoPE
Graph 2: permute → cont → flash_attn_ext → sigmoid(gate) → mul → reshape_2d → mul_mat(W_out) → read output
```

= 2 graph constructions, 2 `allocate_tensors`, 2 `compute` calls, 10+ host↔device transfers.

## After (single-graph)

```
Single graph:
  mul_mat(W_q/W_k/W_v, X)          // 3 projection matmuls
  → reshape_3d + view_3d            // strided deinterleave Q/gate
  → cont                            // materialize non-contiguous views
  → rms_norm + mul (weight)         // per-head QK normalization
  → rope_ext (NeoX mode=2)          // rotary position encoding
  → reshape_4d + permute + cont     // layout for flash_attn_ext
  → flash_attn_ext(mask_f16, scale) // fused scaled dot-product attention
  → sigmoid(gate) → mul             // gated attention
  → reshape_2d → mul_mat(W_out)     // output projection
  → read output

  // Conditional intermediate readback for KV cache:
  → read K_rope, V_proj (only when state capture needed)
```

= 1 graph construction, 1 `allocate_tensors`, 1 `compute` call, single weight/input write + single output read.

## Key techniques

### Strided view_3d for deinterleave

Q_full has interleaved layout `[Q_h0(D), G_h0(D), Q_h1(D), G_h1(D), ...]`
= `[D, 2*H, T]`. Two `view_3d` calls with `nb1 = 2*D*sizeof(f32)` extract
Q (offset=0) and gate (offset=D*sizeof(f32)) as non-contiguous views.
`cont()` materializes them for downstream ops.

### In-graph RMS norm + weight broadcast

`rms_norm` normalizes along `ne[0]=D` for each `(h, t)` pair in a `[D, H, T]`
tensor. `mul` with a `[D]` weight tensor broadcasts automatically via
`ggml_can_repeat`.

### In-graph NeoX RoPE

`rope_ext_with_i32_positions` with `mode=2` (NeoX) and `n_dims=rope_n_dims`.
Positions indexed by `ne[2]=T` on 3D tensors.

### Intermediate readback via build_forward_expand

`build_forward_expand(&k_rope)` and `build_forward_expand(&v_proj)` ensure
these intermediate tensors are computed even though they're not in the output's
dependency chain. After `compute()`, `read_data_backend()` reads them back
for KV cache state capture.

## Memory model note

`allocate_tensors()` statically allocates ALL non-view tensors. No graph-liveness
reuse — all intermediates coexist. Peak memory is higher than the two separate
graphs, but the throughput improvement from eliminating round-trips outweighs this.

## Unchanged paths

- Decode path (`qwen35_full_attention_decode_step`) still uses host-side processing.
  For seq_len=1, graph overhead exceeds any compute benefit.
- `project_qkv_graph`, `project_and_prepare_qkv`, `deinterleave_q_gate`,
  `apply_neox_rope_in_place` retained for decode path use.

## Test results

All 202 tests pass, including the critical parity test
`full_attention_prefill_then_decode_matches_full_reprocess` which verifies
that the fused-graph prefill produces identical results to the host-side decode
path (proving numerical correctness of the in-graph deinterleave, norm, and
RoPE operations).
