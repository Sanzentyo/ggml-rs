# Fused projection + conv graph

**Date**: 2026-04-19
**Commit**: c3d6b96 (exp/oh-my)

## Summary

Merged the linear attention prefill's two-graph pipeline (projection →
host round-trip → conv) into a single ggml compute graph. Eliminates one
host↔device data transfer per linear attention layer per prefill step.

## Previous architecture (2 graphs)

```
Graph 1: mul_mat(W_qkv, X), mul_mat(W_z, X), mul_mat(W_alpha, X), mul_mat(W_beta, X)
  → read QKV, Z, alpha, beta to host

Host: transpose QKV, left-pad with zeros, upload to backend

Graph 2: ssm_conv(padded_qkv, kernel) → silu
  → read conv_silu to host
```

## New architecture (1 graph)

```
Single graph:
  mul_mat(W_qkv, X)  → transpose → cont → concat(zeros, ...) → reshape_3d → ssm_conv → silu
  mul_mat(W_z, X)
  mul_mat(W_alpha, X)
  mul_mat(W_beta, X)

Read: conv_silu, qkv_pre_conv, Z, alpha, beta
```

## In-graph layout transform chain

```
mul_mat output:  ne[0]=conv_channels, ne[1]=seq_len  (channels-fast)
→ transpose:     ne[0]=seq_len, ne[1]=conv_channels  (time-fast, strided)
→ cont:          contiguous copy
→ concat(zeros): ne[0]=padded_len, ne[1]=conv_channels
→ reshape_3d:    ne[0]=padded_len, ne[1]=conv_channels, ne[2]=1  (batch dim)
→ ssm_conv:      ne[0]=conv_channels, ne[1]=seq_len, ne[2]=1
→ silu:          elementwise activation
```

## Key design decisions

1. **Pre-conv QKV must be preserved**: `capture_conv_buffer` needs the last
   `kernel_size-1` rows of pre-conv QKV for decode state continuity. The fused
   graph reads back both `conv_silu` and `qkv_out` (pre-conv projection output).

2. **Buffer lifetime**: The `BackendBuffer` from `allocate_tensors` must live
   until after all `read_data_backend` calls. Initial implementation had the
   buffer inside if/else branches (dropped before reads → SIGSEGV). Fixed by
   hoisting graph build, allocation, upload, compute, and reads to a shared
   scope after the conditional conv-graph construction.

3. **concat 2D limitation**: The `concat` wrapper validates `shape_2d()`, so
   tensors must be 2D during concat. Solution: concat 2D, then `reshape_3d`
   to add the batch dimension for `ssm_conv`.

4. **kernel_size==1 branch**: When `pad==0`, skip concat entirely. Go from
   `cont` → `reshape_3d` → `ssm_conv` → `silu`.

## Memory estimation

Peak memory is higher than separate graphs since projection AND conv
intermediates coexist. The function sums:
- All 4 projection matmul memory estimates
- Conv intermediates: cont, zeros, concat, kernel, ssm_conv output, silu output
- 2× PROJECTION_SLACK_BYTES safety margin

## Files changed

- `llama-rs/src/e2e/linear_attention.rs`:
  - Added `FusedLinearOutputs` struct
  - Added `project_and_conv_fused_graph` function (~120 lines)
  - Rewrote `qwen35_linear_attention_core` to use fused graph
  - `causal_depthwise_conv_graph` demoted to `#[cfg(test)]`
  - Removed unused `Shape3D` import from module level

## Test results

196 tests pass, 1 ignored. No new clippy warnings.
