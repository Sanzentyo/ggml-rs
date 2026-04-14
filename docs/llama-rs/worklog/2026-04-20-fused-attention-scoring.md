# Fused attention scoring graph (flash_attn_ext)

**Date**: 2026-04-20
**Commit**: 57d03d6 (exp/oh-my)

## Summary

Replaced the host-side O(T²·H·D) scoring loop in `qwen35_full_attention_core`
with a single ggml compute graph using `flash_attn_ext`. This is the full
attention counterpart to the linear attention fused projection+conv graph
(2026-04-19).

## Previous architecture (host scoring)

```
Host-side loop per head per token:
  for each query token t:
    for each key token k (up to t):
      score = dot(Q[t,h], K[k,h]) * scale
    weights = softmax(scores)
    O[t,h] = Σ weights[k] * V[k,h]
    O[t,h] *= sigmoid(gate[t,h])
  output = project(O, W_out)
```

Complexity: O(T² · H · D) — dominated by all-pairs dot products.

## New architecture (graph-fused)

```
Single ggml graph:
  Q/K/V [D, H, T, 1] → permute(0,2,1,3) → cont → [D, T, H, 1]
  flash_attn_ext(Q, K, V, mask_f16, scale) → [D, H, T, 1]
  sigmoid(gate [D, H, T, 1])
  mul(attn, sigmoid_gate) → [D, H, T, 1]
  reshape_2d → [H*D, T]
  mul_mat(W_out) → [hidden, T]
```

## Key design decisions

1. **No permute after flash_attn**: Flash output `[D, H, T, 1]` matches the
   gate's host layout directly — d varies fastest, h next, t slowest.
   This is a happy accident of the ggml flash_attn_ext output convention.

2. **f16 causal mask**: The ggml CPU kernel reads the mask tensor as
   `ggml_fp16_t*` (not float). Passing f32 data produces garbage.
   `build_causal_mask_f16_bytes(T)` creates a `T×T` lower-triangular mask
   (0.0 = attend, -inf = block) encoded as IEEE 754 half-precision bytes.
   Written via `DynTensor::write_bytes_backend`.

3. **cont() after permute**: Defensive — some backends may require contiguous
   tensors for flash_attn_ext inputs. The CPU backend works without it,
   but Metal/CUDA may not.

4. **GQA validation**: `h.is_multiple_of(hkv)` checked before graph
   construction. ggml hard-asserts `ne[2] % ne_kv[2] == 0`.

5. **Decode path unchanged**: Single-token decode stays host-side to avoid
   graph construction overhead for T=1. The KV cache + host scoring loop
   is more efficient for individual tokens.

6. **O(T²) mask memory**: Dense causal mask costs 2·T² bytes (f16).
   For T=8k → 128 MiB, T=32k → 2 GiB. Future optimization: cache and
   reuse the mask across layers.

## Layout diagram

```
Host [T, H, D] flat buffer:
  data[t * H * D + h * D + d]
  → d varies fastest = ggml ne[0]=D
  → h varies next    = ggml ne[1]=H
  → t varies slowest = ggml ne[2]=T
  → ggml shape: [D, H, T, 1]

flash_attn_ext wants: [D, T, H, 1]
  → permute(0, 2, 1, 3): swap axes 1 and 2

flash_attn_ext output: [D, H, T, 1]
  → same layout as input gate — element-wise mul works directly
```

## Prerequisites (committed separately)

- `sigmoid` safe wrapper (ggml-rs `src/compute.rs`)
- `flash_attn_ext` safe wrapper with `DynTensor` mask (ggml-rs `src/compute.rs`)
- `DynTensor::write_bytes_backend` for raw byte writes (ggml-rs `src/compute.rs`)
- 5 integration tests (sigmoid CPU/Metal, flash_attn MHA/GQA/shape)

## New code in this commit

- `llama-rs/src/e2e/numeric.rs`:
  - `f32_to_f16_bits(f32) -> u16`: Pure-Rust IEEE 754 f32→f16 conversion
  - `build_causal_mask_f16_bytes(seq_len) -> Vec<u8>`: Causal mask as f16 bytes

- `llama-rs/src/e2e/attention.rs`:
  - `fused_attention_scoring_graph`: ~150 lines, replaces host scoring loop
  - `qwen35_full_attention_core`: rewired to call fused graph
  - Removed unused imports (`head_slice_mut`, `project_sequence_graph`)

## Test results

182 tests pass (down from 196 due to prior reorganization), 0 failures.
Key parity test: `full_attention_prefill_then_decode_matches_full_reprocess` ✅

## Future work

- Cache causal mask across layers (share allocation)
- Move deinterleave + RMS norm + RoPE into the graph (eliminate host↔device round-trip between QKV projection and scoring)
- Graph-level decode_step with flash_attn_ext for longer KV cache sequences
- Size guard for very long contexts (T > ~16k could OOM on mask allocation)
