# Graph-Level Causal Depthwise Conv via ggml_ssm_conv

## Summary

Replaced the host-only scalar `causal_depthwise_conv` in the prefill path
with `causal_depthwise_conv_graph`, which runs `ggml_ssm_conv` + `ggml_silu`
as a compute graph on the available backend (CPU/Metal/CUDA).

## Changes

### ggml-rs (`src/compute.rs`)

- Added `ssm_conv` safe wrapper — typed `f32`-only, wraps `ggml_ssm_conv` FFI
  call with the same error handling pattern as other compute ops.

### llama-rs (`linear_attention.rs`)

- **`causal_depthwise_conv_graph`**: ~100 lines. Builds a 3-tensor graph:
  1. `sx` — pre-padded input `[kernel_size - 1 + seq_len, channels, 1]`
  2. `c` — conv kernel `[kernel_size, channels]`
  3. Output — `ggml_ssm_conv(sx, c)` → `ggml_silu(result)`

- **Layout mapping**:
  - Our Rust data: `input[token * channels + channel]` (channels-fast = ne[0])
  - `ggml_ssm_conv` wants: `sx[ch * padded_len + pad + token]` (time-fast = ne[0])
  - Solution: host-side transpose + left-padding before upload
  - Output `[channels, seq_len]` → channels-fast → matches our convention ✓

- **Memory estimation**: `ggml_ssm_conv` is a direct loop (no im2col), so
  context size = tensor metadata + graph overhead + 4MB slack.

- **Integration**: Prefill core function (`qwen35_linear_attention_core`) now
  calls `causal_depthwise_conv_graph` instead of `causal_depthwise_conv`.
  Decode path (`causal_depthwise_conv_decode_step`) unchanged — single token.

- **Original function**: Demoted to `#[cfg(test)]` reference implementation.

### Tests (4 new)

| Test | Description |
|------|-------------|
| `conv_graph_matches_host_basic` | seq=4, kernel=4, channels=8 — full parity |
| `conv_graph_matches_host_single_token` | seq=1 edge case |
| `conv_graph_matches_host_seq_less_than_kernel` | seq=2 < kernel=4 |
| `conv_graph_rejects_kernel_size_zero` | Validation: returns error |

## Why ggml_ssm_conv over ggml_conv_1d_dw

- `ggml_ssm_conv` handles pre-padded input directly (no symmetric padding issue)
- `ggml_conv_1d_dw` uses im2col internally (F16 intermediates, more complex)
- `ggml_ssm_conv` has optimized Metal/CUDA/Vulkan kernels
- `ggml_ssm_conv` is exactly what llama.cpp uses for Mamba/SSM conv layers

## Known limitation

Current architecture performs host↔backend round-trip:
projection graph → read to host → transpose+pad → upload to conv graph → read back.
This double-transfer may not win over host-only conv for small sequences.
Proper fix: combine projection + conv into single graph (avoiding readback),
but requires in-graph transpose + pad which is complex with current API.

## Verification

- 196 tests pass (4 new + 192 existing), 1 ignored
- clippy clean (3 pre-existing warnings only)
- Commit: `aba371d` on `exp/oh-my`
