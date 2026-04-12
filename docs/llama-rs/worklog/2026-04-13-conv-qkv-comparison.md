# Causal Depthwise Conv & QKV Packing: llama-rs vs llama.cpp Comparison

## Summary

This document compares the causal depthwise convolution and QKV split/packing
implementations between `llama-rs` (Rust, scalar CPU) and `llama.cpp`
(C/C++, graph-based with `ggml_ssm_conv` op). Both achieve identical numerical
results (verified via multi-token parity: `[1088, 35790, 90]`).

## 1. Causal Depthwise Convolution

### 1.1 llama-rs: `causal_depthwise_conv` (e2e.rs ~L1657)

Scalar implementation operating on flat `[seq_len × channels]` f32 buffer.

```
Input layout:  [token_0 × channels, token_1 × channels, ...]
Weight layout: [channel_0 × kernel_size, channel_1 × kernel_size, ...]
```

Key characteristics:
- **Explicit zero-padding**: The `if token + 1 < kernel_size - tap` guard
  implicitly zero-pads the causal boundary (skips taps reaching before token 0).
- **SiLU fused**: `silu_scalar(sum)` is applied per channel per token inside
  the conv loop, fusing activation into the convolution.
- **No state/cache**: Processes the full sequence at once. No conv state
  carry-over between calls (acceptable for prompt-only prefill; decode would
  need conv state management).
- **Channel-major weight indexing**: `weight[channel * kernel_size + tap]` —
  matches ggml's `ssm_conv1d.weight` layout `{d_conv, d_inner}`.

Complexity: O(seq_len × channels × kernel_size) — identical to upstream.

### 1.2 llama.cpp: `ggml_ssm_conv` + state management (qwen35.cpp ~L252-283)

Graph-based implementation using the `GGML_OP_SSM_CONV` operator.

```
conv_input = concat(conv_states, transpose(qkv_mixed), dim=0)
             ^ shape: [d_conv - 1 + n_seq_tokens, conv_channels, n_seqs]
conv_output = ggml_ssm_conv(conv_input, conv_kernel)
             ^ shape: [conv_channels, n_seq_tokens, n_seqs]
conv_output_silu = ggml_silu(conv_output)
```

Key characteristics:
- **Explicit conv state**: Prepends cached `conv_states` (last `d_conv-1`
  tokens from previous calls) to the current sequence via `ggml_concat`.
  This enables autoregressive decode without reprocessing.
- **SiLU separate**: Applied as a distinct graph node (`ggml_silu`) after conv.
- **State update**: After conv, extracts the last `d_conv-1` slices from
  `conv_input` and writes them back to the KV cache for the next call.
- **Transposed input**: `qkv_mixed` is transposed before concat to match
  the `[time, channels, batch]` layout expected by `ggml_ssm_conv`.
- **Kernel math** (ops.cpp ~L9252): Inner loop is a rowwise dot product
  `sum += s[i0 + i1*ncs] * c[i0 + i1*nc]` — numerically identical to
  our `input[src_token * channels + channel] * weight[channel * kernel_size + tap]`.

### 1.3 Comparison Matrix

| Aspect | llama-rs | llama.cpp |
|---|---|---|
| Convolution math | Identical (dot product per channel per token) | Identical |
| SiLU activation | Fused in conv loop | Separate graph node |
| Causal padding | Implicit skip via guard | Explicit state prepend |
| State management | Conv buffer + SSM states | Conv cache in KV store |
| Decode support | Prefill + autoregressive (two-phase) | Prefill + autoregressive |
| Parallelism | Single-threaded scalar | Multi-threaded (row parallel) |
| Weight layout | `[channels, kernel_size]` | `[d_conv, d_inner]` (same) |

### 1.4 Gap: ~~Autoregressive Decode~~ CLOSED

The conv state gap is now fully closed:
- `causal_depthwise_conv_decode_step` convolves a single new token using a
  ring buffer of the last `d_conv - 1` pre-conv activations.
- `LinearAttentionState` holds both the conv buffer and SSM recurrence states.
- `qwen35_linear_attention_prefill` captures state after full-sequence processing.
- `qwen35_linear_attention_decode_step` processes one token using cached state.
- `generation.rs` two-phase loop wires prefill → decode for all Qwen3.5 layers.
- Decode equivalence tests verify numerical parity (prefill+decode = full reprocess).

## 2. QKV Packing / Split

### 2.1 Full Attention (Qwen3.5 layers 3,7,11,...,31)

#### llama-rs: Separate Q(+gate), K, V projections

```rust
q_full = project_sequence(input, ..., query_features * 2, q_weight)  // Q + gate interleaved
k_proj = project_sequence(input, ..., kv_features, k_weight)
v_proj = project_sequence(input, ..., kv_features, v_weight)
```

Then split Q/gate from interleaved layout:
```
ggml layout: [Q_h0(D), G_h0(D), Q_h1(D), G_h1(D), ...] per token
```
This requires head-major extraction: `src_q = token_base + head * 2D + dim`,
`src_g = token_base + head * 2D + D + dim`.

#### llama.cpp: Same structure (`build_layer_attn`)

```cpp
Qcur_full = mm(wq, cur)          // (n_embd_head * 2) * n_head output
Qcur = view_3d(Qcur_full, ...)   // extract Q with stride 2D
gate  = view_3d(Qcur_full, ..., offset=D)  // extract gate with stride 2D
Kcur = mm(wk, cur)
Vcur = mm(wv, cur)
```

**Identical structure**: Both use a joint Q+gate projection then split with
interleaved head-major layout. The only difference is llama.cpp uses `ggml_view_3d`
(zero-copy strided view) while llama-rs does explicit per-element copy.

### 2.2 Linear Attention (Qwen3.5 layers 0-2,4-6,8-10,...,28-30)

#### llama-rs: `build_qkvz` equivalent

```rust
qkv = project_sequence(input, ..., conv_channels, qkv_weight)
z   = project_sequence(input, ..., inner_size, gate_weight)
```

Then after conv + SiLU, split conv output into Q, K, V:
```rust
conv[token * conv_channels .. + qk_features]             → Q
conv[token * conv_channels + qk_features .. + 2*qk_features] → K
conv[token * conv_channels + 2*qk_features .. + inner_size]  → V
```

Where `conv_channels = inner_size + 2 * group_count * state_size`.

#### llama.cpp: `build_qkvz` + `ggml_view_4d`

```cpp
qkvz = build_qkvz(cur, il)  // returns (qkv_mixed, z)
// after conv:
q_conv = view_4d(conv_output, head_k_dim, num_k_heads, ...)         // offset 0
k_conv = view_4d(conv_output, head_k_dim, num_k_heads, ..., offset) // offset = Q size
v_conv = view_4d(conv_output, head_v_dim, num_v_heads, ..., offset) // offset = Q+K size
```

**Identical split logic**: Both split the convolved output into three
contiguous regions `[Q | K | V]` where Q and K have `group_count × state_size`
features and V has `inner_size` features. llama.cpp uses strided views;
llama-rs uses explicit `copy_from_slice`.

### 2.3 Comparison Matrix

| Aspect | llama-rs | llama.cpp |
|---|---|---|
| Full attn Q+gate | Joint projection, head-major split | Same (view_3d) |
| Full attn K, V | Separate projections | Same |
| Linear attn QKV | Joint projection → conv → flat split | Same (view_4d) |
| Linear attn Z | Separate gate projection | Same |
| Split mechanism | Explicit `copy_from_slice` | Zero-copy `ggml_view` |
| Memory overhead | Extra allocation for split copies | None (views) |
| Numerical result | Identical (parity verified) | Identical |

## 3. Key Takeaways

1. **Numerical parity is confirmed** across both paths — the math is identical.
   Extended parity testing: single-token `[5328]`, 5-gen from prompt `[3]`
   `[1088,35790,90,16,14728]`, 3-prompt+5-gen `[31,2,5,1,271]`,
   5-prompt+5-gen `[6,24218,10,4838,1665]` — all match.
2. **llama-rs is functionally complete for prefill** (prompt processing).
3. **Autoregressive decode** for linear attention conv state management is now
   implemented (`causal_depthwise_conv_decode_step` + `LinearAttentionState`).
   Decode equivalence tests verify prefill+decode matches full reprocess.
4. **Full attention decode** is also implemented (`qwen35_full_attention_decode_step`
   + `Qwen35FullAttentionState` KV cache). RoPE `position_offset` parameter
   enables correct position encoding for decode tokens.
5. **Performance**: llama-rs uses scalar loops with explicit copies; llama.cpp
   uses graph-level zero-copy views and multi-threaded kernels. This is
   expected for a reference implementation vs production runtime.
6. **Future optimization**: Replace `copy_from_slice` splits with strided
   iterators or views when the ggml-rs safe API supports it.
