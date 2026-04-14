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
5. **Performance**: llama-rs prefill path now uses `ggml_ssm_conv` via
   `causal_depthwise_conv_graph` — the same graph-level operator used by
   llama.cpp for Mamba/SSM models. Host-side transpose + left-padding maps
   our channel-fast `[seq_len, channels]` layout to ggml's time-fast
   `[padded_len, channels]` convention; the output `[channels, seq_len]` is
   read back channel-fast. Decode path (single token) stays on host scalar
   for latency. QKV splits still use explicit `copy_from_slice` (zero-copy
   views planned). 4 parity tests verify graph vs host-only numerical match.
6. **Future optimization**: The `copy_from_slice` QKV splits now use
   `chunks_exact` iterators, eliminating manual index arithmetic.
   Projection and normalization logic is shared via extracted helpers
   (`project_and_prepare_qkv`, `project_linear_inputs`, `split_and_norm_qk`),
   reducing code duplication between prefill and decode paths. Further
   optimization (strided views) possible when ggml-rs safe API supports it.
   **Update**: `view_3d`, `view_4d`, `reshape_1d`, `reshape_4d` safe wrappers
   are now implemented in ggml-rs with Rust-side validation (contiguity,
   element count, bounds). Existing `view_1d`/`view_2d`/`reshape_2d`/`reshape_3d`
   also gained validation. Graph-level zero-copy QKV splits are now feasible
   as a follow-up optimization.
   **Update 2**: Graph-level projections implemented. Full attention uses a single
   ggml graph with 3 `mul_mat` ops (Q, K, V); linear attention uses 4 (QKV, Z,
   alpha, beta). Output projections also use graph path. Decode (seq_len=1)
   stays host-side. Shared `project_sequence_graph` extracted to `tensor_ops.rs`.
   Parity test confirms host vs graph output matches within 1e-5.
   **Update 3**: Fused projection + conv graph implemented. All 4 projections and
   the causal conv + SiLU are now a single ggml graph (`project_and_conv_fused_graph`),
   eliminating the host↔device round-trip between projection and conv stages.
   See `docs/llama-rs/worklog/2026-04-19-fused-projection-conv.md`.
7. **Multi-layer orchestration verified**: `generation.rs` extracted a
   `GenerationMode` enum (`Auto | FullReprocess | TwoPhase`) and a
   `generate_from_plans` helper.  Integration test
   `two_phase_matches_full_reprocess_multi_layer` runs a 3-layer synthetic
   model (Qwen35Linear → Qwen35Full → Qwen35Linear) through both paths
   and asserts identical token sequences.  This validates the residual
   connections, MLP pass-through, and state orchestration across layers.
8. **Full attention scoring graph-fused**: The host-side O(T²·H·D) scoring
   loop (softmax + dot product + sigmoid gating + output projection) in
   `qwen35_full_attention_core` is now a single ggml graph using
   `flash_attn_ext`:
   ```
   permute(Q/K/V, 0,2,1,3) → cont → flash_attn_ext(mask_f16, scale)
   → sigmoid(gate) → mul → reshape_2d → mul_mat(W_out)
   ```
   Key insight: flash output `[D, H, T, 1]` matches gate layout directly
   — no extra permute for gating. Decode path (seq_len=1) stays host-side.
   `f32_to_f16_bits` + `build_causal_mask_f16_bytes` added to `numeric.rs`.
   See `docs/llama-rs/worklog/2026-04-20-fused-attention-scoring.md`.
9. **Fully fused single-graph full attention**: Merged the two-graph pipeline
   (Graph 1: QKV projection → host round-trip → Graph 2: scoring) into one
   ggml graph that performs everything:
   ```
   mul_mat(W_q/W_k/W_v, X)
   → view_3d strided deinterleave Q/gate
   → rms_norm + weight broadcast
   → rope_ext (NeoX mode=2)
   → permute → cont → flash_attn_ext → sigmoid → mul
   → reshape_2d → mul_mat(W_out)
   ```
   Eliminates 10 host↔device transfers and 2 graph round-trips. Post-RoPE K
   and raw V conditionally read back for KV cache capture. Decode path
   unchanged. 202 tests pass.
   See `docs/llama-rs/worklog/2026-04-20-fully-fused-attention.md`.

10. **Layer pre-norm fusion** (attention + MLP): Moved `rms_norm(X, eps) * weight`
    from host-side `process_all_layers` into each ggml compute graph as the first
    operation. Full attention, linear attention, and MLP graphs all accept un-normed
    hidden state + norm weight, eliminating 2× host↔device round-trips per layer.
    Decode path keeps host-side norm (single-token overhead not worthwhile).
    Standard attention also keeps host-side norm. ggml f32 rms_norm vs host f64
    accumulation verified within 1e-5 tolerance. 201 tests pass.

11. **Q/K L2 norm fusion: skipped** (not worthwhile).
    The conv output must be read back to host anyway for the SSM delta-net
    recurrence (which cannot run in a ggml graph). Q/K L2 norm operates on
    this already-transferred data with small vectors (`state_size` per head).
    ggml has no native L2 norm op — only `rms_norm` which differs in
    normalization factor (`1/sqrt(mean)` vs `1/max(sqrt(sum), eps)`).
    Fusing would add graph complexity and separate readback tensors for minimal
    or negative throughput benefit.

12. **CPU vs Metal benchmark results** (Qwen3.5 0.6B dimensions, release mode,
    warmup=3, iters=20, Apple Silicon):

    | Graph                  | SeqLen | CPU (ms) | Metal (ms) | Speedup |
    |------------------------|-------:|---------:|-----------:|--------:|
    | full_attention_fused   |      1 |    1.865 |      2.654 |   0.70× |
    | full_attention_fused   |      4 |    2.005 |      2.487 |   0.81× |
    | full_attention_fused   |     16 |    2.894 |      2.603 |   1.11× |
    | full_attention_fused   |     64 |    7.005 |      2.865 |   2.45× |
    | linear_attention_fused |      1 |    2.193 |      3.019 |   0.73× |
    | linear_attention_fused |      4 |    2.501 |      2.991 |   0.84× |
    | linear_attention_fused |     16 |    4.026 |      3.426 |   1.18× |
    | linear_attention_fused |     64 |   10.162 |      5.927 |   1.71× |
    | mlp_fused              |      1 |   13.004 |     15.059 |   0.86× |
    | mlp_fused              |      4 |   14.172 |     14.801 |   0.96× |
    | mlp_fused              |     16 |   20.067 |     14.812 |   1.35× |
    | mlp_fused              |     64 |   45.618 |     14.910 |   3.06× |

    Key observations:
    - **Metal overhead dominates for short sequences** (seq_len ≤ 4): CPU is
      faster for single-token decode due to Metal command buffer dispatch
      latency. This validates the decision to keep decode path host-side.
    - **Metal wins at seq_len ≥ 16**: Cross-over occurs around 8–16 tokens.
      At 64 tokens, Metal is 2.4–3.1× faster.
    - **MLP is the bottleneck**: 3 large matmuls (1536×8960) dominate. At
      seq_len=64, MLP takes 45.6ms CPU vs attention's 7.0ms.
    - **Linear attention Metal gain is limited** by host-side SSM recurrence
      (delta-net sequential loop), which doesn't benefit from GPU.
    - Benchmark module: `llama-rs/src/e2e/bench_graphs.rs` (run with
      `--ignored --nocapture`).

## 13. LM Head (Output Projection) Graph Optimization

The LM head is the final projection: rms_norm(last_hidden, eps) × norm_weight →
matmul(output_weight) → logits [vocab_size]. For Qwen3.5 (0.6B), this is a GEMV:
1536 × 151936 ≈ 233M multiply-adds, with ~935MB weight matrix.

### 13.1 Previous: Host-side Naive Loop

`greedy_next_token_id` (generation.rs) performed a scalar host-side matmul:
151,936 dot products of length 1536 in a tight loop. Additionally,
`sample_next_token` wastefully rms-normalized ALL tokens (seq_len × hidden)
when only the last token's hidden state is needed for sampling.

### 13.2 Current: Graph-level Persistent LM Head

The generation loops (`two_phase_loop`, `full_reprocess_loop`) now use a
persistent ggml graph for the LM head:

1. **One-time setup**: Build a ggml context + graph (rms_norm → mul → reshape →
   mul_mat), allocate backend buffer, upload weights (~935MB) once.
2. **Per step**: Upload only the last token's hidden state (~6KB), recompute
   the graph, read back logits (~608KB). Total per-step transfer ≈ 614KB
   vs 935MB for a cold upload.

Key design decisions:
- **No self-referential struct**: Instead of a `PersistentLmHead<'static>` with
  unsafe transmute, the ggml context + tensors live as function-scoped variables
  in the generation loop. This avoids all unsafe code.
- **`build_lm_head_graph`**: Reusable builder (tensor_ops.rs) returns
  `(w_out, norm_w, x_in, logits, graph)` — caller owns the context.
- **`lm_head_sample_step`**: One-liner per decode step (write input → compute →
  read logits → argmax).
- **`graph_sample_at`**: Convenience wrapper that extracts the last token's
  hidden state from a multi-token buffer before calling `lm_head_sample_step`.
- **`sample_next_token` removed**: Was unused after both loops switched to graph.
  `greedy_next_token_id` kept as fallback for tests and `session.rs`.

### 13.3 Files Changed

| File | Changes |
|------|---------|
| `tensor_ops.rs` | Added `recommended_lm_head_memory`, `lm_head_graph` (one-shot), `build_lm_head_graph` (persistent builder), `lm_head_sample_step`, `argmax_token_id` |
| `generation.rs` | Both loops use persistent LM head; added `graph_sample_at`; removed `sample_next_token` |
| `bench_graphs.rs` | Added `bench_lm_head_qwen35`: host vs cold graph vs warm graph |

### 13.4 Parity Tests

- `lm_head_graph_matches_host_sampling`: Verifies one-shot graph argmax matches
  host-side rms_norm + greedy_next_token_id.
- `lm_head_sample_step_matches_one_shot`: Verifies persistent graph step matches
  one-shot graph result.
- `argmax_picks_largest` / `argmax_empty_returns_error`: Unit tests for argmax.

## 14. Decode-Path QKV Backend Offload

### 14.1 Problem

Both `qwen35_full_attention_decode_step` and `qwen35_linear_attention_decode_step`
passed `None` as the backend argument to their projection helpers, forcing host-side
scalar dot-product matmuls for every decode step:

- **Full attention**: 3 sequential matmuls (Q, K, V projections) + 1 output projection
- **Linear attention**: 4 sequential matmuls (QKV, gate, alpha, beta) + 1 output projection

For Qwen3.5 0.6B (hidden=1536), each host matmul at seq_len=1 involves
1536×N multiplications on a single core — the biggest serial bottleneck in the
decode loop.

### 14.2 Solution

Thread `&Backend` into both decode_step functions and pass `Some(backend)` to
the existing projection helpers, which already have graph-path support:

- `project_and_prepare_qkv(…, Some(backend))` — batches 3 matmuls in a single ggml graph
- `project_linear_inputs(…, Some(backend))` — batches 4 matmuls in a single ggml graph
- Output projections (`project_sequence` → `project_sequence_graph`) — single matmul graph

The `DecodeStrategy::process_attention` in `generation.rs` already received
`backend` but discarded it (`_backend`). Now it forwards the reference to both
decode_step callsites.

### 14.3 Per-Step Cost Delta (Qwen3.5 0.6B, seq_len=1)

| Matmul | Dimensions | Before | After |
|--------|-----------|--------|-------|
| Q proj | 1536 → 3072 | Host scalar | ggml graph (CPU/Metal) |
| K proj | 1536 → 256 | Host scalar | ggml graph |
| V proj | 1536 → 256 | Host scalar | ggml graph |
| Output proj | 768 → 1536 | Host scalar | ggml graph |
| Linear QKV | 1536 → conv_ch | Host scalar | ggml graph |
| Linear gate/alpha/beta | 1536 → various | Host scalar | ggml graph |
| Linear output | inner → 1536 | Host scalar | ggml graph |

### 14.4 Files Changed

| File | Changes |
|------|---------|
| `attention.rs` | `qwen35_full_attention_decode_step` gains `backend: &Backend`; QKV + output use graph path |
| `linear_attention.rs` | `qwen35_linear_attention_decode_step` gains `backend: &Backend`; projections + output use graph path |
| `generation.rs` | `DecodeStrategy` passes `backend` (was `_backend`) to both decode_step calls |

### 14.5 Test Updates

- `decode_step_matches_full_reprocess` (attention.rs): passes `&backend` to updated signature
- `linear_decode_step_matches_full_reprocess` (linear_attention.rs): passes `&backend` to updated signature
- All 205 tests pass, 0 new warnings


## 15. Persistent Decode Projections — Eliminate Per-Token Weight Upload

### 15.1 Problem

After offloading decode-path projections to the Metal backend (item 14), each
decode step still creates a fresh ggml `Context`, builds tensors, allocates
backend memory, uploads ALL layer weights (~756 MB for 28 layers), computes,
reads results, and tears down. The per-token overhead is dominated by weight
upload, not computation.

### 15.2 Solution

**Persistent projection graphs**: build one ggml `Context` + graph per layer at
the start of the decode phase, upload weights once, then reuse the graph for
every decode step — only transferring the ~6 KB input/output hidden vectors per
token.

Architecture:
```
PersistentDecodeProjection<'ctx>
├── FullAttention { x_in, q/k/v_out, input_graph, out_x, out_y, output_graph, _buffer }
└── LinearAttention { x_in, qkv/z/alpha/beta_out, input_graph, out_x, out_y, output_graph, _buffer }
```

Each variant holds:
- **Input graph**: `hidden_features → projection outputs` (3 matmuls for full, 4 for linear)
- **Output graph**: `core_result → hidden_features` (1 matmul)
- **BackendBuffer**: keeps allocated Metal memory alive

### 15.3 Phase 1 — Core Extraction (Pure Refactoring)

Extracted reusable decode core logic to enable composition with persistent
projections:

| Function | File | Purpose |
|----------|------|---------|
| `full_attention_hidden_features()` | attention.rs | Derive hidden_features from output weight dims |
| `prepare_qkv_from_raw()` | attention.rs | Post-process raw Q/K/V: deinterleave + per-head RMS norm |
| `full_attention_decode_core()` | attention.rs | RoPE → KV cache → scoring → gating (before output proj) |
| `linear_attention_hidden_features()` | linear_attention.rs | Derive hidden_features from SSM output weight dims |
| `linear_attention_conv_channels()` | linear_attention.rs | Compute conv_channels from plan dimensions |
| `linear_attention_decode_core()` | linear_attention.rs | Conv → split/norm → SSM recurrence → z-gating |

Existing `qwen35_full_attention_decode_step` and
`qwen35_linear_attention_decode_step` refactored to delegate to these cores.
All existing tests pass unchanged.

### 15.4 Phase 2 — Persistent Projection Infrastructure (tensor_ops.rs)

| Component | Purpose |
|-----------|---------|
| `PersistentDecodeProjection` enum | FullAttention / LinearAttention variants holding tensor handles |
| `recommended_persistent_full_attention_memory()` | Context size estimation for full attention |
| `recommended_persistent_linear_attention_memory()` | Context size estimation for linear attention |
| `build_persistent_full_attention_graphs()` | Build input + output matmul graphs in one context |
| `build_persistent_linear_attention_graphs()` | Build 4-input + 1-output matmul graphs in one context |
| `project_input()` method | Write hidden → compute input graph |
| `read_*_projections()` methods | Read raw projection outputs from GPU |
| `project_output()` method | Write core result → compute output graph → read hidden |

### 15.5 Phase 3 — Integration into two_phase_loop (generation.rs)

The decode loop in `two_phase_loop` now:

1. Attempts `try_build_persistent_projections()` for all layers
2. On **success**: uses `persistent_decode_all_layers()` — no per-token weight upload
3. On **failure**: falls back to original `DecodeStrategy` via `process_all_layers()`

Runtime fallback ensures robustness: if any context creation, allocation, or
weight upload fails (e.g., insufficient GPU memory), the loop seamlessly
degrades to the per-token path.

### 15.6 Drop Safety

Uses `transmute` to erase unnameable lifetime from `PersistentDecodeProjection`.
Soundness maintained because:
- `Tensor`/`Graph`/`BackendBuffer` hold `PhantomData<&'ctx Context>`, not real references
- Owning `Vec<Option<Context>>` lives as sibling local in same scope
- Projections drop BEFORE contexts due to declaration order

### 15.7 Performance Impact (Estimated)

| Metric | Before (item 14) | After (item 15) |
|--------|-------------------|------------------|
| Per-token weight upload | ~756 MB (all layers) | ~6 KB (hidden vectors only) |
| Context creation/teardown | 28 × 2 per token | 0 per token (built once) |
| Decode latency | Dominated by weight transfer | Dominated by computation |

### 15.8 Files Changed

| File | Changes |
|------|---------|
| `attention.rs` | `PreparedAttention` pub(super); add `full_attention_hidden_features`, `prepare_qkv_from_raw`, `full_attention_decode_core` with head-count divisibility check |
| `linear_attention.rs` | `LinearProjections` pub(super); add `linear_attention_hidden_features`, `linear_attention_conv_channels`, `linear_attention_decode_core` with group/state/rank validation |
| `tensor_ops.rs` | Add `PersistentDecodeProjection` enum, memory estimators, graph builders, step methods |
| `generation.rs` | Add `try_build_persistent_projections`, `build_one_persistent_full/linear`, `persistent_decode_all_layers`; modify `two_phase_loop` decode phase with fallback |

### 15.9 Test Results

- All existing tests pass (behavior-preserving refactoring + additive feature)
- Fallback path exercised when persistent build is not available

---

## 16. Decode Attention Scoring Offload to GPU (`flash_attn_ext`)

### 16.1 Problem

After persistent decode projections (item 15), the **attention scoring loop**
became the dominant decode bottleneck. For each token at position T, the host
performs O(T × H × D) FLOPs for Q·K dot products, softmax, V aggregation, and
sigmoid gating — ~8.2M FLOPs at T=1000 (34% of decode CPU time).

The host path is a triple-nested scalar loop:
```
for head in 0..H:
  for source in 0..T:       score = dot(Q[head], K_cache[source, kv_head])
  softmax(scores)
  for source in 0..T:       V_agg += V_cache[source, kv_head] * weight[source]
  gate: V_agg *= sigmoid(Q_gate[head])
```

### 16.2 Solution

Offload the entire scoring + gating operation to the backend using
`flash_attn_ext`, which fuses Q·K scoring, softmax, and V aggregation in a
single GPU-optimized kernel. Sigmoid gating is included in the same graph
to avoid an extra host ↔ device readback.

### 16.3 Implementation

**New function** `decode_scoring_gpu` in `attention.rs`:
1. Creates a temporary ggml context per decode step (KV length changes each step)
2. Uploads Q `[D, 1, H, 1]`, gate `[D, H, 1, 1]`, and **live KV cache prefix**
3. Permutes K/V from host cache layout `[D, Hkv, T]` → flash layout `[D, T, Hkv]`
4. Runs `flash_attn_ext` (no mask needed for single-query decode)
5. Applies `sigmoid(gate) ⊙ attn` in the same graph
6. Reads back gated head outputs `[D × H]`

**Modified function** `full_attention_decode_core`:
- New parameter `backend: Option<&Backend>`
- When `Some`: tries GPU scoring first; on any failure, falls back to host loop
- When `None`: host scoring loop (unchanged behavior)

### 16.4 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Per-step temporary context | KV cache length changes each step; can't reuse a persistent graph |
| Upload only live prefix | `k_cache[..total_tokens * kv_features]`, not full buffer (rubber-duck catch) |
| Q shape `[D, 1, H, 1]` | Matches `flash_attn_ext` convention `[D, T_q, H, 1]` (rubber-duck caught initial `[D, H, 1, 1]` bug) |
| No causal mask for T_q=1 | Single-query decode: token attends to self + all past, no future tokens exist |
| Fallback on GPU failure | `full_attention_decode_core` silently falls back to host loop if graph build/compute fails |
| GQA native | `flash_attn_ext` handles H ≠ Hkv natively; no head expansion needed |

### 16.5 KV Cache Layout Transformation

```
Host cache:  k_cache[token * kv_features + kv_head * hd + dim]
             → ggml shape [D, Hkv, T, 1]

flash_attn:  K [D, Tkv, Hkv, 1]

Transform:   permute(0, 2, 1, 3) + cont()
             [D, Hkv, T, 1] → [D, T, Hkv, 1]
```

### 16.6 Performance Impact (Estimated)

| Metric | Before (host) | After (GPU) | Notes |
|--------|---------------|-------------|-------|
| Scoring FLOPs at T=1000 | ~8.2M (scalar) | ~8.2M (SIMD/GPU) | Same work, massively parallel |
| KV upload per step | 0 | ~64KB at T=1000 | Cheap on unified memory |
| Context allocation | 0 | ~2MB metadata | One per step, freed immediately |
| Gating | Separate loop | Fused in graph | One fewer host ↔ device transfer |

### 16.7 Files Changed

| File | Changes |
|------|---------|
| `attention.rs` | Add `decode_scoring_gpu` function; modify `full_attention_decode_core` to accept `backend: Option<&Backend>` with GPU-first + host fallback; update `qwen35_full_attention_decode_step` caller |
| `generation.rs` | Pass `Some(backend)` to `full_attention_decode_core` in `persistent_decode_all_layers` |

### 16.8 Test Results

- New test: `gpu_scoring_matches_host_scoring` — GQA (H=4, Hkv=2), 5+1 tokens,
  verifies GPU path output matches host scoring loop within 1e-4
- All existing tests pass (including `full_attention_prefill_then_decode_matches_full_reprocess`)

## 17. Causal Depthwise Conv Decode Path: Host vs GPU Analysis

### 17.1 Problem

After offloading decode projections (item 15) and attention scoring (item 16),
the **causal depthwise conv decode step** remains host-side. Each decode step
runs a scalar loop over `conv_channels × kernel_size` taps, applies SiLU, and
updates the conv buffer.

Current implementation (`causal_depthwise_conv_decode_step` in `linear_attention.rs`):

```
for channel in 0..conv_channels:
    sum = 0
    for tap in 0..kernel_size:
        value = conv_buffer[lookback] or new_row[channel]
        sum += value * weight[channel * kernel_size + tap]
    output[channel] = silu(sum)
push new_row into conv_buffer
```

### 17.2 Complexity Analysis

For Qwen3.5 0.6B:
- `conv_channels = inner_size + 2 × group_count × state_size`
  (typically ~700 for 0.6B model)
- `kernel_size = 4` (from GGUF metadata `ssm_conv_kernel`)
- FLOPs per decode step: `conv_channels × kernel_size × 2` ≈ **5,600 FLOPs**
- This is 0.07% of the attention scoring FLOPs (~8.2M at T=1000)
- Wall-clock: sub-microsecond on modern CPUs

### 17.3 GPU Offload Feasibility

**Available kernel**: `ggml_ssm_conv` (already used in prefill, item 6):
- Input `sx`: `[padded_len, channels, 1]` — time-fast layout with left-padding
- Weight `c`: `[kernel_size, channels]`
- Output: `[channels, seq_len, 1]`

**For decode (seq_len=1)**:
- `padded_len = kernel_size` (4 tokens: 3 from buffer + 1 new)
- Total data to upload: `4 × conv_channels × 4 bytes` ≈ 11.2 KB
- But requires host-side transpose: `conv_buffer[row * channels + ch]` →
  `sx_data[ch * padded_len + pad + token]` (channel-fast → time-fast)
- Plus graph build + allocate + compute + readback overhead

**Benchmark evidence** (item 12, `bench_graphs.rs`):
- `linear_attention_fused` at seq_len=1: CPU = 2.193 ms, Metal = 3.019 ms
- CPU is **0.73× faster** than Metal for single-token linear attention
- Note: this benchmarks the full fused graph, not a conv-only micro-benchmark.
  However, the conv-only work (~5.6K FLOPs) is orders of magnitude smaller than
  the full graph, so the ~0.8 ms Metal dispatch overhead alone is likely to
  dominate a standalone `ssm_conv(seq_len=d_conv)` graph.

**Verdict**: GPU offload is very likely a **net negative** for decode conv. The
computation is too small (5.6K FLOPs) to amortize dispatch overhead (~0.8 ms on
Metal). Host scalar loop completes in under 1 µs. A dedicated conv-step
microbenchmark could confirm this, but the cost mismatch is ~3 orders of
magnitude.

### 17.4 Fused Projection + Conv Graph

Could we fuse conv into the persistent projection graph (item 15)?

| Consideration | Assessment |
|---------------|------------|
| Conv buffer changes every token | Must upload `(kernel_size-1) × channels` bytes per step |
| Transpose required | Host-side channel-fast → time-fast layout conversion |
| Raw QKV needed before conv | Conv buffer must capture pre-conv QKV for next step's lookback |
| Graph complexity | Adds ssm_conv + silu nodes + extra tensor IO |
| Benefit | Eliminates one host function call — negligible savings |

**Verdict**: Not worth the complexity. The conv buffer state management requires
per-step data upload regardless, and the pre-conv QKV readback (for buffer update)
creates a mandatory host round-trip that fusion cannot eliminate.

### 17.5 Comparison with llama.cpp

| Aspect | llama-rs (decode) | llama.cpp (decode) |
|--------|-------------------|-------------------|
| Conv execution | Host scalar loop | In compute graph (`ggml_ssm_conv`) |
| Conv state | Host-side `conv_buffer` array | Backend-resident `conv_states` tensor |
| State update | `push_conv_row` after conv | Graph builds `concat(conv_states, new_qkv)` |
| Why different? | Host SSM recurrence after conv → must read back | Full-graph architecture → stays on backend |

llama.cpp can keep conv on GPU because their **entire layer** (projection + conv +
SSM + output) is a single compute graph. Our architecture has host-side SSM
recurrence after conv (Delta-Net, see item 19), so fusing just conv doesn't
eliminate the host ↔ device boundary.

### 17.6 Conclusion

The decode conv step is the **smallest** remaining host operation (~5.6K FLOPs).
It stays on host because:
1. GPU dispatch overhead (0.8+ ms) vastly exceeds computation cost (<1 µs)
2. Conv buffer state requires per-step upload regardless
3. Host-side SSM recurrence after conv creates a mandatory readback boundary
4. No measurable impact on end-to-end decode latency

## 18. QKV Packing/Routing in Decode Path

### 18.1 Current Decode QKV Flow

After persistent projections (item 15), the decode QKV flow for linear attention is:

```
GPU: persistent_projection_graph(hidden_input)
     → QKV [conv_channels], Z [inner_size], alpha [time_step_rank], beta [time_step_rank]
       ↓ (readback to host)
Host: causal_depthwise_conv_decode_step(QKV, conv_buffer, conv_weight)
     → conv_output [conv_channels]  (with SiLU)
       ↓
Host: split conv_output into Q_heads, K_heads, V_region
      Q = conv_output[0..qk_features]          → per-group Q vectors
      K = conv_output[qk_features..2*qk_features]  → per-group K vectors
      V = conv_output[2*qk_features..conv_channels] → per-head V vectors
       ↓
Host: RMS norm on Q and K heads (per-group normalization)
       ↓
Host: SSM recurrence per head (Delta-Net, item 19)
       ↓
Host: Z-gating: output[head] = rms_norm(ssm_out) * silu(z[head])
       ↓
GPU: persistent_output_projection(gated_output)
     → layer_output [hidden_features]
```

### 18.2 QKV Split Logic

The post-conv split is identical between prefill and decode:

```
conv_channels = inner_size + 2 × group_count × state_size
qk_features = group_count × state_size

Q_heads: conv_output[0 .. qk_features]
K_heads: conv_output[qk_features .. 2 × qk_features]
V_heads: conv_output[2 × qk_features .. conv_channels]
```

This is **logically contiguous** — just `[Q|K|V]` regions within the conv output
buffer. The V region is borrowed directly (`&conv[qk_features * 2..conv_channels]`),
but Q and K are copied into fresh buffers by `split_and_norm_qk` before applying
per-group RMS normalization. These copies are small (group_count × state_size
elements each) and are not a meaningful bottleneck.

### 18.3 Comparison with llama.cpp QKV Routing

| Aspect | llama-rs (decode) | llama.cpp (decode) |
|--------|-------------------|-------------------|
| QKV split | Host slice indexing (zero-cost) | Graph `view` ops (zero-copy) |
| Post-split norm | Host `rms_norm_single` per group | Graph `rms_norm` + weight broadcast |
| Head routing | `head % group_count` modular mapping | `ggml_repeat_4d` block tiling |
| Group→head expansion | Implicit via index mapping in SSM loop | Explicit tensor repeat in graph |

**Key difference**: llama.cpp's graph-based approach processes all heads in
parallel (SIMD/GPU threads), while our host-side loop is sequential per head.
However, the head count for linear attention is small (`time_step_rank` heads),
so parallelism has limited benefit.

### 18.4 Decode vs Prefill QKV Packing Differences

| Dimension | Prefill | Decode |
|-----------|---------|--------|
| Input to conv | `[seq_len × conv_channels]` | `[1 × conv_channels]` (single row) |
| Conv state | Not needed (full sequence) | Buffer of last `kernel_size-1` rows |
| QKV readback | Entire sequence | Single token's projections |
| Projection path | Fused in single graph (projection + conv + silu) | Persistent projection graph (no conv) |

### 18.5 Optimization Opportunity: Head-Parallel SSM

The current decode path processes heads sequentially:
```rust
for head in 0..time_step_rank {
    ssm_recurrence_step(state[head], q[head], k[head], v[head], ...);
}
```

Each head's recurrence is independent (separate state matrix). A `rayon` parallel
iterator or SIMD batching could process multiple heads simultaneously. However:
- Per-head state is `state_size × state_size` (small matrix, ~128² = 16K floats)
- Memory access pattern is already cache-friendly (contiguous per-head blocks)
- Parallelism overhead may exceed computation for small head counts
- Deferred to future optimization (see item 19 for SSM analysis)

### 18.6 Conclusion

QKV packing/routing in the decode path is **already optimal** — the split is
zero-cost slice indexing, and projections are GPU-offloaded via persistent graphs.
The remaining host work (conv + SSM recurrence) is sequential by nature and too
small for GPU dispatch overhead. The architecture difference from llama.cpp
(host-side SSM vs full-graph) is fundamental and stems from the Delta-Net
recurrence incompatibility with `ggml_ssm_scan` (item 19).

## 19. SSM Recurrence: Delta-Net vs ggml_ssm_scan Compatibility

### 19.1 Problem

The SSM recurrence (`ssm_recurrence_step`) is the **largest** remaining host-side
decode operation (~460K FLOPs at `time_step_rank` heads × `state_size²` per head).
The ggml library provides `ggml_ssm_scan` for GPU-accelerated SSM recurrence.
Can we use it for Qwen3.5?

### 19.2 ggml_ssm_scan Semantics (Linear Selective Scan)

`ggml_ssm_scan` implements a **linear selective scan** supporting Mamba-family
variants (both scalar-decay Mamba-2 and element-wise decay Mamba-1). The key
property is that the update term is **state-independent**:

```
// Input shapes:
s:  [d_state, dim, n_head, n_seqs]    // state matrix per head
x:  [dim, n_head, n_seq_tokens, n_seqs]  // input
dt: [n_head, n_seq_tokens, n_seqs]       // time step
A:  [1, n_head] or [d_state, n_head]     // decay factor
B:  [d_state, n_group, n_seq_tokens, n_seqs]  // input projection
C:  [d_state, n_group, n_seq_tokens, n_seqs]  // output projection

// Per-head recurrence (scalar-decay variant):
dA = exp(softplus(dt[h]) * A[h])           // scalar decay
for each dim i, state j:
    s[j,i,h] = s[j,i,h] * dA + B[j,g] * x[i,h] * softplus(dt[h])
y[i,h] = sum_j(s[j,i,h] * C[j,g])        // linear readout
```

The update term `B[j] * x[i] * dt` depends only on the **input**, not on the
current state. This is the defining property of a linear selective scan.

### 19.3 Qwen3.5 Delta-Net Recurrence

```
// Per-head recurrence (ssm_recurrence_step):
decay = exp(softplus(alpha + dt_bias) * ssm_a)   // scalar decay
beta_value = sigmoid(beta)                         // scalar scale

// Step 1: Decay + compute sk (state-dependent!)
for row, col:
    state[row][col] *= decay
    sk[col] += state[row][col] * k[row]            // sk = K^T · (decayed state)

// Step 2: Delta rule (uses sk → state feedback)
delta[col] = (v[col] - sk[col]) * beta_value       // correction term

// Step 3: Rank-1 update
state[row][col] += k[row] * delta[col]             // s += k ⊗ delta

// Step 4: Linear readout
out[col] = sum_row(state[row][col] * q[row] * scale)  // out = Q^T · s
```

### 19.4 Key Incompatibility

| Property | Linear Selective Scan (`ssm_scan`) | Delta-Net (Qwen3.5) |
|----------|-------------------------------------|---------------------|
| Update rule | `s += B ⊗ (x * dt)` | `s += k ⊗ ((v - k^T·s) * β)` |
| State dependency | Update is **independent** of current state | Update **depends** on current state (via `sk = k^T·s`) |
| Feedback term | None | `sk` feeds back into delta computation |
| Input vectors | Separate B (input) and C (output) projections | Q (read), K (write/feedback), V (target) |
| Activation on dt | `softplus(dt)` applied inside kernel | `softplus(α + dt_bias) * A` applied before, then `exp()` |

The **delta rule feedback** `sk = k^T · (decayed state)` is the fundamental
blocker. `ggml_ssm_scan` computes the state update as `s += B * x_dt` where
`x_dt` depends only on the input, not on the state. Delta-Net's update term
`delta = (v - sk) * β` requires the intermediate value `sk` which depends on
the state AFTER decay.

### 19.5 Alternative: Express as Separate ggml Ops

Could we decompose the Delta-Net recurrence into individual ggml operations?

```
1. scale(state, decay)           → decayed state
2. mul_mat(k^T, state)           → sk vector [state_size]
3. sub(v, sk)                    → (v - sk) [state_size]
4. scale((v-sk), beta)           → delta [state_size]
5. outer_product(k, delta)       → rank-1 matrix [state_size × state_size]
6. add(state, outer)             → updated state
7. mul_mat(q^T * scale, state)   → output [state_size]
```

**Problem**: This requires 7+ graph nodes per head per decode step, with the
state tensor being both read and written within the same graph. ggml graphs
are DAGs (directed acyclic) — in-place state mutation within a graph is not
straightforward. The state would need to flow through the graph as an
intermediate, not as a mutated tensor.

**Additional concerns**:
- `state_size × state_size` matrix is small (e.g., 128 × 128 = 64 KB)
- 7 graph nodes + dispatch overhead likely exceeds host scalar loop cost
- Per-head graphs can't be batched across heads (different state matrices)
- Would need ~7 × time_step_rank graph nodes per decode step

**Verdict**: Not viable. The overhead of building, allocating, and computing
a multi-op graph for a ~460K FLOP computation on small matrices is prohibitive.

### 19.6 Future Opportunities

| Approach | Feasibility | Notes |
|----------|-------------|-------|
| Custom ggml op (`GGML_OP_DELTANET_SCAN`) | Medium | Would need C implementation in ggml, Metal/CUDA kernels. Upstream contribution. |
| Batched BLAS (all heads at once) | Low | Heads have data dependencies on shared QK groups; state_size too small for BLAS efficiency |
| SIMD intrinsics (host-side) | Medium | Hand-vectorize the inner loops with NEON/AVX; avoids graph overhead. Most promising near-term. |
| Rayon parallel heads | Low | Per-head work too small; thread spawn overhead > computation |

### 19.7 Conclusion

`ggml_ssm_scan` **cannot** be used for Qwen3.5's Delta-Net recurrence due to
the state-dependent delta rule feedback term. The linear selective scan assumes
state-independent updates (`s = s * decay + input_term`), while Delta-Net's
correction `delta = (v - k^T·s) * β` creates a read-modify-write dependency
within each recurrence step where the update term itself depends on the decayed
state.

The SSM recurrence stays on host for now. The most promising optimization path
is **SIMD vectorization** of the inner loops (decay + sk accumulation, delta
computation, rank-1 update, readout), which avoids graph dispatch overhead while
exploiting data-level parallelism within each head's `state_size × state_size`
matrix operations.

## 20. RoPE Decode: Host Analysis

### 20.1 Current Implementation

`apply_neox_rope_in_place` (`attention.rs`) applies NeoX-style rotary position
embedding to Q and K vectors. For decode (seq_len=1):

```
// Per-head, half_rot rotations:
for k in 0..half_rot:
    angle = (position * freq_scale) * freq_base^(-2k/n_rot)
    values[k]            = values[k] * cos(angle) - values[k + half_rot] * sin(angle)
    values[k + half_rot] = values[k] * sin(angle) + values[k + half_rot] * cos(angle)
```

### 20.2 Complexity (Qwen3.5 0.6B, seq_len=1)

| Component | Heads | half_rot | FLOPs per rotation | Total FLOPs |
|-----------|-------|----------|-------------------|-------------|
| Q RoPE | 12 | 64 | 6 (4 mul + 1 sub + 1 add) | 4,608 |
| K RoPE | 4 | 64 | 6 | 1,536 |
| cos/sin cache | 1 | 64 | ~2 (per k) | 128 |
| **Total** | | | | **~6,272** |

### 20.3 GPU Offload Feasibility

**Available op**: `rope_ext_with_i32_positions` (already used in prefill graph):
- For decode: graph with single-position tensor, cos/sin computed on GPU
- But total work (~6K FLOPs) is even smaller than conv decode (~5.6K FLOPs)
- Metal dispatch overhead (~0.8 ms) exceeds computation cost by ~4 orders of magnitude

**Integration with persistent projection graph**:
- Could add RoPE as graph nodes after Q/K projection in the persistent graph
- Position would need per-step `write_data_backend` update (1 i32 value)
- Problem: post-RoPE Q/K values need readback for KV cache append, which already
  happens on host. Fusing RoPE into the graph saves the host computation but adds
  graph complexity for negligible savings.

### 20.4 Verdict

RoPE decode stays on host. At ~6K FLOPs, it is the **smallest** per-layer
operation — approximately 0.001% of generation time at T=1000. Further
optimization would be a misallocation of engineering effort.

## 21. Probe-Once GPU Failure Optimization

### 21.1 Problem

`full_attention_decode_core` tries GPU scoring via `decode_scoring_gpu` every
decode step, falling back to the host loop on failure. When the GPU path
consistently fails (e.g., CPU-only backend, unsupported op), this incurs:
- Context allocation attempt + error handling per step
- ~µs overhead per step × hundreds/thousands of steps

### 21.2 Solution

Track GPU scoring availability in `Qwen35FullAttentionState`. After the first
failure, disable GPU attempts for the remainder of the generation. On success,
keep using GPU.

```
state.gpu_scoring_available: Option<bool>
  None  → not probed yet, try GPU
  Some(true)  → GPU works, keep using it
  Some(false) → GPU failed, skip directly to host
```

### 21.3 Implementation

**Modified**: `Qwen35FullAttentionState` gains `gpu_scoring_failed: bool` field
(default `false`). After first GPU attempt:
- Success → leave probed as false (keep trying)
- Failure → set probed to true (skip future attempts)

**Modified**: `full_attention_decode_core` checks `state.gpu_scoring_failed`
before attempting GPU path.

### 21.4 Files Changed

| File | Changes |
|------|---------|
| `state.rs` | Add `gpu_scoring_failed: bool` to `Qwen35FullAttentionState` |
| `attention.rs` | Check flag before GPU attempt, set on failure |

## 22. Persistent Backend-Resident KV Cache Design

### 22.1 Problem

Every decode step in `decode_scoring_gpu` (attention.rs L876–886) uploads the
**entire** KV cache to the backend:

```rust
let kv_prefix_len = t * state.kv_features;
k_raw.write_data_backend(&state.k_cache[..kv_prefix_len])?;
v_raw.write_data_backend(&state.v_cache[..kv_prefix_len])?;
```

For Qwen3.5 0.6B: `kv_features = kv_head_count(4) × head_dim(128) = 512`.
At T=1000 tokens, each step uploads `2 × 1000 × 512 × 4 bytes = 4 MB`.
This grows **linearly** with sequence length.

### 22.2 Proposed Design

Keep KV tensors persistent on-backend, pre-allocated at `max_tokens` size.
Each decode step appends only the **new** K/V pair:

1. `write_data_backend_at(offset=T*kv_features, &new_k)` — 1024 floats = 4 KB
2. Build graph with `view_4d` into the first `T+1` rows of the persistent tensor
3. Run scoring — no full re-upload

**API note**: `Tensor<f32>` already exposes `write_data_backend_at` and
`read_data_backend_at` (compute.rs L1548–1582). No `DynTensor` workaround needed.

### 22.3 Architecture

The persistent KV tensors, their views, and the scoring graph must all live in
the **same** `Context` — ggml views are context-bound (compute.rs L955–1002).
This means the persistent scoring context encompasses:

```
PersistentScoringContext {
    ctx: Context,                      // single ggml context
    k_persistent: Tensor<f32>,         // [D, Hkv, max_T, 1] on backend
    v_persistent: Tensor<f32>,         // [D, Hkv, max_T, 1] on backend
    q_input: Tensor<f32>,              // [D, 1, H, 1]
    gate_input: Tensor<f32>,           // [D, H, 1, 1]
    // Graph rebuilt per step (T changes → view changes)
    _buffer: BackendBuffer,
}
```

**Key constraint**: the graph must be rebuilt each step because the KV prefix
length changes (the `view_4d` parameters depend on `T`). The weights and KV
data stay resident; only the view + graph metadata changes.

### 22.4 Remaining O(T) Work: On-Device Permute + Cont

**Critical**: eliminating the host→backend KV upload does NOT eliminate all O(T)
work per step. The current scoring path does:

```rust
let k_perm = ctx.permute(&k_raw, 0, 2, 1, 3)?;   // [D,Hkv,T] → [D,T,Hkv]
let k = ctx.cont(&k_perm)?;                        // materialize contiguous copy
```

This `permute + cont` runs on-device but still copies O(T × D × Hkv) floats per
step. At T=1000: `1000 × 128 × 4 × 4 bytes = 2 MB` device-local copy per K/V.

**Mitigation options**:
- **Store KV in flash-friendly layout** `[D, T, Hkv]` from the start. Then no
  permute/cont needed, but this changes the append pattern (non-contiguous write).
- **Accept device-local O(T)**: on Metal/unified memory this is a memcpy within
  shared address space (~10 GB/s), not a PCIe transfer. The 4 MB total is ~0.4 ms
  — comparable to Metal dispatch overhead itself.
- **Investigate** whether `flash_attn_ext` can accept permuted (non-contiguous)
  inputs via stride metadata (unlikely given ggml kernel assumptions, but worth
  checking the Metal kernel implementation).

### 22.5 Memory Residency Trade-off

Current state already allocates max-token host KV caches
(`state.rs` L80–81: `vec![0.0; cache_size]`). Adding backend-resident KV
**doubles** memory usage:

| Component | Host (current) | Host + Backend |
|-----------|---------------|----------------|
| K cache (per FA layer) | T×512×4 = 2 MB @ T=1000 | 2 MB + 2 MB |
| V cache (per FA layer) | T×512×4 = 2 MB @ T=1000 | 2 MB + 2 MB |
| Total (8 FA layers) | 32 MB | 64 MB |

Options to mitigate:
- Keep host as canonical source of truth + backend as mirror (simplest; checkpoint
  serialization from `checkpoint.rs` L183–219 stays unchanged)
- Make backend canonical, drop host cache (saves memory, but readback on checkpoint
  save and on GPU-fallback-to-host)
- Lazy eviction: only keep last N tokens on host for fallback, full cache on backend

**Recommendation**: Host-canonical + backend mirror is simplest for the first
implementation. Memory duplication is acceptable for 0.6B model sizes.

### 22.6 Upload Savings Estimate

| Sequence Length | Current Upload/Step | Persistent (append only) | Savings |
|----------------:|--------------------:|-------------------------:|--------:|
| 100 | 400 KB | 4 KB | 100× |
| 500 | 2 MB | 4 KB | 500× |
| 1000 | 4 MB | 4 KB | 1000× |
| 4096 | 16 MB | 4 KB | 4096× |

Note: on Metal/unified memory the "upload" is a shared-memory copy, not a PCIe
DMA. The absolute cost is lower than on discrete GPUs, but the linear growth
still matters for long contexts.

### 22.7 ggml-rs Lifetime Constraint

The safe Rust wrapper ties tensor lifetimes to their creating `Context` via
`Tensor<'ctx, T>`. ggml's C API allows cross-context views: `ggml_view_4d(ctx_a,
tensor_from_ctx_b, ...)` — the view belongs to `ctx_a` but references data from
`ctx_b`. The Rust wrapper requires both to share `'ctx`.

This blocks the natural design of "persistent KV Context + ephemeral scoring
Context". Workarounds:

1. **`'static` transmute** (used by `PersistentDecodeProjection`): transmute
   the persistent KV tensors to `'static`, then transmute back to the ephemeral
   lifetime when creating views. Sound if the persistent context outlives the
   ephemeral one. Requires `unsafe` + careful drop ordering.
2. **Cross-context view API**: Add a safe `view_4d_cross` method to ggml-rs
   that accepts `Tensor<'other, T>` with a bound `'other: 'ctx`. Encodes the
   real safety requirement (source outlives view) in the type system.
3. **Single-context with reset**: Add `Context::reset_graph()` that frees
   graph/view metadata while preserving data tensors. Not supported by ggml C API.

Option 2 is the cleanest long-term solution. Until then, the `'static` transmute
pattern from `generation.rs` can be extended.

### 22.8 Implementation Status

**Not yet implemented** — design documented for future work. The on-device
`permute+cont` O(T) remains the deeper issue; solving only the upload would
give diminishing returns on Metal where the upload is already cheap. The
ggml-rs lifetime constraint adds implementation complexity that warrants a
dedicated cross-context view API (option 2 above) before proceeding.

## 23. SIMD Vectorization of SSM Recurrence

### 23.1 Current Implementation

`ssm_recurrence_step` (linear_attention.rs L710–759) is pure scalar Rust
operating on a `state_size × state_size` matrix (64×64 = 4096 f32 for Qwen3.5).

Four phases:

| Phase | Operation | Inner Size | FLOPs |
|-------|-----------|-----------|-------|
| 1. Decay + sk | `s *= decay; sk[col] += s * k_row` | 64 cols × 64 rows | 8192 |
| 2. Delta | `delta[col] = (v[col] - sk[col]) * beta` | 64 | 128 |
| 3. State update | `s[col] += k_row * delta[col]` | 64 cols × 64 rows | 8192 |
| 4. Output | `out[col] += state[row*ss+col] * (q[row]*scale)` | 64 cols × 64 rows | 8192 |

Total: ~24.7K FLOPs/head/step × 12 heads (Qwen3.5 linear layers) ≈ 296K FLOPs.

### 23.2 Vectorization Analysis

**Phases 1 and 3** are the best SIMD candidates. Inner loops iterate over
contiguous `row_slice` of 64 f32 elements:

```rust
// Phase 1 — contiguous row slice, broadcast k_row
for (col, s) in row_slice.iter_mut().enumerate() {
    *s *= decay;                        // vmul(row_slice, decay_broadcast)
    scratch.sk[col] += *s * k_row;      // vfmadd(sk, row_slice, k_broadcast)
}

// Phase 3 — contiguous row slice, broadcast k_row
row_slice.iter_mut().zip(scratch.delta.iter())
    .for_each(|(s, &d)| *s += k_row * d);  // vfmadd(row_slice, k_broadcast, delta)
```

With `state_size = 64`:
- **NEON (128-bit)**: 4 f32 lanes → 16 iterations per row
- **AVX2/FMA (256-bit)**: 8 f32 lanes → 8 iterations per row
- **AVX-512 (512-bit)**: 16 f32 lanes → 4 iterations per row

**Phase 2** (delta computation) operates on a 64-element vector — trivially
vectorizable but so small it's negligible.

**Phase 4** (output readout) is the **structural problem**. The current loop
is column-major over a row-major matrix:

```rust
for col in 0..state_size {
    for row in 0..state_size {
        scratch.out[col] += state[row * state_size + col] * (q[row] * scale);
    }
}
```

Access pattern `state[row * 64 + col]` reads with stride 64 — **not contiguous**
in the inner loop dimension. LLVM cannot auto-vectorize this efficiently.

**Fix**: Transpose the loop order to row-major:

```rust
for row in 0..state_size {
    let qr_scaled = q[row] * scale;
    for col in 0..state_size {
        scratch.out[col] += state[row * state_size + col] * qr_scaled;
    }
}
```

This makes the inner loop access contiguous `state[row*ss + 0..64]` and
broadcast `qr_scaled` — identical to phases 1/3 and fully SIMD-friendly.

### 23.3 Auto-Vectorization Baseline

Before adding explicit SIMD, check LLVM auto-vectorization in release mode:

```bash
cargo rustc --release -p llama-rs -- --emit asm \
  -C target-cpu=native -C opt-level=3
```

Phases 1/3 likely already auto-vectorize with `-C opt-level=3` on both
x86 (AVX2/FMA) and AArch64 (NEON). Phase 4 almost certainly does NOT
auto-vectorize due to the strided access pattern.

### 23.4 Approach Options

| Option | Pros | Cons |
|--------|------|------|
| Loop reorder + auto-vec | Zero dependencies, safe | Relies on LLVM heuristics |
| `std::simd` (nightly) | Portable, safe | Requires nightly — against project policy |
| `std::arch` intrinsics | Maximum control | `unsafe`, platform-specific `cfg` |
| `pulp` / `safe_arch` crate | Safe wrappers, portable | External dependency |

**Recommended first step**: Reorder phase 4 loop (pure refactoring, measurable),
then verify auto-vectorization via disassembly. Explicit SIMD only if
auto-vectorization proves insufficient.

### 23.5 Expected Gains

The SSM recurrence is already sub-millisecond on host (~296K FLOPs at ~10
GFLOPS/s scalar = ~30 µs). SIMD at 4–16× lane width would reduce to ~2–8 µs.
The absolute saving is small, but this is called 12× per token (once per linear
attention layer), so total savings across all layers: ~250–340 µs/token.

Compared to other per-token costs (MLP ~15 ms, attention scoring ~3 ms on Metal),
SSM recurrence optimization is low priority. The phase 4 loop reorder is the
highest-value change: correct loop order costs nothing and may already unlock
LLVM auto-vectorization.

### 23.6 Implementation Status

**Phase 4 loop reorder**: **IMPLEMENTED** — `ssm_recurrence_step` phase 4 now
uses row-major traversal with iterator zip. All 208 tests pass, including decode
equivalence tests that verify numerical parity.
**Explicit SIMD**: deferred — auto-vectorization analysis needed first.

## 24. End-to-End Decode Per-Token Cost Model (Qwen3.5 0.6B)

### 24.1 Full Cost Breakdown

Per-token decode costs at T=1000, combining findings from items 13–23.
Qwen3.5 0.6B: H=1536, D=128, Hq=24, Hkv=4, FFN=8960, vocab=151936,
8 full attention layers, 24 linear attention layers.

| # | Component | Where | FLOPs | Data I/O | Status |
|---|-----------|-------|-------|----------|--------|
| 1 | QKV projection (full attn) | GPU persistent | 3×H²=7.1M | 6 KB in, ~14 KB out | ✓ Item 15 |
| 2 | QKV projection (linear attn) | GPU persistent | 4×H²=9.4M | 6 KB in, ~18 KB out | ✓ Item 15 |
| 3 | Attention scoring + gating | GPU flash_attn | ~8.2M | 4 MB KV upload | ✓ Item 16 |
| 4 | Output projection (full) | GPU persistent | H²=2.4M | ~6 KB each way | ✓ Item 15 |
| 5 | Output projection (linear) | GPU persistent | H²=2.4M | ~6 KB each way | ✓ Item 15 |
| 6 | MLP | GPU graph | 3×H×FFN=41.3M | ~6 KB each way | ✓ Graph |
| 7 | LM head | GPU persistent | H×V=233M | 6 KB in, 608 KB out | ✓ Item 13 |
| 8 | Conv decode | Host scalar | 5.6K | <1 KB | Item 17 |
| 9 | RoPE decode | Host scalar | 6K | <1 KB | Item 20 |
| 10 | SSM recurrence | Host scalar | 296K | <100 KB | Item 19 |
| 11 | QKV split/norm | Host | ~2K | <4 KB | Item 18 |
| 12 | KV cache upload | Host→Backend | — | 4 MB @ T=1000 | ⚠ O(T) |
| 13 | Temp scoring ctx/graph | Overhead | — | ~2 MB metadata alloc | Per-step |
| 14 | KV permute+cont (device) | Backend | O(T×D×Hkv) | 4 MB device copy | O(T) |
| 15 | Host rms_norm (decode pre-norm) | Host | 2×H=3K | <12 KB | Item 10 |
| 16 | Persistent proj readback | Host←Backend | — | ~20 KB/layer | Per-step |

### 24.2 Per-Layer Subtotals

**Full attention layer** (×8 layers):
- GPU compute: QKV proj (7.1M) + scoring (8.2M) + output proj (2.4M) = 17.7M FLOPs
- Host compute: RoPE (6K) + QKV split (~1K) + rms_norm (3K) ≈ 10K FLOPs
- Data transfer: 6 KB (hidden in) + 4 MB (KV upload) + 4 MB (device permute) + ~20 KB (readbacks) ≈ 8 MB
- **Bottleneck**: KV cache transfer (O(T)), not compute

**Linear attention layer** (×24 layers):
- GPU compute: QKV proj (9.4M) + output proj (2.4M) = 11.8M FLOPs
- Host compute: Conv (5.6K) + SSM recurrence (296K) + split/norm (~2K) + rms_norm (3K) ≈ 307K FLOPs
- Data transfer: 6 KB (hidden in) + ~20 KB (readbacks) ≈ 26 KB
- **Bottleneck**: SSM recurrence (host scalar), but absolute cost is small (~30 µs)

**MLP** (×32 layers):
- GPU compute: 41.3M FLOPs
- Data transfer: ~12 KB round-trip
- No host bottleneck

**LM head** (×1):
- GPU compute: 233M FLOPs
- Data transfer: 6 KB in + 608 KB out = 614 KB

### 24.3 Aggregate Per-Token Estimate

| Category | Total FLOPs | Total Transfer | Wall Time (est.) |
|----------|------------|----------------|-----------------|
| Full attn layers (×8) | 142M | ~64 MB | ~24 ms (KV-dominated) |
| Linear attn layers (×24) | 283M | ~0.6 MB | ~7 ms |
| MLP layers (×32) | 1,322M | ~0.4 MB | ~15 ms |
| LM head (×1) | 233M | ~0.6 MB | ~3 ms |
| **Total** | **1,980M** | **~66 MB** | **~49 ms** |

**Note**: wall time estimates are rough and assume Metal backend on Apple Silicon.
Actual performance depends on pipeline overlap, backend scheduling, and memory
bandwidth. The ~64 MB for full attention layers is the KV re-upload at T=1000 ×
8 layers — this is the dominant data transfer cost.

### 24.4 Key Bottleneck Identification

1. **KV cache data transfer** (O(T) per step): The dominant host↔backend
   transfer bottleneck for long sequences. At T=1000, ~64 MB total across
   8 full attention layers. On Metal/unified memory this is a shared-memory
   copy (not PCIe DMA), but the linear growth still matters at long contexts.
   **Persistent backend-resident KV** (item 22) would reduce this to ~32 KB/step
   for the upload portion, but the on-device `permute+cont` O(T) remains.

2. **MLP compute** (1,322M FLOPs): The largest single compute category, but
   already on GPU with persistent weights. At Metal speeds (~1 TFLOPS f32),
   this is ~1.3 ms — fast.

3. **LM head compute** (233M FLOPs): Large but single-shot per token. Already
   persistent on GPU.

4. **SSM recurrence** (7.1M FLOPs total across 24 layers): Tiny in absolute
   terms (~720 µs total at 10 GFLOPS/s scalar). Phase 4 loop reorder (item 23)
   could reduce by 2–4× through auto-vectorization.

5. **Temporary scoring context allocation**: Creates a new ggml Context +
   BackendBuffer + graph per decode step per full attention layer.
   Persistent scoring context (item 22) would amortize this overhead.

### 24.5 Optimization Priority (Effort vs Impact)

| Priority | Optimization | Impact | Effort |
|----------|-------------|--------|--------|
| 1 | Persistent KV cache (item 22) | Eliminate ~64 MB upload | Medium |
| 2 | Phase 4 loop reorder (item 23) | ~2–4× SSM speedup, free | Trivial |
| 3 | Flash-friendly KV layout | Eliminate O(T) device copy | High |
| 4 | Persistent scoring context | Amortize alloc overhead | Medium |
| 5 | Explicit SIMD (phases 1/3) | ~2–4× additional SSM gain | Medium |
| 6 | Custom DELTANET_SCAN op | Full GPU SSM | Very High (upstream) |

## 25. Cross-Context View API (`view_Nd_of`)

### 25.1 Problem

Persistent KV cache (item 22) requires an ephemeral scoring graph context
to reference tensors owned by a long-lived cache context.  The existing
`view_1d`..`view_4d` methods on `Context` enforce `&Tensor<'ctx, T>` — the
source tensor must belong to the *same* context.  This is a Rust-side
limitation; the ggml C API supports cross-context views natively
(`ggml_view_Nd` only requires a valid pointer, not a matching context).

The workaround used by `PersistentDecodeProjection` (`generation.rs` L314-330)
is `'static` transmute — sound in practice but breaks Rust borrow-checking
guarantees and prevents the compiler from detecting use-after-drop.

### 25.2 Solution: `view_Nd_of` with `'src: 'ctx` Bound

Four new methods on `Context`:

```rust
pub fn view_1d_of<'ctx, 'src: 'ctx, T: GgmlElement>(
    &'ctx self, a: &Tensor<'src, T>, ne0: usize, offset: usize
) -> Result<Tensor<'ctx, T>>
```

(and `view_2d_of`, `view_3d_of`, `view_4d_of` with matching stride/offset params)

**Key lifetime constraint**: `'src: 'ctx` means the source context must
outlive the view context.  The borrow checker enforces this at every call
site — `drop(src_ctx)` while the view is live is a compile error.

**Soundness argument**:
- `Tensor<'src, T>` carries `PhantomData<&'src Context>`, keeping `src_ctx`
  borrowed for `'src`
- The `'src: 'ctx` bound makes the compiler enforce `src_ctx` outlives the
  view context's borrow
- The returned `Tensor<'ctx, T>` lives at most `'ctx`, which is ≤ `'src`
- No unsafe beyond the FFI call to `ggml_view_Nd` (same as existing `view_Nd`)

### 25.3 Validation

All `view_Nd_of` variants reuse the existing `validate_view_extent` helper
(which is already lifetime-agnostic: `&Tensor<'_, T>`), providing the same
OOB bounds checks as same-context views.

### 25.4 Tests (5 new)

| Test | Covers |
|------|--------|
| `cross_context_view_1d_smoke` | Basic cross-context 1D view + offset |
| `cross_context_view_2d_smoke` | Strided 2D view across contexts |
| `cross_context_view_4d_backend` | 4D view with backend-allocated tensors |
| `cross_context_view_oob_rejected` | OOB checks work across contexts |
| `cross_context_view_in_graph` | **Key use case**: persistent ctx tensor + ephemeral graph → `add` → correct output |

### 25.5 Files Changed

| File | Change |
|------|--------|
| `src/compute.rs` | `view_1d_of` through `view_4d_of` (+145 lines) |
| `tests/ggml_tensor_ops.rs` | 5 cross-context view tests (+125 lines) |

### 25.6 Impact on Persistent KV Cache

This API is the safe foundation for item 22's design.  The ephemeral scoring
context can now create views into a persistent KV cache context without
`'static` transmute:

```rust
let persistent_ctx = Context::new_no_alloc(mem)?;
let k_cache = persistent_ctx.new_tensor_3d::<f32>(shape)?;
// ... (persist across decode steps)

// Per-step: ephemeral context views into persistent cache
let graph_ctx = Context::new_no_alloc(mem)?;
let k_prefix = graph_ctx.view_3d_of(&k_cache, d, t, hkv, nb1, nb2, 0)?;
// ... build flash_attn_ext graph using k_prefix ...
```

All 213 tests pass (208 existing + 5 new).

---

## 26. Persistent Backend-Resident KV Cache

### 26.1 Problem: O(T) Per-Step KV Upload

The ephemeral GPU scoring path (`decode_scoring_gpu`) uploads the **entire**
host KV cache to the backend every decode step. Cost per step:

```
bytes = 2 × D × Hkv × T × sizeof(f32)
      = 2 × 128 × 2 × T × 4
      = 2048 × T bytes per FA layer
```

Across 8 full attention layers at T=1000: **~16 MB/step**.
At T=4000 (realistic generation): **~64 MB/step** — PCIe/unified-memory
transfer becomes the dominant bottleneck, not compute.

`PersistentDecodeProjection` already solved the O(1) weight upload for
QKV and output projections (item 21). The KV cache was the last remaining
O(T) transfer.

### 26.2 Solution: `PersistentKvCache` with Incremental Append

**Architecture**: Loop-local parallel container (same pattern as
`PersistentDecodeProjection`) — **not** embedded in `Qwen35FullAttentionState`.

```
two_phase_loop locals (drop order: LIFO, handles before contexts):
├── kv_persistent: Option<(Vec<Option<Context>>, Vec<Option<PersistentKvCache<'static>>>)>
├── decode_projs: Vec<Option<PersistentDecodeProjection<'static>>>
├── _proj_ctxs: Vec<Option<Context>>
└── state: GenerationState   (host KV cache — source of truth)
```

**Key design decisions**:
1. Host `Vec<f32>` remains source of truth (checkpoint serialization, fallback)
2. Backend tensors are a mirrored acceleration structure
3. `'static` transmute follows same safety argument as `PersistentDecodeProjection`
4. Linear attention layers get `None` slots (no quadratic KV cache)

### 26.3 Per-Step Data Flow

| Phase | Data transfer | Direction | Cost |
|-------|--------------|-----------|------|
| Prefill → seed | Full KV prefix upload | Host → Device | O(T_prompt) × once |
| Decode step: host | `state.append_batch(k, v, 1)` | in-memory | O(D×Hkv) |
| Decode step: device | `kv_cache.append_token(k, v, pos)` | Host → Device | O(D×Hkv) |
| Decode step: scoring | `view_4d_of` + permute + flash_attn | Device only | O(T) on-device |

**Improvement**: Host→Device transfer drops from O(T×D×Hkv) to O(D×Hkv) —
constant per step regardless of sequence length.

### 26.4 Tensor Layout

```
Persistent K/V tensors: [D, Hkv, MaxT, 1] — backend-allocated
  D = head_dimension (128)
  Hkv = kv_head_count (2)
  MaxT = total_sequence_length (configurable)

Per-step view: view_4d_of(tensor, D, Hkv, T, 1, nb1, nb2, nb3, offset=0)
  nb1 = D × sizeof(f32)           stride between KV heads
  nb2 = D × Hkv × sizeof(f32)     stride between time steps
  nb3 = nb2 × T                    stride for dim3 (unused, dim3=1)
```

### 26.5 Scoring Path Priority

`full_attention_decode_core` now has a 3-level fallback:

1. **Persistent GPU** (`decode_scoring_gpu_persistent`): O(1) upload,
   cross-context views, flash_attn_ext
2. **Ephemeral GPU** (`decode_scoring_gpu`): O(T) upload, flash_attn_ext
3. **Host scoring**: CPU dot-product loop (reference implementation)

The `gpu_scoring_failed` flag is probe-once: first failure on either GPU
path disables it for all subsequent steps in the same generation.

### 26.6 `PersistentKvCache` API

```rust
pub(super) struct PersistentKvCache<'ctx> {
    k_tensor: Tensor<'ctx, f32>,   // [D, Hkv, MaxT, 1] on backend
    v_tensor: Tensor<'ctx, f32>,
    _buffer: BackendBuffer<'ctx>,
    kv_features: usize,            // D × Hkv
    max_tokens: usize,
}

impl PersistentKvCache<'_> {
    fn append_token(&self, k: &[f32], v: &[f32], pos: usize) -> Result<()>;
    fn seed_from_host(&self, k_cache: &[f32], v_cache: &[f32], len: usize) -> Result<()>;
}

fn build_persistent_kv_cache(
    attention: &Qwen35FullAttentionLayerPlan,
    max_tokens: usize,
    backend: &Backend,
) -> Result<(Context, PersistentKvCache<'static>)>;
```

### 26.7 Integration in `generation.rs`

```rust
fn try_build_persistent_kv_caches(
    layer_plans: &[LayerPlan], max_tokens: usize, backend: &Backend,
) -> Option<PersistentKvCacheSets>
```

- Built after persistent projections succeed
- Seeded from host state after prefill phase
- Passed through `persistent_decode_all_layers` → `full_attention_decode_core`

### 26.8 Memory Budget

```
Per FA layer: 2 × D(128) × Hkv(2) × MaxT × 4B
At MaxT=4096: 2 × 128 × 2 × 4096 × 4 = 8 MB per layer
Across 8 FA layers: ~64 MB total
```

Manageable on unified memory (M-series) and discrete GPU. The max is
determined by `inputs.total_sequence_length`, not a hardcoded constant.

### 26.9 Files Changed

| File | Change |
|------|--------|
| `llama-rs/src/e2e/attention.rs` | `PersistentKvCache` struct, `build_persistent_kv_cache`, `decode_scoring_gpu_persistent`, `full_attention_decode_core` updated |
| `llama-rs/src/e2e/generation.rs` | `try_build_persistent_kv_caches`, `persistent_decode_all_layers` updated, `two_phase_loop` seeding + pass-through |

### 26.10 Relationship to Other Items

- **Item 21** (persistent projections): Same ownership/transmute pattern
- **Item 22** (KV transfer analysis): Identified the bottleneck this solves
- **Item 25** (cross-context view API): Foundation for `decode_scoring_gpu_persistent`
- **Item 27** (next): causal_depthwise_conv vs QKV packing comparison

---

## 27. Causal Depthwise Conv vs QKV Packing: Architecture Comparison

This item is the comparison this worklog was originally scoped for — a
detailed analysis of how the two attention types in Qwen3.5 handle
projection, convolution, and state management, and how llama-rs's approach
relates to the ggml primitives.

### 27.1 Two Attention Types in Qwen3.5

Qwen3.5 interleaves two fundamentally different attention mechanisms:

| Property | Full Attention (FA) | Linear Attention (LA) |
|----------|--------------------|-----------------------|
| Plan struct | `Qwen35FullAttentionLayerPlan` | `Qwen35LinearAttentionLayerPlan` |
| Mechanism | Softmax QKV attention | Delta-net SSM recurrence |
| Complexity (inference) | O(T²·D) per layer | O(T·D·S) per layer |
| Complexity (decode) | O(T·D) per step | O(D·S²) per step |
| KV cache | Quadratic (grows with T) | Constant (SSM state) |
| Uses convolution | No | Yes (causal depthwise) |
| Position encoding | NeoX RoPE | None (implicit via SSM) |

### 27.2 QKV Weight Storage: Separate vs Unified

**Full Attention — Separate matrices:**
```
plan.q_weight_values: [hidden × query_features×2]   // Q + gate interleaved
plan.k_weight_values: [hidden × kv_features]
plan.v_weight_values: [hidden × kv_features]
```

Three independent `mul_mat` ops (batched in one graph):
```
Q_full = W_q · X    → [query_features×2 × seq_len]
K      = W_k · X    → [kv_features × seq_len]
V      = W_v · X    → [kv_features × seq_len]
```

**Linear Attention — Unified matrix:**
```
plan.qkv_weight_values: [hidden × conv_channels]
plan.gate_weight_values: [hidden × inner_size]
plan.alpha_weight_values: [hidden × time_step_rank]
plan.beta_weight_values: [hidden × time_step_rank]
```

Where `conv_channels = inner_size + 2 × group_count × state_size` — Q, K, V
are packed into a single projection, split *after* convolution.

**Design rationale:** In linear attention, Q/K/V pass through a shared causal
conv before they diverge. Projecting into a single vector and convolving once
is cheaper than three separate projections + three convolutions.

### 27.3 Full Attention Q+Gate Interleaving

Qwen3.5 full attention adds a **gating mechanism** where Q and its gate are
interleaved per-head in the weight matrix output:

```
W_q projection output layout (per token):
  [Q_h0_d0..Q_h0_dD, G_h0_d0..G_h0_dD,   ← head 0: Q then gate
   Q_h1_d0..Q_h1_dD, G_h1_d0..G_h1_dD,   ← head 1: Q then gate
   ...]
```

`deinterleave_q_gate` (attention.rs:570-605) separates these into two flat
buffers:
```rust
q_dst[head*hd..(head+1)*hd] = src[head*2*hd .. head*2*hd + hd]     // Q
g_dst[head*hd..(head+1)*hd] = src[head*2*hd + hd .. (head+1)*2*hd] // gate
```

The gate is applied after attention scoring: `output = sigmoid(gate) ⊙ attn`.

**llama.cpp comparison:** llama.cpp typically stores Q and gate as separate
weight tensors (or splits a fused QKV tensor at tensor-load time). llama-rs
stores a single `query_features×2` matrix and splits on host after projection —
this avoids duplicating the hidden→query matmul but adds a strided copy.

### 27.4 Linear Attention: Post-Conv QKV Split

After the unified QKV projection passes through causal depthwise convolution
and SiLU activation, the output is split:

```
conv_output[t, :] = [Q₀..Q_{qk-1}, K₀..K_{qk-1}, V₀..V_{inner-1}]
                     ├─ qk_features ─┤├─ qk_features ─┤├─ inner_size ─┤
```

`split_and_norm_qk` (linear_attention.rs:451-487):
```rust
q_dst.copy_from_slice(&conv_row[..qk_features]);
k_dst.copy_from_slice(&conv_row[qk_features..qk_features * 2]);
// V = conv_row[2*qk_features .. conv_channels] (used in-place)
```

**Normalization difference:**
- Full attention: **RMS norm** per-head on Q and K (pre-RoPE)
- Linear attention: **L2 norm** per-group on Q and K (post-conv)

### 27.5 Causal Depthwise Convolution: Three Implementations

llama-rs implements causal depthwise conv at three levels:

#### (a) Host reference (prefill): `causal_depthwise_conv`

```rust
fn causal_depthwise_conv(
    input: &[f32],          // [seq_len × channels]
    sequence_length, channels, kernel_size,
    weight: &[f32],         // [channels × kernel_size]
) -> Vec<f32>
```

Triple nested loop: token → channel → tap. Causal constraint: `start_tap =
kernel_size - min(kernel_size, t+1)` ensures token `t` only reads
`[max(0, t-K+1), ..., t]`.

#### (b) Graph-accelerated (prefill): `ggml_ssm_conv`

Uses the GGML `ssm_conv` primitive. Requires pre-processing:

1. **Transpose** input from `[seq_len, channels]` (channels-fast) to
   `[channels, padded_len]` (time-fast)
2. **Left-pad** with `kernel_size - 1` zeros per channel (causal mask)
3. **Reshape** to 3D: `[padded_len, channels, 1]` for `ggml_ssm_conv`

```rust
let conv_out = ctx.ssm_conv(&sx, &c)?;   // [channels, seq_len, 1]
let result = ctx.silu(&conv_out)?;
```

**ggml_ssm_conv contract** (ggml.c:5430-5454):
- `sx`: `[d_conv-1+n_t, d_inner, n_s]` — pre-padded input
- `c`: `[d_conv, d_inner]` — per-channel kernel weights
- Output: `[d_inner, n_t, n_s]` — channels-fast, no padding

#### (c) Decode step (single token): `causal_depthwise_conv_decode_step`

```rust
fn causal_depthwise_conv_decode_step(
    new_row: &[f32],         // [conv_channels] — current token
    state: &mut LinearAttentionState,
    weight: &[f32],          // [channels × kernel_size]
) -> Vec<f32>
```

Uses sliding window buffer (`state.conv_buffer`) storing last `kernel_size-1`
pre-conv QKV rows. O(channels × kernel_size) per step.

### 27.6 Conv State: Sliding Window Buffer

```rust
pub struct LinearAttentionState {
    pub conv_buffer: Vec<f32>,   // [(kernel_size-1) × conv_channels]
    pub conv_valid: usize,       // actual filled rows (< kernel_size-1 for short seqs)
    pub conv_channels: usize,
    pub conv_kernel: usize,
    // + ssm_states for recurrence
}
```

**Lifecycle:**
1. **Prefill** → `capture_conv_buffer`: copies last `kernel_size-1` rows of
   raw QKV (pre-conv) from the fused graph's output
2. **Decode** → `push_conv_row`: appends new token, shifts left when full
3. **Decode conv** → reads from buffer + current token, applies kernel

**Key design:** The buffer stores **pre-convolution** QKV. Post-conv values
cannot be reversed, so the raw projection output must be preserved for the
sliding window.

### 27.7 Prefill: Fused Graph vs Separate Round-Trips

**Full attention prefill** (two round-trips):
```
Graph 1: matmul(W_q, X), matmul(W_k, X), matmul(W_v, X) → read back Q,K,V
Host: deinterleave_q_gate, per_head_rms_norm, RoPE, KV cache append
Graph 2: flash_attn_ext(Q, K, V) → read back attention output
Host: sigmoid(gate) ⊙ attn → output projection
```

**Linear attention prefill** (one round-trip for projection+conv):
```
Graph 1 (fused): 
  norm(X) → matmul(W_qkv, X), matmul(W_z, X), matmul(W_α, X), matmul(W_β, X)
  → transpose(QKV) → left_pad → ssm_conv(padded, kernel) → silu
  → read back conv_output, z, alpha, beta, pre_conv_qkv
Host: split Q/K/V, L2 norm, SSM recurrence loop, z-gating
Host: capture_conv_buffer, capture_ssm_states
```

The fused graph eliminates the host↔device round-trip between projection and
convolution — a key advantage of using `ggml_ssm_conv` as a native graph op.

### 27.8 Decode: Host-Heavy vs GPU-Heavy

**Full attention decode** (persistent GPU):
```
GPU: persistent projection matmuls (O(hidden×features))
Host: deinterleave, RMS norm, RoPE
GPU: persistent KV append (O(D×Hkv))
GPU: flash_attn_ext via cross-context views (O(T) on-device)
GPU: output projection
```

**Linear attention decode** (host-dominated):
```
GPU: 4 small projection matmuls (O(hidden×features))
Host: causal_depthwise_conv_decode_step (O(channels×kernel_size))
Host: split_and_norm_qk (O(qk_features))
Host: SSM recurrence loop (O(time_step_rank × state_size²))
GPU: output projection (O(inner_size×hidden))
```

Full attention decode has been progressively offloaded to GPU (items 21, 26).
Linear attention decode remains host-dominated because:
1. Conv is O(channels×K) — tiny, not worth a kernel launch
2. SSM recurrence is inherently sequential (state-dependent)
3. Only the projection matmuls and output projection benefit from GPU

### 27.9 Memory Layout Comparison

| Layout | Full Attention | Linear Attention |
|--------|---------------|-----------------|
| Input | `[hidden × seq_len]` channels-fast | Same |
| Q projection | `[query_features×2 × seq_len]` | Packed in `[conv_channels × seq_len]` |
| K projection | `[kv_features × seq_len]` | Packed in same QKV |
| V projection | `[kv_features × seq_len]` | Packed in same QKV |
| Conv input | N/A | `[channels × padded_len]` time-fast |
| Conv weight | N/A | `[channels × kernel_size]` channel-major |
| KV cache | `[D × Hkv × T]` grows with T | Constant SSM state `[time_step_rank × state_size²]` |

The layout transpose (channels-fast → time-fast) is required because
`ggml_ssm_conv` expects time-fast layout for efficient sliding-window
convolution.

### 27.10 ggml Primitive Usage Summary

| ggml Op | Full Attention | Linear Attention |
|---------|---------------|-----------------|
| `mul_mat` | Q,K,V projection; output proj | QKV,Z,α,β projection; output proj |
| `flash_attn_ext` | Scoring (Q·K+softmax+V) | Not used |
| `ssm_conv` | Not used | Prefill conv (fused graph) |
| `silu` | Not used | Post-conv activation (graph) |
| `permute` | K/V layout for flash_attn | QKV transpose (in fused graph) |
| `cont` | After permute | After transpose |
| `view_4d_of` | Persistent KV cache views | Not used |
| `sigmoid` | Gate activation (scoring graph) | Not used |
| `rms_norm` | Fused graph pre-norm | Fused graph pre-norm |

### 27.11 Key Architectural Differences from llama.cpp

| Aspect | llama-rs | llama.cpp |
|--------|---------|-----------|
| QKV storage | Separate W_q(+gate), W_k, W_v (FA); Unified W_qkv (LA) | Loaded from GGUF as-is (separate or packed per model) |
| Q+gate split | Host-side `deinterleave_q_gate` per-head | Typically tensor-level split at load time |
| Conv weight layout | `[channels × kernel_size]` channel-major | `[d_conv × d_inner]` — same via `ggml_ssm_conv` |
| Conv in decode | Host-side sliding window | Graph-level or host, implementation-dependent |
| Prefill conv | Fused projection+conv graph | Similar (graph-level `ggml_ssm_conv`) |
| KV cache | Dual host+GPU persistent | Typically GPU-only KV cache |
| Scoring | `flash_attn_ext` (same) | `flash_attn_ext` or custom attention |
| Norm per head | Host `per_head_rms_norm` / `per_head_l2_norm` | Often graph-level norm |

### 27.12 Performance Characteristics

**Full Attention:**
- Prefill: GPU-bound (matmuls dominate, O(T²·D))
- Decode bottleneck was KV transfer (O(T) per step) — **solved by item 26**
- Now bottleneck shifts to flash_attn_ext compute time (O(T·D))

**Linear Attention:**
- Prefill: GPU-bound (fused projection+conv graph)
- Decode: CPU-bound (SSM recurrence is O(time_step_rank × state_size²))
- Conv is negligible (O(channels × kernel_size) ≈ O(4K × 4) = 16K flops)
- Potential future offload: batch multiple decode steps for GPU SSM

### 27.13 Files Referenced

| File | Content |
|------|---------|
| `llama-rs/src/e2e/attention.rs` | Full attention: projections, deinterleave, RoPE, scoring |
| `llama-rs/src/e2e/linear_attention.rs` | Linear attention: conv, SSM recurrence, fused graph |
| `llama-rs/src/e2e/state.rs` | Both: KV cache (`Qwen35FullAttentionState`), conv buffer (`LinearAttentionState`) |
| `llama-rs/src/e2e/plan.rs` | Plan structs defining dimensions and weights |
| `llama-rs/src/e2e/generation.rs` | Dispatch: persistent projections, persistent KV cache |
| `vendor/ggml/src/ggml.c` | `ggml_ssm_conv` definition (L5430-5454) |
| `vendor/ggml/src/ggml-cpu/ops.cpp` | CPU `ssm_conv` kernel (L9191-9259) |

---

## 28. Persistent MLP Graphs (Decode-Path Weight-Reuse)

### 28.1 Problem

During autoregressive decode, the MLP stage was the **#1 remaining bottleneck**.
Each decode step created a new ggml context and re-uploaded all three weight
matrices (gate, up, down) per layer per token:

| Matrix | Shape | Bytes (f32) |
|--------|-------|-------------|
| W_gate | `[hidden × ffn]` | `1536 × 8960 × 4 = 55.1 MB` |
| W_up   | `[hidden × ffn]` | `55.1 MB` |
| W_down | `[ffn × hidden]` | `55.1 MB` |
| **Per layer** | | **~165 MB** |
| **32 layers** | | **~5.3 GB** |

This upload happened on *every* decode token, even though the weights are
constant. The actual payload per step is just the hidden vector (~6 KB each
way at hidden=1536).

### 28.2 Solution: `PersistentMlp`

Follows the established persistent graph pattern from items 13 (LM head) and
15 (decode projections): build the graph once, upload weights once, reuse for
every decode token.

```rust
pub(super) struct PersistentMlp<'ctx> {
    x_in: Tensor<'ctx, f32>,     // input (hidden_features × 1)
    y_out: Tensor<'ctx, f32>,    // output (hidden_features × 1)
    graph: Graph<'ctx>,
    _buffer: BackendBuffer<'ctx>,
    hidden_features: usize,
}
```

**Graph topology (fixed at seq_len=1):**
```
rms_norm(x_in, eps) → mul(norm_w) → mul_mat(W_gate) → silu
                                   → mul_mat(W_up)
                                     → mul(silu_gate, up)
                                       → mul_mat(W_down)
                                         → y_out
```

**Per-step cost:**
- Write: ~6 KB (hidden vector to `x_in`)
- Compute: 3 matmuls on device (no transfer)
- Read: ~6 KB (hidden vector from `y_out`)
- **Total: ~12 KB I/O vs ~165 MB ephemeral**

### 28.3 Implementation

#### `build_persistent_mlp()`

Creates a dedicated ggml `Context`, builds the full MLP graph, allocates a
backend buffer, uploads weights (gate, up, down, norm) once, and returns
`(Context, PersistentMlp<'static>)`.

Uses the same `transmute` pattern as `build_one_persistent_full`: `Tensor`,
`Graph`, and `BackendBuffer` all carry `PhantomData<&'ctx Context>` (raw
pointers, not real references). The transmute erases the unnameable local
borrow. Callers maintain LIFO drop order: handles drop before contexts.

#### `PersistentMlp::step()`

```rust
fn step(&mut self, hidden: &[f32], backend: &Backend) -> Result<Vec<f32>, E2eError> {
    self.x_in.write_data_backend(hidden)?;
    backend.compute(&mut self.graph)?;
    self.y_out.read_data_backend()
}
```

#### `try_build_persistent_mlps()`

Per-layer opportunistic builder in `generation.rs`. Unlike
`try_build_persistent_projections` (all-or-nothing), this returns aligned
vecs where individual layers can be `None`. Failed layers fall back to
ephemeral `mlp_sequence_inference_with_weights`.

#### Integration into `two_phase_loop()`

1. Persistent MLPs built **before** the projection decision (same weights
   needed regardless of persistent vs ephemeral projection path)
2. Early-return when `remaining_decode_steps == 0` (skips persistent resource
   construction entirely — adopted from rubber-duck review)
3. MLPs threaded through both persistent and fallback decode branches

### 28.4 API Changes

`process_all_layers()` gained a `persistent_mlps: &mut [Option<PersistentMlp<'static>>]`
parameter. Empty slice (`&mut []`) means disabled. `debug_assert!` validates
that non-empty slices align to `layer_plans.len()`. Safe indexing via `get_mut`.

All callers in `session.rs` and `generation.rs::full_reprocess_loop` pass
`&mut []` (persistent MLPs only apply to the decode phase of `two_phase_loop`).

### 28.5 Performance Impact

| Metric | Before (ephemeral) | After (persistent) |
|--------|-------------------|-------------------|
| Weight upload/step | ~5.3 GB (32 layers) | 0 (one-time) |
| I/O per layer/step | ~165 MB | ~12 KB |
| Context allocation | Per-step per-layer | Once at decode start |
| Graph build | Per-step per-layer | Once at decode start |

Combined with persistent projections (item 15) and persistent KV cache
(item 26), the decode path now has minimal per-step overhead: primarily
compute-bound (matmuls on device) rather than transfer-bound.

### 28.6 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/mlp.rs` | Added `PersistentMlp`, `step()`, `build_persistent_mlp()`, `recommended_persistent_mlp_memory()` |
| `llama-rs/src/e2e/generation.rs` | Added `PersistentMlpSets`, `try_build_persistent_mlps()`; updated `process_all_layers`, `persistent_decode_all_layers`, `two_phase_loop` |
| `llama-rs/src/e2e/session.rs` | Updated 3 `process_all_layers` call sites |
- `DecodeStrategy` preserved as reference implementation for `full_reprocess_loop`

## 29. GraphAllocator Safe Wrapper (ggml_gallocr)

### 29.1 Problem

The ggml-rs crate had no safe wrapper for `ggml_gallocr` — the graph
allocator that pre-reserves a buffer for the maximum graph size and reuses
it across steps. Every decode step called `Context::allocate_tensors()`,
which created a new Metal buffer allocation (~100µs per layer on Apple
Silicon). This overhead accumulated to ~800µs/step across 8 FA layers.

### 29.2 Design

`GraphAllocator` wraps the `ggml_gallocr` C API:

```rust
pub struct GraphAllocator {
    raw: *mut ffi::ggml_gallocr,
    _not_send_sync: PhantomData<*mut ()>,
}
```

Lifecycle:
1. `GraphAllocator::new(&Backend)` — creates allocator with backend's default buffer type
2. `reserve(&Graph)` — pre-allocates buffer for the maximum graph topology
3. `alloc_graph(&mut Graph)` — maps graph tensors to pre-reserved buffer (no new alloc)
4. `buffer_size()` — queries reserved buffer size in bytes

Key properties:
- `!Send + !Sync` via `PhantomData<*mut ()>` (matches `Backend` pattern)
- Skips tensors that already have a backend buffer (safe for persistent KV
  cross-context views)
- `Drop` implementation calls `ggml_gallocr_free`

### 29.3 Error Handling

Added `AllocationFailed { context: String }` variant to `ggml_rs::Error`
with `allocation_failed()` helper for ergonomic construction.

### 29.4 Tests

5 integration tests in `tests/ggml_graph_allocator.rs`:
1. `basic_lifecycle` — new → reserve → alloc → compute → read
2. `reuse_across_steps` — 10 sequential steps with same allocator
3. `skips_preallocated_tensors` — gallocr respects existing backend buffers
4. `matmul_with_gallocr` — matrix multiplication through gallocr
5. `buffer_size_before_reserve` — validates 0 before first reserve

### 29.5 Files Modified

| File | Changes |
|------|---------|
| `src/compute.rs` | Added `GraphAllocator` struct (~80 lines) |
| `src/error.rs` | Added `AllocationFailed` variant + helper |
| `src/lib.rs` | Re-export `GraphAllocator` in public API and prelude |
| `tests/ggml_graph_allocator.rs` | New: 5 integration tests |

## 30. PersistentScoringContext (Decode-Path Buffer Reuse)

### 30.1 Problem

`decode_scoring_gpu_persistent()` called `ctx.allocate_tensors(backend)`
every step, triggering a Metal buffer allocation for the scoring graph
(Q, gate, intermediates from permute+cont+flash_attn+sigmoid+mul). With
8 FA layers, this cost ~800µs/step — roughly 3% of total decode time at
T=1000.

### 30.2 Design

#### `build_scoring_graph()`

Extracted the common graph construction into a standalone function:

```rust
fn build_scoring_graph<'ctx>(
    ctx: &'ctx Context,
    attention: &Qwen35FullAttentionLayerPlan,
    total_tokens: usize,
    kv_cache: &PersistentKvCache<'static>,
) -> Result<ScoringGraph<'ctx>, E2eError>
```

Returns a `ScoringGraph` struct holding Q, gate, gated output, and the
compute graph. This is used by both `PersistentScoringContext::new()`
(for reservation) and `decode_scoring_gpu_persistent()` (per-step).

#### `PersistentScoringContext`

```rust
pub(super) struct PersistentScoringContext {
    gallocr: GraphAllocator,
}
```

- `new()` builds a reservation graph at `max_tokens`, calls
  `gallocr.reserve()` to pre-allocate the worst-case buffer
- One context is shared across all FA layers (they run serially and
  share identical dimensions in Qwen3.5)

#### Integration

`decode_scoring_gpu_persistent()` now accepts `Option<&mut PersistentScoringContext>`:
- When `Some`: uses `gallocr.alloc_graph()` (no new buffer allocation)
- When `None`: falls back to `ctx.allocate_tensors()` (original behavior)

Threading:
- `full_attention_decode_core()` → new `scoring_ctx` parameter
- `persistent_decode_all_layers()` → new `scoring_ctx` parameter
- `two_phase_loop()` → builds context after KV caches, passes to decode loop

### 30.3 Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Metal buffer alloc/step | 8 allocations (~100µs each) | 0 (pre-reserved) |
| Overhead/step | ~800µs | ~0µs |
| Over 1000 steps | ~800ms (~3%) | negligible |
| Memory | Transient per-step | One persistent buffer |

### 30.4 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/attention.rs` | Added `PersistentScoringContext`, `ScoringGraph`, `build_scoring_graph()`; refactored `decode_scoring_gpu_persistent()`; updated `full_attention_decode_core()` signature |
| `llama-rs/src/e2e/generation.rs` | Updated `persistent_decode_all_layers()` and `two_phase_loop()` to build and thread `PersistentScoringContext` |

## 31. Flash-Friendly KV Layout (Eliminate Permute+Cont)

### 31.1 Problem

`PersistentKvCache` stored tensors in `[D, Hkv, MaxT, 1]` (head-major)
layout, but `flash_attn_ext` expects `[D, T, Hkv, 1]` (time-major). Every
decode step performed `permute(0,2,1,3) + cont` — an O(T) on-device data
copy — to transpose the KV tensors. At T=1000, this cost ~136µs/step.

### 31.2 Design

Changed storage layout to `[D, MaxT, Hkv, 1]` (time-major, flash-friendly):

**Before:**
```
Storage: [D, Hkv, MaxT, 1]  →  view [D, Hkv, T, 1]
                             →  permute(0,2,1,3)  [D, T, Hkv, 1]
                             →  cont               [D, T, Hkv, 1] (copy)
```

**After:**
```
Storage: [D, MaxT, Hkv, 1]  →  view_4d_of [D, T, Hkv, 1]  (zero-copy)
```

The view uses strided access: `nb2 = D × MaxT × sizeof(f32)` to skip
between heads without copying. `flash_attn_ext` handles non-contiguous
K/V via stride metadata (verified in both `ggml.c` and Metal backend — no
contiguity assertions on K or V inputs).

#### `append_token()` changes

Old: one contiguous `write_data_backend_at` per tensor (offset = `t × kv_features`).

New: Hkv small writes per tensor, each writing D floats to the head's stride:
```
offset_for_head_h = h × D × MaxT + t × D
```

For Qwen3.5 0.6B (Hkv=4, D=64): 4 writes of 256 bytes instead of 1 write
of 1024 bytes — negligible overhead increase.

#### `seed_from_host()` changes

Host layout is `[D×Hkv, T]` (contiguous per-token). Now transposes to
`[D, MaxT, Hkv, 1]` during bulk upload (T × Hkv individual writes).

### 31.3 Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Per-step permute+cont | O(T) device copy (~136µs at T=1000) | Eliminated |
| Per-step append overhead | 1 contiguous write | Hkv small writes (negligible) |
| Seed overhead | 2 bulk writes | T×Hkv writes (one-time) |
| Graph ops | view→permute→cont→flash_attn | view→flash_attn (2 fewer ops) |

### 31.4 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/attention.rs` | Changed `PersistentKvCache` layout from `[D, Hkv, MaxT, 1]` to `[D, MaxT, Hkv, 1]`; updated `append_token()`, `seed_from_host()`, `build_persistent_kv_cache()`, `build_scoring_graph()` |

## 32. Auto-Vectorization Verification + Decode Allocation Elimination

### 32.1 Auto-Vectorization Analysis

#### Method

Emitted release ASM with native CPU targeting to verify LLVM auto-vectorization:

```bash
cargo rustc --release -p llama-rs --features link-system \
  -- --emit asm -C target-cpu=native -C opt-level=3
```

Output: `target/release/deps/llama_rs-*.s`

#### Findings: `ssm_recurrence_step`

| Aspect | Detail |
|--------|--------|
| Vectorization | LLVM auto-vectorizes SSM phases 1, 3, 4 with NEON |
| Instructions | `fmul.4s` + `fadd.4s` pairs (NOT fused `fmla.4s`) |
| Throughput | 8 floats/iteration via register pairs (`ldp`/`stp q1, q2`) |
| Total NEON instructions | 129 across `ssm_recurrence_step` |
| `dot()` | Fully inlined by LLVM — no separate symbol |

#### FMA Rejection

Fused multiply-add (`f32::mul_add` / `vfmaq_f32`) was **rejected** for the
default path because:

1. FMA uses single rounding (round-once) vs. separate mul+add (round-twice)
2. This changes the least-significant bits of intermediate results
3. In autoregressive SSM recurrence, small differences accumulate across
   hundreds of steps on 64×64 state matrices
4. Accumulated drift can flip argmax decisions in greedy decoding
5. Our decode equivalence tests verify exact parity with the prefill path,
   which uses ggml graph execution (no FMA in host-side fallback)

**Decision**: Keep LLVM's auto-vectorized mul+add pairs. No `std::arch`
intrinsics, no `mul_add()`. Parity trumps throughput for correctness.

#### Non-target: `dot()` / `project_sequence()`

The `dot()` function in `numeric.rs` is used by `project_sequence()` for
host-side QKV projection. However, in the production decode path, projections
execute through ggml compute graphs on Metal/CPU backend — `dot()` is only
used in the reference/test path. Optimizing it yields no runtime benefit.

### 32.2 Decode Allocation Elimination

#### Problem

In the persistent decode loop, `linear_attention_decode_core()` allocated
fresh buffers on every call:

- 3 `Vec<f32>` for `SsmScratch` (sk, delta, out) — per head × 12 heads
- 1 `Vec<f32>` for output accumulator (inner_size=768)
- 12 `Vec<f32>` for `rms_norm_single()` results (one per head)

Total: **16 allocations per layer × 24 linear layers = 384 heap allocations
per decode step**.

#### Design

**New allocation-free normalization functions:**

```rust
fn rms_norm_single_into(input: &[f32], weight: &[f32], eps: f32, dst: &mut [f32])
fn rms_norm_single_in_place(data: &mut [f32], weight: &[f32], eps: f32)
```

- `rms_norm_single_into`: writes result into caller-provided buffer (separate
  input/output)
- `rms_norm_single_in_place`: reads and writes the same buffer. Needed because
  Rust's borrow checker prevents passing the same slice as both `&[f32]` and
  `&mut [f32]` to `rms_norm_single_into`
- `rms_norm_single()` refactored to delegate to `rms_norm_single_into()`
- `per_head_rms_norm()` updated to use `rms_norm_single_in_place()`

**`LinearDecodeScratch` struct:**

```rust
pub(crate) struct LinearDecodeScratch {
    pub ssm: SsmScratch,      // sk[state_size], delta[time_step_rank], out[state_size]
    pub output: Vec<f32>,     // inner_size accumulator
    pub norm_buf: Vec<f32>,   // state_size for rms_norm scratch
}
```

- Built once before decode loop from first linear attention layer's dimensions
- Passed as `&mut Option<LinearDecodeScratch>` through `persistent_decode_all_layers()`
- All 24 linear layers share dimensions, so one scratch serves all

**Threading:**

```
two_phase_loop()
  └── builds LinearDecodeScratch from layer dims
      └── persistent_decode_all_layers(&mut linear_scratch)
          └── linear_attention_decode_core(Some(&mut scratch))
              ├── reuses scratch.ssm for ssm_recurrence_step
              ├── reuses scratch.output for accumulation
              └── reuses scratch.norm_buf for per_head_rms_norm
```

### 32.3 Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Heap allocations per decode step | 384 (16/layer × 24 layers) | 0 (scratch reused) |
| Allocation overhead | ~38µs/step (est. 100ns/alloc) | Eliminated |
| Auto-vectorization | Already optimal (NEON confirmed) | No change needed |
| FMA | Not used | Rejected (parity) |

### 32.4 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/tensor_ops.rs` | Added `rms_norm_single_into()`, `rms_norm_single_in_place()`; refactored `rms_norm_single()` to delegate; updated `per_head_rms_norm()` to in-place |
| `llama-rs/src/e2e/linear_attention.rs` | Added `LinearDecodeScratch` struct; updated `linear_attention_decode_core()` to accept optional scratch; added scratch construction helper |
| `llama-rs/src/e2e/generation.rs` | Builds `LinearDecodeScratch` in `two_phase_loop()`; threads through `persistent_decode_all_layers()` |
- `DecodeStrategy` preserved as reference implementation for `full_reprocess_loop`

## 33. Resumable GenerationSession + Binary Checkpoint Serialization

### 33.1 Problem

The existing `generate_token_ids_from_model()` API is one-shot: it runs the
full generation loop and returns all tokens. There is no way to:

- Pause generation mid-stream and resume later
- Save inference state (KV caches, conv buffers, SSM states) to disk
- Resume from a checkpoint with a new model instance
- Step through generation one token at a time for streaming UIs

This blocks the `save-load-state` parity matrix entry and prevents
interactive/streaming use cases.

### 33.2 Design

**`GenerationSession`** (`session.rs`, 679 lines):

```rust
pub struct GenerationSession {
    // Owned model data (embeddings, weights, norms, plans)
    layer_plans: Vec<LayerPlan>,
    token_embedding_values: Vec<f32>,
    output_weight_values: Vec<f32>,
    output_norm_values: Vec<f32>,
    // Token tracking
    prompt_token_ids: Vec<i32>,
    all_token_ids: Vec<i32>,       // pad-filled, prompt + generated
    generated_token_ids: Vec<i32>,
    // State
    state: GenerationState,        // per-layer KV/conv/SSM
    effective_mode: GenerationMode, // TwoPhase or FullReprocess
    fingerprint: ModelFingerprint,
    backend: Backend,
    // ...
}
```

Key API:
- `new(model, config)` — resolve model, validate tokens, init state
- `next_token()` → `Option<i32>` — prefill on first call, decode thereafter
- `checkpoint()` → `GenerationCheckpoint` — snapshot entire session state
- `resume(model, backend, policy, checkpoint)` — restore from saved checkpoint
- `generated_tokens()`, `all_tokens()`, `is_finished()`, `generated_count()`

Execution modes:
- **TwoPhase** (Qwen3.5 linear layers): Prefill captures KV/conv/SSM state,
  then decode uses cached state. Checkpoints preserve performance.
- **FullReprocess** (Standard attention or zero max_new_tokens): All tokens
  reprocessed each step. Checkpoints save IDs and position but state is
  recomputed on resume.

**`GenerationCheckpoint`** (`checkpoint.rs`, 737 lines):

```rust
pub struct GenerationCheckpoint {
    inner: CheckpointV1,  // versioned DTO
}
```

Binary format:
- Magic bytes `LRCK` + postcard-serialized `CheckpointV1`
- Versioned envelope (CHECKPOINT_VERSION = 1) for forward compatibility
- `save_to(impl Write)` / `load_from(impl Read)` for I/O

DTO architecture:
- `CheckpointV1` decoupled from runtime types — internal refactoring
  won't break serialized checkpoint compatibility
- `ModelFingerprint` validates model compatibility on load:
  layer_count, hidden_features, vocab_size, rms_norm_eps, per-layer type tags
- `LayerStateDto` serializes per-layer state (KV cache, conv buffer, SSM states)
- `validate_invariants()` prevents panics from malformed/corrupted data

**Error handling:**

New error variants:
- `CheckpointDeserialize(String)` — format/parse errors
- `CheckpointVersionMismatch { file_version, expected_version }`
- `CheckpointModelMismatch { reason: String }` — fingerprint mismatch

### 33.3 Model Compatibility Validation

`ModelFingerprint` captures:
- `layer_count`, `hidden_features`, `vocab_size`, `rms_norm_eps_bits`
- Per-layer `LayerTypeTag` with attention-specific dimensions:
  - `Qwen35Full { kv_head_count, head_dimension }`
  - `Qwen35Linear { conv_kernel, conv_channels, time_step_rank, state_size }`
  - `Standard`, `None`

On `resume()`, the checkpoint fingerprint is validated against the
current model's fingerprint. Mismatch on any field produces
`CheckpointModelMismatch` with a detailed reason string.

Cross-backend resume (CPU → Metal) is allowed but not guaranteed to
produce identical tokens due to floating-point precision differences.

### 33.4 Test Coverage (18 tests)

**checkpoint.rs (14 tests):**
- Roundtrip save/load preserves all fields
- Bad magic bytes rejected
- Invalid version rejected
- Corrupted data rejected
- Token count invariant violations caught
- Token out-of-vocab caught
- Layer state count mismatch caught
- KV cache data length validation
- Conv buffer length validation
- Fingerprint mismatch on layer_count, hidden_features, vocab_size, eps, layer types

**session.rs (4 tests):**
- Step-by-step generation produces correct token count
- Zero max_new_tokens → immediately finished
- Checkpoint roundtrip preserves generated tokens
- Two independent sessions with same config produce identical tokens

### 33.5 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/session.rs` | New file: `GenerationSession` with `new()`, `resume()`, `next_token()`, `checkpoint()` |
| `llama-rs/src/e2e/checkpoint.rs` | New file: `GenerationCheckpoint`, `CheckpointV1`, `ModelFingerprint`, `LayerStateDto`, save/load, validation |
| `llama-rs/src/e2e.rs` | Added `mod session`, `mod checkpoint`, public re-exports |
| `llama-rs/src/lib.rs` | Added `GenerationSession`, `GenerationCheckpoint` to re-exports |
| `llama-rs/src/e2e/error.rs` | Added `CheckpointDeserialize`, `CheckpointVersionMismatch`, `CheckpointModelMismatch` variants |

---

## Item 34: PersistentDecodeResources — unified session decode optimization

### 34.1 Problem

The `two_phase_loop` function built persistent resources (projections, KV
caches, scoring context, linear scratch, MLPs) using ~170 lines of inline
resource management with complex ownership (5 separate `_ctx` variables,
2 fallback branches, manual lifetime transmutes scattered across the
function body).

Meanwhile, `GenerationSession.step_two_phase()` used **none** of these
optimizations — every decode step rebuilt attention graphs, re-uploaded
weights, and computed MLPs ephemerally, resulting in an estimated
20-200x slowdown per token compared to the standalone generation path.

### 34.2 Design

**`PersistentDecodeResources`** encapsulates all GPU-resident state in a
single struct with safety-critical field ordering:

```
Resources (dropped first — top of struct):
  scoring_ctx: Option<PersistentScoringContext>
  linear_scratch: Option<LinearDecodeScratch>
  persistent_mlps: Vec<Option<PersistentMlp<'static>>>
  decode_projs: Option<Vec<Option<PersistentDecodeProjection<'static>>>>
  kv_caches: Vec<Option<PersistentKvCache<'static>>>
  lm_head: LmHeadResources  (self-contained; see item 35)

Contexts (dropped last — bottom of struct):
  _mlp_ctxs, _proj_ctxs, _kv_ctxs
```

**Granular optionality**: LM head is always present (try_build returns
None if it fails). All other resource groups are independently optional.
Failure in one category (e.g., KV caches) does not prevent others
(e.g., MLPs) from being used.

**`decode_step()` fallback**: When `decode_projs` is None, falls back
to `process_all_layers` with `DecodeStrategy` but still uses persistent
MLPs — partial optimization rather than all-or-nothing.

**`'static` transmute pattern**: All persistent types use
`PhantomData<&'ctx Context>` (raw pointers, not real references).
The transmute erases unnameable local lifetimes. Sound because struct
field ordering ensures contexts outlive their derived handles.

### 34.3 Changes to `two_phase_loop`

Before: ~170 lines of inline resource building with 2 branches
(persistent projections succeed -> full fast path; fail -> fallback with
persistent MLPs only). Duplicate LM head build (one for prefill sampling,
one for decode loop).

After: ~50 lines. Resources built once via `try_build()`, shared between
prefill sampling and decode loop. Single fallback path using
`graph_sample_fallback()` for CPU-only greedy sampling when persistent
LM head unavailable.

### 34.4 GenerationSession integration

- Added `persistent_resources: Option<PersistentDecodeResources>` field
  **before** `backend` field (drop ordering: resources drop before backend)
- Lazy initialization via `ensure_persistent_resources()`:
  - After prefill completes (first `next_token()` call)
  - On first decode step after checkpoint resume (if `prefill_done = true`)
- KV caches seeded from host prefill state after build
- Fallback: if `try_build()` returns None, session uses the slow
  `DecodeStrategy` + `sample_next()` path (existing behavior preserved)

### 34.5 New helper: `graph_sample_fallback`

CPU-only greedy sampling for when persistent LM head is unavailable:
`rms_norm_with_weight` then `greedy_next_token_id`. Used in both
`two_phase_loop` fallback and as conceptual reference for the session
slow path.

### 34.6 Test results

All 218 tests pass (0 failures, 7 ignored). Session tests exercise the
new `persistent_resources` field initialization and fallback path
(test sessions use CPU backend with tiny models where `try_build()` may
succeed or fail depending on backend capabilities).

### 34.7 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/generation.rs` | Added `PersistentDecodeResources` struct (~220 lines), `graph_sample_fallback()`, refactored `two_phase_loop` to use unified resources |
| `llama-rs/src/e2e/session.rs` | Added `persistent_resources` field, `ensure_persistent_resources()`, refactored `step_two_phase()` to use persistent resources with lazy init |

## Item 35: LmHeadResources extraction — self-contained LM head struct

### 35.1 Problem

The LM head graph resources (ggml context, buffer, input/output tensors,
compute graph) were duplicated in two places:

1. **`PersistentDecodeResources`**: 6 inline fields (`lm_x_in`, `lm_logits_t`,
   `lm_graph`, `_lm_buffer`, `_lm_ctx`) with transmute logic in `try_build()`.
2. **`full_reprocess_loop`**: ~20 lines of inline LM head construction with
   separate error handling (different error messages, no transmute needed).

The `graph_sample_at` helper was also a thin wrapper around
`lm_head_sample_step` with offset calculation — duplicated with
`PersistentDecodeResources::sample_token()`.

### 35.2 Design

**`LmHeadResources`** — self-contained struct owning all LM head graph state:

```
LmHeadResources
├── x_in: Tensor<'static, f32>       (graph input — dropped first)
├── logits_t: Tensor<'static, f32>   (graph output — dropped first)
├── graph: Graph<'static>            (compute graph — dropped first)
├── _buffer: BackendBuffer<'static>  (device memory — dropped second)
└── _ctx: Context                    (ggml context — dropped last)
```

- `try_build()` takes model dimensions + weight slices + backend, returns
  `Option<Self>` (None on any failure).
- `sample_hidden(&mut self, hidden_state: &[f32], backend: &Backend) -> Result<i32>`
  runs one sampling step through the graph.
- Drop ordering (declaration order) ensures soundness: tensors and graph
  reference `_ctx` memory, so they drop before `_ctx` and `_buffer`.

### 35.3 Changes

| Area | Before | After |
|------|--------|-------|
| `PersistentDecodeResources` | 6 inline LM head fields + inline transmute | Embedded `lm_head: LmHeadResources` field |
| `PersistentDecodeResources::try_build()` | ~25 lines of LM head construction | `LmHeadResources::try_build(...)?` (1 line) |
| `PersistentDecodeResources::sample_token()` | Called `lm_head_sample_step` directly | Delegates to `self.lm_head.sample_hidden()` |
| `full_reprocess_loop` | ~20 lines inline LM head construction | `LmHeadResources::try_build(...)` with fallback |
| `graph_sample_at` helper | Standalone function | **Removed** — replaced by `LmHeadResources::sample_hidden()` |

### 35.4 Behavioral improvement

`full_reprocess_loop` now has graceful fallback: if `LmHeadResources::try_build()`
returns `None` (e.g., on a backend that cannot allocate the graph), the loop falls
back to `graph_sample_fallback()` (host-side norm + matmul + argmax). Previously
it would return `Err` immediately.

### 35.5 Test results

All 222 tests pass (0 failures, 7 ignored). No behavioral changes — only
internal struct organization.

### 35.6 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/generation.rs` | Added `LmHeadResources` struct (~70 lines), embedded in `PersistentDecodeResources`, refactored `full_reprocess_loop`, removed `graph_sample_at` |

---

## 36. Attention dispatch helpers — `is_standard()` predicate

### 36.1 Goal

Reduce repetitive `matches!(p.attention, Some(AttentionLayerPlan::Standard(_)))`
patterns. Provide a predicate method on `AttentionLayerPlan` for boolean-query
use sites while preserving exhaustive pattern matching at dispatch sites.

### 36.2 Analysis

Explore agent identified 15+ match sites on `AttentionLayerPlan` across 5 files
(~40 individual match arms). Categorized as:

| Category | Count | Refactorable? |
|----------|-------|---------------|
| DISPATCH | 3 | ✅ via trait (Phase 2, deferred) |
| PAIRED_DISPATCH | 4 | ⚠️ Partial (borrow checker) |
| CONSTRUCTION | 5+ | ❌ Keep as-is (structural) |
| PREDICATE_CHECK | 2 | ✅ Done — `is_standard()` |
| QUERY_ACCESSOR | 1 | Already a method |

### 36.3 Implementation

Added `is_standard()` method to `AttentionLayerPlan` in `plan.rs`:

```rust
pub(super) fn is_standard(&self) -> bool {
    matches!(self, Self::Standard(_))
}
```

Applied at 2 predicate-check sites in `generate_from_plans()`:

```rust
// Before:
.any(|p| matches!(p.attention, Some(AttentionLayerPlan::Standard(_))))

// After:
.any(|p| p.attention.as_ref().is_some_and(|a| a.is_standard()))
```

**Important**: Dispatch match arms (e.g., `try_build_persistent_projections`)
keep explicit `AttentionLayerPlan::Standard(_)` patterns. Using `is_standard()`
as a match guard would require a catch-all `_ =>` arm, weakening exhaustiveness
checking — the compiler would no longer force updating the match when new
variants are added.

### 36.4 Phase 2 (deferred)

Trait dispatch for `InferenceStrategy::process_attention()` would replace 3
match arms with per-variant methods. Deferred because only 3 variants exist
and new variants are infrequent.

### 36.5 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/plan.rs` | Added `is_standard()` method to `AttentionLayerPlan` |
| `llama-rs/src/e2e/generation.rs` | Replaced 2 `matches!()` predicates with `is_standard()` |

---

## Item 37 — Dead Code Cleanup

**Commit:** `f63af61 refactor(llama-rs): remove unused _hidden_features param + demote lm_head_graph to #[cfg(test)] (item 37)`

### 37.1 Motivation

The explore-agent analysis (item 38 prep) flagged two dead-code issues:

1. `process_all_layers` had an unused `_hidden_features` parameter (8 args, 7 call sites).
2. The one-shot `lm_head_graph` function was superseded by `LmHeadResources` (item 35) but still
   `pub(super)` — only test/bench code used it.

### 37.2 Changes

- **`process_all_layers`**: Removed `_hidden_features: usize` from signature and all 7 call sites
  (4 in `generation.rs`, 3 in `session.rs`). Function now has 7 parameters (at clippy limit).
- **`lm_head_graph`**: Demoted from `pub(super)` to `#[cfg(test)]`. Still used by parity tests
  and `bench_graphs.rs` benchmarks.

### 37.3 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/tensor_ops.rs` | Removed `_hidden_features` from `process_all_layers`, gated `lm_head_graph` with `#[cfg(test)]` |
| `llama-rs/src/e2e/generation.rs` | Updated 4 `process_all_layers` call sites |
| `llama-rs/src/e2e/session.rs` | Updated 3 `process_all_layers` call sites |

---

## Item 38 — Tuple-to-Struct for Persistent Graph Builders

**Commit:** `0625a25 refactor(llama-rs): replace tuple returns with named structs for persistent graph builders (item 38)`

### 38.1 Motivation

Three persistent graph builder functions returned large anonymous tuples (5, 12, and 14 elements).
Two required `#[allow(clippy::type_complexity)]` annotations. Positional destructuring was fragile
and unreadable — swapping two fields of the same type would compile silently but produce wrong results.

### 38.2 New Structs

```rust
pub(super) struct LmHeadGraphParts<'ctx> {
    pub w_out: Tensor<'ctx, f32>,
    pub norm_w: Tensor<'ctx, f32>,
    pub x_in: Tensor<'ctx, f32>,
    pub logits: Tensor<'ctx, f32>,
    pub graph: Graph<'ctx>,
}

pub(super) struct FullAttentionGraphParts<'ctx> {
    pub x_in: Tensor<'ctx, f32>,
    pub w_q: Tensor<'ctx, f32>,
    pub w_k: Tensor<'ctx, f32>,
    pub w_v: Tensor<'ctx, f32>,
    pub q_out: Tensor<'ctx, f32>,
    pub k_out: Tensor<'ctx, f32>,
    pub v_out: Tensor<'ctx, f32>,
    pub input_graph: Graph<'ctx>,
    pub out_x: Tensor<'ctx, f32>,
    pub w_out: Tensor<'ctx, f32>,
    pub out_y: Tensor<'ctx, f32>,
    pub output_graph: Graph<'ctx>,
}

pub(super) struct LinearAttentionGraphParts<'ctx> {
    pub x_in: Tensor<'ctx, f32>,
    pub w_qkv: Tensor<'ctx, f32>,
    pub w_z: Tensor<'ctx, f32>,
    pub w_alpha: Tensor<'ctx, f32>,
    pub w_beta: Tensor<'ctx, f32>,
    pub qkv_out: Tensor<'ctx, f32>,
    pub z_out: Tensor<'ctx, f32>,
    pub alpha_out: Tensor<'ctx, f32>,
    pub beta_out: Tensor<'ctx, f32>,
    pub input_graph: Graph<'ctx>,
    pub out_x: Tensor<'ctx, f32>,
    pub w_out: Tensor<'ctx, f32>,
    pub out_y: Tensor<'ctx, f32>,
    pub output_graph: Graph<'ctx>,
}
```

### 38.3 Call Site Migration Pattern

```rust
// Before (fragile positional destructuring):
let (x_in, w_q, w_k, w_v, q_out, k_out, v_out, input_graph,
     out_x, w_out, out_y, output_graph) =
    build_persistent_full_attention_graphs(&ctx, ...)?;
w_q.write_data_backend(&attn.q_weight_values)?;

// After (named field access):
let g = build_persistent_full_attention_graphs(&ctx, ...)?;
g.w_q.write_data_backend(&attn.q_weight_values)?;
```

### 38.4 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/tensor_ops.rs` | Added 3 structs, updated 3 builder return types, updated 3 test call sites |
| `llama-rs/src/e2e/generation.rs` | Updated 3 call sites (LmHeadResources::try_build, build_one_persistent_full, build_one_persistent_linear) |
| `llama-rs/src/e2e/bench_graphs.rs` | Updated lm_head benchmark call site |

---

## Item 39 — Inline `persistent_decode_all_layers` as Method

**Commit:** `5ac76a7 refactor(llama-rs): inline persistent_decode_all_layers into PersistentDecodeResources method (item 39)`

### 39.1 Motivation

`persistent_decode_all_layers` was a free function taking **11 parameters**, 5 of which were
fields of `PersistentDecodeResources`. Clippy flagged it as `too_many_arguments(11/7)`.
Converting to a method on `PersistentDecodeResources` eliminates 5 explicit params via `self`.

### 39.2 Before → After

```rust
// Before (free function, 11 params):
fn persistent_decode_all_layers(
    hidden: &mut Vec<f32>,
    layer_plans: &[LayerPlan],
    state: &mut GenerationState,
    hidden_features: usize,
    rms_norm_eps: f32,
    backend: &Backend,
    projections: &[Option<PersistentDecodeProjection<'static>>],
    kv_caches: &[Option<PersistentKvCache<'static>>],
    persistent_mlps: &[Option<PersistentMlp<'static>>],
    scoring_ctx: &Option<PersistentScoringContext>,
    linear_scratch: &mut Option<LinearDecodeScratch>,
) -> Result<(), E2eError>

// After (method, 6 params — 5 come from self):
impl PersistentDecodeResources {
    fn persistent_decode_all_layers(
        &mut self,
        hidden: &mut Vec<f32>,
        layer_plans: &[LayerPlan],
        state: &mut GenerationState,
        hidden_features: usize,
        rms_norm_eps: f32,
        backend: &Backend,
    ) -> Result<(), E2eError>
}
```

### 39.3 Call Site Change

```rust
// Before:
persistent_decode_all_layers(
    hidden, layer_plans, state, hidden_features, rms_norm_eps, backend,
    &projs, &self.kv_caches, &self.persistent_mlps,
    &self.scoring_ctx, &mut self.linear_scratch,
)?;

// After:
self.persistent_decode_all_layers(
    hidden, layer_plans, state, hidden_features, rms_norm_eps, backend,
)?;
```

### 39.4 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/generation.rs` | Moved function into `impl PersistentDecodeResources`, updated `decode_step` call site |

---

## Item 40 — Extract `RopeParams` Struct

**Commit:** `00d9b51 refactor(llama-rs): extract RopeParams struct from apply_neox_rope_in_place (item 40)`

### 40.1 Motivation

`apply_neox_rope_in_place` took **8 parameters**, 4 of which were logically grouped RoPE
configuration values. Clippy flagged `too_many_arguments(8/7)`.

### 40.2 New Struct

```rust
#[derive(Debug, Clone, Copy)]
pub(super) struct RopeParams {
    pub n_rot: usize,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub position_offset: usize,
}
```

### 40.3 Signature Change

```rust
// Before (8 params):
fn apply_neox_rope_in_place(
    data: &mut [f32], head_dim: usize, n_heads: usize, seq_len: usize,
    n_rot: usize, freq_base: f32, freq_scale: f32, position_offset: usize,
) -> Result<(), E2eError>

// After (5 params):
fn apply_neox_rope_in_place(
    data: &mut [f32], head_dim: usize, n_heads: usize, seq_len: usize,
    rope: RopeParams,
) -> Result<(), E2eError>
```

### 40.4 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/attention.rs` | Added `RopeParams` struct, updated function signature, updated 2 production call sites + 7 test call sites |

---

## Item 41 — Further Clippy `too_many_arguments` Reduction

**Commit:** `f9b6e36 refactor(llama-rs): reduce clippy too_many_arguments in project_qkv_graph and try_build (item 41)`

### 41.1 Motivation

Two remaining `too_many_arguments` warnings in llama-rs:
1. `project_qkv_graph` (9 params): Accepted 3 individual weight slice params (`q_weights`, `k_weights`, `v_weights`).
2. `PersistentDecodeResources::try_build` (8 params): Accepted raw weight slices + dimensions to build `LmHeadResources` internally.

### 41.2 `project_qkv_graph` Change (9→8 params)

Replaced 3 individual weight slice params with a single `&Qwen35FullAttentionLayerPlan` reference.
The function extracts `q_weight_values`, `k_weight_values`, `v_weight_values` from the plan.

### 41.3 `try_build` Change (8→5 params)

Moved LM head construction to the caller. `try_build` now accepts a pre-built `LmHeadResources`
instead of raw `hidden_features`, `vocab_size`, `output_weight_values`, `output_norm_values`.
Return type changed from `Option<Self>` to `Self` (LM head failure now handled by caller).

```rust
// Before (8 params, returns Option<Self>):
fn try_build(
    layer_plans, hidden_features, vocab_size, rms_norm_eps,
    total_seq_len, output_weight_values, output_norm_values, backend,
) -> Option<Self>

// After (5 params, returns Self):
fn try_build(
    layer_plans, lm_head: LmHeadResources, rms_norm_eps,
    total_seq_len, backend,
) -> Self
```

Callers build `LmHeadResources` first, then use `Option::map` to create resources:
```rust
let mut resources = LmHeadResources::try_build(...)
    .map(|lm_head| PersistentDecodeResources::try_build(layer_plans, lm_head, ...));
```

### 41.4 Result

All `too_many_arguments` warnings eliminated from `llama-rs` crate. Remaining warnings are
in `ggml-rs` (`flash_attn_ext`: 8/7, ggml API surface) and test helpers (low priority).

### 41.5 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/attention.rs` | `project_qkv_graph` accepts `&Qwen35FullAttentionLayerPlan` instead of 3 weight slices |
| `llama-rs/src/e2e/generation.rs` | `try_build` accepts `LmHeadResources`, returns `Self`; `two_phase_loop` caller updated |
| `llama-rs/src/e2e/session.rs` | `ensure_persistent_resources` builds LM head first, uses `Option::map` pattern |

---

## Item 42 — `QkvProjections` Struct (type_complexity elimination)

**Commit:** `48f8b99 refactor(llama-rs): extract QkvProjections struct to eliminate type_complexity warning (item 42)`

### 42.1 Motivation

`project_qkv_graph` returned `Result<(Vec<f32>, Vec<f32>, Vec<f32>), E2eError>`. Clippy
flagged this as `type_complexity`. Positional tuple access (`result.0`, `result.1`, etc.)
was also error-prone.

### 42.2 New Struct

```rust
#[derive(Debug)]
struct QkvProjections {
    q_full: Vec<f32>,   // [H * D * 2, T] — Q + gate interleaved
    k_proj: Vec<f32>,   // [Hkv * D, T] — K projection
    v_proj: Vec<f32>,   // [Hkv * D, T] — V projection
}
```

### 42.3 Result

- Eliminated `type_complexity` warning
- Both branches (graph and host-side fallback) construct `QkvProjections` directly
- Call site uses `qkv.q_full`, `qkv.k_proj`, `qkv.v_proj` (self-documenting)
- **llama-rs crate now has ZERO clippy warnings**

### 42.4 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/attention.rs` | Added `QkvProjections` struct, updated return type + call site |

---

## Item 43 — `FullAttentionDims` Struct (dimension consolidation)

**Commit:** `dbbb883 refactor(llama-rs): extract FullAttentionDims struct from fully_fused_attention_graph (item 43)`

### 43.1 Motivation

`fully_fused_attention_graph` (338 lines, 7 params) inlined ~30 lines of dimension validation
and memory estimation at the top of the function. The same dimension pattern (`d, h, hkv, hidden,
qf, qf2, kvf`) appeared in at least 4 places across attention.rs. Consolidating into a struct:
- Makes validation reusable across functions
- Eliminates inline dimension derivation from weight tensor sizes
- Replaces ad-hoc memory estimation with a named method

### 43.2 New Struct

```rust
#[derive(Debug, Clone, Copy)]
struct FullAttentionDims {
    d: usize,      // Per-head feature dimension (D)
    h: usize,      // Number of query/gate heads (H)
    hkv: usize,    // Number of KV heads (Hkv ≤ H, for GQA)
    hidden: usize,  // Model hidden size (H * D, from output weight matrix)
    qf: usize,     // Total query features (H * D)
    qf2: usize,    // Q+gate interleaved features (H * D * 2)
    kvf: usize,    // KV features per tensor (Hkv * D)
}
```

### 43.3 Key Methods

- `FullAttentionDims::new(attention)` — validates GQA divisibility, derives hidden size from
  output weight matrix, returns `Result<Self, E2eError>`
- `estimate_memory(t)` — conservative memory estimate for the fully-fused attention graph:
  weight tensors + data tensors + f16 causal mask + 1 MB overhead headroom

### 43.4 Integration

`fully_fused_attention_graph` now starts with:
```rust
let dims = FullAttentionDims::new(attention)?;
let FullAttentionDims { d, h, hkv, hidden, qf, qf2, kvf } = dims;
```
replacing ~30 lines of inline validation + memory calculation with 2 lines + a destructure.
The function body continues to use the destructured locals — no `dims.` prefix noise.

### 43.5 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/attention.rs` | Added `FullAttentionDims` struct + `new()` + `estimate_memory()`, updated `fully_fused_attention_graph` |

---

## Item 44 — `LinearAttentionDims` Struct (dimension consolidation for linear attention)

**Commit:** `57d949b refactor(llama-rs): extract LinearAttentionDims struct from project_and_conv_fused_graph (item 44)`

### 44.1 Motivation

`project_and_conv_fused_graph` (251 lines, 8 params) had `#[allow(clippy::too_many_arguments)]`.
The same dimension derivation (`hidden_features` from output weight matrix, `conv_channels` from
`inner_size + 2 * group_count * state_size`) was duplicated in three places:
- `qwen35_linear_attention_core` (inline derivation)
- `project_linear_inputs` (inline derivation)
- `project_and_conv_fused_graph` (received as params)

### 44.2 New Struct

```rust
#[derive(Debug, Clone, Copy)]
struct LinearAttentionDims {
    hidden: usize,         // Hidden features (H) — from output weight matrix
    inner_size: usize,     // Inner size (IS) — from plan
    conv_channels: usize,  // IS + 2*G*S — total conv channel width
    time_step_rank: usize, // Timestep rank (R) — from plan
    kernel_size: usize,    // Conv kernel size (K) — from plan
}
```

### 44.3 Key Methods

- `LinearAttentionDims::new(attention)` — reuses existing `linear_attention_hidden_features` +
  `linear_attention_conv_channels` helpers for validation
- `estimate_fused_memory(seq_len)` — replaces ~40 lines of inline memory estimation in
  `project_and_conv_fused_graph` (projection matmul memory + conv intermediates + slack)

### 44.4 Changes

| What | Before | After |
|------|--------|-------|
| `project_and_conv_fused_graph` params | 8 (with `#[allow]`) | 7 (no `#[allow]`) |
| `qwen35_linear_attention_core` dim derivation | 16 lines inline | `LinearAttentionDims::new(attention)?` |
| `project_linear_inputs` dim derivation | 13 lines inline | `LinearAttentionDims::new(attention)?` |
| `project_linear_inputs_graph` | spurious `#[allow]` (only 6 params) | `#[allow]` removed |

### 44.5 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/linear_attention.rs` | Added `LinearAttentionDims` struct + methods; refactored 3 callers; removed 2 `#[allow]` annotations |

---

## Item 45 — Remove Stale `#[allow(clippy::...)]` Annotations

### 45.1 Motivation

After items 42–44 extracted dimension structs and reduced parameter counts,
three `#[allow(clippy::...)]` annotations became stale:

| File | Function | Annotation | Reason stale |
|------|----------|-----------|--------------|
| `tensor_ops.rs` | `build_lm_head_graph` | `type_complexity` | Already returns `LmHeadGraphParts<'ctx>` struct |
| `attention.rs` | `fully_fused_attention_graph` | `too_many_arguments` | Now 7 params (limit 7, warns at 8+) |
| `linear_attention.rs` | `qwen35_linear_attention_core` | `too_many_arguments` | Now 7 params |

### 45.2 Changes

- Removed all three `#[allow]` lines
- Fixed `build_lm_head_graph` doc comment: was "Returns `(x_input_tensor, logits_tensor, graph)`",
  now correctly references `LmHeadGraphParts`

### 45.3 Remaining `#[allow]` in llama-rs

Only `ssm_recurrence_step` in `linear_attention.rs` retains `#[allow(clippy::too_many_arguments)]`
(10 params). This is a math kernel whose signature directly mirrors the recurrence formula —
bundling parameters into a struct would obscure the mathematical relationship.

---

## Item 46 — Extract Projection Result Structs

### 46.1 Motivation

Two functions in `tensor_ops.rs` had `#[allow(clippy::type_complexity)]` because
they returned multi-element tuples:

```rust
// Before
fn read_full_attention_projections(&self) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), E2eError>
fn read_linear_attention_projections(&self) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), E2eError>
```

### 46.2 Design Decision — Reuse vs New Type

**Full attention**: `QkvProjections { q_full, k_proj, v_proj }` already existed in
`attention.rs` (item 42) with identical fields. Per rubber-duck recommendation,
made it `pub(super)` and reused it from `tensor_ops.rs`, avoiding a duplicate type.

**Linear attention**: `LinearProjections` exists but carries extra dimension fields
(`conv_channels`, `hidden_features`) that the raw GPU readback doesn't have.
Creating dummy values would make the type lie. Instead, introduced a minimal
`RawLinearProjections { qkv, z, alpha, beta }` — the caller in `generation.rs`
adds the derived dimensions when constructing the full `LinearProjections`.

### 46.3 After

```rust
// After
fn read_full_attention_projections(&self) -> Result<QkvProjections, E2eError>
fn read_linear_attention_projections(&self) -> Result<RawLinearProjections, E2eError>
```

### 46.4 Clippy `#[allow]` Status After Item 46

| Scope | `type_complexity` | `too_many_arguments` |
|-------|-------------------|---------------------|
| llama-rs production | **0** | **1** (`ssm_recurrence_step`, deliberate) |
| ggml-rs | 0 | 1 (`flash_attn_ext`, ggml API surface) |

### 46.5 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/tensor_ops.rs` | Removed 2 stale `#[allow(type_complexity)]`; added `RawLinearProjections` struct; updated return types + tests |
| `llama-rs/src/e2e/attention.rs` | Made `QkvProjections` `pub(super)` with `#[derive(Debug)]`; removed stale `#[allow(too_many_arguments)]` |
| `llama-rs/src/e2e/linear_attention.rs` | Removed stale `#[allow(too_many_arguments)]` |
| `llama-rs/src/e2e/generation.rs` | Updated callers to use struct destructuring |

---

## Item 47 — `sum_matmul_memories` Helper

### 47.1 Motivation

`recommended_persistent_full_attention_memory` and `recommended_persistent_linear_attention_memory`
in `tensor_ops.rs` follow identical patterns: compute `recommended_backend_matmul_memory` for
each weight×input pair, chain `checked_add`, add slack. The only differences are the number
of projections (4 vs 5) and their dimensions.

### 47.2 Implementation

Extracted `sum_matmul_memories(projections: &[(Shape2D, Shape2D, &'static str)])`:
- Uses `try_fold` for clean accumulation with `checked_add`
- Adds `PROJECTION_SLACK_BYTES * 2` at the end
- Each caller now passes a declarative slice of `(weight_shape, input_shape, label)` tuples

### 47.3 Before/After

```rust
// Before: 4 × (3 lines of memory query + map_err) + 5 lines of checked_add chain = ~17 lines
// After: declarative 4-element slice, 1 function call = ~6 lines
```

Both `recommended_persistent_*_memory` functions reduced from ~35 lines each to ~10 lines.

---

## Item 48 — `upload_weight` Helper

### 48.1 Motivation

`build_one_persistent_full` and `build_one_persistent_linear` in `generation.rs` each
have 4-5 repetitive `write_data_backend` + `map_err` calls for weight upload.

### 48.2 Implementation

Extracted `upload_weight(tensor: &Tensor<f32>, data: &[f32], label: &'static str)`:
- Preserves per-weight error labels (`write<W_Q>(pfa)`, `write<W_ALPHA>(pla)`, etc.)
- Each upload call reduced from 3 lines to 1 line

### 48.3 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/tensor_ops.rs` | Added `sum_matmul_memories`; refactored both `recommended_persistent_*_memory` to use it |
| `llama-rs/src/e2e/generation.rs` | Added `upload_weight`; refactored both `build_one_persistent_*` to use it |

---

## Item 49 — `OutputProjectionGraph` Sub-Struct Extraction

### 49.1 Motivation

`FullAttentionGraphParts` and `LinearAttentionGraphParts` each contain identical output
projection fields (`out_x`, `w_out`, `out_y`, `output_graph`) and their builders duplicate
~15 lines of `new_tensor_2d` → `mul_mat` → `new_graph` → `build_forward_expand` code.
The `PersistentDecodeProjection` enum also duplicates these 4 fields in both variants.

### 49.2 Implementation

Extracted `OutputProjectionGraph<'ctx>` struct with fields `{ w, x, y, graph }` and a
`build_output_projection_graph(ctx, input_features, output_features, label)` helper.

Both `FullAttentionGraphParts` and `LinearAttentionGraphParts` now contain an
`output: OutputProjectionGraph<'ctx>` field instead of 4 separate fields. The
`project_output` method on `PersistentDecodeProjection` now extracts the shared
sub-struct via a single match, eliminating the 3-field multi-arm destructure.

### 49.3 Files Modified

| File | Changes |
|------|---------|
| `llama-rs/src/e2e/tensor_ops.rs` | Added `OutputProjectionGraph`, `build_output_projection_graph`; refactored both `*GraphParts` structs, both builders, `PersistentDecodeProjection` enum, `project_output`, and test callers |
| `llama-rs/src/e2e/generation.rs` | Updated `build_one_persistent_full` and `build_one_persistent_linear` to use `g.output.w` and compose `output: g.output` |
| `llama-rs/Cargo.toml` | Added `postcard` + `serde` dependencies |