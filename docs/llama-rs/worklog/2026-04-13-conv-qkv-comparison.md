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


