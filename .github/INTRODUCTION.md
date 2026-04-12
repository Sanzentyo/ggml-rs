# Ongoing execution policy

This repository is currently following a strict execution policy to avoid losing intent across long refactor loops.

**Docs-update policy**: After every significant change (API refactor, bug fix, feature addition), update this file and relevant docs under `./docs/` to reflect the new state. Read all markdown in `.github/` and `./docs/` at session start.

## Skills you should use
- rust-best-practices
And you should write rusty code(ADT, enum, type state pattern)

## Current branch

- `exp/oh-my` — dedicated branch for review_1 + review_3 refactor of `ggml-rs`.

## Immediate priority

1. ~~Close the remaining Qwen3.5 strict token-id parity gap in `llama-rs`.~~ **DONE** — parity achieved.
2. ~~Expand `Type` enum to all ggml types, seal `HostElement`, update decode APIs.~~ **DONE** — zero clippy warnings.
3. ~~Implement MRoPE for full attention layers (required for multi-token prompts).~~ **DONE** — multi-token parity achieved.
4. ~~Causal depthwise conv & QKV packing comparison.~~ **DONE** — documented in `docs/llama-rs/worklog/2026-04-13-conv-qkv-comparison.md`.
5. ~~Continue review_3 refactor items (generic inference, ND tensor, semantic wrapper dedup).~~ **10/12 DONE** — test coverage 122 tests (zero warnings), backend examples exist.
6. ~~Autoregressive decode state management (prefill/decode split).~~ **DONE** — KV cache for full attention, conv buffer + SSM states for linear attention, decode equivalence tests pass.
7. ~~Two-phase generation loop (prefill + incremental decode).~~ **DONE** — generation.rs branches on layer types: all-Qwen3.5 → two-phase (prefill all prompt tokens, then decode one-at-a-time), otherwise → full-reprocess fallback.
8. Backend example enhancement (review_3 item 11) + README (item 12).
9. Merge back to `main` only after validation and runtime checks pass.

## Completed refactor items

- `AsRef<str>` for GGUF string arguments (`find_key`, `kv_value_by_key`, `set_value`, `remove_key`).
- `TryFromGgufValue` trait and `kv_value_as::<T>()` convenience method on `GgufFile`.
- `GgufTypeMismatch` error variant for type-safe GGUF value extraction.
- `Tensor<'ctx, T>` typestate pattern and `DynTensor<'ctx>` runtime-typed handle.
- `TensorExpr<'ctx, T>` typed expression wrapper.
- `rope_ext_with_i32_positions` mixed-type RoPE helper for `f32` data + `i32` positions.
- Backend-path / ND tensor / error-path test expansion.
- `llama-rs` migration to the typed `Tensor<'ctx, T>` / `DynTensor<'ctx>` API.
- **Type consolidation**: `Type` expanded from 2 variants (F32, I32) to all 32+ ggml
  tensor types (quantized Q4_K, Q8_0, etc. + native floats/ints + Unknown(i32)).
  `GgufTensorInfo` now stores `ggml_type: Type` instead of raw `i32` + `String`.
  Decode APIs (`decode_tensor_data_to`, `tensor_element_count`) accept `Type`.
  `HostElement` sealed via private `Sealed` supertrait (eliminates `private_bounds` warning).
- **e2e.rs module split**: Monolithic 2412-line file split into 13 focused submodules
  (error, config, numeric, tensor_ops, resolve, decode, plan, planner, attention,
  linear_attention, mlp, generation, state). Public API unchanged.
- **Autoregressive decode infrastructure**: `state.rs` with `Qwen35FullAttentionState`
  (KV cache), `LinearAttentionState` (conv buffer + SSM states), `GenerationState`.
  `attention.rs` gains `qwen35_full_attention_prefill` + `decode_step`.
  `linear_attention.rs` gains `qwen35_linear_attention_prefill` + `decode_step` +
  `causal_depthwise_conv_decode_step`. RoPE `position_offset` parameter added.
  Decode equivalence tests verify prefill+decode = full reprocess.
- **Two-phase generation loop**: `generation.rs` branches on layer types: all-Qwen3.5
  → prefill + incremental decode (one token at a time with cached state); Standard
  attention present → full-reprocess fallback. Handles `max_new_tokens==0`, EOS on
  first generated token, MLP-only layers.

## Validation checkpoints completed on this branch

- `cargo fmt --all`
- `cargo clippy --workspace --all-targets`
- `cargo test --workspace`
- `cargo test --features link-system`
- CPU perf gate: `cargo run --example bench_matmul --features link-system -- cpu -n 10`
  - current checkpoint result: `avg=0.256 ms`

## Current parity investigation status

- **Qwen3.5 strict parity: ACHIEVED** (single-token AND multi-token, up to 5-token prompts).
  - Single-token: prompt `[1]`, `max_new_tokens=1` → both produce `[5328]`.
  - Multi-token (single prompt): prompt `[3]`, `max_new_tokens=5` → both produce `[1088, 35790, 90, 16, 14728]`.
  - Multi-token (3 prompt tokens): prompt `[1,2,3]`, `max_new_tokens=5` → both produce `[31, 2, 5, 1, 271]`.
  - Multi-token (5 prompt tokens): prompt `[1,2,3,4,5]`, `max_new_tokens=5` → both produce `[6, 24218, 10, 4838, 1665]`.
  - Known precision edge: prompt `[5]` diverges at token 5 only (23 vs 24 — adjacent logits,
    numerical precision difference, not a systematic bug).
  - Bug 1 (linear attention): Head-group mapping used `head / repeat_factor` (interleaved),
    while llama.cpp's `ggml_repeat_4d` tiles block-by-block. Fixed to `head % group_count`.
  - Bug 2 (full attention): Q/gate split treated ggml's interleaved layout
    `[Q_h0(D), G_h0(D), Q_h1(D), ...]` as two flat halves with dim-major indexing.
    Fixed to head-major interleaved extraction: `head * 2D + dim` for Q, `head * 2D + D + dim` for gate.
  - `causal_depthwise_conv` verified correct (comparison documented).
  - Delta-net recurrence math verified correct.
  - NeoX RoPE for full attention implemented and verified at non-zero positions.
  - `causal_depthwise_conv` and QKV packing comparison completed — see
    `docs/llama-rs/worklog/2026-04-13-conv-qkv-comparison.md`.

## Already-implemented items (review_3 items done before this branch)

- `GgmlElement` trait with `f32`/`i32` impls.
- `GgmlType` trait mapping Rust types to `Type`.
- `BackendElement` / `HostElement` trait hierarchy.
- `with_context` / `with_no_alloc_context` scoped helpers.
- `Shape3D`, `Shape4D`, `Dims<const N>`, `new_tensor<const N>`, `rank()`, `dims()`, `shape_nd()`.
- `decode_tensor_data_to::<T>()` generic GGUF decode.
- `MaybeUninit` / `spare_capacity_mut` optimization on hot tensor readout paths.
- `define_semantic_usize!` macro for newtype dedup.
- Typed tensor wrappers (`Tensor1D`..`Tensor4D`).

## Safety rule

- If `unsafe` is required, include explicit safety comments proving why the usage is valid.

## Validation rule

- Always run:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
- Re-run CPU/Metal runtime smoke (`--features link-system`) for performance-sensitive paths.

## Project objective lock

- Keep `llama-rs` reproducing llama.cpp behavior on top of `ggml-rs` safe APIs.
- Improve `ggml-rs` architecture and performance in parallel.

## After parity closure

- Return to broader `llama-rs` trait/ADT continuation tasks.
