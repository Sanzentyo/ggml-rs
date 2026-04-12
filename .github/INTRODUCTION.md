# After you read this introduction, you read markdown in directory allocated this md, then, you read some new markdown in ./docs dir. Then, update docs content and, unify these introduction. the intro should have policy that you update intro and docs content.

# Ongoing execution policy (review_1 + review_3 complete, parity follow-up active)

This repository is currently following a strict execution policy to avoid losing intent across long refactor loops.

## Skills you should use
- rust-best-practices
And you should write rusty code(ADT, enum, type state pattern)

## Current branch

- `exp/oh-my` — dedicated branch for review_1 + review_3 refactor of `ggml-rs`.

## Immediate priority

1. ~~Close the remaining Qwen3.5 strict token-id parity gap in `llama-rs`.~~ **DONE** — parity achieved.
2. Implement MRoPE for full attention layers (required for multi-token prompts).
3. Keep the `ggml-rs` review_1/review_3 refactor branch validated while parity work proceeds.
4. Merge back to `main` only after validation and runtime checks pass.

## Completed refactor items

- `AsRef<str>` for GGUF string arguments (`find_key`, `kv_value_by_key`, `set_value`, `remove_key`).
- `TryFromGgufValue` trait and `kv_value_as::<T>()` convenience method on `GgufFile`.
- `GgufTypeMismatch` error variant for type-safe GGUF value extraction.
- `Tensor<'ctx, T>` typestate pattern and `DynTensor<'ctx>` runtime-typed handle.
- `TensorExpr<'ctx, T>` typed expression wrapper.
- `rope_ext_with_i32_positions` mixed-type RoPE helper for `f32` data + `i32` positions.
- Backend-path / ND tensor / error-path test expansion.
- `llama-rs` migration to the typed `Tensor<'ctx, T>` / `DynTensor<'ctx>` API.

## Validation checkpoints completed on this branch

- `cargo fmt --all`
- `cargo clippy --workspace --all-targets`
- `cargo test --workspace`
- `cargo test --features link-system`
- CPU perf gate: `cargo run --example bench_matmul --features link-system -- cpu -n 10`
  - current checkpoint result: `avg=0.256 ms`

## Current parity investigation status

- **Qwen3.5 strict parity: ACHIEVED** (prompt `[1]`, `max_new_tokens=1`).
  - Both llama-rs and llama.cpp produce `[5328]`.
  - Bug 1 (linear attention): Head-group mapping used `head / repeat_factor` (interleaved),
    while llama.cpp's `ggml_repeat_4d` tiles block-by-block. Fixed to `head % group_count`.
  - Bug 2 (full attention): Q/gate split treated ggml's interleaved layout
    `[Q_h0(D), G_h0(D), Q_h1(D), ...]` as two flat halves with dim-major indexing.
    Fixed to head-major interleaved extraction: `head * 2D + dim` for Q, `head * 2D + D + dim` for gate.
  - `causal_depthwise_conv` verified correct. Delta-net recurrence math verified correct.
  - Known gap: RoPE (MRoPE) not yet implemented for full attention layers.
    At position 0, RoPE is identity, so single-token parity passes. Multi-token and
    autoregressive decode beyond step 0 will require RoPE implementation.

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
