# Ongoing execution policy (review_1 + review_3 complete, parity follow-up active)

This repository is currently following a strict execution policy to avoid losing intent across long refactor loops.

## Current branch

- `exp/oh-my` — dedicated branch for review_1 + review_3 refactor of `ggml-rs`.

## Immediate priority

1. Close the remaining Qwen3.5 strict token-id parity gap in `llama-rs`.
2. Keep the `ggml-rs` review_1/review_3 refactor branch validated while parity work proceeds.
3. Merge back to `main` only after validation and runtime checks pass.

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

- Qwen3.5 strict parity: head-group mapping bug identified and fixed.
  - Root cause: Q/K head-to-group mapping used integer division (`head / repeat_factor`)
    producing an interleaved pattern `[g0,g0,g1,g1,...]`, while llama.cpp's `ggml_repeat_4d`
    tiles block-by-block `[g0,g1,...,gN,g0,g1,...,gN]`.
  - Fix: changed to `head % group_count` to match upstream tiled repeat semantics.
  - Added shape invariant assertions (`time_step_rank % group_count == 0`,
    `inner_size == time_step_rank * state_size`) and a regression test.
  - `causal_depthwise_conv` was verified correct for initial-prompt (zero-state) case.
  - Strict parity rerun pending after this fix.

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
