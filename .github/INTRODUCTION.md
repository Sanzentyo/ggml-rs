# Ongoing execution policy (review_1 + review_3 refactor)

This repository is currently following a strict execution policy to avoid losing intent across long refactor loops.

## Current branch

- `exp/oh-my` — dedicated branch for review_1 + review_3 refactor of `ggml-rs`.

## Immediate priority

1. Apply `docs/third_reviews/review_1.md` and `docs/third_reviews/review_3.md` refactor recommendations to `ggml-rs`.
2. Enforce no-regression performance gating (baseline vs post-change).
3. Iterate until performance is at least baseline (preferably improved).
4. Merge back to `main` only after validation and runtime checks pass.

## Completed refactor items

- `AsRef<str>` for GGUF string arguments (`find_key`, `kv_value_by_key`, `set_value`, `remove_key`).
- `TryFromGgufValue` trait and `kv_value_as::<T>()` convenience method on `GgufFile`.
- `GgufTypeMismatch` error variant for type-safe GGUF value extraction.

## Remaining refactor items (from review_3)

- `Tensor<'ctx, T>` typestate pattern — carry element type at compile time, eliminate runtime type-specific methods.
- `GgmlElement` / `GgmlType` trait unification with `Tensor<'ctx, T>`.
- `TensorExpr<'ctx, T>` typed expression wrapper.
- Backend path test coverage (CPU/Metal).
- ND tensor test coverage.
- Error path and boundary tests.

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

## After review refactor completion

- Return to `llama-rs` trait/ADT continuation tasks.
