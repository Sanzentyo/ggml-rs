# llama-rs worklog

## Current session milestones

- Verified `ggml-rs` safe API matmul on CPU and Metal.
- Added local `llama.cpp` checkout for migration reference.
- Added structured migration docs:
  - `docs/llama-rs/README.md`
  - `docs/llama-rs/PARITY_MATRIX.md`
  - `docs/llama-rs/KNOWLEDGE_BASE.md`
- Created initial `llama-rs` crate skeleton and backend smoke example powered by `ggml-rs` safe API.
- Validated `llama-rs/examples/backend_smoke.rs` on both CPU and Metal (`[CPU] ... OK`, `[MTL0] ... OK`).
- Added safe GGUF reader API to `ggml-rs` and `llama-rs/examples/gguf_inspect.rs`.
- Verified `gguf_inspect` against a generated sample GGUF file from upstream `llama-gguf`.
- Added `llama-rs/examples/gguf_hash.rs` with layer/global hashing and manifest verification.
- Verified `gguf_hash` generation and `--check` round-trip success on sample GGUF manifest.
- Expanded `ggml-rs` safe op surface for llama runtime needs: `add`, `mul`, `silu`, `rms_norm`, `scale`, `get_rows`, `repeat`, `cpy`, `cont`, `reshape_*`, `view_*`, `permute`, `diag_mask_inf`, `soft_max(_ext)`, `rope_ext`.
- Added tensor naming (`set_name` / `name`) and backend `i32` tensor transfer helpers.
- Re-ran CPU + Metal execution checks after the op-surface expansion:
  - `ggml-rs/examples/backend_matmul.rs`
  - `llama-rs/examples/backend_smoke.rs`
- Applied Rust API polishing pass:
  - split core definitions into `src/error.rs` and `src/types.rs`,
  - added trait-based typed backend tensor I/O (`BackendElement`),
  - added arithmetic expression abstraction (`TensorExpr`) with `+ - * /` operator support,
  - added `prelude` re-export module.
- Added `ggml-rs/examples/arithmetic_expr.rs` and verified runtime execution.
- Added `docs/ggml-rs/README.md` and `docs/ggml-rs/KNOWLEDGE_BASE.md`.
- Addressed `docs/third_reviews/review_0.md` feedback:
  - switched error definition to `thiserror`,
  - changed error string payloads to `Cow<'static, str>`,
  - added `TryFrom<c_int> for Type` with `UnsupportedType` fallback,
  - replaced integer conversion helper functions with `TryFrom`/`TryInto`-based extension trait,
  - updated FFI opaque types to include pinned marker metadata while keeping FFI-safe opaque layout.
- Per follow-up review feedback, further refactored to Rust-idiomatic style:
  - replaced nested `checked_add`/`checked_mul` helper style with `num-traits` (`CheckedAdd`/`CheckedMul`) trait-based checked arithmetic,
  - switched backend name init path from raw `*const c_char` plumbing to `&CStr` internal API,
  - optimized path-to-C-string conversion to avoid unnecessary lossy conversion on Unix.
- Added zero-overhead newtype layer (`Cols`, `Rows`, `Shape2D`) and typed API entrypoints for 2D tensor creation / shape handling / matmul memory estimation.
- Continued module split to reduce `lib.rs` concentration:
  - moved GGUF implementation to `src/gguf.rs`,
  - moved numeric conversion/checked-arithmetic traits to `src/num_ext.rs`,
  - moved expression DSL (`TensorExpr`, operator impls) to `src/tensor_expr.rs`.
- Reworked error flow to remove `field` string threading:
  - `Error` now uses source-based variants (`IntConversion`, `CString`, `Utf8`) and explicit structured variants (`NullPointer { api }`, `Overflow`),
  - conversion helpers now expose `try_into_checked()` / checked arithmetic methods without field-name arguments.
- Per additional modularization request, moved compute-layer implementation into `src/compute.rs` and made `src/lib.rs` a compact re-export façade.
- Expanded newtype usage further with `Length`, `TensorIndex`, `ThreadCount` and typed entrypoints (`new_*_1d_len`, `compute_with_threads`, `get_f32_at`).
- Migrated ggml test/bench assets (safe API scope):
  - added `tests/ggml_simple_ctx.rs` and `tests/ggml_test_cont.rs` (feature-gated with `link-system`),
  - added `examples/bench_matmul.rs` benchmark harness for CPU/Metal timing.
- Added full vendor suite harnesses:
  - `tests/ggml_upstream_suite.rs` to build/run all upstream ggml test targets,
  - `examples/bench_upstream_suite.rs` to run upstream perf-focused targets.

## Next concrete steps

1. Expand `llama-rs` core API beyond smoke-level matrix operations.
2. Start implementing feature parity per example target from the parity matrix.
3. Mark each target as in-progress/done with CPU+Metal verification evidence.
