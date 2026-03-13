# ggml-rs

`ggml-rs` is a safe Rust wrapper around a focused subset of
[`ggml-org/ggml`](https://github.com/ggml-org/ggml).

The crate intentionally keeps unsafe details internal and exposes a safe API
centered on `Context`, `Tensor`, `Graph`, and `Backend`.

## API design

- Explicit low-level-style API through `Context` methods (`mul_mat`, `add`,
  `rms_norm`, `rope_ext`, ...).
- Expression API via `TensorExpr` with operator overloading (`+`, `-`, `*`,
  `/`) that still returns `Result<_>`.
- Typed backend I/O through `BackendElement`.
- Shape-safe APIs with newtypes (`Shape2D`, `Length`, `TensorIndex`,
  `ThreadCount`, `Bytes`) and const-generic typed shapes
  (`StaticShape2D<const C, const R>` + `Tensor2D`).

```rust
use ggml_rs::prelude::*;

let expr = ((ctx.expr(a) + ctx.expr(b))? * ctx.expr(c))? / ctx.expr(d);
let out = expr?.into_tensor();
```

## Module map

- `src/error.rs`: crate error type (`Error`, `Result`)
- `src/shape.rs`: semantic newtypes and static-shape traits
- `src/types.rs`: `Type`, `BackendKind`, `RopeExtParams`
- `src/num_ext.rs`: checked conversion/arithmetic helpers
- `src/gguf.rs`: safe GGUF inspection helpers
- `src/tensor_expr.rs`: expression wrapper and operator impls
- `src/typed_tensor.rs`: typed tensor wrappers (`Tensor2D`, `Tensor2DConst`)
- `src/compute.rs`: context/backend/tensor/graph implementations
- `src/ffi.rs`: minimal C FFI declarations
- `src/lib.rs`: public façade and re-exports

## Error handling policy

- `thiserror`-based structured errors.
- Conversion failures are propagated as `Error::IntConversion`.
- API-level null returns are represented by `Error::NullPointer { api }`.
- Numeric overflow is handled by checked helpers (`Error::Overflow`).

## Test and benchmark coverage

Rust-native parity coverage:

- `tests/ggml_simple_ctx.rs`: parity with `simple-ctx` style matmul
- `tests/ggml_test_cont.rs`: parity for `transpose` + `cont`
- `examples/bench_matmul.rs`: CPU/Metal matmul benchmark path

Upstream-suite harnesses:

- `tests/ggml_upstream_suite.rs` (ignored by default): builds/runs upstream
  `ggml` test targets
- `examples/bench_upstream_suite.rs`: builds/runs selected upstream perf targets

## Upstream-suite operation controls

Both suite harnesses support:

- `GGML_UPSTREAM_BUILD_DIR`: override build directory
- `GGML_UPSTREAM_SKIP_BUILD=1`: skip `cmake --build`
- `GGML_UPSTREAM_BUILD_JOBS=<n>`: parallel build jobs
- `GGML_UPSTREAM_LIST_ONLY=1`: print selected targets and exit
- `GGML_UPSTREAM_EXCLUDE_TARGETS=target1,target2,...`: remove targets
- `GGML_UPSTREAM_SUMMARY_PATH=<file>`: persist run summary

Test suite specific:

- `GGML_UPSTREAM_TEST_TARGETS=target1,target2,...`
- `GGML_UPSTREAM_KEEP_GOING=1`: continue after failures and report summary

Bench suite specific:

- `GGML_UPSTREAM_BENCH_TARGETS=target1,target2,...`
- CLI args can directly pass target names and override env/default targets
- CLI flags: `--skip-build`, `--list-only`, `--keep-going`, `--fail-fast`

Detailed operations runbook:

- `docs/ggml-rs/SUITE_OPERATIONS.md`

## Examples

- `examples/simple_ctx.rs`: simple context-based matmul
- `examples/backend_matmul.rs`: backend compute on CPU/Metal
- `examples/arithmetic_expr.rs`: expression-style tensor arithmetic
- `examples/bench_matmul.rs`: end-to-end backend matmul benchmark

## Suggested import

```rust
use ggml_rs::prelude::*;
```
