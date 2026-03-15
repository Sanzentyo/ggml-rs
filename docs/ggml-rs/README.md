# ggml-rs

`ggml-rs` is a safe Rust wrapper around a focused subset of
[`ggml-org/ggml`](https://github.com/ggml-org/ggml).

The crate intentionally keeps unsafe details internal and exposes a safe API
centered on `Context`, `Tensor`, `Graph`, and `Backend`.

## API design

- Explicit low-level-style API through `Context` methods (`mul_mat`, `add`,
  `rms_norm`, `rope_ext`, ...).
- Backend execution includes explicit synchronization support via
  `Backend::synchronize()` for stable benchmark timing/readback boundaries.
- Expression API via `TensorExpr` with operator overloading (`+`, `-`, `*`,
  `/`) that still returns `Result<_>`.
- Typed tensor I/O through `GgmlElement` (`Tensor::write_data`,
  `Tensor::read_data`, `Tensor::get_data`) plus backend-slice traits
  (`BackendElement`).
- Unified typed memory estimation through
  `Context::recommended_matmul_memory::<T>(...)` and
  `Context::recommended_backend_matmul_memory::<T>(...)`.
- Shape-safe APIs with newtypes (`Shape2D`, `Length`, `TensorIndex`,
  `ThreadCount`, `Bytes`) and const-generic typed shapes
  (`StaticShape2D<const C, const R>` + `Tensor2D`).
- Generic tensor-construction entrypoints:
  `Context::new_tensor_typed::<T, N>(...)`, `new_tensor_1d::<T>(Length)`,
  `new_tensor_2d::<T>(Shape2D)`, `new_tensor_3d::<T>(Shape3D)`,
  `new_tensor_4d::<T>(Shape4D)`.

```rust
use ggml_rs::prelude::*;

let expr = ((ctx.expr(a) + ctx.expr(b))? * ctx.expr(c))? / ctx.expr(d);
let out = expr?.into_tensor();
```

## Module map

- `src/error.rs`: crate error type (`Error`, `Result`)
- `src/shape.rs`: semantic newtypes and static-shape traits
- `src/types.rs`: `Type`, `BackendKind`, `BackendDeviceType`, `ComputeStatus`, `RopeExtParams`
- `src/num_ext.rs`: checked conversion/arithmetic helpers
- `src/gguf.rs`: safe GGUF inspection and writing helpers (`GgufFile`, `GgufWriter`)
- `src/tensor_expr.rs`: expression wrapper, element traits (`BackendElement`,
  `GgmlElement`) and operator impls
- `src/typed_tensor.rs`: typed tensor wrappers (`Tensor2D`, `Tensor2DConst`)
- `src/compute.rs`: context/backend/tensor/graph implementations
- `src/ffi.rs`: minimal C FFI declarations
- `src/lib.rs`: public façade and re-exports

## FFI generation policy

The default path uses `bindgen` (common Rust C-FFI workflow):

- `build.rs` generates bindings from ggml headers into `$OUT_DIR/ffi_bindings.rs`
- `src/ffi.rs` includes those generated bindings

- Header resolution order:
  1. `GGML_RS_GGML_INCLUDE_DIR`
  2. `vendor/ggml/include` (submodule layout)
  3. `target/vendor/ggml/include` (legacy local layout)

If headers are missing, initialize submodules:

```bash
git submodule update --init --recursive
```

## Error handling policy

- `thiserror`-based structured errors.
- Conversion failures are propagated as `Error::IntConversion { context, source }`.
- API-level null returns are represented by `Error::NullPointer { context }`.
- Numeric overflow is handled by checked helpers (`Error::Overflow`).

Detailed policy:

- `docs/ggml-rs/ERROR_CONTEXT_POLICY.md`

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
- `examples/perf_metal.rs`: Metal backend stress graph equivalent to upstream `perf-metal`
- `examples/arithmetic_expr.rs`: expression-style tensor arithmetic
- `examples/bench_matmul.rs`: end-to-end backend matmul benchmark
- synthetic parity counterparts for upstream higher-level examples:
  - `examples/gptj_main_synth.rs`
  - `examples/gptj_quantize_synth.rs`
  - `examples/magika_main_synth.rs`
  - `examples/mnist_eval.rs`
  - `examples/mnist_train.rs`
  - `examples/sam.rs`
  - `examples/yolov3_tiny.rs`

Current upstream target coverage and parity state:

- `docs/ggml-rs/EXAMPLE_PARITY_MATRIX.md`

## Suggested import

```rust
use ggml_rs::prelude::*;
```
