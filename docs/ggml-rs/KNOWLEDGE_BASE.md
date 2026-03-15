# ggml-rs knowledge base

## Local verification environment

- ggml checkout (git submodule): `vendor/ggml`
- legacy fallback path (older artifacts): `target/vendor/ggml`
- Typical local build with CPU + Metal enabled:

```bash
cmake -S vendor/ggml -B vendor/ggml/build \
  -DGGML_METAL=ON -DGGML_CPU=ON \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/ggml/build -j
```

## Link variables for `--features link-system`

```bash
GGML_RS_LIB_DIRS=vendor/ggml/build/src:vendor/ggml/build/src/ggml-metal:vendor/ggml/build/src/ggml-blas
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas
DYLD_LIBRARY_PATH=vendor/ggml/build/src:vendor/ggml/build/src/ggml-metal:vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH
```

## FFI generation modes

Bindgen-only mode (default and required):

- `build.rs` uses `bindgen` and emits `$OUT_DIR/ffi_bindings.rs`
- `src/ffi.rs` includes generated bindings

Header lookup order:

```bash
GGML_RS_GGML_INCLUDE_DIR=<path>/include     # explicit override
vendor/ggml/include                          # submodule layout
target/vendor/ggml/include                   # legacy local layout
```

Submodule setup command:

```bash
git submodule update --init --recursive
```

## Backend behavior notes

- In this environment the Metal backend name may appear as `MTL0` rather than
  a fixed `Metal`.
- Device enumeration plus backend type fallback is more stable than relying on
  a single backend-name string.
- `ggml_metal_device_init` may log warnings while still producing valid matmul
  execution results.
- For benchmark timing, call `Backend::synchronize()` after compute loops to
  ensure queued backend work is fully completed before measuring/reporting.
- Quantized tensor decode helpers are available in safe API:
  - `decode_tensor_data_to::<T>(ggml_type_raw, payload) -> Vec<T>`
  - `tensor_element_count(ggml_type_raw, payload_bytes)`
  These use GGML type traits (`ggml_get_type_traits`) and are useful for GGUF
  model paths that need typed views over quantized payloads.
- Decode API ownership refactor was assembly-verified (`2026-03-15`):
  - focused snippets: `target/benchmarks/review3_decode_asm_snippets.md`
  - before/after line-count summary: `target/benchmarks/review3_decode_asm_compare.md`
  - final judgement: `target/benchmarks/review3_decode_asm_judgement.md`
  - max-opt helper line count moved `122 -> 115` (`after/before ~0.943`), and
    the release caller fast path remains contiguous copy (`__rust_alloc + memcpy`)
    for native payloads.
  - policy: keep ownership-returning decode API as default (`Result<Vec<T>>`).
- Backend tensors support partial safe writes through typed APIs:
  - `Tensor::write_data_backend_at::<f32>(offset, values)`
  - `Tensor::write_data_backend_at::<i32>(offset, values)`
  Use these when only a contiguous region changes (for example, stepwise mask
  delta updates) to reduce host-device transfer overhead.
- GGUF now has a safe write path:
  - `GgufWriter::new()`
  - `GgufWriter::set_value(key, &GgufValue)` / `set_values(iter)` (typed scalars/arrays)
  - `GgufWriter::remove_key(key)` / `merge_kv_from(&GgufFile)`
  - `GgufWriter::add_tensor(&Tensor)`
  - `GgufWriter::write_to_file(path, only_meta)` plus mode-specific helpers:
    - `write_data_to_file(path)`
    - `write_metadata_to_file(path)`
  This allows fixture generation and round-trip checks without exposing unsafe pointers.
- Tensor constructor wrappers were consolidated into generic APIs (`2026-03-15`):
  - use `new_tensor_typed::<T, N>(Dims<N>)` for rank-generic paths,
  - use semantic helpers `new_tensor_1d::<T>(Length)`, `new_tensor_2d::<T>(Shape2D)`,
    `new_tensor_3d::<T>(Shape3D)`, `new_tensor_4d::<T>(Shape4D)` for common shapes.
- Typed-shape wrappers were expanded to rank-complete generic forms (`2026-03-15`):
  - wrappers: `Tensor1D/2D/3D/4D` and const aliases `Tensor1DConst..Tensor4DConst`,
  - constructor helpers: `new_tensor_1d_typed::<T, S>()` ... `new_tensor_4d_typed::<T, S>()`.
- Root synthetic parity examples are now clap-based and re-verified on CPU/Metal:
  - artifact: `target/benchmarks/review3_constructor_clap_runtime_smoke.txt`.

## Error context policy

- Low-level/FFI-adjacent code must use call-site `map_err` / `ok_or_else`
  context attachment.
- See: `docs/ggml-rs/ERROR_CONTEXT_POLICY.md`

## Baseline validation commands

```bash
cargo fmt
cargo clippy --workspace --all-targets
cargo test --workspace
cargo run --example backend_matmul --features link-system
cargo run --example perf_metal --features link-system -- 256 16
cargo run -p llama-rs --example backend_smoke --features link-system
cargo run --example arithmetic_expr --features link-system
```

## Full upstream test/bench workflow

```bash
# Full upstream test-target harness (ignored test)
cargo test --features link-system --test ggml_upstream_suite -- --ignored

# Upstream bench-target harness
cargo run --example bench_upstream_suite --features link-system
```

## Suite operation controls

Common controls:

- `GGML_UPSTREAM_BUILD_DIR=<path>`
- `GGML_UPSTREAM_SKIP_BUILD=1`
- `GGML_UPSTREAM_BUILD_JOBS=<n>`
- `GGML_UPSTREAM_LIST_ONLY=1`
- `GGML_UPSTREAM_EXCLUDE_TARGETS=test-opt,test-quantize-fns`
- `GGML_UPSTREAM_SUMMARY_PATH=target/upstream-suite-summary.txt`

Test-suite controls:

- `GGML_UPSTREAM_TEST_TARGETS=test-cont,test-opt`
- `GGML_UPSTREAM_KEEP_GOING=1` (collect all failures instead of fail-fast)

Bench-suite controls:

- `GGML_UPSTREAM_BENCH_TARGETS=test-backend-ops,test-quantize-perf`
- Direct CLI targets are also supported:
  `cargo run --example bench_upstream_suite --features link-system -- test-backend-ops`
- Bench CLI flags are available: `--skip-build`, `--list-only`,
  `--keep-going`, `--fail-fast`

Detailed runbook:

- `docs/ggml-rs/SUITE_OPERATIONS.md`
- `docs/ggml-rs/EXAMPLE_PARITY_MATRIX.md` (upstream example-by-example parity status)
