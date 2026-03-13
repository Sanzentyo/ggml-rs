# ggml-rs knowledge base

## Local verification environment

- ggml checkout: `target/vendor/ggml`
- Typical local build with CPU + Metal enabled:

```bash
cmake -S target/vendor/ggml -B target/vendor/ggml/build \
  -DGGML_METAL=ON -DGGML_CPU=ON \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build target/vendor/ggml/build -j
```

## Link variables for `--features link-system`

```bash
GGML_RS_LIB_DIRS=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas
DYLD_LIBRARY_PATH=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH
```

## Backend behavior notes

- In this environment the Metal backend name may appear as `MTL0` rather than
  a fixed `Metal`.
- Device enumeration plus backend type fallback is more stable than relying on
  a single backend-name string.
- `ggml_metal_device_init` may log warnings while still producing valid matmul
  execution results.

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
