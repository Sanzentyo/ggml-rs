# llama-rs knowledge base

## Local reference checkouts

- `target/vendor/ggml`
- `target/vendor/llama.cpp`

## ggml local build used for verification

```bash
cmake -S target/vendor/ggml -B target/vendor/ggml/build \
  -DGGML_METAL=ON -DGGML_CPU=ON \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build target/vendor/ggml/build -j
```

## Link/run env used by Rust examples

```bash
GGML_RS_LIB_DIRS=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas
DYLD_LIBRARY_PATH=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH
```

## Backend-specific notes

- On this machine, Metal backend name is reported as `MTL0`.
- `ggml_metal_device_init` may print:
  - `tensor API disabled for pre-M5 and pre-A19 devices`
  - This did not block matmul execution in backend smoke tests.
- Backend init by literal name (`Metal`) can fail depending on registry naming; device-type and device-enumeration fallback is needed.

## Existing validated baseline

- `ggml-rs/examples/simple_ctx.rs`: CPU path runs and reproduces expected matrix result.
- `ggml-rs/examples/backend_matmul.rs`: CPU and Metal both run and report expected matrix result.
- `ggml-rs/examples/bench_matmul.rs`: CPU and Metal matmul benchmark path runs with stable checksum.
- `ggml-rs/examples/arithmetic_expr.rs`: trait-based `TensorExpr` arithmetic (`+ - * /`) runs and validates expected output.
- `llama-rs/examples/simple.rs`: safe `simple-ctx` parity path runs and validates expected output on the host compute path.
- `llama-rs/examples/backend_smoke.rs`: CPU and Metal both run and report expected matrix result.
- `llama-rs/examples/bench_matmul.rs`: CPU and Metal benchmark path runs via `ggml-rs` safe API only.
- `llama-rs/examples/gguf_inspect.rs`: can read and print metadata/tensor info from a sample GGUF file.

## Error-context rollout policy

- `llama-rs` layers that orchestrate ggml-backed operations follow the same
  call-site context policy as `ggml-rs`.
- Reference: `docs/ggml-rs/ERROR_CONTEXT_POLICY.md`

## Recent ggml-rs API expansion (safe wrappers)

- Added safe wrappers for llama-runtime-critical ops:
  - `add`, `mul`, `silu`, `rms_norm`, `scale`
  - `get_rows`, `repeat`, `cpy`, `cont`
  - `reshape_2d`, `reshape_3d`, `view_1d`, `view_2d`, `permute`
  - `diag_mask_inf`, `soft_max`, `soft_max_ext`, `rope_ext`
- Added tensor naming (`set_name` / `name`) and backend `i32` tensor transfer (`set_i32_backend` / `to_vec_i32_backend`).

## GGUF sample generation command

```bash
cmake -S target/vendor/llama.cpp -B target/vendor/llama.cpp/build -DGGML_METAL=ON -DGGML_CPU=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build target/vendor/llama.cpp/build --target llama-gguf -j
target/vendor/llama.cpp/build/bin/llama-gguf target/vendor/llama.cpp/build/sample.gguf w
```

## GGUF hash parity check command

```bash
cargo run -p llama-rs --example gguf_hash --features link-system -- --all target/vendor/llama.cpp/build/sample.gguf > target/vendor/llama.cpp/build/sample.gguf.manifest
cargo run -p llama-rs --example gguf_hash --features link-system -- --all --check target/vendor/llama.cpp/build/sample.gguf.manifest target/vendor/llama.cpp/build/sample.gguf
```

## Benchmark parity snapshot (same host, 256x256, `--iters 10`)

Observed samples:

- Run A
  - `ggml-rs`: CPU `0.272 ms`, Metal `0.386 ms`
  - `llama-rs`: CPU `0.276 ms`, Metal `0.293 ms`
- Run B
  - `ggml-rs`: CPU `0.285 ms`, Metal `0.379 ms`
  - `llama-rs`: CPU `0.286 ms`, Metal `0.443 ms`
- Run C
  - `ggml-rs`: CPU `0.287 ms`, Metal `0.187 ms`
  - `llama-rs`: CPU `0.282 ms`, Metal `0.344 ms`

All runs produced identical checksum `956.435547`.
Latency fluctuates run-to-run on Metal, but both binaries stay in the same
order of magnitude with no systemic overhead from the `llama-rs` wrapper path.
