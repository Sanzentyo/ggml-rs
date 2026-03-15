# foundation batch log

- scope: `simple-ctx`, `simple-backend`, `perf-metal`
- owner: parallel subagent

## 2026-03-15 inspection
- reviewed upstream scopes:
  - `vendor/ggml/examples/simple/simple-ctx.cpp`
  - `vendor/ggml/examples/simple/simple-backend.cpp`
  - `vendor/ggml/examples/perf-metal/perf-metal.cpp`
- reviewed Rust counterparts:
  - `examples/simple_ctx.rs` (present, aligned to simple-ctx matrix case)
  - `examples/backend_matmul.rs` (present, CPU/Metal backend parity check)
  - no dedicated `perf_metal` Rust example found (gap identified)
- planned implementation: add `examples/perf_metal.rs`, register in `Cargo.toml`, then run locked C++/Rust parity runs with benchmark artifact output under `target/benchmarks/`.

## 2026-03-15 implementation
- added safe API helpers for custom graph sizing in `ggml-rs`:
  - `graph_overhead_custom(size, grads)`
  - `Context::new_graph_custom(size, grads)`
- added new Rust example `examples/perf_metal.rs` as a safe counterpart of upstream `perf-metal.cpp`.
  - positional args: `n_op`, `n_iter`
  - backend: Metal (`BackendKind::Metal`)
  - prints graph node count, avg time, checksum, and full 8x8 output matrix.
- registered the example in `Cargo.toml` (`[[example]] name = "perf_metal"`, `required-features = ["link-system"]`).
- updated docs references (`docs/ggml-rs/README.md`, `docs/ggml-rs/KNOWLEDGE_BASE.md`).

## 2026-03-15 validation
- `cargo fmt --all` executed under lock:
  - command: `./scripts/agent_lock.sh cargo cargo fmt --all`
  - note: baseline workspace has a missing optional example path (`examples/magika_main_synth.rs`); to complete mandatory fmt, a temporary stub file was created and removed in the same locked command.
- targeted clippy (locked):
  - `GGML_RS_LIB_DIRS=target/vendor/ggml-foundation/build/src:target/vendor/ggml-foundation/build/src/ggml-metal:target/vendor/ggml-foundation/build/src/ggml-blas GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas ./scripts/agent_lock.sh cargo cargo clippy --features link-system --lib --tests --example simple_ctx --example backend_matmul --example perf_metal`
- targeted tests (locked):
  - `GGML_RS_LIB_DIRS=target/vendor/ggml-foundation/build/src:target/vendor/ggml-foundation/build/src/ggml-metal:target/vendor/ggml-foundation/build/src/ggml-blas GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas ./scripts/agent_lock.sh cargo cargo test --features link-system --test ggml_simple_ctx --test ggml_tensor_ops`

## 2026-03-15 parity run
- built C++ targets with lock:
  - `./scripts/agent_lock.sh cpp cmake -S vendor/ggml -B target/vendor/ggml-foundation/build -DGGML_METAL=ON -DGGML_CPU=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF -DCMAKE_BUILD_TYPE=Release`
  - `./scripts/agent_lock.sh cpp cmake --build target/vendor/ggml-foundation/build --target simple-ctx simple-backend perf-metal --parallel 8`
- built Rust targets with lock:
  - `GGML_RS_LIB_DIRS=target/vendor/ggml-foundation/build/src:target/vendor/ggml-foundation/build/src/ggml-metal:target/vendor/ggml-foundation/build/src/ggml-blas GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas ./scripts/agent_lock.sh cargo cargo build --features link-system --example simple_ctx --example backend_matmul --example perf_metal`
- runtime parity (locked bench):
  - simple context + backend parity outputs captured for C++/Rust.
  - perf parity run: `perf-metal` vs `perf_metal` with `n_op=512`, `n_iter=64`.
- artifact outputs under `target/benchmarks/`:
  - `foundation_simple_ctx_cpp.txt`
  - `foundation_simple_ctx_rust.txt`
  - `foundation_backend_matmul_cpp.txt`
  - `foundation_backend_matmul_rust.txt`
  - `foundation_perf_metal_cpp.txt`
  - `foundation_perf_metal_rust.txt`
  - `foundation_parity_report.md`
  - `foundation_parity_report.json`
- parity summary snapshot:
  - simple-ctx checksum delta: `0.000000`
  - simple-backend checksum delta (CPU/Metal): `0.000000` / `0.000000`
  - perf-metal checksum delta: `0.000000`
  - perf-metal time delta (Rust vs C++): `-2.048%`
