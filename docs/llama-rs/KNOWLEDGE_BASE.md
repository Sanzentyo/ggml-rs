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
- `llama-rs/examples/batched.rs`: CPU and Metal batched execution path runs with reusable backend graph and deterministic checksum.
- `llama-rs/examples/gguf_inspect.rs`: can read and print metadata/tensor info from a sample GGUF file.
- `llama-rs/examples/model_catalog.rs`: can load GGUF bytes + tensor catalog, validate all tensor payload ranges, and query tensors by name.
- `llama-rs/examples/embedding_probe.rs`: can decode an f32 tensor payload from GGUF and compute embedding-style summary stats.
- `llama-rs/examples/min_infer_linear.rs`: can run a minimal linear inference path (`Y = W * X`) from GGUF weights on selected backend.
- `llama-rs/examples/min_infer_mlp.rs`: can run a minimal MLP block (`down(silu(gate(x)) * up(x))`) on selected backend.
- `llama-rs/examples/min_infer_mlp_layer.rs`: can run MLP inference by resolved layer index from GGUF tensor names.
- `llama-rs/examples/min_infer_attention_layer.rs`: can run minimal attention inference by resolved layer index from GGUF tensor names.
- `llama-rs/examples/bench_mlp_layer.rs`: can benchmark the layer-MLP path on CPU/Metal with deterministic checksum output.
- `llama-rs/examples/resolve_tensor_names.rs`: can resolve canonical LLaMA tensor roles from GGUF names and report missing mappings.
- `llama-rs/tests/mlp_cpp_parity.rs` (feature `link-system`): validates rust MLP CPU output against an equivalent C++ reference program.
- `llama-rs/tests/attention_parity.rs` (feature `link-system`): validates attention output parity between CPU and Metal backends.

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

## GGUF model catalog command

```bash
cargo run -p llama-rs --example model_catalog --features link-system -- target/vendor/llama.cpp/build/sample.gguf --head 8 --check-tensor tensor_0
```

Sample result (current local `sample.gguf`):

- `file_size=2016`, `n_kv=15`, `n_tensors=10`
- head tensor payload checks pass (`tensor_0`..`tensor_4`)

## Embedding probe command

```bash
cargo run -p llama-rs --example embedding_probe --features link-system -- target/vendor/llama.cpp/build/sample.gguf tensor_0
```

Sample result (`tensor_0` in local sample):

- `len=2`, `mean=100.0`, `l2_norm=141.421356`, `min=max=100.0`

## Batched execution command

```bash
cargo run -p llama-rs --example batched --features link-system -- --batch 8 --repeats 5 --readback-every 1 --size 256 cpu metal
```

API notes:

- `BatchedConfig::validated()` enforces shape invariants.
- Scheduling knobs are strongly typed and non-zero: `BatchSize`, `RepeatCount`, `ReadbackEvery`.
- `BatchedWorkload` separates synthetic/custom workload construction from execution.
- Batch inputs are kept in contiguous storage (`batch_input` / `batch_inputs` views) for cleaner ownership and better locality.

Sample result (`--batch 8 --repeats 5 --readback-every 2 --size 128`):

- CPU: `avg_item` around `0.085-0.108 ms`, `readbacks=20`, checksum `2310.615234`
- Metal (`MTL0`): `avg_item` around `0.169-0.371 ms`, `readbacks=20`, checksum `2310.615234`

## Minimal inference command

```bash
cargo run -p llama-rs --example min_infer_linear --features link-system -- target/vendor/llama.cpp/build/sample.gguf tensor_2 --in 6 --out 6 --repeats 3 cpu
```

API notes:

- `LinearInferenceConfig::new(in_features, out_features)` clarifies domain naming.
- `LinearWeights::from_model(...)` decodes once and enables reusable repeated inference.
- `LinearInferenceConfig::builder()` provides a type-state builder (`Missing`/`Present`) so required dimensions are set before build.

Sample result (`tensor_2`, `--in 6 --out 6`):

- CPU preview: `[191.25, 191.25, 191.25, 191.25, 191.25, 191.25]`
- Metal preview: `[191.25, 191.25, 191.25, 191.25, 191.25, 191.25]`

## Model API notes

- `GgufModel::find_tensor` returns an opaque `TensorHandle` for type-safe repeated lookup.
- Handle-based accessors (`tensor_info_by_handle`, `tensor_payload_by_handle`, `decode_tensor_f32_into_by_handle`) avoid repeated string lookups in hot paths.
- `GgufModel::kv_value` exposes typed GGUF metadata values (`GgufValue`) for architecture-aware config parsing.
- `resolve_transformer_metadata(_from_kv)` parses `{architecture}.*` metadata namespaces (e.g. llama/mistral prefixes).
- `resolve_llama_layer_tensor_names` resolves one layer directly (not only full-catalog resolution).
- `resolve_llama_layer_dimensions` derives hidden/ffn/head dimensions from metadata + tensor lengths.
- If full metadata is missing, head topology falls back to tensor-length heuristics (`attn_q/attn_k`) with preferred head-dimension candidates.
- `resolve_mlp_weights_for_layer_auto` / `resolve_attention_weights_for_layer_auto` use the dimension resolver and reduce manual shape flags.

## Unified error boundary

- `llama-rs` now exports `LlamaError` and `LlamaResult<T>`.
- `From` conversions are provided from module-level errors (`BenchError`, `BatchedError`, `ModelError`, `EmbeddingError`, `InferenceError`, `NamingError`, `SimpleError`, `SmokeError`, `GgufHashError`, `ParseBackendError`, `ggml_rs::Error`) to simplify app-level error handling.

## Tensor-name resolver command

```bash
cargo run -p llama-rs --example resolve_tensor_names --features link-system -- target/vendor/llama.cpp/build/sample.gguf --head 2
```

Notes:

- Resolver supports common naming families:
  - `blk.{layer}.*` (current llama.cpp GGUF style)
  - `layers.{layer}.*` (legacy style)
  - `model.layers.{layer}.*` (HF-like style)
- In non-strict mode, unresolved models still return success so inspection pipelines can proceed.

## Minimal MLP-block inference command

```bash
cargo run -p llama-rs --example min_infer_mlp --features link-system -- cpu metal --hidden 64 --ffn 128 --repeats 2
```

API notes:

- `MlpInferenceConfig` introduces typed dimensions (`HiddenFeatures`, `FfnFeatures`).
- `MlpWeights::deterministic` provides backend/runtime smoke-test weights without GGUF dependency.
- `MlpWeights::from_model` and `run_mlp_inference(...)` provide a GGUF-backed path for future model wiring.

Sample result (`--hidden 64 --ffn 128 --repeats 2`):

- CPU preview: `[2856.8298, 3016.3652, 2761.5999, 3007.5938, 2820.4155, 2915.0156, 2953.4194, 2814.0574]`
- Metal preview: `[2856.8296, 3016.3652, 2761.6, 3007.5938, 2820.4155, 2915.0156, 2953.4194, 2814.0574]`

## Synthetic layer-fixture generation (uv)

To test layer-index resolver + MLP execution without a full model, generate a
small GGUF fixture via `uv run`:

```bash
uv run --with numpy --with pyyaml python path/to/create_fixture.py
```

Fixture used in this session:

- `/tmp/mlp_layer_sample.gguf` (contains `blk.0.*` attention/ffn tensors and required global tensors)
- `/tmp/mlp_layer_sample_meta.gguf` (same tensor set + full `llama.*` metadata for auto-shape validation)

## Layer-index GGUF-backed MLP inference command

```bash
cargo run -p llama-rs --example min_infer_mlp_layer --features link-system -- /tmp/mlp_layer_sample.gguf --layer 0 --repeats 2 cpu metal
```

API notes:

- `run_mlp_inference_for_layer(_repeats)` now uses `resolve_mlp_weights_for_layer_auto` by default.
- Hidden width is derived from GGUF metadata when present, and falls back to tensor-size inference for sparse fixtures.

Sample result (`--layer 0 --repeats 2`):

- CPU preview: `[435.15863, 1039.7737, 1644.3887, 2249.0037, 2853.6187, 3458.2336, 4062.8489, 4667.464]`
- Metal preview: `[435.15866, 1039.7737, 1644.3888, 2249.004, 2853.619, 3458.234, 4062.849, 4667.464]`
- output now includes `resolution_mode=FullMetadata|TensorHeuristic` for visibility.

## Layer-index attention inference command

```bash
cargo run -p llama-rs --example min_infer_attention_layer --features link-system -- /tmp/mlp_layer_sample.gguf --layer 0 --seq 4 --repeats 2 cpu metal
```

API notes:

- Attention config is ADT-based: `AttentionLayout` + `AttentionMaskPolicy` + `RotaryEmbedding`.
- `resolve_attention_weights_for_layer_auto` derives layout from metadata/tensor lengths.
- `run_attention_inference_with_weights_repeats` executes multi-head grouped attention on backend tensors.
- Causal path is supported via `AttentionMaskPolicy::Causal` (`soft_max_ext` mask). Verified on CPU; CPU/Metal parity is validated on the non-causal path.
- Optional flags in example: `--causal`, `--no-rope`.

Sample result (`--layer 0 --seq 4 --repeats 2`):

- CPU preview: `[0.1777783, 0.47217602, 0.76657367, 1.0609715, 1.3553691, 1.6497668, 1.9441648, 2.2385623]`
- Metal preview: `[0.1777783, 0.472176, 0.76657367, 1.0609714, 1.355369, 1.6497668, 1.9441643, 2.238562]`
- output now includes `resolution_mode` and inferred/metadata head topology (`heads=q/kv`).

## Layer-MLP benchmark command

```bash
cargo run -p llama-rs --example bench_mlp_layer --features link-system -- --cases 64x128,96x192 --warmup 1 --iters 3 cpu metal
```

Sample result (`--cases 64x128,96x192 --warmup 1 --iters 3`):

- `64x128`: CPU `~27.1 ms`, Metal `~27.6 ms`, checksum `48763.663x`
- `96x192`: CPU `~27.3 ms`, Metal `~28.2 ms`, checksum `167016.84x`

## Rust vs C++ parity test command

```bash
cargo test -p llama-rs --features link-system --test mlp_cpp_parity
```

Notes:

- C++ reference source: `llama-rs/tests/cpp/mlp_reference.cpp`
- C++ reference now executes the same graph with ggml CPU backend (`ggml_mul_mat`/`ggml_silu`/`ggml_mul`).
- Rust test compiles this source at runtime and compares outputs against
  `run_mlp_inference_with_weights_repeats(..., LlamaBackend::Cpu, 1)`.
- Coverage now includes multiple shape cases (`8x16`, `16x32`, `32x64`) and
  an additional CPU-vs-Metal parity test across the same matrix.

## Attention parity test command

```bash
cargo test -p llama-rs --features link-system --test attention_parity
```

This test validates CPU-vs-Metal parity for the minimal attention path and also includes a causal-mask CPU execution check.

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
