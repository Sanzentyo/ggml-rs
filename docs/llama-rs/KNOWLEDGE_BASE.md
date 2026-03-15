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
- `llama-rs/examples/gguf.rs`: can write a deterministic GGUF fixture (`w`) and run upstream-style read mode splits (`r0` metadata-only, `r1` tensor-data pass, `r` combined) with optional data checks.
- `llama-rs/examples/idle.rs`: can run decode-proxy idle timings with pause schedules on CPU/Metal and report pause-vs-latency statistics.
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

## Example CLI policy (clap unification)

- `llama-rs/examples` argument parsing is standardized on typed `clap` derive structs.
- This pass includes `bench_attention_layer`, `bench_attention_decode_cpp_compare`, `bench_compare_report`, `gguf`, `gguf_hash`, and `idle`.
- Error boundaries for the clap-unified examples are standardized with `thiserror`-based typed enums.
- `gguf` keeps legacy shorthand compatibility for read mode (`r1 n`) in addition to `--check/--no-check`.

Clap-refactor runtime smoke artifact (CPU+Metal where applicable):

- `target/benchmarks/llama_rs_clap_refactor_runtime_smoke.txt`

## Inference modularization notes (stepwise track)

- Stepwise decode core is now split into dedicated modules:
  - `llama-rs/src/inference/stepwise_plan.rs` (`DecodeStepPlan` ADT + type-state builder + static-dispatch trait),
  - `llama-rs/src/inference/stepwise_decode.rs` (stepwise config/report types + execution runners),
  - `llama-rs/src/inference/attention_ops.rs` (RoPE/concat helpers).
- `llama-rs/src/inference.rs` acts as the re-export and orchestration surface for these submodules.
- Runtime smoke artifact after the deeper split:
  - `target/benchmarks/llama_rs_stepwise_refactor_stepwisecore_smoke.txt`.
- API naming migration:
  - long `run_*` public entrypoints were renamed to direct operation names (for example: `mlp_inference_*`, `attention_inference_*`, `attention_decode_proxy_*`, `backend_smoke`, `simple_ctx`, `idle_decode_proxy`).
  - stepwise execution now routes through `DecodeStepPlan::{execute_single, bench}`.
- Runtime artifacts after ADT + naming migration:
  - `target/benchmarks/llama_rs_stepwise_decodeplan_smoke.txt`,
  - `target/benchmarks/llama_rs_stepwise_decodeplan_cppcompare_smoke.txt`.
- Final post-rename runtime checks:
  - `target/benchmarks/llama_rs_backend_smoke_decodeplan_postrename.txt`,
  - `target/benchmarks/llama_rs_stepwise_decodeplan_postrename_smoke.txt`.
- Repo-level auto-loaded policy file:
  - `.github/copilot-instructions.md`.

## Current sequential refactor policy (resume-safe)

For the active llama.cpp reproduction + ggml-rs refinement track, refactors are executed in this fixed order and validated at each step:

1. Backend/context lifecycle traitization
2. Stepwise sequence state traitization (mask/position delta updates)
3. Decode projection/cache traitization
4. Then continue proactive hotspot exploration with the same trait-first ADT policy

Completed artifacts for 1-3 (CPU/Metal runtime smoke):

- `target/benchmarks/llama_rs_backend_runtime_trait_refactor_smoke_cpu_metal.txt`
- `target/benchmarks/llama_rs_sequence_state_trait_refactor_smoke_cpu_metal.txt`
- `target/benchmarks/llama_rs_projection_trait_refactor_smoke_cpu_metal.txt`

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

## GGUF read/write parity command (safe API)

```bash
GGML_RS_LIB_DIRS=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas \
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas \
DYLD_LIBRARY_PATH=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH \
cargo run -q -p llama-rs --example gguf --features link-system -- /tmp/llama_rs_gguf_fixture_safeapi.gguf w

GGML_RS_LIB_DIRS=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas \
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas \
DYLD_LIBRARY_PATH=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH \
cargo run -q -p llama-rs --example gguf --features link-system -- /tmp/llama_rs_gguf_fixture_safeapi.gguf r0

GGML_RS_LIB_DIRS=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas \
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas \
DYLD_LIBRARY_PATH=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH \
cargo run -q -p llama-rs --example gguf --features link-system -- /tmp/llama_rs_gguf_fixture_safeapi.gguf r1 --check

GGML_RS_LIB_DIRS=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas \
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas \
DYLD_LIBRARY_PATH=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH \
cargo run -q -p llama-rs --example gguf --features link-system -- /tmp/llama_rs_gguf_fixture_safeapi.gguf r --check
```

GGUF writer usability notes in this pass:

- `GgufWriter::set_values(...)` for batch KV insertion.
- `GgufWriter::remove_key(...)` for explicit KV cleanup before write.
- `GgufWriter::write_data_to_file(...)` / `write_metadata_to_file(...)` convenience APIs.
- `gguf` example mode parity:
  - `r0`: metadata/key/tensor-info read pass (`gguf_ex_read_0` style),
  - `r1`: tensor-data preview pass with optional `--check` (`gguf_ex_read_1` style),
  - `r`: combined `r0 + r1`.
- verification artifact: `target/benchmarks/llama_rs_gguf_mode_parity_r0_r1.txt`.

## Idle proxy command (CPU/Metal)

```bash
GGML_RS_LIB_DIRS=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas \
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas \
DYLD_LIBRARY_PATH=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH \
cargo run -q -p llama-rs --example idle --features link-system -- \
  target/models/elyza_llama3_jp_8b_q4_k_m/Llama-3-ELYZA-JP-8B-q4_k_m.gguf \
  --layer 0 --decode-kv 128 --past 127 --iters 2 --pauses 0,800 cpu metal
```

Artifact:

- `target/benchmarks/llama_rs_idle_elyza_layer0_cpu_metal.txt`
- `target/benchmarks/llama_rs_idle_qwen35_cpu_metal_fallback.txt`
- `target/benchmarks/llama_rs_idle_elyza_cpu_metal_post_qwen_fix.txt`
- `idle` now tries the requested layer first, then scans detected layers for attention-capable tensors.
- For non-llama architectures where real per-layer attention weights cannot be resolved, `idle` falls back to metadata-derived deterministic attention weights and reports `weights_mode=MetadataDeterministic`.
- Llama-style models keep real tensor decoding (`weights_mode=ModelLayer`).

Qwen validation command:

```bash
GGML_RS_LIB_DIRS=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas \
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas \
DYLD_LIBRARY_PATH=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH \
cargo run -q -p llama-rs --example idle --features link-system -- \
  target/models/qwen3_5_4b_q4_k_m/Qwen3.5-4B-Q4_K_M.gguf \
  --decode-kv 128 --iters 1 --pauses 0 cpu metal
```

## Stepwise optimization loop (resumed after idle+gguf pass)

```bash
GGML_RS_LIB_DIRS=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas \
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas \
DYLD_LIBRARY_PATH=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH \
cargo run -q -p llama-rs --example bench_attention_layer --features link-system -- \
  --cases 4096x32x8x1 \
  --decode-kv 128 --decode-steps 16 --past 127 --causal --rope \
  --decode-stepwise-profile-outproj-fused-layerx5 \
  --decode-stepwise-kv-proj --decode-stepwise-kv-cache-write --decode-stepwise-block \
  --block-mlp-model target/models/elyza_llama3_jp_8b_q4_k_m/Llama-3-ELYZA-JP-8B-q4_k_m.gguf \
  --block-mlp-layer-range 0:7 \
  --warmup 1 --iters 5 cpu metal
```

Artifacts:

- raw: `target/benchmarks/llama_rs_stepwise_resume_elyza_layers0_7.txt`
- ranked summary: `target/benchmarks/llama_stepwise_resume_elyza_layers0_7_summary.{csv,md}`

Post-clap+thiserror resume artifacts (same ELYZA slice):

- raw: `target/benchmarks/llama_rs_stepwise_resume_after_clap_elyza_layers0_7.txt`
- ranked summary: `target/benchmarks/llama_stepwise_resume_after_clap_elyza_layers0_7_summary.{csv,md}`
- hotspot A/B impacts (`block_layer=2..7`):
  - `target/benchmarks/llama_stepwise_resume_after_clap_elyza_layers2_7_maskhost_impact.md`
  - `target/benchmarks/llama_stepwise_resume_after_clap_elyza_layers2_7_headstage_impact.md`
  - `target/benchmarks/llama_stepwise_resume_after_clap_elyza_layers2_7_blockgateup_impact.md`
  - `target/benchmarks/llama_stepwise_resume_after_clap_elyza_layers2_7_balconcat_impact.md`
  - `target/benchmarks/llama_stepwise_resume_after_clap_elyza_layers2_7_maskdelta_impact.md`
  - `target/benchmarks/llama_stepwise_resume_after_clap_elyza_layers2_7_positiondelta_impact.md`
  - stability rerun: `target/benchmarks/llama_stepwise_resume_after_clap_elyza_layers2_7_delta_stability_r3.md`
- phase-init-elision follow-up (`block_layer=5..7`, same lock):
  - raw: `target/benchmarks/llama_rs_stepwise_phase_init_elision_elyza_layers5_7.txt`
  - smoke: `target/benchmarks/llama_rs_stepwise_phase_init_elision_smoke.txt`
  - impact: `target/benchmarks/llama_stepwise_phase_init_elision_elyza_layers5_7_impact.md`
- backend-load-once cleanup remeasure (`block_layer=5..7`, same lock):
  - raw: `target/benchmarks/llama_rs_stepwise_backend_load_once_elyza_layers5_7.txt`
  - impact: `target/benchmarks/llama_stepwise_backend_load_once_elyza_layers5_7_impact.md`
- layer-loop reuse cache smoke (`bench_attention_layer` block-MLP cache path):
  - `target/benchmarks/llama_rs_stepwise_layer_loop_reuse_cache_smoke.txt`
- graph-level reuse planning baseline (`setup` instrumentation):
  - raw: `target/benchmarks/llama_rs_stepwise_graph_reuse_setup_baseline_elyza_layers5_7.txt`
  - summary: `target/benchmarks/llama_stepwise_graph_reuse_setup_baseline_elyza_layers5_7.md`
- graph-level layer-sweep reuse prototype (shared setup across `--block-mlp-layer-range`):
  - runtime (CPU+Metal):
    - `target/benchmarks/llama_rs_stepwise_graph_reuse_layer_sweep_elyza_layers5_7.txt`
  - impact:
    - `target/benchmarks/llama_stepwise_graph_reuse_layer_sweep_elyza_layers5_7_impact.md`
  - stepwise output markers:
    - `graph_reuse_sweep=true`,
    - `setup_shared=... ms` (single setup per backend call),
    - `setup=... ms` (amortized per-layer setup).
- query-RoPE multi-head hotspot pass (on top of graph-reuse sweep):
  - runtime (CPU+Metal):
    - `target/benchmarks/llama_rs_stepwise_graph_reuse_layer_sweep_elyza_layers5_7_qrope_multihead.txt`
  - impact:
    - `target/benchmarks/llama_stepwise_graph_reuse_qrope_multihead_elyza_layers5_7_impact.md`
  - measured post/base `avg_token`:
    - CPU `~1.001` (near-neutral),
    - MTL0 `~0.944` (improved),
    - overall `~0.977` (improved),
    - checksum parity: `max abs delta = 0.0`.
- CPU-side follow-up check (`head_stage_buf=true`) after qrope pass:
  - impact artifact:
    - `target/benchmarks/llama_stepwise_headstage_after_qrope_refactor_smoke_impact.md`
  - sampled result (`variant/base`):
    - CPU `~1.022`,
    - MTL0 `~0.995`,
    - overall `~1.011`,
    - decision: keep `head_stage_buf=false`.
- Refactor progress (ADT/type-state + module split):
  - added decode-step ADT with type-state builder and trait-based dispatch:
    - `llama-rs/src/inference/stepwise_plan.rs`,
    - `DecodeStepPlan` / `DecodeStepPlanBuilder`,
    - trait: `DecodeStepBenchSet` (`single`/`sweep` static dispatch).
  - `bench_attention_layer` now uses the plan ADT for stepwise bench execution.
  - split helper ops from `inference.rs`:
    - `llama-rs/src/inference/attention_ops.rs`.
  - profile precedence fix:
    - explicit toggles now override profile presets (e.g. `--decode-stepwise-head-stage-buffer` with `--decode-stepwise-profile-outproj-fused-layerx5`).
  - runtime smoke artifacts:
    - `target/benchmarks/llama_rs_stepwise_refactor_plan_smoke.txt`,
    - `target/benchmarks/llama_rs_refactor_profile_override_smoke.txt`,
    - `target/benchmarks/llama_rs_refactor_attention_ops_smoke.txt`.
  - API naming policy update:
    - long `run_*` public function names were removed in favor of ADT-first and direct operation names (`mlp_inference`, `attention_inference_*`, `attention_decode_proxy_*`, etc.).

Hotspot follow-up (`block_layer=5..7`, same profile lock):

```bash
GGML_RS_LIB_DIRS=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas \
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas \
DYLD_LIBRARY_PATH=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH \
cargo run -q -p llama-rs --example bench_attention_layer --features link-system -- \
  --cases 4096x32x8x1 \
  --decode-kv 128 --decode-steps 16 --past 127 --causal --rope \
  --decode-stepwise-profile-outproj-fused-layerx5 \
  --decode-stepwise-kv-proj --decode-stepwise-kv-cache-write --decode-stepwise-block \
  --block-mlp-model target/models/elyza_llama3_jp_8b_q4_k_m/Llama-3-ELYZA-JP-8B-q4_k_m.gguf \
  --block-mlp-layer-range 5:7 --warmup 1 --iters 5 cpu metal \
  | tee target/benchmarks/llama_rs_stepwise_elyza_layers5_7_headstage_base.txt

GGML_RS_LIB_DIRS=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas \
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas \
DYLD_LIBRARY_PATH=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH \
cargo run -q -p llama-rs --example bench_attention_layer --features link-system -- \
  --cases 4096x32x8x1 \
  --decode-kv 128 --decode-steps 16 --past 127 --causal --rope \
  --decode-stepwise-profile-outproj-fused-layerx5 \
  --decode-stepwise-kv-proj --decode-stepwise-kv-cache-write --decode-stepwise-block \
  --decode-stepwise-head-stage-buffer \
  --block-mlp-model target/models/elyza_llama3_jp_8b_q4_k_m/Llama-3-ELYZA-JP-8B-q4_k_m.gguf \
  --block-mlp-layer-range 5:7 --warmup 1 --iters 5 cpu metal \
  | tee target/benchmarks/llama_rs_stepwise_elyza_layers5_7_headstage_on.txt
```

- impact: `target/benchmarks/llama_stepwise_profile_layerx5_headstage_ab_elyza_layers5_7_impact.md`
- checksum parity: `target/benchmarks/llama_stepwise_profile_layerx5_headstage_ab_elyza_layers5_7_checksum_check.md` (`max abs delta = 0.0`)
- snapshot: `on/base` CPU `~1.009`, MTL0 `~1.000`, overall `~1.005` (kept default-off).

Block gate/up follow-up (`block_layer=5..7`, same profile lock):

```bash
GGML_RS_LIB_DIRS=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas \
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas \
DYLD_LIBRARY_PATH=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH \
cargo run -q -p llama-rs --example bench_attention_layer --features link-system -- \
  --cases 4096x32x8x1 \
  --decode-kv 128 --decode-steps 16 --past 127 --causal --rope \
  --decode-stepwise-profile-outproj-fused-layerx5 \
  --decode-stepwise-kv-proj --decode-stepwise-kv-cache-write --decode-stepwise-block \
  --block-mlp-model target/models/elyza_llama3_jp_8b_q4_k_m/Llama-3-ELYZA-JP-8B-q4_k_m.gguf \
  --block-mlp-layer-range 5:7 --warmup 1 --iters 5 cpu metal \
  | tee target/benchmarks/llama_rs_stepwise_elyza_layers5_7_blockgateup_base.txt

GGML_RS_LIB_DIRS=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas \
GGML_RS_LIBS=ggml,ggml-base,ggml-cpu,ggml-metal,ggml-blas \
DYLD_LIBRARY_PATH=target/vendor/ggml/build/src:target/vendor/ggml/build/src/ggml-metal:target/vendor/ggml/build/src/ggml-blas:$DYLD_LIBRARY_PATH \
cargo run -q -p llama-rs --example bench_attention_layer --features link-system -- \
  --cases 4096x32x8x1 \
  --decode-kv 128 --decode-steps 16 --past 127 --causal --rope \
  --decode-stepwise-profile-outproj-fused-layerx5 \
  --decode-stepwise-kv-proj --decode-stepwise-kv-cache-write --decode-stepwise-block \
  --decode-stepwise-fuse-block-gate-up \
  --block-mlp-model target/models/elyza_llama3_jp_8b_q4_k_m/Llama-3-ELYZA-JP-8B-q4_k_m.gguf \
  --block-mlp-layer-range 5:7 --warmup 1 --iters 5 cpu metal \
  | tee target/benchmarks/llama_rs_stepwise_elyza_layers5_7_blockgateup_on.txt
```

- impact: `target/benchmarks/llama_stepwise_profile_layerx5_blockgateup_ab_elyza_layers5_7_impact.md`
- checksum parity: `target/benchmarks/llama_stepwise_profile_layerx5_blockgateup_ab_elyza_layers5_7_checksum_check.md` (`max abs delta = 0.0`)
- snapshot: `on/base` CPU `~1.013`, MTL0 `~1.012`, overall `~1.012` (kept default-off).

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
- `MlpWeights::from_model` and `mlp_inference(...)` provide a GGUF-backed path for future model wiring.

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

- `mlp_inference_for_layer(_repeats)` now uses `resolve_mlp_weights_for_layer_auto` by default.
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
- `attention_inference_with_weights_repeats` executes multi-head grouped attention on backend tensors.
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

## Layer-attention benchmark command

```bash
cargo run -p llama-rs --example bench_attention_layer --features link-system -- --cases 64x8x8x8 --warmup 1 --iters 2 cpu metal
```

Case format:

- `HxQxKxS` = hidden features, query heads, KV heads, sequence length.

Sample result (`--cases 64x8x8x8 --warmup 1 --iters 2`):

- CPU: `~60.9 ms`, checksum `701.095203`
- Metal (`MTL0`): `~62.5 ms`, checksum `701.095192`

Decode-like mode:

```bash
cargo run -p llama-rs --example bench_attention_layer --features link-system -- \
  --cases 4096x32x8x1 --decode-kv 128 --past 127 --causal --rope --warmup 1 --iters 3 cpu metal
```

Sample result:

- CPU: `~48.7 ms`, checksum `2724244.625000`
- Metal (`MTL0`): `~52.3 ms`, checksum `2724244.609375`
- output line includes `cache_reuse=true` to indicate pre-projected KV cache reuse in decode mode.

Stepwise decode-growth mode:

```bash
cargo run -p llama-rs --example bench_attention_layer --features link-system -- \
  --cases 4096x32x8x1 --decode-kv 128 --decode-steps 16 --past 127 --causal --rope --warmup 2 --iters 10 cpu metal
```

Sample result:

- CPU: `~4.18 ms/token`, checksum `2720587.859375`
- Metal (`MTL0`): `~3.29 ms/token`, checksum `2720587.812500`
- output line includes `stepwise=true`, `kv_start=...`, `steps=...`, `setup=... ms`, and `avg_token=... ms`.
- optional `--decode-stepwise-kv-proj` enables per-step KV projection cost modeling in the persistent runner (`kv_proj=true` in output).
- optional `--decode-stepwise-block` enables residual+RMSNorm+MLP-shaped block-scope cost modeling (`block=true` in output).
- optional `--decode-stepwise-sync-step` inserts per-step backend synchronization (currently applied to Metal path in the benchmark runner).
- optional `--decode-stepwise-readback-step` inserts per-step output readback (currently applied to Metal path in the benchmark runner).
- optional `--decode-stepwise-kv-cache-write` adds explicit projected-K/V copy nodes (requires `--decode-stepwise-kv-proj`) to model cache-write cost.
- optional `--decode-stepwise-kv-cache-write-to-cache` rewires KV-write copies into per-step views of the persistent K/V cache tensor (requires `--decode-stepwise-kv-cache-write`).
- optional `--decode-stepwise-layer-repeat <n>` multiplies per-step graph execution count (`n >= 1`) to model extra stacked decode work while keeping the same graph structure.
- optional `--decode-stepwise-layer-repeat-model` derives `layer_repeat` from `block_count` metadata of `--block-mlp-model` (for model-shaped sweeps).
- optional `--decode-stepwise-no-mask-delta` disables incremental mask-delta updates and forces full per-step mask uploads (for matched A/B checks).
- optional `--decode-stepwise-elide-mask-host-buffer` enables experimental host-buffer elision for incremental mask updates (`query_length=1`); default is disabled.
- optional `--decode-stepwise-head-stage-buffer` enables an experimental per-head staging-buffer path before fused output projection; default is disabled.
- optional `--decode-stepwise-fuse-block-gate-up` enables experimental fusion of block-MLP gate/up projection into one matmul in stepwise block mode; default is disabled.
- optional `--block-mlp-model <gguf> --block-mlp-layer <n>` wires model-derived block-MLP topology into block mode (`block_mlp_real=true/false` in output).
- optional `--block-mlp-layer-range <start:end>` runs the same benchmark across multiple block layers in one command and emits `block_layer=<n>` per line.
- stepwise output includes `mask_host_elide=<true|false>` to keep this A/B condition explicit in artifacts.
- stepwise output includes `head_stage_buf=<true|false>` to keep this A/B condition explicit in artifacts.
- stepwise output includes `block_gateup_fused=<true|false>` to keep this A/B condition explicit in artifacts.
- quantized GGUF block-MLP tensors are now decoded through GGML type-traits (`to_float`) in `GgufModel::tensor_f32_values`, so q4/q5/q6 models can report `block_mlp_real=true`.
- benchmark runner uses persistent graph/tensor allocations across all decode steps and only updates causal-mask/query-position tensors per step.
- benchmark runner also performs one untimed backend preflight per backend to reduce first-case kernel compile bias.
- in `--decode-steps` mode, warmup and measured iterations now share one persistent stepwise allocation path (`DecodeStepPlan::bench`) to avoid duplicate setup in benchmark sweeps.
- in the same path, the measured phase now skips redundant KV/precompute reinit when warmup does not mutate persistent KV cache tensors (`kv_write_cache=false`), reducing phase boundary overhead.
- backend registry loading in `llama-rs` runtime paths is now guarded by one-time init (`ensure_backends_loaded`) to avoid repeated `Backend::load_all()` calls.
- when `--block-mlp-model` is used, `bench_attention_layer` now caches resolved block-MLP weights by `(hidden_features, block_layer)` across the run to reuse layer decodes in repeated case/layer sweeps.
- argument parsing in `bench_attention_layer` is refactored to iterator-driven `next_arg(...)` handling; complex-flag smoke artifact: `target/benchmarks/llama_rs_parser_refactor_smoke.txt`.

Canonical refresh sequence (parity default + variants):

```bash
# canonical parity snapshot (default: steps=16)
cargo run -p llama-rs --example bench_attention_layer --features link-system -- \
  --cases 2560x16x4x1,3072x32x8x1,3584x28x4x1,3840x16x8x1,4096x32x8x1 \
  --decode-kv 128 --decode-steps 16 --past 127 --causal --rope --warmup 2 --iters 10 cpu metal \
  | tee target/benchmarks/llama_rs_bench_attention_decode_stepwise_models.txt

# sensitivity variants
for S in 8 16 32; do
  cargo run -p llama-rs --example bench_attention_layer --features link-system -- \
    --cases 2560x16x4x1,3072x32x8x1,3584x28x4x1,3840x16x8x1,4096x32x8x1 \
    --decode-kv 128 --decode-steps ${S} --past 127 --causal --rope --warmup 2 --iters 10 cpu metal \
    | tee target/benchmarks/llama_rs_bench_attention_decode_stepwise_s${S}.txt
done
```

Layerwise profiling command pattern (per-layer optimization loop):

```bash
cargo run -q -p llama-rs --features link-system --example bench_attention_layer -- \
  --cases 2560x16x4x1 \
  --decode-kv 128 --decode-steps 16 --past 127 --causal --rope \
  --decode-stepwise-kv-proj --decode-stepwise-kv-cache-write --decode-stepwise-block \
  --decode-stepwise-layer-repeat 3 \
  --block-mlp-model target/models/qwen3_5_4b_q4_k_m/Qwen3.5-4B-Q4_K_M.gguf \
  --block-mlp-layer-range 0:31 \
  --warmup 2 --iters 10 cpu metal
```

This emits one benchmark row per backend per layer (`block_layer=<n>`), which is used as the primary input for layer-by-layer optimization.

## llama.cpp baseline capture notes (real GGUF models)

Build baseline tool:

```bash
cmake --build target/vendor/llama.cpp/build --target llama-bench -j
```

Model download (session command, `uv` + HF Hub):

```bash
uv run --with huggingface_hub python - <<'PY'
from huggingface_hub import hf_hub_download
# download selected GGUF files into target/models/*
PY
```

Baseline run profile used in this session:

```bash
target/vendor/llama.cpp/build/bin/llama-bench \
  -m <model.gguf> -r 1 -o jsonl -t 8 \
  -pg 256,0 -pg 0,128 -ngl 0
target/vendor/llama.cpp/build/bin/llama-bench \
  -m <model.gguf> -r 1 -o jsonl -t 8 \
  -pg 256,0 -pg 0,128 -ngl 99
```

Note:

- `llama-bench` also emits its built-in default pair (`512/0`) in addition to explicit `-pg` pairs.

Captured artifacts:

- `target/benchmarks/llama_cpp_baseline_all.jsonl`
- `target/benchmarks/llama_cpp_baseline_extra.jsonl`

Observed constraint:

- `target/vendor/llama.cpp/build/sample.gguf` is a synthetic fixture and `llama-bench` rejects it as a benchmark model (`failed to load model`).

Snapshot (`tok/s`):

| Model | Pair (`prompt/gen`) | CPU | Metal | Metal/CPU |
| --- | --- | ---:| ---:| ---:|
| Qwen3.5-4B-Q4_K_M.gguf | `256/0` | 184.528 | 609.166 | 3.301 |
| Qwen3.5-4B-Q4_K_M.gguf | `0/128` | 52.538 | 46.680 | 0.888 |
| Qwen3-8B-Q4_K_M.gguf | `256/0` | 102.153 | 359.133 | 3.516 |
| Qwen3-8B-Q4_K_M.gguf | `0/128` | 35.198 | 39.866 | 1.133 |
| Llama-3.1-Minitron-4B-Width-Base-Q4_0.gguf | `256/0` | 226.868 | 695.732 | 3.067 |
| Llama-3.1-Minitron-4B-Width-Base-Q4_0.gguf | `0/128` | 68.321 | 74.523 | 1.091 |
| Llama-3-ELYZA-JP-8B-q4_k_m.gguf | `256/0` | 96.085 | 360.596 | 3.753 |
| Llama-3-ELYZA-JP-8B-q4_k_m.gguf | `0/128` | 30.486 | 41.075 | 1.347 |
| KaLM-Embedding-Gemma3-12B-2511.Q2_K.gguf | `256/0` | 73.619 | 220.334 | 2.993 |
| KaLM-Embedding-Gemma3-12B-2511.Q2_K.gguf | `0/128` | 19.789 | 20.552 | 1.039 |
| InternVL3-8B-Q4_K_M.gguf | `256/0` | 91.668 | 388.066 | 4.233 |
| InternVL3-8B-Q4_K_M.gguf | `0/128` | 38.365 | 42.768 | 1.115 |

## Model-shaped llama-rs proxy benchmark notes

Metadata extraction command used for case derivation:

```bash
cargo run -q -p llama-rs --features link-system --example gguf_inspect -- <model.gguf> \
  | rg 'general\\.architecture|embedding_length|feed_forward_length|attention\\.head_count'
```

Bench commands used in this session:

```bash
cargo run -q -p llama-rs --features link-system --example bench_mlp_layer -- \
  --cases 4096x14336,3840x15360,3584x18944,3072x9216,2560x9216,4096x12288 \
  --warmup 1 --iters 3 cpu metal

cargo run -q -p llama-rs --features link-system --example bench_attention_layer -- \
  --cases 4096x32x8x256,3840x16x8x256,3584x28x4x256,3072x32x8x256,2560x16x4x256 \
  --causal --rope --warmup 1 --iters 3 cpu metal
```

MLP snapshot (`ms/iter`):

| Case (`hidden x ffn`) | CPU | Metal | CPU/Metal |
| --- | ---:| ---:| ---:|
| `2560x9216` | 49.199 | 50.481 | 0.975 |
| `3072x9216` | 51.302 | 49.314 | 1.040 |
| `3584x18944` | 63.366 | 64.241 | 0.986 |
| `3840x15360` | 60.370 | 58.821 | 1.026 |
| `4096x12288` | 56.759 | 56.556 | 1.004 |
| `4096x14336` | 58.974 | 57.152 | 1.032 |

Attention snapshot (`ms/iter`, `seq=256`, causal+RoPE):

| Case (`HxQxKxS`) | CPU | Metal | CPU/Metal |
| --- | ---:| ---:| ---:|
| `2560x16x4x256` | 117.039 | 55.816 | 2.097 |
| `3072x32x8x256` | 155.085 | 57.334 | 2.705 |
| `3584x28x4x256` | 177.645 | 54.664 | 3.250 |
| `3840x16x8x256` | 227.090 | 61.189 | 3.711 |
| `4096x32x8x256` | 265.821 | 60.378 | 4.403 |

Decode-like attention proxy snapshot (`ms/iter`, `q_seq=1`, `kv_seq=128`, `past=127`, causal+RoPE):

| Case (`HxQxKxQ/KV`) | CPU | Metal | CPU/Metal |
| --- | ---:| ---:| ---:|
| `2560x16x4x1/128` | 46.081 | 47.909 | 0.962 |
| `3072x32x8x1/128` | 47.493 | 50.076 | 0.948 |
| `3584x28x4x1/128` | 48.665 | 53.078 | 0.917 |
| `3840x16x8x1/128` | 49.014 | 51.095 | 0.959 |
| `4096x32x8x1/128` | 48.686 | 52.277 | 0.931 |

Stepwise decode-growth snapshot (`ms/token`, `q_seq=1`, `kv_start=128`, `steps=16`, causal+RoPE):

| Case (`HxQxKxQ/KV+steps`) | CPU | Metal | CPU/Metal |
| --- | ---:| ---:| ---:|
| `2560x16x4x1/128+16` | 3.228 | 2.438 | 1.324 |
| `3072x32x8x1/128+16` | 4.067 | 3.043 | 1.337 |
| `3584x28x4x1/128+16` | 3.910 | 2.884 | 1.356 |
| `3840x16x8x1/128+16` | 4.520 | 2.783 | 1.624 |
| `4096x32x8x1/128+16` | 4.181 | 3.291 | 1.270 |

Interpretation notes:

- `llama.cpp` and `llama-rs` proxy benches are different layers of the stack. `tok/s` and `ms/iter` are not directly interchangeable.
- The useful comparison here is trend direction: Metal advantage is strong on prefill-like and attention-heavy paths, while MLP-only proxy cases are closer to parity on this host.
- Decode-like proxy (`q_seq=1`, `kv_seq=128`) is close to parity on this host, and in this snapshot CPU is slightly faster across these cases.
- Stepwise decode-growth proxy (`kv_start=128`, `steps=16`) with persistent runner + backend preflight shows consistent Metal advantage across the sampled model-shaped cases on this host.

Automated report command:

```bash
cargo run -p llama-rs --example bench_compare_report -- \
  --output target/benchmarks/llama_proxy_vs_cpp_report.md
```

Decode-focused attention report example:

```bash
cargo run -p llama-rs --example bench_compare_report -- \
  --llama-rs-attention target/benchmarks/llama_rs_bench_attention_decode_all_models.txt \
  --output target/benchmarks/llama_proxy_vs_cpp_decode_report.md
```

Stepwise-vs-`llama.cpp` calibration artifact (generated in-session via `uv run python`):

Calibration output:

- `target/benchmarks/llama_stepwise_vs_cpp_calibration.md`
- metric definition: `llama.cpp ms/token = 1000 / tok/s` for `0/128`, compared against persistent stepwise proxy `avg_token` (`--decode-kv 128 --decode-steps 16 --warmup 2 --iters 10`).
- `target/benchmarks/llama_stepwise_variant_sweep.md` (variant sweep for `steps=8/16/32` under the same warmup/repeat policy).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_kvproj.md` (same calibration in KV-projection mode, `--decode-stepwise-kv-proj`).
- `target/benchmarks/llama_stepwise_kvproj_impact.md` (matched-environment delta table between `kv_proj=false` and `kv_proj=true`).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj.md` (block-scope + KV-projection mode: `--decode-stepwise-block --decode-stepwise-kv-proj`).
- `target/benchmarks/llama_stepwise_block_scope_impact.md` (matched-env table across base / kv / block+kv modes).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_sync.md` (block+kv mode with `--decode-stepwise-sync-step`).
- `target/benchmarks/llama_stepwise_sync_step_impact.md` (matched-env table for block+kv vs block+kv+sync).
- `target/benchmarks/llama_stepwise_realmlp_qwen35_calibration.md` (Qwen3.5 one-case check with model-derived block-MLP shape).
- `target/benchmarks/rust_style_quantized_realmlp_qwen35_comparison.md` (control vs real-MLP comparison showing `block_mlp_real=true` on q4 model).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_realmlp_s16_models.txt` (6-model block+kv+real-MLP sweep with `block_mlp_real=true`).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_realmlp.md` (same calibration in block+kv+real-MLP mode).
- `target/benchmarks/llama_stepwise_block_realmlp_impact.md` (delta table: deterministic fallback vs real-MLP decode).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_realmlp_s16_models_maskdelta.txt` (rerun after mask-delta optimization pass).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_realmlp_maskdelta.md` (updated 6-model calibration after mask-delta pass).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_realmlp_maskdelta_impact.md` (old vs new real-MLP calibration delta table).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_s16_models.txt` (6-model sweep with optional `--decode-stepwise-kv-cache-write` enabled).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_maskdelta.md` (calibration in block+kv+kvwrite+real-MLP mode).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_maskdelta_impact.md` (delta vs block+kv+real-MLP mask-delta base).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models.txt` (same sweep with `--decode-stepwise-layer-repeat 3`).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx4_s16_models.txt` (same sweep with `--decode-stepwise-layer-repeat 4`).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_maskdelta.md` (`layer_repeat=3` calibration).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat4_maskdelta.md` (`layer_repeat=4` calibration).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat_impact.md` (base vs `layer_repeat=3/4` comparison).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_r3_i15_raw.txt` (`layer_repeat=3`, 3 reruns, `iters=15`, raw capture).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_r3_i15_median.txt` (`layer_repeat=3` stable median artifact).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_maskdelta_stable.md` (`layer_repeat=3` stable median calibration).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_stable_impact.md` (single-run vs stable median impact for `layer_repeat=3`).
- `target/benchmarks/llama_stepwise_layerx3_stability_r3_i15.md` (`layer_repeat=3` min/median/max stability table).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx4_s16_models_r3_i15_raw.txt` (`layer_repeat=4`, 3 reruns, `iters=15`, raw capture).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx4_s16_models_r3_i15_median.txt` (`layer_repeat=4` stable median artifact).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat4_maskdelta_stable.md` (`layer_repeat=4` stable median calibration).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_vs_4_stable_impact.md` (stable median comparison: `layer_repeat=3` vs `4`).
- `target/benchmarks/llama_stepwise_layerx4_stability_r3_i15.md` (`layer_repeat=4` min/median/max stability table).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_qwen35_layers0_31_layerx3.txt` (Qwen3.5 per-layer sweep, layers `0..31`).
- `target/benchmarks/llama_stepwise_qwen35_layers0_31_layerx3_profile.{md,csv}` (Qwen3.5 per-layer profile summary/data).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_qwen3_8b_layers0_35_layerx3.txt` (Qwen3-8B per-layer sweep, layers `0..35`).
- `target/benchmarks/llama_stepwise_qwen3_8b_layers0_35_layerx3_profile.{md,csv}` (Qwen3-8B per-layer profile summary/data).
- `target/benchmarks/llama_stepwise_layerwise_profile_summary_layerx3.md` (cross-model layerwise summary and initial hotspot interpretation).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_qwen35_layers0_31_layerx3_maskdelta_hostless.txt` (Qwen3.5 rerun for mask host-buffer elision experiment).
- `target/benchmarks/llama_stepwise_qwen35_layers0_31_layerx3_maskdelta_hostless_profile.{md,csv}` (Qwen3.5 profile from the host-buffer-elision pass).
- `target/benchmarks/llama_stepwise_qwen35_layers0_31_layerx3_maskdelta_hostless_impact.md` (Qwen3.5 impact snapshot vs prior profiles).
- `target/benchmarks/llama_stepwise_mask_host_elide_ab_qwen35_layers_sample_impact.md` (Qwen3.5 sampled-layer A/B with order balancing).
- `target/benchmarks/llama_stepwise_mask_host_elide_ab_qwen3_8b_layers_sample_impact.md` (Qwen3-8B sampled-layer A/B with order balancing).
- `target/benchmarks/llama_stepwise_mask_host_elide_sampled_impact.md` (cross-model sampled A/B summary for `mask_host_elide`).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_maskhost_base_balanced.txt` (6-model balanced-order sweep with `mask_host_elide=false`).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_maskhost_elide_balanced.txt` (same 6-model sweep with `mask_host_elide=true`).
- `target/benchmarks/llama_stepwise_mask_host_elide_full_sweep_impact.md` (full-sweep on/off impact table and backend averages).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_maskhost_{base,elide}_balanced_r2.txt` (stability rerun #2 raw artifacts).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_maskhost_{base,elide}_balanced_r3.txt` (stability rerun #3 raw artifacts).
- `target/benchmarks/llama_stepwise_mask_host_elide_full_sweep_stability_r3.md` (`r=3` median stability table for full-sweep on/off).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_kvheadcache_post.txt` (6-model post-change sweep after KV-head cache reuse optimization).
- `target/benchmarks/llama_stepwise_kvhead_cache_impact_vs_maskhost_base_r3_median.md` (post-change impact vs pre-change `r=3` median baseline).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_kvheadcache_post_r2.txt` (post-change rerun #2).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_kvheadcache_post_r3.txt` (post-change rerun #3).
- `target/benchmarks/llama_stepwise_kvhead_cache_stability_r3.md` (`r=3` stability table for KV-head cache optimization).
- `target/benchmarks/llama_stepwise_kvhead_cache_impact_vs_maskhost_base_r3_median_stable.md` (stable median impact vs pre-change baseline).
- `target/benchmarks/llama_stepwise_kvhead_cache_checksum_check.md` (checksum parity check for KV-head cache reuse optimization).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_maskdelta_kvheadcache_stable.md` (updated stable calibration vs llama.cpp after KV-head cache optimization).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_maskdelta_kvheadcache_stable_impact.md` (old/new stable calibration impact after KV-head cache optimization).
- `target/benchmarks/llama_stepwise_outproj_fuse_smoke_ab.txt` (CPU/Metal smoke A/B for stepwise output-projection fusion on/off).
- `target/benchmarks/llama_stepwise_outproj_fuse_smoke_impact.md` (smoke A/B impact summary for output-projection fusion).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_outproj_base_balanced.txt` (6-model balanced-order base run for output-projection fusion pass).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_outproj_fused_balanced.txt` (same full sweep with `--decode-stepwise-fuse-output-proj`).
- `target/benchmarks/llama_stepwise_outproj_fuse_full_sweep_impact.md` (full-sweep fused/base impact table).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_outproj_{base,fused}_balanced_r2.txt` (stability rerun #2).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_outproj_{base,fused}_balanced_r3.txt` (stability rerun #3).
- `target/benchmarks/llama_stepwise_outproj_fuse_full_sweep_stability_r3.md` (`r=3` median stability table for output-projection fusion).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_maskdelta_kvheadcache_outprojfuse_stable.md` (stable llama.cpp calibration with fused output projection).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_maskdelta_kvheadcache_outprojfuse_stable_impact.md` (old/new calibration impact for fused output projection).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx4_s16_models_outprojfused.txt` (retune sweep with `outproj_fused + layer_repeat=4`).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx5_s16_models_outprojfused.txt` (retune sweep with `outproj_fused + layer_repeat=5`).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx6_s16_models_outprojfused.txt` (retune sweep with `outproj_fused + layer_repeat=6`).
- `target/benchmarks/llama_stepwise_outproj_fused_layerrepeat456_calibration.md` (repeat-parameter retune table against llama.cpp baseline).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx5_s16_models_outprojfused_r{2,3}.txt` (`layer_repeat=5` stability reruns).
- `target/benchmarks/llama_stepwise_outproj_fused_layerx5_stability_r3.md` (`layer_repeat=5` stability medians for fused output projection).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat5_outprojfuse_stable.md` (stable calibration using `outproj_fused + layer_repeat=5`).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat5_outprojfuse_stable_impact.md` (impact vs prior stable references).
- `target/benchmarks/llama_stepwise_profile_outprojfused_layerx5_smoke.txt` (CPU/Metal smoke check for preset profile flag).
- `target/benchmarks/llama_stepwise_profile_outprojfused_balanced_smoke.txt` (CPU/Metal smoke check for balanced preset profile).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_balanced_s16_models.txt` (6-model run for balanced preset profile).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_profile_outprojfused_balanced.md` (calibration table for balanced preset profile).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_profile_outprojfused_balanced_impact.md` (impact vs prior stable references for balanced preset profile).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_balanced_statickv_s16_models.txt` (6-model run for balanced preset with static KV-head precompute).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_profile_outprojfused_balanced_statickv.md` (calibration table for balanced+static-KV profile).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_profile_outprojfused_balanced_statickv_impact.md` (impact vs prior balanced calibration).
- `target/benchmarks/llama_stepwise_profile_layerx5_statickvhead_ab_qwen35_layer0.txt` (representative CPU/Metal A/B for static KV-head precompute on Qwen3.5 layer0).
- `target/benchmarks/llama_stepwise_profile_layerx5_statickvhead_ab_qwen35_layer0_impact.md` (representative A/B impact summary).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_layerx5_statickv_on_s16_models.txt` (6-model run with static KV-head precompute on).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_layerx5_statickv_off_s16_models.txt` (same 6-model run with static KV-head precompute off).
- `target/benchmarks/llama_stepwise_profile_outprojfused_layerx5_statickv_impact.md` (6-model on/off impact table for static KV-head precompute).
- `target/benchmarks/llama_stepwise_profile_outprojfused_layerx5_statickv_checksum_check.md` (checksum parity check for static KV-head precompute on/off).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat5_outprojfuse_statickv.md` (calibration table with static KV-head precompute enabled).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat5_outprojfuse_statickv_impact.md` (impact vs prior `layer_repeat=5 outproj_fused` stable calibration).
- `target/benchmarks/llama_stepwise_profile_layerx5_balancedconcat_ab_qwen35_layer0.txt` (representative A/B for fused-output balanced head concat).
- `target/benchmarks/llama_stepwise_profile_layerx5_balancedconcat_ab_qwen35_layer0_impact.md` (representative balanced-concat A/B impact summary).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_layerx5_statickv_balancedconcat_on_s16_models.txt` (6-model run with balanced head concat on).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_layerx5_statickv_balancedconcat_off_s16_models.txt` (6-model run with balanced head concat off).
- `target/benchmarks/llama_stepwise_profile_outprojfused_layerx5_statickv_balancedconcat_impact.md` (6-model balanced-concat on/off impact + checksum deltas).
- `target/benchmarks/llama_stepwise_profile_layerx5_positiondelta_ab_qwen35_layer0.txt` (representative A/B for QUERY_POS position-delta update).
- `target/benchmarks/llama_stepwise_profile_layerx5_positiondelta_ab_qwen35_layer0_impact.md` (representative position-delta A/B impact summary).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_layerx5_statickv_positiondelta_on_s16_models.txt` (6-model run with position-delta updates on).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_layerx5_statickv_positiondelta_off_s16_models.txt` (6-model run with position-delta updates off).
- `target/benchmarks/llama_stepwise_profile_outprojfused_layerx5_statickv_positiondelta_impact.md` (6-model position-delta on/off impact + checksum deltas).
- `target/benchmarks/llama_stepwise_models_layerx3_maskdelta_on_vs_off.md` (`layer_repeat=3` matched A/B for `mask_delta` on/off).
- `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwritecache_realmlp_layerx3_s16_models.txt` (`layer_repeat=3` with `--decode-stepwise-kv-cache-write-to-cache`).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwritecache_realmlp_layerrepeat3_maskdelta.md` (cache-view KV-write calibration).
- `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwritecache_realmlp_layerrepeat3_impact.md` (`layer_repeat=3` base vs cache-view KV-write).

Variant note:

- On this host, larger step windows (`16`/`32`) yielded more stable Metal advantage than `steps=8` for the model-shaped set.
- Canonical parity reporting keeps `--decode-steps 16` as default (balance between stability and avoiding extra amortization drift); use `steps=32` as a sensitivity check.
- `--decode-stepwise-no-mask-delta` can disable the delta-update path for matched A/B checks; in-session A/B stayed near parity on model-shaped stepwise (`on/off ~1.001` overall, CPU `~1.009`, MTL0 `~0.991`).
- `--decode-stepwise-kv-proj` increased measured `ms/token` in most sampled cases (~`+4%` to `+12%` on CPU and mostly positive on Metal in the matched run), but did not close the proxy/cpp gap by itself.
- Adding `--decode-stepwise-kv-cache-write` on top of block+kv+real-MLP nudged MTL0 proxy/cpp (`~0.252 -> ~0.256`) while keeping total drift small (mostly within ~`±3%` per case in the matched run).
- `--decode-stepwise-block --decode-stepwise-kv-proj` substantially increases modeled cost; in the current 6-model set average proxy/cpp ratio moved to ~`0.89x` (CPU) and ~`0.36x` (Metal), so Metal-side gap remains the primary next target.
- Adding `--decode-stepwise-sync-step` on top of block+kv nudged Metal average to ~`0.38x` in the sampled set, but still leaves a substantial Metal-side gap.
- Quantized Qwen3.5 (`Q4_K_M`) block-MLP wiring now runs with `block_mlp_real=true` on both CPU and Metal in the sampled one-case check.
- In the full 6-model real-MLP sweep (`--block-mlp-model`), average proxy/cpp moved to ~`0.31x` (CPU) and ~`0.26x` (Metal), indicating the deterministic-fallback block path had been overestimating cost relative to model-derived values.
- In the mask-delta rerun of the same 6-model real-MLP sweep, average proxy/cpp is `~0.306x` (CPU) and `~0.252x` (MTL0).
- In the same mode with `--decode-stepwise-layer-repeat 3`, average proxy/cpp moves to `~0.941x` (CPU), `~0.965x` (MTL0), `~0.953x` overall.
- `--decode-stepwise-layer-repeat 4` overshoots in aggregate (`~1.363x` CPU, `~1.152x` MTL0, `~1.258x` overall), so `3` is the closer calibration target in this pass.
- `layer_repeat=3` mask-delta A/B (`target/benchmarks/llama_stepwise_models_layerx3_maskdelta_on_vs_off.md`) shows overall `on/off ~0.965` in the sampled rerun, with wider per-case noise than earlier non-layer-repeat A/B runs.
- Enabling `--decode-stepwise-kv-cache-write-to-cache` under `layer_repeat=3` reduced modeled cost too far in this host run (`overall proxy/cpp ~0.717`, CPU `~0.820`, MTL0 `~0.615`), so this variant is currently not selected as parity default.
- Legacy layerx3 calibration default: `--decode-stepwise-layer-repeat 3` with `--decode-stepwise-kv-cache-write` (without `--decode-stepwise-kv-cache-write-to-cache`).
- Current active optimization track (user-selected): `--decode-stepwise-profile-outproj-fused-layerx5` with static KV-head precompute enabled.
- Stability reruns (`r=3`, `iters=15`) show significant command-to-command variance on this host:
  - `layer_repeat=3` stable median calibration moved to CPU `~0.869`, MTL0 `~0.634`, overall `~0.751`.
  - `layer_repeat=4` stable median calibration moved to CPU `~1.140`, MTL0 `~0.802`, overall `~0.971`.
  - stable impact artifact: `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_vs_4_stable_impact.md`.
- Layer-by-layer sweeps (`--block-mlp-layer-range`) are now the primary optimization input:
  - Qwen3.5-4B (`0..31`): CPU mean/std `~12.607/0.104`, MTL0 `~9.053/0.076` ms/token.
  - Qwen3-8B (`0..35`): CPU mean/std `~23.824/0.169`, MTL0 `~16.342/0.091` ms/token.
  - Current layerwise CV is low (roughly `~0.5%` to `~0.8%`), suggesting no extreme outlier layers under this synthetic decode proxy condition.
- Experimental `--decode-stepwise-elide-mask-host-buffer` (host-buffer elision for incremental mask updates) is currently **opt-in**:
  - full 6-model balanced-order sweep average: CPU `elide/base ~0.951`, MTL0 `~0.983`,
  - full-sweep stability reruns (`r=3`, median-of-runs) average: CPU `~0.937`, MTL0 `~0.995`,
  - per-model direction is mixed (e.g., InternVL/Gemma improve on CPU, ELYZA regresses on CPU),
  - therefore default remains `mask_host_elide=false`; use it as an explicit experiment knob.
- KV-head cache reuse optimization (build rotated K and transposed/contiguous V once per KV head, then reuse across grouped query heads) now applies in stepwise decode path:
  - stable (`r=3`) post-change vs pre-change median baseline:
    - CPU `post/base ~0.840`,
    - MTL0 `post/base ~0.935`,
  - stable calibration vs `llama.cpp` moved:
    - CPU avg proxy/cpp `0.869 -> 0.777`,
    - MTL0 avg proxy/cpp `0.634 -> 0.630`,
    - overall `0.752 -> 0.704`,
  - output checksum parity remained exact in the sampled 6-model check (`max abs delta = 0.0`).
- Experimental `--decode-stepwise-fuse-output-proj` is available for stepwise output projection:
  - implementation: concatenate per-head attention outputs (`Context::concat(..., dim=0)`) then run one `W_O * HEADS` matmul,
  - preset flag: `--decode-stepwise-profile-outproj-fused-layerx5` (`outproj_fused=true`, `layer_repeat=5`),
  - backend-balanced preset flag: `--decode-stepwise-profile-outproj-fused-balanced` (`CPU layer_repeat=5`, `MTL layer_repeat=6`, `outproj_fused=true`),
  - stepwise output tags preset runs with `profile=outproj_fused_layerx5`,
  - static-KV precompute controls:
    - `--decode-stepwise-static-kv-head-precompute`,
    - `--decode-stepwise-no-static-kv-head-precompute`,
    - output includes `kvhead_static_precompute=<true|false>`,
  - default remains `outproj_fused=false` (opt-in),
  - full 6-model balanced sweep:
    - CPU `fused/base ~0.884`,
    - MTL0 `fused/base ~0.941`,
    - overall `~0.912`,
  - `r=3` stability medians:
    - CPU `fused/base ~0.885`,
    - MTL0 `fused/base ~0.940`,
    - overall `~0.913`,
  - calibration impact at current `layer_repeat=3` baseline:
    - CPU proxy/cpp `0.777 -> 0.693`,
    - MTL0 proxy/cpp `0.630 -> 0.595`,
    - overall `0.704 -> 0.644`,
  - parity retune sweep (`layer_repeat=4/5/6`, single-run):
    - `repeat4` overall avg proxy/cpp `~0.807`,
    - `repeat5` overall avg proxy/cpp `~0.985`,
    - `repeat6` overall avg proxy/cpp `~1.156`,
    - best overall parity distance in this pass: `layer_repeat=5`,
  - `layer_repeat=5` stability reruns (`r=3`) with `outproj_fused=true`:
    - CPU avg proxy/cpp `~1.073`,
    - MTL0 avg proxy/cpp `~0.908`,
    - overall `~0.991`,
  - balanced preset (`CPU=5`, `MTL0=6`) full-run calibration:
    - CPU avg proxy/cpp `~1.076`,
    - MTL0 avg proxy/cpp `~1.063`,
    - overall `~1.070`,
  - static-KV precompute A/B under `outproj_fused_layerx5` (6-model, s16):
    - CPU avg `on/off ~0.933`,
    - MTL0 avg `on/off ~0.964`,
    - overall `on/off ~0.949`,
    - checksum parity: exact (`max abs delta = 0.0`),
  - calibration refresh with static-KV precompute (same cpp reference):
    - CPU avg proxy/cpp `~1.016`,
    - MTL0 avg proxy/cpp `~0.889`,
    - overall `~0.953`,
  - balanced preset + static-KV calibration:
    - CPU avg proxy/cpp `~1.025`,
    - MTL0 avg proxy/cpp `~1.041`,
    - overall `~1.033`,
  - fused-output balanced head-concat A/B (`outproj_fused_layerx5`, static-KV on, 6-model):
    - CPU avg `on/off ~1.000`,
    - MTL0 avg `on/off ~0.983`,
    - overall `on/off ~0.992`,
    - checksum parity: exact (`max abs delta = 0.0`),
  - QUERY_POS position-delta A/B (`outproj_fused_layerx5`, static-KV on, 6-model):
    - CPU avg `on/off ~0.990`,
    - MTL0 avg `on/off ~1.001`,
    - overall `on/off ~0.995`,
    - checksum parity: exact (`max abs delta = 0.0`),
  - policy:
    - keep `outproj_fused` default-off for canonical layerx3 path,
    - active optimization track (user-selected) is `outproj_fused + layer_repeat=5`,
    - static-KV precompute is now enabled by default in this track,
    - balanced + static-KV is a near-parity alternative when closer-to-1.0 calibration is prioritized,
    - keep `head_concat_balanced=false` as default (marginal/mixed impact), but keep the A/B toggle available,
    - keep `position_delta=true` as default (small positive overall impact), with explicit A/B toggle retained.

Output:

- `target/benchmarks/llama_proxy_vs_cpp_report.md` (markdown summary with all parsed pairs/cases).
- The attention parser supports decode-mode lines (`attn decode bench ... q_seq=... kv_seq=...`) and stepwise decode lines (`attn decode stepwise bench ... kv_start=... steps=... avg_token=...`).

## Rust vs C++ parity test command

```bash
cargo test -p llama-rs --features link-system --test mlp_cpp_parity
```

Notes:

- C++ reference source: `llama-rs/tests/cpp/mlp_reference.cpp`
- C++ reference now executes the same graph with ggml CPU backend (`ggml_mul_mat`/`ggml_silu`/`ggml_mul`).
- Rust test compiles this source at runtime and compares outputs against
  `mlp_inference_with_weights_repeats(..., LlamaBackend::Cpu, 1)`.
- Coverage now includes multiple shape cases (`8x16`, `16x32`, `32x64`) and
  an additional CPU-vs-Metal parity test across the same matrix.

## Attention parity test command

```bash
cargo test -p llama-rs --features link-system --test attention_parity
```

This test validates CPU-vs-Metal parity for the minimal attention path and also includes a causal-mask CPU execution check.

## Same-workload C++ vs Rust decode comparator

To avoid comparing unlike workloads, use the dedicated same-graph comparator:

```bash
cargo run -p llama-rs --example bench_attention_decode_cpp_compare --features link-system -- \
  --decode-kv 128 --warmup 2 --iters 10 cpu metal
```

What it does:

- builds and runs `llama-rs/tests/cpp/attention_decode_proxy_reference.cpp` (ggml C++ reference),
- runs `llama-rs` decode-proxy path with the same deterministic graph and shape cases,
- reports `rust_avg`, `cpp_avg`, `rust/cpp`, and checksum deltas.
- C++ include resolution respects `GGML_RS_GGML_INCLUDE_DIR` when set, then falls back to
  `target/vendor/ggml/include` from the workspace.

Artifacts:

- raw output: `target/benchmarks/llama_attention_decode_samework_cpp_vs_rust.txt`
- markdown summary: `target/benchmarks/llama_attention_decode_samework_cpp_vs_rust.md`

Current snapshot:

- `Max checksum_rel` is about `0.000391` (same-workload numerical drift is small).

## Same-workload C++ vs Rust stepwise comparator

The same comparator now supports stepwise decode growth with causal mask updates:

```bash
cargo run -p llama-rs --example bench_attention_decode_cpp_compare --features link-system -- \
  --decode-kv 143 --stepwise-start 128 --stepwise-steps 16 --past 127 --warmup 2 --iters 10 cpu metal
```

What it does:

- runs Rust and C++ through the same deterministic per-step graph (`q_len=1`),
- advances `kv_len` from `128` across `16` steps with synchronized causal mask/past updates,
- reports per-token timing (`rust_avg` / `cpp_avg`) and checksum deltas.
- argument constraints: `--stepwise-start == --past + 1` and
  `--decode-kv == --stepwise-start + --stepwise-steps - 1`.

Artifacts:

- raw output: `target/benchmarks/llama_attention_decode_stepwise_samework_cpp_vs_rust.txt`
- markdown summary: `target/benchmarks/llama_attention_decode_stepwise_samework_cpp_vs_rust.md`

Current snapshot:

- `CPU avg rust/cpp`: about `0.727`
- `MTL0 avg rust/cpp`: about `0.669`
- `Max checksum_rel`: about `0.000089`

Latest optimization pass:

- added backend partial tensor-write APIs (`set_*_backend_at`) and switched stepwise causal-mask updates to delta writes on the canonical decode path (`q_len=1`, `past+1==kv_start`),
- impact artifact: `target/benchmarks/llama_attention_decode_stepwise_samework_maskdelta_impact.md`,
- observed `post/pre` on `rust_avg_ms/token`: about `0.805` overall (`CPU ~0.804`, `MTL0 ~0.806`).
- model-shaped A/B toggle artifact (`--decode-stepwise-no-mask-delta`): `target/benchmarks/llama_stepwise_models_maskdelta_on_vs_off.md` (`on/off ~1.001` overall).
- release-asm spot-check (`cargo rustc -p llama-rs --lib --release -- -C codegen-units=1 --emit=asm`):
  - `fill_causal_mask_values` is lowered to a vectorized loop,
  - stepwise hot path calls `Tensor::set_f32_backend_at` only on delta-update branches,
  - no steady-state allocator calls were observed in the successful stepwise loop path.

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
