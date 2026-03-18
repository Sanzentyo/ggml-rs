# 2026-03-15 migration log

Detailed entries migrated from `docs/llama-rs/WORKLOG.md` during worklog compaction.

## Detailed summary

- Metadata-driven auto-resolution path is validated with explicit `resolution_mode` output (`FullMetadata` / `TensorHeuristic`).
- MLP and attention layer examples are verified on CPU and Metal for synthetic fixtures.
- Link-system parity tests (`mlp_cpp_parity`, `attention_parity`) pass after RoPE integration fixes.
- Unified example argument parsing on `clap` derive + typed CLI structs across `llama-rs/examples` (including `gguf`, `idle`, `bench_attention_layer`, and `bench_attention_decode_cpp_compare`).
- Removed remaining constructor-wrapper usage in `ggml-rs` call sites and switched to
  generic constructor APIs:
  - `new_tensor_typed::<T, N>(Dims<N>)`,
  - `new_tensor_1d::<T>(Length)`,
  - `new_tensor_2d::<T>(Shape2D)`,
  - `new_tensor_3d::<T>(Shape3D)`,
  - `new_tensor_4d::<T>(Shape4D)`.

## 2026-03-18 balanced profile fine-tune (unblocked)

- Unblocked `fine-tune-balanced-profile` and re-ran balanced profile calibration against:
  - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_profile_outprojfused_balanced_statickv.md`.
- Executed 7 repeat-pair candidates:
  - `cpu4_mtl5`, `cpu4_mtl6`, `cpu5_mtl5`, `cpu5_mtl6`, `cpu5_mtl7`, `cpu6_mtl6`, `cpu6_mtl7`.
- Candidate raw artifacts:
  - `target/benchmarks/review4_finetune_balanced_profile_cpu4_mtl5.txt`
  - `target/benchmarks/review4_finetune_balanced_profile_cpu4_mtl6.txt`
  - `target/benchmarks/review4_finetune_balanced_profile_cpu5_mtl5.txt`
  - `target/benchmarks/review4_finetune_balanced_profile_cpu5_mtl6_refresh.txt`
  - `target/benchmarks/review4_finetune_balanced_profile_cpu5_mtl7.txt`
  - `target/benchmarks/review4_finetune_balanced_profile_cpu6_mtl6.txt`
  - `target/benchmarks/review4_finetune_balanced_profile_cpu6_mtl7.txt`
- Selection summary:
  - `target/benchmarks/review4_finetune_balanced_profile_sweep_summary.md`
  - `target/benchmarks/review4_finetune_balanced_profile_sweep_summary.json`
  - `target/benchmarks/review4_finetune_balanced_profile_cpu6_mtl7_calibration.md`
  - `target/benchmarks/review4_finetune_balanced_profile_cpu6_mtl7_impact.md`
  - selected pair: `cpu6_mtl7` with averages:
    - CPU `~1.024`, MTL0 `~0.965`, overall `~0.995`, balance gap `~0.059`.
- Applied preset update in `llama-rs/examples/bench_attention_layer.rs`:
  - `--decode-stepwise-profile-outproj-fused-balanced` now resolves to
    `CPU layer_repeat=6`, `MTL0 layer_repeat=7`,
  - stepwise tag now emits `profile=outproj_fused_balanced_cpu6_mtl7`.

## 2026-03-18 cpp baseline refresh + balanced retune

- Refreshed llama.cpp baselines via rebuilt `llama-bench` using:
  - `target/benchmarks/llama_cpp_baseline_all.jsonl`
  - `target/benchmarks/llama_cpp_baseline_extra.jsonl`
  - run trace: `target/benchmarks/review4_llama_cpp_baseline_refresh.log`
- Baseline coverage check: all 6 models captured with both `n_gpu_layers={0,99}` and decode rows (`n_prompt=0,n_gen=128`) present for calibration.
- Re-ranked the same 7 balanced repeat-pair candidates against refreshed decode baselines:
  - `target/benchmarks/review4_finetune_balanced_profile_sweep_summary.md`
  - `target/benchmarks/review4_finetune_balanced_profile_sweep_summary.json`
- New selected pair: `cpu5_mtl7`:
  - calibration table: `target/benchmarks/review4_finetune_balanced_profile_cpu5_mtl7_calibration.md`
  - impact vs previous tuned preset (`cpu6_mtl7`):
    - `target/benchmarks/review4_finetune_balanced_profile_cpu5_mtl7_impact.md`
  - averages: CPU `~0.981`, MTL0 `~1.010`, overall `~0.995`, balance gap `~0.029`.
- Updated balanced preset wiring/tag in `bench_attention_layer`:
  - `--decode-stepwise-profile-outproj-fused-balanced` now resolves to
    `CPU layer_repeat=5`, `MTL0 layer_repeat=7`,
  - stepwise tag now emits `profile=outproj_fused_balanced_cpu5_mtl7`.

## 2026-03-18 step1 continuation on refreshed balanced preset

- Resumed step `1` layer sweep on ELYZA (`block_layer=0..7`) with refreshed balanced preset (`cpu5_mtl7`):
  - raw: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers0_7.txt`
  - summary: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers0_7_summary.{md,csv}`
  - aggregate avg_token: CPU `~28.427 ms`, MTL0 `~28.061 ms`, overall `~28.244 ms`.
- Follow-up hotspot-window A/B (`layers 5..7`) for `head_stage_buf`:
  - base: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers5_7_headstage_base.txt`
  - on: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers5_7_headstage_on.txt`
  - impact: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers5_7_headstage_impact.md`
  - aggregate (`on/base`):
    - token ratio: CPU `~0.997`, MTL0 `~0.995`, overall `~0.996`,
    - setup ratio: CPU `~1.006`, MTL0 `~1.004`, overall `~1.005`,
    - checksum delta max: `0.0`.
- Decision on this refreshed slice: keep `head_stage_buf=false` as default (token gain is small and setup regresses).

## 2026-03-18 step2 cross-model recheck (`head_stage_buf`)

- Ran cross-model operator A/B (`6` models, CPU/Metal) for `head_stage_buf` under refreshed balanced preset (`cpu5_mtl7`):
  - base: `target/benchmarks/review4_step2_balanced_cpu5_mtl7_headstage_base.txt`
  - variant: `target/benchmarks/review4_step2_balanced_cpu5_mtl7_headstage_on.txt`
  - impact: `target/benchmarks/review4_step2_balanced_cpu5_mtl7_headstage_impact.md`
- Aggregate (`on/base`) from this pass:
  - token ratio: CPU `~0.995`, MTL0 `~0.994`, overall `~0.994`,
  - checksum delta max: `0.0`.
- Relative to refreshed cpp baselines, parity objective (`proxy/cpp -> 1.0`) moved:
  - overall `~0.993 -> ~0.987` (further from `1.0`),
  - CPU `~0.986 -> ~0.981`,
  - MTL0 `~1.000 -> ~0.994`.
- Decision on refreshed 6-model set: keep `head_stage_buf=false` for parity-focused default policy.

## 2026-03-18 step3 upstream-suite refresh

- Ran refreshed upstream example suite on the same lock:
  - run log: `target/benchmarks/review4_step3_upstream_suite_refresh.txt`
  - summary: `target/benchmarks/review4_step3_upstream_suite_refresh_summary.md`
- Status:
  - `passed=3`, `failed=0`, `skipped_run_targets=13`.
- Skipped runs remained model/data-argument dependent targets and were explicitly reported with `GGML_UPSTREAM_RUN_ARGS_<TARGET>` guidance.
- Unified root `ggml-rs/examples` on clap derive argument parsing (including
  `backend_matmul`, `bench_matmul`, `perf_metal`, synthetic GPT-J/Magika/MNIST/SAM/YOLO,
  and `bench_upstream_suite`) and re-verified runtime CPU/Metal smoke:
  - `target/benchmarks/review3_constructor_clap_runtime_smoke.txt`.
- Added loop-reuse synthetic performance pass for vision/MNIST proxies by reusing
  context/graph across `--synthetic-iters`:
  - re-run artifacts:
    - `target/benchmarks/vision_mnist/rust_mnist_eval_post_loopreuse.txt`,
    - `target/benchmarks/vision_mnist/rust_mnist_train_post_loopreuse.txt`,
    - `target/benchmarks/vision_mnist/rust_sam_post_loopreuse.txt`,
    - `target/benchmarks/vision_mnist/rust_yolo_post_loopreuse.txt`,
  - impact summary:
    - `target/benchmarks/vision_mnist/loopreuse_impact.md`
    - `post/pre` per-iter: `mnist-eval ~0.347`, `mnist-train ~0.914`, `sam ~0.282`, `yolo ~0.735`.
- Refactored clap-unified example error surfaces to `thiserror`-based typed errors (while preserving the existing validation/error-policy behavior).
- Re-verified the clap-unified runtime surfaces on CPU/Metal with `--features link-system` and recorded:
  - `target/benchmarks/llama_rs_clap_refactor_runtime_smoke.txt`.
- Resumed step `1` (layer-by-layer optimization loop) on ELYZA (`block_layer=0..7`) with the current lock (`outproj_fused_layerx5 + static_kv + kv_proj + kv_write + block`) and recorded:
  - raw: `target/benchmarks/llama_rs_stepwise_resume_after_clap_elyza_layers0_7.txt`
  - summary: `target/benchmarks/llama_stepwise_resume_after_clap_elyza_layers0_7_summary.{csv,md}`
- Hotspot A/B results on `block_layer=2..7`:
  - `mask_host_elide`: regression (`~1.008` on/base),
  - `head_stage_buf`: near-neutral (`~1.000` on/base),
  - `block_gateup_fused`: near-neutral (`~1.000` on/base),
  - `head_concat_balanced`: regression (`~1.010` on/base).
- Delta toggle recheck on the same hotspot window:
  - `--decode-stepwise-no-mask-delta`: `~0.994` variant/base,
  - `--decode-stepwise-no-position-delta`: `~0.994` variant/base.
- Stability rerun (`r=3` median) for the same delta toggles:
  - `target/benchmarks/llama_stepwise_resume_after_clap_elyza_layers2_7_delta_stability_r3.md`
  - `no-mask/base ~0.998`, `no-pos/base ~0.999` overall (near-neutral), so defaults stay unchanged.
- Post-review3 decode-API continuation pass on ELYZA (`block_layer=5..7`, `decode-kv=129`, `steps=8`) rechecked delta toggles under the latest lock:
  - base: `target/benchmarks/review3_decodeapi_step2_elyza_layers5_7_delta_base.txt`,
  - no-mask: `target/benchmarks/review3_decodeapi_step2_elyza_layers5_7_delta_nomask.txt`,
  - no-position: `target/benchmarks/review3_decodeapi_step2_elyza_layers5_7_delta_nopos.txt`,
  - impact: `target/benchmarks/review3_decodeapi_step2_elyza_layers5_7_delta_impact.md`,
  - checksum: `target/benchmarks/review3_decodeapi_step2_elyza_layers5_7_delta_checksum_check.md`.
  - means (`variant/base`): `no-mask ~1.018`, `no-position ~1.041` overall, checksum deltas `0.0`;
    defaults remain `mask_delta=true`, `position_delta=true`.
- Added file-based lock helper (`cargo +nightly -Zscript scripts/agent_lock.rs -- ...`) and parallel subagent logging policy:
  - lock protocol doc: `docs/llama-rs/KNOWLEDGE_BASE.md` (`cargo|cpp|bench` locks),
  - subagent logs: `docs/llama-rs/worklog/subagents/{foundation,gpt2,gptj_magika,vision_mnist}.md`.
- Executed parallel subagent implementation/validation for ggml upstream example reproduction:
  - direct parity batch artifacts:
    - `target/benchmarks/foundation_parity_report.{md,json}`,
  - synthetic parity batches:
    - `target/benchmarks/gpt2_parity_summary.{txt,json}`,
    - `target/benchmarks/gptj_magika_synth_{parity_summary,perf_summary}.txt`,
    - `target/benchmarks/vision_mnist/parity_perf_{summary.md,json}`,
  - consolidated matrix:
    - `docs/ggml-rs/EXAMPLE_PARITY_MATRIX.md`.
  - consolidated rollout summary:
    - `target/benchmarks/parallel_subagent_rollout_summary.md`.
- Expanded `thiserror` rollout across remaining `llama-rs/examples` (`run()` + typed `ExampleError` boundary) and re-verified link-system runtime surfaces:
  - `target/benchmarks/llama_rs_thiserror_rollout_runtime_smoke.txt`.
- Added stepwise warmup+bench reuse path to reduce repeated backend/context setup during layer sweeps:
  - new API: `run_attention_decode_stepwise_bench_with_cache_repeats_with_block_mlp`,
  - `bench_attention_layer` now uses the new API in `--decode-steps` mode,
  - runtime smoke: `target/benchmarks/llama_rs_stepwise_backend_context_reuse_smoke.txt`.
- Added phase-init elision on top of the warmup+bench reuse path (skip redundant KV/precompute reinit between warmup and bench when cache writes do not mutate persistent KV tensors):
  - runtime smoke: `target/benchmarks/llama_rs_stepwise_phase_init_elision_smoke.txt`,
  - ELYZA `block_layer=5..7` remeasure and impact:
    - raw: `target/benchmarks/llama_rs_stepwise_phase_init_elision_elyza_layers5_7.txt`,
    - summary: `target/benchmarks/llama_stepwise_phase_init_elision_elyza_layers5_7_impact.md`,
    - `new/base` avg_token: CPU `~0.629`, MTL0 `~0.549` (baseline from `llama_rs_stepwise_resume_after_clap_elyza_layers0_7.txt` subset).
- Added one-time backend loader guard (`ensure_backends_loaded`) across `llama-rs` runtime paths (`bench`, `batched`, `smoke`, `inference`) to remove repeated `Backend::load_all()` calls.
  - validation/remeasure:
    - raw: `target/benchmarks/llama_rs_stepwise_backend_load_once_elyza_layers5_7.txt`,
    - impact: `target/benchmarks/llama_stepwise_backend_load_once_elyza_layers5_7_impact.md`,
    - `new/base` avg_token vs phase-init-elision baseline: CPU `~1.013`, MTL0 `~1.000` (near-neutral; keep as cleanup/hardening).
- Added block-MLP layer-loop cache in `bench_attention_layer` (`HashMap<(hidden_features, block_layer), (MlpWeights, block_mlp_real)>`) so repeated case/layer sweeps reuse resolved model-layer weights instead of re-decoding each time.
  - smoke (CPU+Metal, duplicated case to exercise cache path): `target/benchmarks/llama_rs_stepwise_layer_loop_reuse_cache_smoke.txt`.
- Added stepwise setup-time instrumentation (`setup=... ms`) to separate one-time graph/context preparation cost from token compute cost in decode-stepwise output.
  - baseline artifact for graph-level reuse planning:
    - raw: `target/benchmarks/llama_rs_stepwise_graph_reuse_setup_baseline_elyza_layers5_7.txt`,
    - summary: `target/benchmarks/llama_stepwise_graph_reuse_setup_baseline_elyza_layers5_7.md`,
    - `block_layer=5..7` averages: setup `~456.6 ms` (CPU), `~462.3 ms` (MTL0) per call.
- Added graph-level layer-sweep reuse path for stepwise decode benchmarking.
  - new API/report:
    - `run_attention_decode_stepwise_bench_sweep_with_cache_repeats_with_block_mlp`,
    - `AttentionDecodeStepwiseBenchSweepReport`.
  - `bench_attention_layer` now uses the shared-setup sweep path for decode-stepwise + model-backed multi-layer sweeps.
  - output markers:
    - `graph_reuse_sweep=true`,
    - `setup_shared=... ms` (shared one-time setup),
    - `setup=... ms` (amortized per-layer setup).
  - runtime artifact: `target/benchmarks/llama_rs_stepwise_graph_reuse_layer_sweep_elyza_layers5_7.txt`.
  - impact summary: `target/benchmarks/llama_stepwise_graph_reuse_layer_sweep_elyza_layers5_7_impact.md`.
  - measured setup ratio (`new/base`): CPU `~0.315`, MTL0 `~0.309`, overall `~0.312`.
- Added token-compute hotspot optimization for query-side RoPE application in stepwise decode:
  - apply RoPE once on reshaped multi-head query tensor (`head_dim x n_heads x seq`) instead of per-head repeated RoPE nodes.
  - runtime artifact:
    - `target/benchmarks/llama_rs_stepwise_graph_reuse_layer_sweep_elyza_layers5_7_qrope_multihead.txt`.
  - impact summary:
    - `target/benchmarks/llama_stepwise_graph_reuse_qrope_multihead_elyza_layers5_7_impact.md`.
  - measured post/base ratios:
    - CPU: `avg_token ~1.001` (near-neutral),
    - MTL0: `avg_token ~0.944` (improved),
    - overall: `avg_token ~0.977` (improved),
    - checksum parity: `max abs delta = 0.0`.
- Ran CPU-side follow-up check (`head_stage_buf=true`) after the qrope pass:
  - impact: `target/benchmarks/llama_stepwise_headstage_after_qrope_refactor_smoke_impact.md`,
  - sampled ratio (`variant/base`): CPU `~1.022`, MTL0 `~0.995`, overall `~1.011`,
  - decision: keep `head_stage_buf=false`.
- Started user-requested refactor pass toward ADT/type-state/trait modularization:
  - added `llama-rs/src/inference/stepwise_plan.rs`:
    - `StepwiseBenchPlan` ADT,
    - type-state builder (`backend/config` required before `build()`),
    - trait-based static dispatch (`StepwiseBlockMlpRunSet`) for single vs sweep runs.
  - updated `bench_attention_layer` to run stepwise benchmarks via `StepwiseBenchPlan` instead of direct `run_*_bench*` calls.
  - split helper ops from `inference.rs` into `llama-rs/src/inference/attention_ops.rs`.
  - fixed option precedence so explicit decode-stepwise toggles override profile presets.
  - runtime smoke artifacts:
    - `target/benchmarks/llama_rs_stepwise_refactor_plan_smoke.txt`,
    - `target/benchmarks/llama_rs_refactor_profile_override_smoke.txt`,
    - `target/benchmarks/llama_rs_refactor_attention_ops_smoke.txt`.
- Continued the same refactor by extracting stepwise decode core from `inference.rs`:
  - added `llama-rs/src/inference/stepwise_decode.rs` and moved:
    - `AttentionDecodeStepwiseConfig`,
    - `AttentionDecodeStepwiseReport`,
    - stepwise bench reports and all `run_attention_decode_stepwise_*` runners.
  - `llama-rs/src/inference.rs` now re-exports stepwise public APIs through the new module.
  - re-validated full workspace (`fmt`, `clippy`, `test`) and runtime CPU/Metal smoke:
    - `target/benchmarks/llama_rs_stepwise_refactor_stepwisecore_smoke.txt`.
- Completed ADT-first API naming pass to remove long `run_*` public entrypoints across `llama-rs`:
  - stepwise plan rename:
    - `StepwiseBenchPlan` -> `DecodeStepPlan`,
    - `StepwiseBenchPlanBuilder` -> `DecodeStepPlanBuilder`,
    - `StepwiseBlockMlpRunSet` -> `DecodeStepBenchSet`,
  - stepwise benchmarks and preflight now route through `DecodeStepPlan::{bench, execute_single}`.
  - crate-wide `run_*` public function prefixes were removed (linear/mlp/attention/decode/batched/smoke/simple/idle), and examples/tests were updated accordingly.
- Added repository-level Rust policy file for auto-loaded guidance:
  - `.github/copilot-instructions.md` (ADT-first, type-state, static dispatch, validation/runbook defaults).
- Runtime re-validation after ADT + naming migration:
  - `target/benchmarks/llama_rs_stepwise_decodeplan_smoke.txt`,
  - `target/benchmarks/llama_rs_stepwise_decodeplan_cppcompare_smoke.txt`.
- Final post-rename runtime recheck (CPU/Metal):
  - `target/benchmarks/llama_rs_backend_smoke_decodeplan_postrename.txt`,
  - `target/benchmarks/llama_rs_stepwise_decodeplan_postrename_smoke.txt`.
- Continued ADT consolidation for decode-proxy execution:
  - added `llama-rs/src/inference/decode_proxy_plan.rs` with:
    - `AttentionDecodePlan` + type-state builder,
    - trait-based static dispatch (`AttentionDecodeSource`),
    - source ADTs (`AttentionDecodeCacheInput`, `AttentionDecodeWeightsInput`).
  - migrated decode-proxy call sites in:
    - `bench_attention_layer`,
    - `bench_attention_decode_cpp_compare`,
    - `idle`.
  - reduced duplicated decode proxy public variants from `lib.rs` export surface and kept shared execution internals private.
- Runtime re-validation for the decode-proxy ADT pass (CPU/Metal, link-system):
  - `target/benchmarks/llama_rs_decode_proxy_plan_smoke_cpu_metal.txt`,
  - `target/benchmarks/llama_rs_backend_smoke_decode_proxy_plan_postrefactor.txt`.
- Reduced function-heavy attention helper code by introducing trait-driven attention ops in `llama-rs/src/inference/attention_ops.rs`:
  - `RotaryApplier` + `LlamaRotaryApplier` (single-head and multi-head RoPE application),
  - `HeadConcatStrategy` with concrete `LeftFoldHeadConcat` and `BalancedHeadConcat`,
  - `HeadConcatMetadata` for typed concat error context.
- Migrated attention call sites to the new trait APIs:
  - `llama-rs/src/inference.rs`,
  - `llama-rs/src/inference/stepwise_decode.rs`.
- Re-validated after the attention-ops trait refactor:
  - `cargo fmt --all`,
  - `cargo clippy --workspace --all-targets`,
  - `cargo test --workspace`,
  - runtime smoke (CPU/Metal, link-system):
    - `target/benchmarks/llama_rs_attention_ops_trait_refactor_smoke_cpu_metal.txt`.
- User-directed sequential trait-first refactor policy (`1 -> 2 -> 3`) executed and recorded for resume safety:
  1. Backend/context lifecycle abstraction:
     - added `llama-rs/src/inference/backend_runtime.rs`,
     - `BackendRuntimeBuilder` + `DefaultBackendRuntimeBuilder`,
     - migrated backend/context setup in linear/MLP/attention/decode paths.
  2. Stepwise sequence state abstraction:
     - added `SequenceStateUpdater` + `DeltaSequenceStateUpdater` in `stepwise_decode.rs`,
     - migrated mask/position delta update branches to trait-backed methods.
  3. Decode projection/cache abstraction:
     - added `llama-rs/src/inference/projection_ops.rs`,
     - `TensorProjector` + `F32MatmulProjector`,
     - `DecodeCacheBuilder` + `StandardDecodeCacheBuilder`,
     - routed `build_attention_decode_cache` through trait-based builder.
- Runtime smoke artifacts for each sequential stage (CPU/Metal, link-system):
  - `target/benchmarks/llama_rs_backend_runtime_trait_refactor_smoke_cpu_metal.txt`,
  - `target/benchmarks/llama_rs_sequence_state_trait_refactor_smoke_cpu_metal.txt`,
  - `target/benchmarks/llama_rs_projection_trait_refactor_smoke_cpu_metal.txt`.
- llama-bench proxy now includes `bench_attention_layer` (`HxQxKxS` cases) and uses explicit backend synchronization for benchmark timing stability.
- llama.cpp baseline capture is now completed on CPU/Metal with six real GGUF models; results are recorded in `target/benchmarks/`.
- llama-rs proxy comparison is now captured using metadata-derived MLP/attention shape sets from the same GGUF models.
- Added `bench_compare_report` automation to generate a consolidated markdown comparison report from benchmark artifacts.
- Added decode-like attention proxy mode (`q_seq != kv_seq`) with reusable projected KV cache for more direct comparison with llama.cpp decode-profile behavior.
- Added stepwise decode-growth benchmark mode (`--decode-steps`) and a persistent stepwise runner (single backend/context/graph allocation with per-step mask/position updates). Decode report now includes a dedicated stepwise section (`ms/token`).
- Added `llama.cpp` decode (`0/128`) vs persistent-stepwise calibration artifact: `target/benchmarks/llama_stepwise_vs_cpp_calibration.md`.
- Added per-backend untimed preflight in `bench_attention_layer` to reduce first-case kernel compile bias; validated with reordered-case runs.
- Added step-window variant sweep artifact (`steps=8/16/32`): `target/benchmarks/llama_stepwise_variant_sweep.md`.
- Refreshed canonical stepwise snapshot (`target/benchmarks/llama_rs_bench_attention_decode_stepwise_models.txt`) from `steps=16` and regenerated calibration table to keep parity artifacts aligned with the default policy.
- Added safe GGUF write support to `ggml-rs` (`GgufWriter`) and new `llama-rs/examples/gguf` read/write parity flow (`w` / `r --check`).
- Added feature-gated integration coverage (`tests/gguf_roundtrip.rs`) for GGUF typed KV + tensor metadata round-trip.
- Added GGUF writer usability APIs: `set_values`, `remove_key`, `write_data_to_file`, and `write_metadata_to_file`.
- Re-validated CPU/Metal runtime execution with `llama-rs/examples/backend_smoke` after GGUF write-path expansion.
- Added `llama-rs/examples/idle` (decode-proxy idle timing path) with state-typed pause schedule (`IdlePauseSchedule<PauseScheduleReady>`), then validated CPU/Metal on `Llama-3-ELYZA-JP-8B-q4_k_m`.
- Returned to stepwise optimization loop and captured refreshed layerwise profile on ELYZA (`block_layer=0..7`):
  - raw: `target/benchmarks/llama_rs_stepwise_resume_elyza_layers0_7.txt`,
  - ranked summary: `target/benchmarks/llama_stepwise_resume_elyza_layers0_7_summary.md`.
- Added optional decode-stepwise KV projection cost modeling (`--decode-stepwise-kv-proj`) to include per-step `Wk/Wv` projection kernels in the persistent runner benchmark graph.
- Verified KV-projection mode on real CPU + Metal runs and added artifacts:
  - benchmark output: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_kvproj_s16_models.txt`,
  - calibration table: `target/benchmarks/llama_stepwise_vs_cpp_calibration_kvproj.md`,
  - matched-env impact table: `target/benchmarks/llama_stepwise_kvproj_impact.md`.
- Added optional block-scope mode (`--decode-stepwise-block`) with residual + RMSNorm + MLP-shaped compute on top of stepwise attention proxy, then validated CPU/Metal runtime and generated:
  - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_s16_models.txt`,
  - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj.md`,
  - `target/benchmarks/llama_stepwise_block_scope_impact.md`.
- Added optional stepwise sync/readback controls (`--decode-stepwise-sync-step`, `--decode-stepwise-readback-step`) and captured sync-mode calibration artifacts:
  - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_sync_s16_models.txt`,
  - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_sync.md`,
  - `target/benchmarks/llama_stepwise_sync_step_impact.md`.
- Current takeaway: scheduling-level controls alone are insufficient for Metal-side parity; topology/kernel composition alignment remains the main gap.
- Added model-derived block-MLP wiring options for stepwise block mode:
  - `--block-mlp-model <gguf> --block-mlp-layer <n>`,
  - relaxed MLP-name resolver and quantized fallback (`block_mlp_real=false`) using real metadata shape + deterministic values.
- Verified one-case Qwen3.5 run and generated `target/benchmarks/llama_stepwise_realmlp_qwen35_calibration.md`.
- Added Rust-idiomatic stepwise config construction helpers (`AttentionDecodeStepwiseConfig::new(...).with_*`) and centralized benchmark wiring in `bench_attention_layer`.
- Reduced stepwise hot-loop host churn by reusing per-step `QUERY_POS`/`CAUSAL_MASK` buffers (in-place fill) instead of allocating vectors every step.
- Re-validated after refactor with `cargo fmt`, `cargo clippy -p llama-rs --all-targets`, `cargo test -p llama-rs`, plus CPU/Metal runtime benchmarks (`target/benchmarks/rust_style_perf_stepwise_{base,block_kv}.txt`).
- Added quantized GGUF dequant decode path via GGML type traits (`decode_tensor_data_to::<T>` / `tensor_element_count`) and wired `GgufModel` tensor decode to use it.
- Added deep decode-API assembly inspection artifacts for the ownership refactor:
  - focused sections: `target/benchmarks/review3_decode_asm_snippets.md`,
  - before/after summary: `target/benchmarks/review3_decode_asm_compare.md`,
  - final judgement: `target/benchmarks/review3_decode_asm_judgement.md`.
  - key finding: release `decode_tensor_by_handle` keeps the contiguous
    `__rust_alloc + memcpy` fast path for native payloads, with no new
    per-element conversion overhead on that path.
- Verified Qwen3.5 `Q4_K_M` block-MLP run with `block_mlp_real=true` on both CPU and Metal and captured control-vs-real comparison (`target/benchmarks/rust_style_quantized_realmlp_qwen35_comparison.md`).
- Completed 6-model `--block-mlp-model` sweep and generated updated calibration/impact artifacts:
  - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_realmlp_s16_models.txt`,
  - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_realmlp.md`,
  - `target/benchmarks/llama_stepwise_block_realmlp_impact.md`.
- Added an apples-to-apples decode comparator so Rust/C++ run the same graph before optimization work:
  - C++ reference: `llama-rs/tests/cpp/attention_decode_proxy_reference.cpp`,
  - runner: `llama-rs/examples/bench_attention_decode_cpp_compare.rs`,
  - outputs: `target/benchmarks/llama_attention_decode_samework_cpp_vs_rust.{txt,md}`.
- Extended the same-workload comparator with a stepwise mode (`--stepwise-start/--stepwise-steps/--past`) and validated CPU + Metal:
  - command: `bench_attention_decode_cpp_compare --decode-kv 143 --stepwise-start 128 --stepwise-steps 16 --past 127 --warmup 2 --iters 10 cpu metal`,
  - outputs: `target/benchmarks/llama_attention_decode_stepwise_samework_cpp_vs_rust.{txt,md}`,
  - snapshot: `CPU avg rust/cpp ~0.727`, `MTL0 avg rust/cpp ~0.669`, `max checksum_rel ~8.9e-5`.
- Added backend partial tensor-write APIs (`set_f32_backend_at` / `set_i32_backend_at`) and applied stepwise mask delta updates in `run_attention_decode_stepwise_with_cache_repeats_with_block_mlp`.
  - pre/post impact artifact: `target/benchmarks/llama_attention_decode_stepwise_samework_maskdelta_impact.md`,
  - measured `rust_avg_ms/token` post/pre: `~0.805` overall (`CPU ~0.804`, `MTL0 ~0.806`).
- Added explicit stepwise mask-delta A/B toggle (`--decode-stepwise-no-mask-delta`) for matched measurements.
  - artifact: `target/benchmarks/llama_stepwise_models_maskdelta_on_vs_off.md`,
  - A/B summary: `on/off ~1.001` overall (`CPU ~1.009`, `MTL0 ~0.991`).
- Re-ran 6-model block+kv+real-MLP calibration after the mask-delta pass.
  - sweep: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_realmlp_s16_models_maskdelta.txt`,
  - calibration: `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_realmlp_maskdelta.md`,
  - old/new impact: `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_realmlp_maskdelta_impact.md`,
  - average proxy/cpp moved from `0.310 -> 0.306` (CPU), `0.257 -> 0.252` (MTL0).
- Added optional KV-write fidelity nodes (`--decode-stepwise-kv-cache-write`, gated by `--decode-stepwise-kv-proj`) and ran a 6-model matched sweep.
  - sweep: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_s16_models.txt`,
  - calibration: `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_maskdelta.md`,
  - impact vs block+kv+real-MLP mask-delta base: `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_maskdelta_impact.md`,
  - average proxy/cpp moved from `0.252 -> 0.256` on MTL0 (overall `0.279 -> 0.281`) with small per-case drift.
- Added configurable stepwise layer-repeat fidelity (`--decode-stepwise-layer-repeat <n>`) and benchmark support for model-derived repeat count (`--decode-stepwise-layer-repeat-model`).
  - 6-model sweeps:
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models.txt`,
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx4_s16_models.txt`,
  - calibrations:
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_maskdelta.md`,
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat4_maskdelta.md`,
  - condition comparison:
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat_impact.md`,
  - result summary:
    - `layer_repeat=3` reached near-parity (`CPU ~0.941`, `MTL0 ~0.965`, overall `~0.953`),
    - `layer_repeat=4` overshot (`CPU ~1.363`, `MTL0 ~1.152`, overall `~1.258`).
- Added KV-write cache-view fidelity mode (`--decode-stepwise-kv-cache-write-to-cache`) and evaluated it on the same 6-model `layer_repeat=3` setup.
  - sweep:
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwritecache_realmlp_layerx3_s16_models.txt`,
  - calibration:
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwritecache_realmlp_layerrepeat3_maskdelta.md`,
  - impact vs `layer_repeat=3` baseline:
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwritecache_realmlp_layerrepeat3_impact.md`,
  - summary:
    - average proxy/cpp moved from `0.953 -> 0.717` (CPU `0.941 -> 0.820`, MTL0 `0.965 -> 0.615`), so this variant is not adopted as default.
- Per user request, ran noise-reduction reruns (same condition, `r=3`, `iters=15`) and generated stable-median artifacts.
  - `layer_repeat=3` stability + calibration artifacts:
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_r3_i15_raw.txt`,
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_r3_i15_median.txt`,
    - `target/benchmarks/llama_stepwise_layerx3_stability_r3_i15.md`,
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_maskdelta_stable.md`,
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_stable_impact.md`.
  - `layer_repeat=4` stability + calibration artifacts:
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx4_s16_models_r3_i15_raw.txt`,
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx4_s16_models_r3_i15_median.txt`,
    - `target/benchmarks/llama_stepwise_layerx4_stability_r3_i15.md`,
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat4_maskdelta_stable.md`,
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_vs_4_stable_impact.md`.
  - stable-median summary:
    - `layer_repeat=3`: CPU `~0.869`, MTL0 `~0.634`, overall `~0.751`,
    - `layer_repeat=4`: CPU `~1.140`, MTL0 `~0.802`, overall `~0.971`.
- Reframed optimization policy to layer-by-layer measurement first, and extended `bench_attention_layer` with layer sweep support:
  - new CLI option: `--block-mlp-layer-range <start:end>`,
  - output now includes `block_layer=<n>` for stepwise block runs.
- Captured first per-layer profile artifacts under the current decode-stepwise condition (`layer_repeat=3`, block+kv+kvwrite):
  - Qwen3.5-4B:
    - raw: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_qwen35_layers0_31_layerx3.txt`,
    - summary/data: `target/benchmarks/llama_stepwise_qwen35_layers0_31_layerx3_profile.{md,csv}`,
    - stats: CPU mean/std `~12.607/0.104`, MTL0 mean/std `~9.053/0.076` ms/token.
  - Qwen3-8B:
    - raw: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_qwen3_8b_layers0_35_layerx3.txt`,
    - summary/data: `target/benchmarks/llama_stepwise_qwen3_8b_layers0_35_layerx3_profile.{md,csv}`,
    - stats: CPU mean/std `~23.824/0.169`, MTL0 mean/std `~16.342/0.091` ms/token.
  - cross-model summary:
    - `target/benchmarks/llama_stepwise_layerwise_profile_summary_layerx3.md`.
- Added experimental host-buffer elision toggle for incremental mask updates in stepwise mode:
  - config/API: `AttentionDecodeStepwiseConfig::with_mask_host_buffer_elision(bool)`,
  - benchmark CLI:
    - `--decode-stepwise-elide-mask-host-buffer` (enable),
    - `--decode-stepwise-keep-mask-host-buffer` (explicitly disable),
  - stepwise output now includes `mask_host_elide=<true|false>`.
- Hardened the incremental mask-delta path implementation:
  - removed the unnecessary host-buffer mutation dependency from the elision path,
  - removed the previous unsafe host write from this path.
- Ran sampled A/B experiments (order-balanced true->false and false->true):
  - Qwen3.5 sampled layers impact:
    - `target/benchmarks/llama_stepwise_mask_host_elide_ab_qwen35_layers_sample_impact.md`,
  - Qwen3-8B sampled layers impact:
    - `target/benchmarks/llama_stepwise_mask_host_elide_ab_qwen3_8b_layers_sample_impact.md`,
  - cross-model summary:
    - `target/benchmarks/llama_stepwise_mask_host_elide_sampled_impact.md`.
- Ran user-requested full 6-model validation sweep for `mask_host_elide` with balanced ordering:
  - base (`mask_host_elide=false`):
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_maskhost_base_balanced.txt`,
  - elide (`mask_host_elide=true`):
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_maskhost_elide_balanced.txt`,
  - impact summary:
    - `target/benchmarks/llama_stepwise_mask_host_elide_full_sweep_impact.md`.
- Current policy after full sweep:
  - backend averages improved slightly (`CPU ~0.951`, `MTL0 ~0.983`), but direction remains model-sensitive, so `mask_host_elide` stays opt-in (default off).
- Added stability reruns for the same full-sweep condition (`r=3` total including baseline run):
  - run2 raw:
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_maskhost_{base,elide}_balanced_r2.txt`,
  - run3 raw:
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_maskhost_{base,elide}_balanced_r3.txt`,
  - stability summary:
    - `target/benchmarks/llama_stepwise_mask_host_elide_full_sweep_stability_r3.md`.
- Stability (`r=3` median) policy check:
  - backend averages: CPU `~0.937`, MTL0 `~0.995` (`elide/base`),
  - model-level direction remains mixed, so default policy is unchanged (`mask_host_elide=false`, opt-in only).
- Implemented next common hot-path optimization in attention decode:
  - cache rotated K heads and transposed/contiguous V heads per KV head once, then reuse them across grouped query heads (instead of recomputing per query head).
- Post-change 6-model sweep (default `mask_host_elide=false`):
  - raw:
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_kvheadcache_post.txt`,
  - stability reruns:
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_kvheadcache_post_r2.txt`,
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_kvheadcache_post_r3.txt`,
    - `target/benchmarks/llama_stepwise_kvhead_cache_stability_r3.md`,
  - stable impact vs pre-change `r=3` baseline:
    - `target/benchmarks/llama_stepwise_kvhead_cache_impact_vs_maskhost_base_r3_median_stable.md`,
    - averages: CPU `~0.840`, MTL0 `~0.935` (`post/base`).
- Correctness guard:
  - checksum parity report:
    - `target/benchmarks/llama_stepwise_kvhead_cache_checksum_check.md`,
  - sampled 6-model check stayed exact (`max abs delta = 0.0`).
- Updated stable `llama.cpp` calibration after KV-head cache optimization:
  - calibration:
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_maskdelta_kvheadcache_stable.md`,
  - old/new calibration impact:
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_maskdelta_kvheadcache_stable_impact.md`,
  - averages:
    - CPU proxy/cpp `0.869 -> 0.777`,
    - MTL0 proxy/cpp `0.634 -> 0.630`,
    - overall `0.752 -> 0.704`.
- Added layer-repeat=3 mask-delta A/B artifact:
  - `target/benchmarks/llama_stepwise_models_layerx3_maskdelta_on_vs_off.md`,
  - sampled rerun summary: overall `on/off ~0.965` (`CPU ~0.981`, `MTL0 ~0.949`).
- Performed release-asm spot checks for the mask update hotspot:
  - command: `cargo rustc -p llama-rs --lib --release -- -C codegen-units=1 --emit=asm`,
  - `fill_causal_mask_values` is vectorized,
  - stepwise delta path lowers to `Tensor::set_f32_backend_at` branch with no steady-state allocator calls in the success loop.
- Started the next hot-path pass on attention output projection with an opt-in fused mode:
  - `AttentionDecodeStepwiseConfig::with_fused_output_projection(bool)` (default `false`),
  - benchmark CLI:
    - `--decode-stepwise-fuse-output-proj`,
    - `--decode-stepwise-no-fuse-output-proj`,
    - `--decode-stepwise-profile-outproj-fused-layerx5` (preset: `outproj_fused=true` + `layer_repeat=5`),
  - stepwise output now includes `outproj_fused=<true|false>`.
- Added safe `Context::concat` wrapper in `ggml-rs` and wired fused mode to:
  - concatenate per-head attention outputs (`dim=0`) and execute one `W_O * HEADS` matmul.
- CPU/Metal runtime smoke A/B completed (`4096x32x8x1`, `decode-kv=128`, `steps=16`, `layer_repeat=3`):
  - raw: `target/benchmarks/llama_stepwise_outproj_fuse_smoke_ab.txt`,
  - impact: `target/benchmarks/llama_stepwise_outproj_fuse_smoke_impact.md`,
  - sample ratio (`fused/base`): CPU `~1.051`, MTL0 `~0.963`.
- Ran full 6-model balanced-order sweep for `outproj_fused` on/off:
  - base: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_outproj_base_balanced.txt`,
  - fused: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_outproj_fused_balanced.txt`,
  - impact: `target/benchmarks/llama_stepwise_outproj_fuse_full_sweep_impact.md`,
  - average `fused/base`: CPU `~0.884`, MTL0 `~0.941`, overall `~0.912`.
- Added `r=3` stability reruns for the same full-sweep condition:
  - run2:
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_outproj_base_balanced_r2.txt`,
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_outproj_fused_balanced_r2.txt`,
  - run3:
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_outproj_base_balanced_r3.txt`,
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx3_s16_models_outproj_fused_balanced_r3.txt`,
  - stability summary:
    - `target/benchmarks/llama_stepwise_outproj_fuse_full_sweep_stability_r3.md`,
  - stable median `fused/base` averages: CPU `~0.885`, MTL0 `~0.940`, overall `~0.913`.
- Updated stable calibration vs `llama.cpp` for the fused medians:
  - calibration:
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_maskdelta_kvheadcache_outprojfuse_stable.md`,
  - old/new impact:
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat3_maskdelta_kvheadcache_outprojfuse_stable_impact.md`,
  - averages moved:
    - CPU proxy/cpp `0.777 -> 0.693`,
    - MTL0 proxy/cpp `0.630 -> 0.595`,
    - overall `0.704 -> 0.644`.
- Current policy:
  - `outproj_fused` remains default-off (opt-in) in this pass because it is a strong speedup but increases parity drift against the current `layer_repeat=3` calibration target.
- Ran parity-retune sweeps for `outproj_fused` with `layer_repeat=4/5/6`:
  - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx4_s16_models_outprojfused.txt`,
  - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx5_s16_models_outprojfused.txt`,
  - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx6_s16_models_outprojfused.txt`,
  - comparison: `target/benchmarks/llama_stepwise_outproj_fused_layerrepeat456_calibration.md`.
- Retune snapshot:
  - overall avg proxy/cpp:
    - `repeat4 ~0.807`,
    - `repeat5 ~0.985`,
    - `repeat6 ~1.156`,
  - best overall parity distance in this pass: `layer_repeat=5`.
- Added `layer_repeat=5` stability reruns (`r=3`) for `outproj_fused`:
  - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx5_s16_models_outprojfused_r2.txt`,
  - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_layerx5_s16_models_outprojfused_r3.txt`,
  - stability table: `target/benchmarks/llama_stepwise_outproj_fused_layerx5_stability_r3.md`.
- Stable calibration with `outproj_fused + layer_repeat=5`:
  - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat5_outprojfuse_stable.md`,
  - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat5_outprojfuse_stable_impact.md`,
  - averages:
    - CPU proxy/cpp `~1.073`,
    - MTL0 proxy/cpp `~0.908`,
    - overall `~0.991`.
- Profile smoke check:
  - `target/benchmarks/llama_stepwise_profile_outprojfused_layerx5_smoke.txt` confirms CPU+Metal runtime with `profile=outproj_fused_layerx5` output tagging.
- Added backend-balanced preset profile wiring:
  - new CLI flag: `--decode-stepwise-profile-outproj-fused-balanced`,
  - preset behavior:
    - CPU uses `layer_repeat=5`,
    - Metal uses `layer_repeat=6`,
    - `outproj_fused=true`,
  - output tag: `profile=outproj_fused_balanced_cpu5_mtl6`.
- Profile smoke check:
  - `target/benchmarks/llama_stepwise_profile_outprojfused_balanced_smoke.txt` confirms CPU+Metal runtime and per-backend repeat selection.
- Captured full 6-model calibration run for the balanced preset:
  - run:
    - `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_balanced_s16_models.txt`,
  - calibration:
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_profile_outprojfused_balanced.md`,
  - impact vs prior references:
    - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_profile_outprojfused_balanced_impact.md`,
  - averages:
    - CPU proxy/cpp `~1.076`,
    - MTL0 proxy/cpp `~1.063`,
    - overall `~1.070`.
- User selected the near-overall parity track (`outproj_fused_layerx5`) as the active canonical profile for continued optimization.
- Implemented a new hot-path optimization in stepwise decode:
  - precompute static per-KV-head transforms (`RoPE(K)`, `transpose+cont(V)`) once and reuse them,
  - API: `AttentionDecodeStepwiseConfig::with_static_kv_head_view_precompute(bool)`,
  - benchmark flags:
    - `--decode-stepwise-static-kv-head-precompute`,
    - `--decode-stepwise-no-static-kv-head-precompute`,
  - stepwise output now includes `kvhead_static_precompute=<true|false>`.
- CPU/Metal representative A/B (Qwen3.5, layer0, `outproj_fused_layerx5`) shows consistent speedup:
  - raw: `target/benchmarks/llama_stepwise_profile_layerx5_statickvhead_ab_qwen35_layer0.txt`,
  - impact: `target/benchmarks/llama_stepwise_profile_layerx5_statickvhead_ab_qwen35_layer0_impact.md`,
  - `on/off`: CPU `~0.961`, MTL0 `~0.983`.
- Full 6-model CPU/Metal A/B sweep under `outproj_fused_layerx5`:
  - on: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_layerx5_statickv_on_s16_models.txt`,
  - off: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_layerx5_statickv_off_s16_models.txt`,
  - impact: `target/benchmarks/llama_stepwise_profile_outprojfused_layerx5_statickv_impact.md`,
  - averages (`on/off`): CPU `~0.933`, MTL0 `~0.964`, overall `~0.949`.
- Checksum parity remained exact across all 6 models:
  - `target/benchmarks/llama_stepwise_profile_outprojfused_layerx5_statickv_checksum_check.md` (`max abs delta = 0.0`).
- Calibration refresh using the same cpp reference:
  - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat5_outprojfuse_statickv.md`,
  - `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_layerrepeat5_outprojfuse_statickv_impact.md`,
  - averages moved:
    - CPU proxy/cpp `~1.073 -> ~1.016`,
    - MTL0 proxy/cpp `~0.908 -> ~0.889`,
    - overall `~0.991 -> ~0.953`.
- Re-evaluated the balanced preset under the new static-KV baseline:
  - run: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_balanced_statickv_s16_models.txt`,
  - calibration: `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_profile_outprojfused_balanced_statickv.md`,
  - impact: `target/benchmarks/llama_stepwise_vs_cpp_calibration_block_kvproj_kvwrite_realmlp_profile_outprojfused_balanced_statickv_impact.md`,
  - averages moved (`prior balanced -> balanced + static KV`):
    - CPU proxy/cpp `~1.076 -> ~1.025`,
    - MTL0 proxy/cpp `~1.063 -> ~1.041`,
    - overall `~1.070 -> ~1.033`.
- Current interpretation:
  - `outproj_fused_layerx5 + static_kv` remains the user-selected active optimization track,
  - balanced + static-KV is now a nearer-to-1.0 calibration alternative than the previous balanced run.
- Ran next hotspot experiment on fused-output head concatenation strategy:
  - added balanced-concat option for fused output projection:
    - API: `AttentionDecodeStepwiseConfig::with_balanced_head_concat(bool)`,
    - CLI:
      - `--decode-stepwise-balanced-head-concat`,
      - `--decode-stepwise-no-balanced-head-concat`,
    - stepwise output now includes `head_concat_balanced=<true|false>`.
- Representative A/B (Qwen3.5 layer0, `outproj_fused_layerx5`, static-KV on):
  - raw: `target/benchmarks/llama_stepwise_profile_layerx5_balancedconcat_ab_qwen35_layer0.txt`,
  - impact: `target/benchmarks/llama_stepwise_profile_layerx5_balancedconcat_ab_qwen35_layer0_impact.md`,
  - `on/off`: CPU `~1.012`, MTL0 `~1.003` (no clear win).
- Full 6-model A/B (`outproj_fused_layerx5`, static-KV on):
  - on: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_layerx5_statickv_balancedconcat_on_s16_models.txt`,
  - off: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_layerx5_statickv_balancedconcat_off_s16_models.txt`,
  - impact: `target/benchmarks/llama_stepwise_profile_outprojfused_layerx5_statickv_balancedconcat_impact.md`,
  - averages (`on/off`): CPU `~1.000`, MTL0 `~0.983`, overall `~0.992`,
  - checksum parity remained exact (`max abs delta = 0.0`).
- Policy for this pass:
  - keep `head_concat_balanced=false` as default (marginal/mixed impact),
  - keep the switch available for explicit A/B and future retests.
- Ran next hotspot experiment on QUERY_POS updates for incremental decode:
  - added position-delta toggle for stepwise path:
    - API: `AttentionDecodeStepwiseConfig::with_position_deltas(bool)`,
    - CLI:
      - `--decode-stepwise-position-delta`,
      - `--decode-stepwise-no-position-delta`,
    - stepwise output now includes `position_delta=<true|false>`.
- Representative A/B (Qwen3.5 layer0, `outproj_fused_layerx5`, static-KV on):
  - raw: `target/benchmarks/llama_stepwise_profile_layerx5_positiondelta_ab_qwen35_layer0.txt`,
  - impact: `target/benchmarks/llama_stepwise_profile_layerx5_positiondelta_ab_qwen35_layer0_impact.md`,
  - `on/off`: CPU `~0.997`, MTL0 `~0.977`.
- Full 6-model A/B (`outproj_fused_layerx5`, static-KV on):
  - on: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_layerx5_statickv_positiondelta_on_s16_models.txt`,
  - off: `target/benchmarks/llama_rs_bench_attention_decode_stepwise_block_kvproj_kvwrite_realmlp_profile_outprojfused_layerx5_statickv_positiondelta_off_s16_models.txt`,
  - impact: `target/benchmarks/llama_stepwise_profile_outprojfused_layerx5_statickv_positiondelta_impact.md`,
  - averages (`on/off`): CPU `~0.990`, MTL0 `~1.001`, overall `~0.995`,
  - checksum parity remained exact (`max abs delta = 0.0`).
- Policy for this pass:
  - keep `position_delta=true` as default (small but positive overall impact),
  - keep the toggle for explicit A/B and future stability checks.
- Ran next hotspot experiment on fused-output head staging:
  - added head-staging toggle for stepwise fused output projection:
    - API: `AttentionDecodeStepwiseConfig::with_head_output_staging_buffer(bool)`,
    - CLI:
      - `--decode-stepwise-head-stage-buffer`,
      - `--decode-stepwise-no-head-stage-buffer`,
    - stepwise output now includes `head_stage_buf=<true|false>`.
- Targeted hotspot A/B (ELYZA, `block_layer=5..7`, profile `outproj_fused_layerx5`, static-KV on):
  - base: `target/benchmarks/llama_rs_stepwise_elyza_layers5_7_headstage_base.txt`,
  - on: `target/benchmarks/llama_rs_stepwise_elyza_layers5_7_headstage_on.txt`,
  - impact: `target/benchmarks/llama_stepwise_profile_layerx5_headstage_ab_elyza_layers5_7_impact.md`,
  - result (`on/base`): CPU `~1.009`, MTL0 `~1.000`, overall `~1.005` (no speed win),
  - checksum parity: `target/benchmarks/llama_stepwise_profile_layerx5_headstage_ab_elyza_layers5_7_checksum_check.md` (`max abs delta = 0.0`).
- Policy for this pass:
  - keep `head_stage_buf=false` as default,
  - keep the switch available for focused re-tests on future hotspot slices.
- Ran next hotspot experiment on block-MLP gate/up projection fusion:
  - added fused gate/up toggle for stepwise block mode:
    - API: `AttentionDecodeStepwiseConfig::with_fused_block_gate_up_projection(bool)`,
    - CLI:
      - `--decode-stepwise-fuse-block-gate-up`,
      - `--decode-stepwise-no-fuse-block-gate-up`,
    - stepwise output now includes `block_gateup_fused=<true|false>`.
- Targeted hotspot A/B (ELYZA, `block_layer=5..7`, profile `outproj_fused_layerx5`, static-KV on):
  - base: `target/benchmarks/llama_rs_stepwise_elyza_layers5_7_blockgateup_base.txt`,
  - on: `target/benchmarks/llama_rs_stepwise_elyza_layers5_7_blockgateup_on.txt`,
  - impact: `target/benchmarks/llama_stepwise_profile_layerx5_blockgateup_ab_elyza_layers5_7_impact.md`,
  - result (`on/base`): CPU `~1.013`, MTL0 `~1.012`, overall `~1.012` (regression),
  - checksum parity: `target/benchmarks/llama_stepwise_profile_layerx5_blockgateup_ab_elyza_layers5_7_checksum_check.md` (`max abs delta = 0.0`).
- Policy for this pass:
  - keep `block_gateup_fused=false` as default,
  - keep the switch available for focused re-tests on future hotspot slices.
- Refactored argument processing in `bench_attention_layer` before next example expansion:
  - replaced pending-flag state-machine parsing with iterator-driven `next_arg(...)` flow,
  - kept the existing CLI surface and post-parse validation rules,
  - reduced parser branching complexity and improved maintainability.
- Re-validated parser refactor:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
  - runtime parser smoke with complex options:
    - `target/benchmarks/llama_rs_parser_refactor_smoke.txt`.
- Hardened `idle` for mixed-architecture GGUF models (Qwen follow-up):
  - changed `resolve_llama_layer_tensor_names_from_names` to resolve only the requested layer (instead of resolving every detected layer first), so mixed layer dialects do not fail at layer `0` when probing another layer.
  - made `resolve_llama_layer_dimensions` treat non-llama architecture metadata as tensor-heuristic fallback instead of hard failure (`UnsupportedArchitecture` no longer aborts this resolver path).
  - added naming regression test `resolves_requested_layer_even_when_other_layers_are_incomplete` to lock layer-scoped behavior for mixed-layer models.
  - added `idle` fallback policy:
    - try requested layer + detected layer scan for real attention weights (`weights_mode=ModelLayer`),
    - if none resolve on non-llama metadata, run with metadata-derived deterministic attention weights (`weights_mode=MetadataDeterministic`).
  - `IdleReport` now includes `requested_layer` and `weights_mode`; `examples/idle` output includes both fields.
- Validation for the qwen idle pass:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
  - runtime:
    - `target/benchmarks/llama_rs_idle_qwen35_cpu_metal_fallback.txt` (Qwen3.5 CPU/Metal, deterministic metadata fallback),
    - `target/benchmarks/llama_rs_idle_elyza_cpu_metal_post_qwen_fix.txt` (ELYZA CPU/Metal, real model-layer weights unchanged).
- Tightened GGUF residual compatibility in `llama-rs/examples/gguf`:
  - expanded mode surface to `w | r0 | r1 | r`:
    - `r0`: metadata-only read pass (`gguf_ex_read_0`-style),
    - `r1`: tensor-data preview pass (`gguf_ex_read_1`-style) with `--check|--no-check`,
    - `r`: combined `r0 + r1`.
  - added shared tensor payload slicing helper to keep bounds checks centralized for preview + validation paths.
- Validation for GGUF mode compatibility:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
  - runtime combined artifact:
    - `target/benchmarks/llama_rs_gguf_mode_parity_r0_r1.txt`.

## Latest continuation (review1 merge + trait queue resume)

- Merged review1 worktree performance/hardening pass into `master`:
  - `src/compute.rs` tensor I/O fast paths and allocation cleanup,
  - C++ reference include resolution now honors `GGML_RS_GGML_INCLUDE_DIR` in:
    - `llama-rs/examples/bench_attention_decode_cpp_compare.rs`,
    - `llama-rs/tests/mlp_cpp_parity.rs`.
- Added targeted tensor I/O perf evidence:
  - `target/benchmarks/review1_tensorio_microbench_summary.md`.
- Resumed trait/ADT queue in `stepwise_decode`:
  - Candidate A complete:
    - introduced `KvCacheWriteStrategy` abstraction (`shared` vs `step-specific`),
    - runtime smoke artifact: `target/benchmarks/llama_rs_kv_write_strategy_trait_smoke.txt`.
  - Candidate B complete:
    - introduced head-output projection mode ADT + assembler (`per-head`, `fused concat`, `fused staging`),
    - runtime smoke artifact: `target/benchmarks/llama_rs_head_output_mode_adt_smoke.txt`.
  - Candidate C complete:
    - introduced `StepwiseGraphBuilder` + `KvPolicyStepwiseGraphBuilder` and `StepwiseGraphBuildInput`,
    - centralized stepwise graph construction (graph count, prereq nodes, KV write wiring) under one ADT-driven builder,
    - runtime smoke artifact: `target/benchmarks/llama_rs_stepwise_graph_builder_adt_smoke.txt`.
- Continued stepwise execution-loop cleanup after Candidate C:
  - added `StepGraphSchedule` to precompute per-step graph indices once and reuse them across warmup/bench loops.
  - runtime smoke artifact:
    - `target/benchmarks/llama_rs_stepwise_graph_schedule_smoke.txt`.
- Refreshed layer hotspot profile on the active lock (`outproj_fused_layerx5`, ELYZA layers `5..7`):
  - raw:
    - `target/benchmarks/llama_rs_stepwise_post_graphbuilder_elyza_layers5_7.txt`,
  - summaries:
    - `target/benchmarks/llama_stepwise_post_graphbuilder_elyza_layers5_7_summary.csv`,
    - `target/benchmarks/llama_stepwise_post_graphbuilder_elyza_layers5_7_summary.md`,
  - current hotspot snapshot:
    - CPU max in this slice: `layer=7` (`29.283 ms/token`),
    - MTL0 max in this slice: `layer=6` (`20.165 ms/token`).
- Rechecked `block_gateup_fused` after the graph-builder/schedule refactors on the same ELYZA hotspot slice:
  - base:
    - `target/benchmarks/llama_rs_stepwise_post_graphbuilder_elyza_layers5_7_blockgateup_base.txt`,
  - variant(on):
    - `target/benchmarks/llama_rs_stepwise_post_graphbuilder_elyza_layers5_7_blockgateup_on.txt`,
  - impact:
    - `target/benchmarks/llama_stepwise_post_graphbuilder_elyza_layers5_7_blockgateup_impact.md`,
  - checksum check:
    - `target/benchmarks/llama_stepwise_post_graphbuilder_elyza_layers5_7_blockgateup_checksum_check.md` (`max abs delta = 0.0`),
  - means (`on/base`):
    - CPU `~0.988`,
    - MTL0 `~0.995`,
    - overall `~0.991`.
- Broader validation pass (`Qwen3.5`, layers `0..7`) for `block_gateup_fused` under the same profile lock:
  - base:
    - `target/benchmarks/llama_rs_stepwise_post_graphbuilder_qwen35_layers0_7_blockgateup_base.txt`,
  - variant(on):
    - `target/benchmarks/llama_rs_stepwise_post_graphbuilder_qwen35_layers0_7_blockgateup_on.txt`,
  - impact:
    - `target/benchmarks/llama_stepwise_post_graphbuilder_qwen35_layers0_7_blockgateup_impact.md`,
  - checksum check:
    - `target/benchmarks/llama_stepwise_post_graphbuilder_qwen35_layers0_7_blockgateup_checksum_check.md` (`max abs delta = 0.0`),
  - means (`on/base`):
    - CPU `~1.029`,
    - MTL0 `~0.980`,
    - overall `~1.004`.
- Current pass decision:
  - keep `block_gateup_fused=false` as default (cross-model direction remains mixed after broader validation),
  - keep the flag available for backend-specific or model-specific A/B exploration.
- Rechecked `head_stage_buf` after graph-builder/schedule passes on ELYZA layers `5..7`:
  - base:
    - `target/benchmarks/llama_rs_stepwise_post_graphbuilder_elyza_layers5_7.txt`,
  - variant(on):
    - `target/benchmarks/llama_rs_stepwise_post_graphbuilder_elyza_layers5_7_headstage_on.txt`,
  - impact:
    - `target/benchmarks/llama_stepwise_post_graphbuilder_elyza_layers5_7_headstage_impact.md`,
  - checksum check:
    - `target/benchmarks/llama_stepwise_post_graphbuilder_elyza_layers5_7_headstage_checksum_check.md` (`max abs delta = 0.0`),
  - means (`on/base`):
    - CPU `~1.000`,
    - MTL0 `~1.004`,
    - overall `~1.002`.
- Policy for this pass:
  - keep `head_stage_buf=false` as default (near-neutral to slight regression on the current hotspot slice).
- Rechecked `mask_host_elide` after graph-builder/schedule passes on ELYZA layers `5..7`:
  - base:
    - `target/benchmarks/llama_rs_stepwise_post_graphbuilder_elyza_layers5_7.txt`,
  - variant(on):
    - `target/benchmarks/llama_rs_stepwise_post_graphbuilder_elyza_layers5_7_maskhost_on.txt`,
  - impact:
    - `target/benchmarks/llama_stepwise_post_graphbuilder_elyza_layers5_7_maskhost_impact.md`,
  - checksum check:
    - `target/benchmarks/llama_stepwise_post_graphbuilder_elyza_layers5_7_maskhost_checksum_check.md` (`max abs delta = 0.0`),
  - means (`on/base`):
    - CPU `~1.018`,
    - MTL0 `~0.998`,
    - overall `~1.008`.
- Policy for this pass:
  - keep `mask_host_elide=false` as default (CPU-side regression dominates on the current hotspot slice).

## Review2 worktree refactor pass

- Added review2-focused API unification in `ggml-rs` while preserving existing behavior:
  - new generic memory APIs:
    - `Context::recommended_matmul_memory::<T>(lhs, rhs) -> Result<Bytes>`,
    - `Context::recommended_backend_matmul_memory::<T>(lhs, rhs) -> Result<Bytes>`,
  - legacy `*_f32*_shapes*_bytes` helpers kept as compatibility wrappers.
- Added typed tensor I/O dispatch API:
  - `GgmlElement` trait (`f32`, `i32`) and generic methods on `Tensor`:
    - `write_data`, `write_data_backend`, `write_data_backend_at`,
    - `read_data`, `read_data_backend`, `get_data`.
  - updated ggml examples to exercise the new generic surface.
- Migrated llama-rs memory-sizing call sites to the generic helpers (`::<f32>`).
- Validation completed in review2 worktree:
  - `cargo fmt --all`,
  - `cargo clippy --workspace --all-targets`,
  - `cargo test --workspace`,
  - runtime smoke:
    - `examples/simple_ctx`,
    - `examples/backend_matmul` (CPU + Metal),
    - `llama-rs/examples/backend_smoke` (CPU + Metal).
- Regression guard snapshots (`master` vs `review2`):
  - `target/benchmarks/review2_bench_matmul_master_vs_refactor.md`,
  - `target/benchmarks/review2_llamars_bench_matmul_master_vs_refactor.md`,
  - both show no regression signal in this pass (sampled ratios stay within expected run-to-run noise envelope, with several improvements).
- Step `1` resumed after review2 merge:
  - captured post-merge hotspot slice run (ELYZA `block_layer=5..7`):
    - `target/benchmarks/review2_postmerge_step1_elyza_layers5_7.txt`,
  - compared against pre-review2 baseline:
    - `target/benchmarks/review2_postmerge_step1_elyza_layers5_7_impact.md`,
  - mean `post/base`:
    - CPU `~1.002`,
    - MTL0 `~0.997`,
    - overall `~0.999`,
  - interpretation:
    - no measurable regression from review2 refactor in the active stepwise hotspot slice.
- Follow-up: completed full compatibility-wrapper cleanup in `ggml-rs` public API:
  - removed remaining scalar-specific tensor I/O aliases and old `*_f32*` memory helper surface,
  - removed 1D/2D raw convenience constructors in favor of semantic newtype APIs (`Length`, `Shape2D`),
  - migrated all in-tree call sites (`ggml-rs` examples/tests + `llama-rs` inference/example paths).
- Validation after cleanup:
  - `cargo fmt --all`,
  - `cargo clippy --workspace --all-targets`,
  - `cargo test --workspace`,
  - runtime smoke (`--features link-system`, CPU+Metal):
    - `target/benchmarks/review2_cleanup_runtime_smoke.txt`.
- Step `1` guard rerun after cleanup (ELYZA `block_layer=5..7`):
  - raw:
    - `target/benchmarks/review2_postclean_step1_elyza_layers5_7.txt`,
  - impact:
    - `target/benchmarks/review2_postclean_step1_elyza_layers5_7_impact.md`,
  - mean `post/base`:
    - CPU `~1.013`,
    - MTL0 `~0.998`,
    - overall `~1.005`,
  - checksum deltas: all `0.0`.
- Follow-up API polish (Rust-generic host I/O internals):
  - generalized tensor host-side internals from `f32`-specific methods to type-driven helpers:
    - `Tensor::write_host_data<T>()`,
    - `Tensor::read_host_data<T>()`,
    - `Tensor::read_host_at<T>()`,
  - introduced internal `HostElement` trait (`f32` + `i32`) for direct ggml 1D host accessor dispatch,
  - updated `GgmlElement` implementations to use generic host helpers for both `f32` and `i32`.
- Verification after generic host-I/O refactor:
  - `cargo fmt --all`,
  - `cargo clippy --workspace --all-targets`,
  - `cargo test --workspace`,
  - runtime smoke (`--features link-system`, CPU+Metal):
    - `target/benchmarks/review2_generic_host_io_runtime_smoke.txt`.
- Step `1` guard rerun after generic host-I/O refactor (ELYZA `block_layer=5..7`):
  - raw:
    - `target/benchmarks/review2_generic_host_io_step1_elyza_layers5_7.txt`,
  - impact:
    - `target/benchmarks/review2_generic_host_io_step1_elyza_layers5_7_impact.md`,
  - mean `post/base`:
    - CPU `~1.002`,
    - MTL0 `~1.009`,
    - overall `~1.006`,
  - checksum deltas: all `0.0`.
- Continued step `1` per-layer loop on top of the generic host-I/O pass:
  - refreshed ELYZA layer sweep (`block_layer=0..7`):
    - raw: `target/benchmarks/review2_generic_host_io_step1_elyza_layers0_7.txt`,
    - summary:
      - `target/benchmarks/review2_generic_host_io_step1_elyza_layers0_7_summary.md`,
      - `target/benchmarks/review2_generic_host_io_step1_elyza_layers0_7_summary.csv`.
  - current hotspot snapshot (this run):
    - CPU max: `layer=0` (`30.048 ms/token`),
    - MTL0 max: `layer=7` (`19.965 ms/token`).
- Next hotspot check on this refreshed baseline: `block_gateup_fused` (`layers 0..7`, ELYZA):
  - variant(on):
    - `target/benchmarks/review2_generic_host_io_step1_elyza_layers0_7_blockgateup_on.txt`,
  - impact:
    - `target/benchmarks/review2_generic_host_io_step1_elyza_layers0_7_blockgateup_impact.md`,
  - means (`on/base`):
    - CPU `~1.010`,
    - MTL0 `~0.995`,
    - overall `~1.003`,
  - checksum deltas: all `0.0`,
  - decision: keep `block_gateup_fused=false` default (CPU regression dominates).
- Follow-up hotspot check: `head_stage_buf` (`layers 0..7`, ELYZA):
  - variant(on):
    - `target/benchmarks/review2_generic_host_io_step1_elyza_layers0_7_headstage_on.txt`,
  - impact:
    - `target/benchmarks/review2_generic_host_io_step1_elyza_layers0_7_headstage_impact.md`,
  - means (`on/base`):
    - CPU `~1.008`,
    - MTL0 `~0.998`,
    - overall `~1.003`,
  - checksum deltas: all `0.0`,
  - decision: keep `head_stage_buf=false` default (CPU regression dominates).
- Requested follow-up `1`: `mask_host_elide` recheck (`layers 0..7`, ELYZA):
  - variant(on):
    - `target/benchmarks/review2_generic_host_io_step1_elyza_layers0_7_maskhost_on.txt`,
  - impact:
    - `target/benchmarks/review2_generic_host_io_step1_elyza_layers0_7_maskhost_impact.md`,
  - means (`on/base`):
    - CPU `~0.972`,
    - MTL0 `~1.012`,
    - overall `~0.992`,
  - checksum deltas: all `0.0`.
- Requested follow-up `2`: cross-model check on Qwen3.5 (`layers 0..7`) under the same mask-host condition:
  - base:
    - `target/benchmarks/review2_generic_host_io_step1_qwen35_layers0_7_maskhost_base.txt`,
  - variant(on):
    - `target/benchmarks/review2_generic_host_io_step1_qwen35_layers0_7_maskhost_on.txt`,
  - impact:
    - `target/benchmarks/review2_generic_host_io_step1_qwen35_layers0_7_maskhost_impact.md`,
  - means (`on/base`):
    - CPU `~0.997`,
    - MTL0 `~0.995`,
    - overall `~0.996`,
  - checksum deltas: all `0.0`.
- Dependency hygiene update:
  - added `ggml` as repository submodule at `vendor/ggml` (`.gitmodules` + `vendor/ggml`).
  - updated docs to treat `llama.cpp` as optional external reference (no submodule dependency), with explicit reproduction/setup steps based on `LLAMA_CPP_DIR`.

- User-confirmed continuation: full 6-model `mask_host_elide` sweep on the current lock (`outproj_fused_layerx5 + static_kv + position_delta`, `block_layer=0`, `steps=16`):
  - local ggml link-system build:
    - `cmake -S vendor/ggml -B vendor/ggml/build ... && cmake --build vendor/ggml/build -j`
  - balanced-order A/B artifacts:
    - base:
      - `target/benchmarks/review2_generic_host_io_step1_models6_maskhost_base_balanced.txt`
    - variant(on):
      - `target/benchmarks/review2_generic_host_io_step1_models6_maskhost_on_balanced.txt`
    - impact:
      - `target/benchmarks/review2_generic_host_io_step1_models6_maskhost_impact.md`
    - checksum parity:
      - `target/benchmarks/review2_generic_host_io_step1_models6_maskhost_checksum_check.md`
- means (`on/base`):
  - CPU `~1.001`
  - MTL0 `~1.004`
  - overall `~1.002`
- checksum deltas:
  - `max abs delta = 0.0`
- decision:
  - keep `mask_host_elide=false` default under the current lock (small overall regression).

## Typed-tensor/API polish + GPT synthetic continuation

- Rustified `ggml-rs` typed tensor surface:
  - rewrote `src/typed_tensor.rs` to generic rank-complete wrappers:
    - `Tensor1D/2D/3D/4D` + `Tensor1DConst..Tensor4DConst`,
  - added typed constructors:
    - `new_tensor_1d_typed::<T, S>()` ... `new_tensor_4d_typed::<T, S>()`,
  - removed remaining scalar-specific typed constructor variants in call sites.
- Public export/prelude sync:
  - updated `src/lib.rs` re-exports for new shape/spec traits and typed tensor wrappers.
- Lock-script hardening:
  - `scripts/agent_lock.rs` now strips an optional leading `--` argument from `cargo -Zscript` invocation style.
  - smoke:
    - `./scripts/agent_lock.sh cargo cargo --version`,
    - `./scripts/agent_lock.sh bench echo lock-script-ok`.
- Step `1` continuation (GPT synthetic perf track):
  - investigated GPT2/GPTJ synthetic loop hotspots.
  - trialed full `run_ctx` reuse in `llama-rs/src/gpt2_synthetic.rs`; measured regression and reverted to baseline flow.
    - trial impact artifact:
      - `target/benchmarks/review4_gpt2_ctx_loopreuse_trial_impact.md` (`trial/base ~1.944`).
  - implemented low-risk reuse in `examples/gptj_main_synth.rs`:
    - build context/weights/graph once with fixed token capacity (`prompt_len + n_predict`),
    - per-step update only token buffer + compute + current-position readback.
  - parity-config result (`seed=17`, `n_predict=6`):
    - baseline: `target/benchmarks/gptj_main_synth_rust.txt` (`elapsed_us=1064`),
    - post: `target/benchmarks/review4_gptj_main_synth_post.txt` (`elapsed_us=985`),
    - impact: `target/benchmarks/review4_gptj_main_synth_impact.md` (`post/base ~0.926`),
    - generated tokens and checksum remained identical.
- Validation and runtime:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
  - `cargo check --workspace --all-targets --features link-system`
  - runtime smoke (`--features link-system`, CPU/Metal):
    - `target/benchmarks/review4_runtime_smoke_post_perfpass.txt`.

## Step2 follow-up (`uv` model assets + real-model CPU/Metal smoke)

- Used `uv` Python workflow to verify required six real GGUF assets under `target/models/*`:
  - command style: `uv run python - <<'PY' ...`,
  - artifact: `target/benchmarks/review4_model_asset_uv_check.txt`,
  - result: `missing_count=0` (`present_count=6`).
- Cleared the model-acquisition blocker based on confirmed local asset presence.
- Re-ran real-model runtime smoke (`llama-rs/examples/idle`) on both backends:
  - Qwen3.5-4B (`target/models/qwen3_5_4b_q4_k_m/...`) and
  - ELYZA 8B (`target/models/elyza_llama3_jp_8b_q4_k_m/...`),
  - command profile:
    - `--layer 0 --decode-kv 64 --iters 1 --pauses 0 cpu metal`,
  - artifact:
    - `target/benchmarks/review4_idle_realmodel_cpu_metal_smoke.txt`.
- Outcome:
  - CPU and Metal execution succeeded with real model assets in both runs.

## Llama-rs modularization pass (role split + ADT/trait)

- Refactored `llama-rs` inference internals to reduce monolithic responsibilities:
  - extracted layer-dimension resolution into
    `llama-rs/src/inference/layer_dimensions.rs`.
- Moved and isolated these surfaces into the new module:
  - `MetadataResolutionMode`,
  - `LlamaLayerDimensions`,
  - `resolve_llama_layer_dimensions(...)`,
  - all helper validations for hidden/FFN/projection shape inference.
- Added explicit strategy trait for attention layout inference:
  - `HeadLayoutStrategy`,
  - `PreferredHeadLayoutStrategy` (default heuristic implementation).
- `inference.rs` now re-exports the public layer-dimension ADTs/functions and
  keeps execution-path code focused on graph/runtime behavior.
- Validation:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
  - `cargo check --workspace --all-targets --features link-system`
- Runtime smoke after split:
  - `target/benchmarks/review4_llamars_modularization_runtime_smoke.txt`
  - includes:
    - `llama-rs/examples/backend_smoke -- cpu metal`,
    - `llama-rs/examples/idle <Qwen3.5 model> --layer 0 --decode-kv 64 --iters 1 --pauses 0 cpu metal`.

## Llama-rs modularization continuation (attention runtime extraction)

- Continued the same `inference.rs` role-split by extracting attention runtime
  execution into:
  - `llama-rs/src/inference/attention_runtime.rs`.
- Moved the following surfaces from `inference.rs` into the new module:
  - `resolve_attention_weights_for_layer(_auto)`,
  - `attention_inference_for_layer(_auto)(_repeats)`,
  - `attention_inference_with_weights(_repeats)`,
  - `build_attention_decode_cache`,
  - decode-proxy execution core and attention backend memory sizing helpers.
- `llama-rs/src/inference.rs` now re-exports the public attention APIs and keeps
  only orchestration-level visibility (`pub use` / `pub(crate) use`) for shared
  internals consumed by `decode_proxy_plan` / `stepwise_decode`.
- Full validation after extraction:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
  - `cargo check --workspace --all-targets --features link-system`
- Rebuilt local `vendor/ggml` shared libs and re-ran CPU/Metal runtime smoke:
  - `target/benchmarks/review4_attention_runtime_modularization_runtime_smoke.txt`
  - includes:
    - `ggml-rs/examples/backend_matmul -- cpu`,
    - `ggml-rs/examples/backend_matmul -- metal`,
    - `llama-rs/examples/backend_smoke -- cpu metal`.
- Post-split synthetic guard (`gptj_main_synth`, `seed=17`, `n_predict=6`):
  - run artifact:
    - `target/benchmarks/review4_attention_runtime_modularization_gptj_guard.txt`,
  - impact summary:
    - `target/benchmarks/review4_attention_runtime_modularization_gptj_guard_impact.md`,
  - parity remained exact (`generated_tokens` and `logit_checksum` matched the pre-split baseline).

## Step1 continuation (GPT-2 synthetic host-allocation cleanup)

- Implemented a low-risk synthetic optimization in `llama-rs/src/gpt2_synthetic.rs`:
  - replaced per-step `lhs` vector allocation with a reusable preallocated host
    buffer in both:
    - `run_ctx`,
    - `run_backend_for_steps` (therefore also covering alloc/backend/sched/batched paths).
  - new helper:
    - `fill_lhs_batch(&mut [f32], seed, step)`.
- Validation + parity-config re-run:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
  - `cargo check --workspace --all-targets --features link-system`
  - parity-config run artifact:
    - `target/benchmarks/review4_gpt2_fillreuse_paritycfg.txt`
  - impact artifact:
    - `target/benchmarks/review4_gpt2_fillreuse_impact.md`
    - `ctx avg_item_ms`: `0.043166 -> 0.017536` (`~0.406`)
    - `alloc avg_item_ms`: `0.018706 -> 0.016009` (`~0.856`)
    - checksum parity: unchanged for both rows.
- CPU/Metal runtime smoke for backend path:
  - `target/benchmarks/review4_gpt2_fillreuse_runtime_smoke.txt`
  - includes:
    - `llama-rs/examples/gpt2_backend cpu`,
    - `llama-rs/examples/gpt2_backend metal`.

## Step1 continuation (backend sampled-read pass)

- Extended `ggml-rs` safe tensor backend I/O surface:
  - added `Tensor::read_data_backend_at::<T>(element_offset, element_count)`.
- Added coverage in `tests/ggml_tensor_ops.rs`:
  - backend slice read success case (`offset=2, len=3`),
  - out-of-bounds read error case.
- Updated `llama-rs/src/gpt2_synthetic.rs` backend runner to use sampled readback:
  - `run_backend_for_steps` now reads only checksum-required prefix via
    `read_data_backend_at::<f32>(0, sample_len)`.
- Validation:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
  - `cargo test -p ggml-rs --features link-system --test ggml_tensor_ops`
  - `cargo check --workspace --all-targets --features link-system`
- Runtime/parity artifacts:
  - `target/benchmarks/review4_gpt2_readslice_paritycfg.txt`
  - `target/benchmarks/review4_gpt2_readslice_impact_vs_fillreuse.md`
  - `target/benchmarks/review4_gpt2_readslice_backend_impact_vs_fillreuse.md`
  - stability summaries:
    - `target/benchmarks/review4_gpt2_readslice_stability_r3_summary.md`
    - `target/benchmarks/review4_gpt2_readslice_ctx_stability_r3_summary.md`
- Follow-up decision in the same pass:
  - attempted a context-path sampled checksum (`Tensor::get_data`) optimization for `run_ctx`,
  - stability evidence showed no reliable gain, so this part was reverted.
- Added/kept host range-read API in `ggml-rs`:
  - `Tensor::read_data_at::<T>(offset, len)`,
  - validated via `ggml_tensor_ops`; this API remains available even though
    `gpt2 run_ctx` stayed on full readback for stable behavior.
- Final retained state:
  - backend sampled-read API + backend-path usage stays enabled,
  - final artifacts:
    - `target/benchmarks/review4_gpt2_readslice_final_paritycfg.txt`
    - `target/benchmarks/review4_gpt2_readslice_final_stability_r3_summary.md`
    - `target/benchmarks/review4_gpt2_readslice_final_impact_vs_original.md`
  - final impact vs original parity baseline:
    - `ctx avg_item_ms`: `0.043166 -> 0.021701` (`~0.503`, median),
    - `alloc avg_item_ms`: `0.018706 -> 0.014245` (`~0.762`, median),
    - checksum parity remained stable across `r=3`.

## Step1 continuation (GPT-J logits range-read)

- Updated `examples/gptj_main_synth.rs` to avoid full logits readback per step:
  - replaced full tensor read + slice (`read_data::<f32>()` then `[start..]`) with
    direct range read:
    - `read_data_at::<f32>(start, GPTJ_VOCAB)`.
- Validation:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
  - `cargo test -p ggml-rs --features link-system --test ggml_tensor_ops`
  - `cargo check --workspace --all-targets --features link-system`
- Parity/perf artifacts:
  - guard run:
    - `target/benchmarks/review4_gptj_slice_parity_guard.txt`
  - stability:
    - `target/benchmarks/review4_gptj_slice_stability_r5_summary.md`
  - baseline impact:
    - `target/benchmarks/review4_gptj_slice_impact_vs_post_baseline.md`
    - median `elapsed_us`: `985 -> 874` (`~0.887`),
    - generated tokens/checksum remained stable across `r=5`.
- Follow-up host-write incremental update trial:
  - attempted to update only the newest token position via
    `Tensor::write_data_at` in the loop,
  - rejected after regression evidence:
    - `target/benchmarks/review4_gptj_hostwrite_stability_r5_summary.md`
    - `target/benchmarks/review4_gptj_hostwrite_impact_vs_slice_baseline.md`
    - trial/baseline median ratio: `~2.127`,
  - final code keeps full token-buffer write per iteration and only retains
    logits range-read optimization.

## Step2 continuation (stepwise decode toggle stability, canonical lock)

- Canonical condition for this pass:
  - `--cases 4096x32x8x1 --causal --decode-kv 129 --decode-steps 8 --past 128`
  - `--decode-stepwise-profile-outproj-fused-layerx5`
  - `--decode-stepwise-kv-proj --decode-stepwise-kv-cache-write --decode-stepwise-block --block-mlp-layer 0`
  - backends: `cpu metal`, `warmup=1`, `iters=2`.
- First A/B artifact for `no-mask-delta`:
  - `target/benchmarks/review4_step2_stepwise_nomask_impact.md`
  - snapshot (`variant/base`): CPU `~0.942`, MTL0 `~1.002`.
- Stability reruns (`r=3`) for `no-mask-delta`:
  - raw:
    - `target/benchmarks/review4_step2_stepwise_baseline_cpu_metal{,_r2,_r3}.txt`,
    - `target/benchmarks/review4_step2_stepwise_nomask_cpu_metal{,_r2,_r3}.txt`,
  - summary:
    - `target/benchmarks/review4_step2_stepwise_nomask_stability_r3.md`,
  - median summary (`variant/base`):
    - CPU `~1.006`,
    - MTL0 `~0.998`,
    - overall mean of backend medians `~1.002`.
- Added `no-position-delta` A/B and stability run:
  - first-run impact:
    - `target/benchmarks/review4_step2_stepwise_nopos_impact.md`,
  - raw reruns:
    - `target/benchmarks/review4_step2_stepwise_nopos_base_cpu_metal{,_r2,_r3}.txt`,
    - `target/benchmarks/review4_step2_stepwise_nopos_variant_cpu_metal{,_r2,_r3}.txt`,
  - stability summary:
    - `target/benchmarks/review4_step2_stepwise_nopos_stability_r3.md`,
  - median summary (`variant/base`):
    - CPU `~1.014`,
    - MTL0 `~1.001`,
    - overall mean of backend medians `~1.007`.
- Added projection-hotspot recheck (`--decode-stepwise-no-fuse-output-proj`):
  - `target/benchmarks/review4_step2_stepwise_outproj_nofuse_impact.md`,
  - result (`variant/base`): CPU `~1.026`, MTL0 `~1.074`, overall `~1.050`.
- Added static-KV-head precompute ablation (`--decode-stepwise-no-static-kv-head-precompute`):
  - `target/benchmarks/review4_step2_stepwise_statickv_off_impact.md`,
  - token ratio (`variant/base`): CPU `~0.992`, MTL0 `~1.007`, overall `~0.999`,
  - setup ratio (`variant/base`): CPU `~1.014`, MTL0 `~1.150`, overall `~1.082`,
  - checksum parity remained exact (`0.0` delta on both backends).
- Added head-stage-buffer recheck on the same lock:
  - first-run impact:
    - `target/benchmarks/review4_step2_stepwise_headstage_on_impact.md`,
  - stability reruns (`r=3`):
    - raw:
      - `target/benchmarks/review4_step2_stepwise_headstage_base_cpu_metal{,_r2,_r3}.txt`,
      - `target/benchmarks/review4_step2_stepwise_headstage_on_cpu_metal{,_r2,_r3}.txt`,
    - summary:
      - `target/benchmarks/review4_step2_stepwise_headstage_stability_r3.md`,
  - median summary (`head-stage/base`):
    - token: CPU `~0.979`, MTL0 `~0.996`, overall `~0.988`,
    - setup: CPU `~1.014`, MTL0 `~1.163`, overall `~1.089`,
  - checksum parity remained exact (`0.0` delta on both backends).
- Per user-selected next step, ran broader layer sweep (`block_layer=0..7`) for head-stage:
  - base:
    - `target/benchmarks/review4_step2_headstage_broadsweep_layers0_7_base.txt`,
  - head-stage on:
    - `target/benchmarks/review4_step2_headstage_broadsweep_layers0_7_on.txt`,
  - impact summary:
    - `target/benchmarks/review4_step2_headstage_broadsweep_layers0_7_impact.md`,
    - `target/benchmarks/review4_step2_headstage_broadsweep_layers0_7_impact.csv`,
  - broad-sweep means (`head-stage/base`):
    - CPU token `~1.004` (3/8 layers faster),
    - MTL0 token `~0.998` (5/8 layers faster),
    - overall token `~1.001`,
    - overall setup `~1.003`,
  - checksum parity remained exact (`0.0` delta for all rows).
- Ran next broader layer sweep for block gate/up fusion (`block_layer=0..7`):
  - first broad impact:
    - `target/benchmarks/review4_step2_blockgateup_broadsweep_layers0_7_impact.md`,
    - `target/benchmarks/review4_step2_blockgateup_broadsweep_layers0_7_impact.csv`,
  - additional paired rerun (`r2`) with stability summary:
    - `target/benchmarks/review4_step2_blockgateup_broadsweep_layers0_7_base_r2.txt`,
    - `target/benchmarks/review4_step2_blockgateup_broadsweep_layers0_7_on_r2.txt`,
    - `target/benchmarks/review4_step2_blockgateup_broadsweep_layers0_7_stability_r2.md`,
    - `target/benchmarks/review4_step2_blockgateup_broadsweep_layers0_7_stability_r2.csv`,
  - stability (`r=2`) summary (`gateup-fused/base`):
    - CPU token `~0.996`,
    - MTL0 token `~1.001`,
    - overall token `~0.998`,
    - overall setup `~1.018`,
  - checksum parity remained exact (`0.0` delta for all rows).
- Pass decision:
  - keep defaults unchanged on this lock:
    - `mask_delta=true`,
    - `position_delta=true`,
    - `outproj_fused=true`,
    - `kvhead_static_precompute=true`,
    - `head_stage_buf=false` (broader sweep stayed near-neutral overall),
    - `block_gateup_fused=false` (broader stability stayed near-neutral overall).
- Per follow-up directive (`1,2 then 3`), ran two more broad-sweep passes (`block_layer=0..7`, `r=2`) on the same lock:
  1. Head-concat balanced (`--decode-stepwise-balanced-head-concat`):
     - variant runs:
       - `target/benchmarks/review4_step2_headconcat_broadsweep_layers0_7_on.txt`,
       - `target/benchmarks/review4_step2_headconcat_broadsweep_layers0_7_on_r2.txt`,
     - stability summary:
       - `target/benchmarks/review4_step2_headconcat_broadsweep_layers0_7_stability_r2.md`,
       - overall token `~1.006`, overall setup `~1.057` (regression).
  2. Mask-host elision (`--decode-stepwise-elide-mask-host-buffer`):
     - variant runs:
       - `target/benchmarks/review4_step2_maskhost_broadsweep_layers0_7_on.txt`,
       - `target/benchmarks/review4_step2_maskhost_broadsweep_layers0_7_on_r2.txt`,
     - stability summary:
       - `target/benchmarks/review4_step2_maskhost_broadsweep_layers0_7_stability_r2.md`,
       - overall token `~0.996`, overall setup `~1.083` (small token-side win offset by setup cost).
- Then moved to alternate condition (`3`) and remeasured with `--decode-steps 16` on layers `0..7`:
  - base:
    - `target/benchmarks/review4_step2_alt_steps16_layers0_7_base.txt`,
  - mask-host variant:
    - `target/benchmarks/review4_step2_alt_steps16_layers0_7_maskhost_on.txt`,
    - `target/benchmarks/review4_step2_alt_steps16_layers0_7_maskhost_impact.md`,
  - block-gateup variant:
    - `target/benchmarks/review4_step2_alt_steps16_layers0_7_blockgateup_on.txt`,
    - `target/benchmarks/review4_step2_alt_steps16_layers0_7_blockgateup_impact.md`,
  - combined summary:
    - `target/benchmarks/review4_step2_alt_steps16_layers0_7_summary.md`,
    - `maskhost-elide`: token `~1.008`, setup `~1.012`,
    - `block-gateup-fused`: token `~1.004`, setup `~1.025`.
- Updated decision after `1,2,3` pass:
  - keep defaults unchanged on this lock:
    - `head_concat_balanced=false`,
    - `mask_host_elide=false`,
    - `block_gateup_fused=false`,
    - `head_stage_buf=false`,
  - checksum parity remained exact (`0.0` deltas) across these sweeps.
- Next candidate pass on the same canonical lock (`block_layer=0..7`):
  - enabled `--decode-stepwise-kv-cache-write-to-cache`,
  - artifact:
    - `target/benchmarks/review4_step2_kvwritecache_broadsweep_layers0_7_impact.md`,
    - `target/benchmarks/review4_step2_kvwritecache_broadsweep_layers0_7_impact.csv`,
  - summary (`variant/base`):
    - token: CPU `~1.024`, MTL0 `~1.012`, overall `~1.018`,
    - setup: CPU `~1.111`, MTL0 `~1.114`, overall `~1.112`,
  - checksum note:
    - CPU delta remained `0.0`,
    - MTL0 rows showed non-zero checksum delta (`+104`) and require no further pursuit because performance regressed clearly.
- Decision update:
  - keep `kv_cache_write_to_cache=false` on this lock.
- Next candidate pass on the same canonical lock (`block_layer=0..7`):
  - enabled `--decode-stepwise-no-static-kv-head-precompute`,
  - artifact:
    - `target/benchmarks/review4_step2_statickv_off_broadsweep_layers0_7_impact.md`,
    - `target/benchmarks/review4_step2_statickv_off_broadsweep_layers0_7_impact.csv`,
  - summary (`variant/base`):
    - token: CPU `~0.998`, MTL0 `~1.013`, overall `~1.006`,
    - setup: CPU `~1.163`, MTL0 `~1.156`, overall `~1.160`,
  - checksum parity remained exact (`0.0` delta for all rows).
- Decision update:
  - keep `kvhead_static_precompute=true` on this lock.
- Next candidate pass (`sync/readback`) started from layer-0 probe:
  - `target/benchmarks/review4_step2_sync_readback_layer0_impact.md`,
  - quick layer-0 snapshot:
    - `sync_step`: CPU `~1.003`, MTL0 `~0.994`,
    - `readback_step`: CPU `~1.015`, MTL0 `~0.999`.
- Continued with broad sync-step sweep (`block_layer=0..7`) and stability rerun:
  - first broad run:
    - `target/benchmarks/review4_step2_syncstep_broadsweep_layers0_7_impact.md`,
  - r2 broad run:
    - `target/benchmarks/review4_step2_syncstep_broadsweep_layers0_7_on_r2.txt`,
  - stability summary:
    - `target/benchmarks/review4_step2_syncstep_broadsweep_layers0_7_stability_r2.md`,
    - aggregate (`sync-step/base`, r=2 medians):
      - CPU token `~0.916`,
      - MTL0 token `~1.019`,
      - overall `~0.967`,
      - setup overhead increased (overall `~1.236`).
- Decision update:
  - keep `sync_step=false` as default on this lock for now (strong backend split and setup cost),
  - keep backend-specific policy exploration as a separate follow-up item.

## Idle fallback runtime confirmation + sync-policy reassessment

- Verified the `idle` metadata fallback fix directly on the previously failing model:
  - command target:
    - `target/models/llama_minitron_4b_q4_0/Llama-3.1-Minitron-4B-Width-Base-Q4_0.gguf`
  - artifact:
    - `target/benchmarks/review4_idle_minitron_post_fallback_fix.txt`
  - result:
    - both `CPU` and `MTL0` now complete with `weights_mode=MetadataDeterministic`.
- Re-ran full post-fallback idle refresh for six target models (CPU + Metal):
  - raw:
    - `target/benchmarks/review4_model_inference_refresh_idle_cpu_metal_post_fallback_fix.txt`
  - summary:
    - `target/benchmarks/review4_model_inference_refresh_idle_cpu_metal_post_fallback_fix_summary.md`
  - result:
    - all six models emitted both backend rows successfully.

- Follow-up on pending backend-specific sync policy task:
  - first pass used model-hidden-matched cases (`2560/3072/3584/3840/4096`) and revealed a runner nuance:
    - `build_stepwise_config` applied `--decode-stepwise-sync-step` only when backend was Metal.
    - this made CPU rows under `sync_step=true` effectively baseline/no-op.
  - implemented runner fix in `llama-rs/examples/bench_attention_layer.rs`:
    - `with_sync_per_step(parsed.decode_stepwise_sync_step)`
    - `with_readback_per_step(parsed.decode_stepwise_readback_step)`
    - removed now-unused `backend` parameter in `build_stepwise_config`.
  - revalidated after code change:
    - `cargo fmt --all`
    - `cargo clippy --workspace --all-targets`
    - `cargo test --workspace`

- Re-ran 6-model model-hidden-matched sync-step A/B after the runner fix:
  - base:
    - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_cpu_base.txt`
    - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_mtl_base.txt`
  - sync on:
    - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_cpu_sync_on.txt`
    - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_mtl_sync_on.txt`
  - impact:
    - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_impact.md`
    - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_impact.csv`
  - aggregates (`sync/base`):
    - CPU token `~1.002`,
    - MTL0 token `~0.996`,
    - overall token `~0.999`,
    - checksum deltas remained `0.0` for all rows.

- Decision update:
  - keep `sync_step=false` default under the current lock.
  - keep the sync-step switch for explicit A/B use, now with backend-consistent semantics.

## Next operator pass after sync-policy reassessment (`readback_step`)

- Candidate:
  - enable `--decode-stepwise-readback-step` on the same canonical lock used for sync-policy follow-up.
- Base reused from the post-fix canonical runs:
  - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_cpu_base.txt`
  - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_mtl_base.txt`
- Variant runs:
  - `target/benchmarks/review4_step2_readback_models6_cpu_on.txt`
  - `target/benchmarks/review4_step2_readback_models6_mtl_on.txt`
- Impact artifacts:
  - `target/benchmarks/review4_step2_readback_models6_impact.md`
  - `target/benchmarks/review4_step2_readback_models6_impact.csv`
- Aggregate (`readback/base`) summary:
  - CPU token `~0.987`,
  - MTL0 token `~1.005`,
  - overall token `~0.996`,
  - setup means improved (`CPU ~0.977`, `MTL0 ~0.982`, overall `~0.980`),
  - checksum deltas remained `0.0` for all rows.
- Decision update:
  - keep `readback_step=false` default on the current lock (backend split remains).
  - retain the switch for targeted A/B and backend-specific investigations.

## Subsequent operator pass (`kv_cache_write_to_cache`) after readback check

- Candidate:
  - enable `--decode-stepwise-kv-cache-write-to-cache` on the same canonical lock.
- Base reused from canonical post-fix runs:
  - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_cpu_base.txt`
  - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_mtl_base.txt`
- Variant runs:
  - `target/benchmarks/review4_step2_kvwritecache_models6_cpu_on.txt`
  - `target/benchmarks/review4_step2_kvwritecache_models6_mtl_on.txt`
- Impact artifacts:
  - `target/benchmarks/review4_step2_kvwritecache_models6_impact.md`
  - `target/benchmarks/review4_step2_kvwritecache_models6_impact.csv`
- Aggregate (`variant/base`) summary:
  - CPU token `~1.008`,
  - MTL0 token `~1.019`,
  - overall token `~1.014` (regression),
  - setup means were near-neutral overall (`~0.994`) but do not offset token slowdown.
- Fidelity note:
  - checksum deltas were non-zero for multiple rows (both CPU and Metal).
- Decision update:
  - keep `kv_cache_write_to_cache=false` on the current lock.

## Additional operator pass (`head_stage_buf`) after kv-write-to-cache check

- Candidate:
  - enable `--decode-stepwise-head-stage-buffer` on the same canonical lock.
- Base reused from canonical post-fix runs:
  - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_cpu_base.txt`
  - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_mtl_base.txt`
- Variant runs:
  - `target/benchmarks/review4_step2_headstage_models6_cpu_on.txt`
  - `target/benchmarks/review4_step2_headstage_models6_mtl_on.txt`
- Impact artifacts:
  - `target/benchmarks/review4_step2_headstage_models6_impact.md`
  - `target/benchmarks/review4_step2_headstage_models6_impact.csv`
- Aggregate (`variant/base`) summary:
  - CPU token `~1.020`,
  - MTL0 token `~1.002`,
  - overall token `~1.011` (regression),
  - setup overhead also increased (`overall ~1.026`).
- Fidelity note:
  - checksum parity remained exact (`0.0` deltas for all rows).
- Decision update:
  - keep `head_stage_buf=false` on the current lock.

## Next operator pass (`block_gateup_fused`) on canonical lock

- Candidate:
  - enable `--decode-stepwise-fuse-block-gate-up` on the same canonical lock.
- Base reused from canonical post-fix runs:
  - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_cpu_base.txt`
  - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_mtl_base.txt`
- Variant runs:
  - `target/benchmarks/review4_step2_blockgateup_models6_cpu_on.txt`
  - `target/benchmarks/review4_step2_blockgateup_models6_mtl_on.txt`
- Impact artifacts:
  - `target/benchmarks/review4_step2_blockgateup_models6_impact.md`
  - `target/benchmarks/review4_step2_blockgateup_models6_impact.csv`
- Aggregate (`variant/base`) summary:
  - CPU token `~0.993`,
  - MTL0 token `~1.003`,
  - overall token `~0.998`,
  - setup overhead increased (`overall ~1.022`).
- Fidelity note:
  - checksum parity remained exact (`0.0` deltas for all rows).
- Decision update:
  - keep `block_gateup_fused=false` on the current lock.

## `perf-close-cpp-gap` trial: gpt2 synthetic last-node lookup caching

- Trialed a micro-optimization in `llama-rs/src/gpt2_synthetic.rs`:
  - cache `graph.last_node()` outside the backend step loop.
- Measured with repeated runs (`r=5`) in A/B style:
  - pre:
    - `target/benchmarks/review4_perf_close_gap_gpt2_backend_lastnodecache_r5_pre.txt`
  - post:
    - `target/benchmarks/review4_perf_close_gap_gpt2_backend_lastnodecache_r5_post.txt`
  - impact:
    - `target/benchmarks/review4_perf_close_gap_gpt2_backend_lastnodecache_r5_impact.md`
  - median (`post/pre`) summary:
    - CPU `~0.991`,
    - MTL0 `~1.101`.
- Decision update:
  - reject and revert this micro-optimization (backend impact is not consistently positive).

## Additional operator pass (`mask_host_elide`) after block-gateup check

- Candidate:
  - enable `--decode-stepwise-elide-mask-host-buffer` on the same canonical lock.
- Base reused from canonical post-fix runs:
  - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_cpu_base.txt`
  - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_mtl_base.txt`
- Variant runs:
  - `target/benchmarks/review4_step2_maskhost_models6_cpu_on.txt`
  - `target/benchmarks/review4_step2_maskhost_models6_mtl_on.txt`
- Impact artifacts:
  - `target/benchmarks/review4_step2_maskhost_models6_impact.md`
  - `target/benchmarks/review4_step2_maskhost_models6_impact.csv`
- Aggregate (`variant/base`) summary:
  - CPU token `~0.988`,
  - MTL0 token `~1.004`,
  - overall token `~0.996`,
  - setup remained near-neutral (`overall ~0.998`).
- Fidelity note:
  - checksum parity remained exact (`0.0` deltas for all rows).
- Decision update:
  - keep `mask_host_elide=false` on the current lock (backend split remains).

## `parallel-remaining-examples` harness hardening pass

- Updated `examples/bench_upstream_suite.rs`:
  - default benchmark target list now matches currently available target names in local ggml build (`simple-*`, `gpt-2-*`, `gpt-j*`, `magika`, `mnist-*`, `sam`, `yolov3-tiny`, `perf-metal`).
  - added dynamic target discovery via:
    - `cmake --build <dir> --target help`
  - selected targets are now filtered against discovered available targets;
    unavailable targets are reported as `skipped_targets` in summary output.
- Added run-skip policy for argument/model-data-dependent targets:
  - targets that require explicit external model/data arguments are now marked as `skipped_runs` (not failures) in this harness.
- Validation and suite run:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
  - suite artifact:
    - `target/benchmarks/review4_parallel_remaining_examples_suite_post_hardening.txt`
    - summary:
      - `target/benchmarks/review4_parallel_remaining_examples_suite_post_hardening_summary.md`
  - current run result:
    - `passed=4`, `failed=12` (remaining failures are run-time errors on model-asset dependent targets).
- Post skip-rule rerun:
  - suite artifact:
    - `target/benchmarks/review4_parallel_remaining_examples_suite_post_skiprules.txt`
  - summary:
    - `target/benchmarks/review4_parallel_remaining_examples_suite_post_skiprules_summary.md`
  - result:
    - `passed=3`, `failed=0`, `skipped_run_targets=13`.

## Loop continuation (`1 -> 2 -> 3`) on canonical lock (`c01..c04`)

- Baseline reused for all step2 operator A/B passes:
  - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_cpu_base.txt`
  - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_mtl_base.txt`

### Cycle `c01`

1. step2-next-operator-pass (`head_concat_balanced=true`)
   - variants:
     - `target/benchmarks/review4_c01_20260317T165532_step2_headconcat_models6_cpu_on.txt`
     - `target/benchmarks/review4_c01_20260317T165532_step2_headconcat_models6_mtl_on.txt`
   - impact:
     - `target/benchmarks/review4_c01_20260317T165532_step2_headconcat_models6_impact.md`
     - `target/benchmarks/review4_c01_20260317T165532_step2_headconcat_models6_impact.csv`
   - aggregates (`variant/base`): CPU token `~1.042`, MTL0 token `~1.007`, overall token `~1.024`, setup overall `~1.090`.
   - decision: reject; keep `head_concat_balanced=false`.
2. perf-close-cpp-gap pass (`gpt2_backend --threads 2`, `r=5`)
   - pre/post:
     - `target/benchmarks/review4_c01_20260317T170130_perf_close_gap_gpt2_backend_threads2_r5_pre.txt`
     - `target/benchmarks/review4_c01_20260317T170130_perf_close_gap_gpt2_backend_threads2_r5_post.txt`
   - impact:
     - `target/benchmarks/review4_c01_20260317T170130_perf_close_gap_gpt2_backend_threads2_r5_impact.md`
   - medians (`post/pre`): CPU `~0.985`, MTL0 `~0.912`.
   - decision: accept as runtime tuning signal (no code/default flip).
3. parallel-remaining-examples pass (`bench_upstream_suite --keep-going`)
   - raw/summary:
     - `target/benchmarks/review4_c01_20260317T170512_parallel_remaining_examples_suite.txt`
     - `target/benchmarks/review4_c01_20260317T170512_parallel_remaining_examples_suite_summary.md`
   - status: `passed=3`, `failed=0`, `skipped_run_targets=13`.

### Cycle `c02`

1. step2-next-operator-pass (`--decode-stepwise-no-static-kv-head-precompute`)
   - variants:
     - `target/benchmarks/review4_c02_20260317T170617_step2_statickv_off_models6_cpu_on.txt`
     - `target/benchmarks/review4_c02_20260317T170617_step2_statickv_off_models6_mtl_on.txt`
   - impact:
     - `target/benchmarks/review4_c02_20260317T170617_step2_statickv_off_models6_impact.md`
     - `target/benchmarks/review4_c02_20260317T170617_step2_statickv_off_models6_impact.csv`
   - aggregates (`variant/base`): CPU token `~1.063`, MTL0 token `~1.013`, overall token `~1.038`, setup overall `~1.126`.
   - decision: reject; keep `kvhead_static_precompute=true`.
2. perf-close-cpp-gap pass (`gpt2_backend --n-batch 16`, `r=5`)
   - pre/post:
     - `target/benchmarks/review4_c02_20260317T171023_perf_close_gap_gpt2_backend_batch16_r5_pre.txt`
     - `target/benchmarks/review4_c02_20260317T171023_perf_close_gap_gpt2_backend_batch16_r5_post.txt`
   - impact:
     - `target/benchmarks/review4_c02_20260317T171023_perf_close_gap_gpt2_backend_batch16_r5_impact.md`
   - medians (`post/pre`): CPU `~0.599`, MTL0 `~0.538`.
   - decision: accept as runtime tuning signal (workload-shape dependent).
3. parallel-remaining-examples pass (`bench_upstream_suite --keep-going`)
   - raw/summary:
     - `target/benchmarks/review4_c02_20260317T171305_parallel_remaining_examples_suite.txt`
     - `target/benchmarks/review4_c02_20260317T171305_parallel_remaining_examples_suite_summary.md`
   - status: `passed=3`, `failed=0`, `skipped_run_targets=13`.

### Cycle `c03`

1. step2-next-operator-pass (`--decode-stepwise-no-position-delta`)
   - variants:
     - `target/benchmarks/review4_c03_20260317T171408_step2_nopos_models6_cpu_on.txt`
     - `target/benchmarks/review4_c03_20260317T171408_step2_nopos_models6_mtl_on.txt`
   - impact:
     - `target/benchmarks/review4_c03_20260317T171408_step2_nopos_models6_impact.md`
     - `target/benchmarks/review4_c03_20260317T171408_step2_nopos_models6_impact.csv`
   - aggregates (`variant/base`): CPU token `~1.036`, MTL0 token `~1.001`, overall token `~1.019`, setup overall `~1.167`.
   - decision: reject; keep `position_delta=true`.
2. perf-close-cpp-gap pass (`gpt2_backend --n-batch 16 --threads 2`, `r=5`)
   - pre/post:
     - `target/benchmarks/review4_c03_20260317T171810_perf_close_gap_gpt2_backend_threads2_batch16_r5_pre.txt`
     - `target/benchmarks/review4_c03_20260317T171810_perf_close_gap_gpt2_backend_threads2_batch16_r5_post.txt`
   - impact:
     - `target/benchmarks/review4_c03_20260317T171810_perf_close_gap_gpt2_backend_threads2_batch16_r5_impact.md`
   - medians (`post/pre`): CPU `~0.993`, MTL0 `~0.542`.
   - decision: accept as runtime tuning signal.
3. parallel-remaining-examples pass (`bench_upstream_suite --keep-going`)
   - raw/summary:
     - `target/benchmarks/review4_c03_20260317T172057_parallel_remaining_examples_suite.txt`
     - `target/benchmarks/review4_c03_20260317T172057_parallel_remaining_examples_suite_summary.md`
   - status: `passed=3`, `failed=0`, `skipped_run_targets=13`.

### Cycle `c04`

1. step2-next-operator-pass (`--decode-stepwise-no-mask-delta`)
   - variants:
     - `target/benchmarks/review4_c04_20260317T172201_step2_nomask_models6_cpu_on.txt`
     - `target/benchmarks/review4_c04_20260317T172201_step2_nomask_models6_mtl_on.txt`
   - impact:
     - `target/benchmarks/review4_c04_20260317T172201_step2_nomask_models6_impact.md`
     - `target/benchmarks/review4_c04_20260317T172201_step2_nomask_models6_impact.csv`
   - aggregates (`variant/base`): CPU token `~1.037`, MTL0 token `~1.000`, overall token `~1.019`, setup overall `~1.202`.
   - decision: reject; keep `mask_delta=true`.
2. perf-close-cpp-gap pass (`gpt2_backend --n-batch 16 --threads 4`, `r=5`)
   - pre/post:
     - `target/benchmarks/review4_c04_20260317T172614_perf_close_gap_gpt2_backend_threads4_batch16_r5_pre.txt`
     - `target/benchmarks/review4_c04_20260317T172614_perf_close_gap_gpt2_backend_threads4_batch16_r5_post.txt`
   - impact:
     - `target/benchmarks/review4_c04_20260317T172614_perf_close_gap_gpt2_backend_threads4_batch16_r5_impact.md`
   - medians (`post/pre`): CPU `~1.085`, MTL0 `~0.982`.
   - decision: reject (`CPU` regression dominates).
3. parallel-remaining-examples pass (`bench_upstream_suite --keep-going`)
   - raw/summary:
     - `target/benchmarks/review4_c04_20260317T172855_parallel_remaining_examples_suite.txt`
     - `target/benchmarks/review4_c04_20260317T172855_parallel_remaining_examples_suite_summary.md`
   - status: `passed=3`, `failed=0`, `skipped_run_targets=13`.

### Loop-wide status update

- step2 canonical single-toggle operator queue on this lock is now exhausted across:
  - sync/readback/kvwritecache/headstage/blockgateup/maskhost/headconcat/statickv/nopos/nomask.
- defaults remain unchanged for stepwise lock due repeated regressions or backend split.
- perf pass produced reproducible runtime-tuning wins (`threads=2`, `n_batch=16`) but no code-level candidate accepted in this loop segment.

## Loop continuation (`1 -> 2 -> 3`) on canonical lock (`c05..c08`)

### Cycle `c05`

1. step2-next-operator-pass (`head_concat_balanced=true`)
   - variants:
     - `target/benchmarks/review4_c05_20260317T174327_step2_headconcat_models6_cpu_on.txt`
     - `target/benchmarks/review4_c05_20260317T174327_step2_headconcat_models6_mtl_on.txt`
   - impact:
     - `target/benchmarks/review4_c05_20260317T174327_step2_headconcat_models6_impact.md`
     - `target/benchmarks/review4_c05_20260317T174327_step2_headconcat_models6_impact.csv`
   - aggregates (`variant/base`): CPU token `~1.028`, MTL0 token `~1.000`, overall token `~1.014`.
   - decision: reject; keep `head_concat_balanced=false`.
2. perf-close-cpp-gap pass (`gpt2_backend --threads 2`, `r=5`)
   - pre/post:
     - `target/benchmarks/review4_c05_20260317T174700_perf_close_gap_gpt2_backend_threads2_r5_pre.txt`
     - `target/benchmarks/review4_c05_20260317T174700_perf_close_gap_gpt2_backend_threads2_r5_post.txt`
   - impact:
     - `target/benchmarks/review4_c05_20260317T174700_perf_close_gap_gpt2_backend_threads2_r5_impact.md`
   - medians (`post/pre`): CPU `~0.998`, MTL0 `~0.975`.
   - decision: accept as runtime tuning signal.
3. parallel-remaining-examples pass (`bench_upstream_suite --keep-going`)
   - raw/summary:
     - `target/benchmarks/review4_c05_20260317T174914_parallel_remaining_examples_suite.txt`
     - `target/benchmarks/review4_c05_20260317T174914_parallel_remaining_examples_suite_summary.md`
   - status: `passed=3`, `failed=0`, `skipped_run_targets=13`.

### Cycle `c06`

1. step2-next-operator-pass (`--decode-stepwise-no-static-kv-head-precompute`)
   - variants:
     - `target/benchmarks/review4_c06_20260317T174922_step2_statickv_off_models6_cpu_on.txt`
     - `target/benchmarks/review4_c06_20260317T174922_step2_statickv_off_models6_mtl_on.txt`
   - impact:
     - `target/benchmarks/review4_c06_20260317T174922_step2_statickv_off_models6_impact.md`
     - `target/benchmarks/review4_c06_20260317T174922_step2_statickv_off_models6_impact.csv`
   - aggregates (`variant/base`): CPU token `~1.058`, MTL0 token `~1.012`, overall token `~1.035`.
   - decision: reject; keep `kvhead_static_precompute=true`.
2. perf-close-cpp-gap pass (`gpt2_backend --n-batch 16`, `r=5`)
   - pre/post:
     - `target/benchmarks/review4_c06_20260317T175302_perf_close_gap_gpt2_backend_batch16_r5_pre.txt`
     - `target/benchmarks/review4_c06_20260317T175302_perf_close_gap_gpt2_backend_batch16_r5_post.txt`
   - impact:
     - `target/benchmarks/review4_c06_20260317T175302_perf_close_gap_gpt2_backend_batch16_r5_impact.md`
   - medians (`post/pre`): CPU `~0.602`, MTL0 `~0.516`.
   - decision: accept as runtime tuning signal.
3. parallel-remaining-examples pass (`bench_upstream_suite --keep-going`)
   - raw/summary:
     - `target/benchmarks/review4_c06_20260317T175516_parallel_remaining_examples_suite.txt`
     - `target/benchmarks/review4_c06_20260317T175516_parallel_remaining_examples_suite_summary.md`
   - status: `passed=3`, `failed=0`, `skipped_run_targets=13`.

### Cycle `c07`

1. step2-next-operator-pass (`--decode-stepwise-no-position-delta`)
   - variants:
     - `target/benchmarks/review4_c07_20260317T175524_step2_nopos_models6_cpu_on.txt`
     - `target/benchmarks/review4_c07_20260317T175524_step2_nopos_models6_mtl_on.txt`
   - impact:
     - `target/benchmarks/review4_c07_20260317T175524_step2_nopos_models6_impact.md`
     - `target/benchmarks/review4_c07_20260317T175524_step2_nopos_models6_impact.csv`
   - aggregates (`variant/base`): CPU token `~1.029`, MTL0 token `~1.001`, overall token `~1.015`.
   - decision: reject; keep `position_delta=true`.
2. perf-close-cpp-gap pass (`gpt2_backend --n-batch 16 --threads 2`, `r=5`)
   - pre/post:
     - `target/benchmarks/review4_c07_20260317T175908_perf_close_gap_gpt2_backend_threads2_batch16_r5_pre.txt`
     - `target/benchmarks/review4_c07_20260317T175908_perf_close_gap_gpt2_backend_threads2_batch16_r5_post.txt`
   - impact:
     - `target/benchmarks/review4_c07_20260317T175908_perf_close_gap_gpt2_backend_threads2_batch16_r5_impact.md`
   - medians (`post/pre`): CPU `~0.977`, MTL0 `~1.065`.
   - decision: reject (Metal regression in this pass).
3. parallel-remaining-examples pass (`bench_upstream_suite --keep-going`)
   - raw/summary:
     - `target/benchmarks/review4_c07_20260317T180127_parallel_remaining_examples_suite.txt`
     - `target/benchmarks/review4_c07_20260317T180127_parallel_remaining_examples_suite_summary.md`
   - status: `passed=3`, `failed=0`, `skipped_run_targets=13`.

### Cycle `c08`

1. step2-next-operator-pass (`--decode-stepwise-no-mask-delta`)
   - variants:
     - `target/benchmarks/review4_c08_20260317T180135_step2_nomask_models6_cpu_on.txt`
     - `target/benchmarks/review4_c08_20260317T180135_step2_nomask_models6_mtl_on.txt`
   - impact:
     - `target/benchmarks/review4_c08_20260317T180135_step2_nomask_models6_impact.md`
     - `target/benchmarks/review4_c08_20260317T180135_step2_nomask_models6_impact.csv`
   - aggregates (`variant/base`): CPU token `~1.035`, MTL0 token `~1.000`, overall token `~1.018`.
   - decision: reject; keep `mask_delta=true`.
2. perf-close-cpp-gap pass (`gpt2_backend --n-batch 16 --threads 4`, `r=5`)
   - pre/post:
     - `target/benchmarks/review4_c08_20260317T180520_perf_close_gap_gpt2_backend_threads4_batch16_r5_pre.txt`
     - `target/benchmarks/review4_c08_20260317T180520_perf_close_gap_gpt2_backend_threads4_batch16_r5_post.txt`
   - impact:
     - `target/benchmarks/review4_c08_20260317T180520_perf_close_gap_gpt2_backend_threads4_batch16_r5_impact.md`
   - medians (`post/pre`): CPU `~0.997`, MTL0 `~0.532`.
   - decision: keep conservative and hold defaults until this contradictory signal is confirmed by additional stability reruns.
3. parallel-remaining-examples pass (`bench_upstream_suite --keep-going`)
   - raw/summary:
     - `target/benchmarks/review4_c08_20260317T180742_parallel_remaining_examples_suite.txt`
     - `target/benchmarks/review4_c08_20260317T180742_parallel_remaining_examples_suite_summary.md`
   - status: `passed=3`, `failed=0`, `skipped_run_targets=13`.

### Runtime progress snapshot

- phase2 elapsed after `c08`: `1462 s` (`~24.4 min`).
- cumulative loop estimate (`phase1 ~44 min` + phase2): `4102 s` (`~68.4 min`).
- remaining to the `>= ~3 h` target: `~6698 s` (`~111.6 min`) at cycle boundary.

## Loop continuation (`1 -> 2 -> 3`) on canonical lock (`c09..c18`)

- Generated cycle-indexed artifacts for all ten cycles:
  - step2 impacts:
    - `target/benchmarks/review4_c09_20260317T181241_step2_headconcat_models6_impact.md`
    - `target/benchmarks/review4_c10_20260317T181859_step2_statickv_off_models6_impact.md`
    - `target/benchmarks/review4_c11_20260317T182524_step2_nopos_models6_impact.md`
    - `target/benchmarks/review4_c12_20260317T183156_step2_nomask_models6_impact.md`
    - `target/benchmarks/review4_c13_20260317T183836_step2_headconcat_models6_impact.md`
    - `target/benchmarks/review4_c14_20260317T184517_step2_statickv_off_models6_impact.md`
    - `target/benchmarks/review4_c15_20260317T185202_step2_nopos_models6_impact.md`
    - `target/benchmarks/review4_c16_20260317T185851_step2_nomask_models6_impact.md`
    - `target/benchmarks/review4_c17_20260317T190542_step2_headconcat_models6_impact.md`
    - `target/benchmarks/review4_c18_20260317T191237_step2_statickv_off_models6_impact.md`
  - perf impacts:
    - `target/benchmarks/review4_c09_20260317T181627_perf_close_gap_gpt2_backend_threads2_r5_impact.md`
    - `target/benchmarks/review4_c10_20260317T182311_perf_close_gap_gpt2_backend_batch16_r5_impact.md`
    - `target/benchmarks/review4_c11_20260317T182916_perf_close_gap_gpt2_backend_threads2_batch16_r5_impact.md`
    - `target/benchmarks/review4_c12_20260317T183610_perf_close_gap_gpt2_backend_threads4_batch16_r5_impact.md`
    - `target/benchmarks/review4_c13_20260317T184121_perf_close_gap_gpt2_backend_threads2_r5_impact.md`
    - `target/benchmarks/review4_c14_20260317T185001_perf_close_gap_gpt2_backend_batch16_r5_impact.md`
    - `target/benchmarks/review4_c15_20260317T185617_perf_close_gap_gpt2_backend_threads2_batch16_r5_impact.md`
    - `target/benchmarks/review4_c16_20260317T190307_perf_close_gap_gpt2_backend_threads4_batch16_r5_impact.md`
    - `target/benchmarks/review4_c17_20260317T190955_perf_close_gap_gpt2_backend_threads2_r5_impact.md`
    - `target/benchmarks/review4_c18_20260317T191649_perf_close_gap_gpt2_backend_batch16_r5_impact.md`
  - suite summaries:
    - `target/benchmarks/review4_c09_20260317T181851_parallel_remaining_examples_suite_summary.md`
    - `target/benchmarks/review4_c10_20260317T182517_parallel_remaining_examples_suite_summary.md`
    - `target/benchmarks/review4_c11_20260317T183127_parallel_remaining_examples_suite_summary.md`
    - `target/benchmarks/review4_c12_20260317T183830_parallel_remaining_examples_suite_summary.md`
    - `target/benchmarks/review4_c13_20260317T184332_parallel_remaining_examples_suite_summary.md`
    - `target/benchmarks/review4_c14_20260317T185212_parallel_remaining_examples_suite_summary.md`
    - `target/benchmarks/review4_c15_20260317T185830_parallel_remaining_examples_suite_summary.md`
    - `target/benchmarks/review4_c16_20260317T190520_parallel_remaining_examples_suite_summary.md`
    - `target/benchmarks/review4_c17_20260317T191202_parallel_remaining_examples_suite_summary.md`
    - `target/benchmarks/review4_c18_20260317T191935_parallel_remaining_examples_suite_summary.md`

### Aggregated outcomes for `c09..c18`

- step2-next-operator-pass remained reject-only on this lock:
  - `headconcat` (`c09/c13/c17`): overall token `~1.023/~1.024/~1.017`.
  - `statickv_off` (`c10/c14/c18`): overall token `~1.033/~1.035/~1.030`.
  - `nopos` (`c11/c15`): overall token `~1.012/~1.018`.
  - `nomask` (`c12/c16`): overall token `~1.019/~1.022`.
- perf-close-cpp-gap was mixed in this span:
  - `threads2@batch8` (`c09/c13/c17`): ratios varied (`~1.028/1.184`, `~1.000/0.702`, `~1.003/1.006` CPU/MTL0).
  - `batch16@threads2` (`c10/c14/c18`): repeated robust wins (`CPU ~0.598/0.599/0.541`, `MTL0 ~0.530/0.522/0.515`).
  - `threads2@batch16` (`c11/c15`): mixed/near-neutral (`CPU ~1.009/0.999`, `MTL0 ~0.991/1.004`).
  - `threads4@batch16` (`c12/c16`): non-robust (`CPU ~1.003/0.989`, `MTL0 ~0.980/1.031`), keep conservative defaults.
- `parallel-remaining-examples` stayed stable in all ten cycles:
  - each summary reported `passed=3`, `failed=0`, `skipped_run_targets=13`.

### Runtime progress snapshot

- phase2 elapsed after `c18`: `5481 s` (`~91.4 min`).
- cumulative loop estimate (`phase1 ~44 min` + phase2): `8121 s` (`~135.4 min`).
- remaining to the `>= ~3 h` target: `~2679 s` (`~44.7 min`) at cycle boundary.

## Loop completion block (`1 -> 2 -> 3`, `c19..c22`) and stop boundary

- Final cycle artifacts:
  - step2 impacts:
    - `target/benchmarks/review4_c19_20260317T192108_step2_nopos_models6_impact.md`
    - `target/benchmarks/review4_c20_20260317T193027_step2_nomask_models6_impact.md`
    - `target/benchmarks/review4_c21_20260317T194103_step2_headconcat_models6_impact.md`
    - `target/benchmarks/review4_c22_20260317T195158_step2_statickv_off_models6_impact.md`
  - perf impacts:
    - `target/benchmarks/review4_c19_20260317T192548_perf_close_gap_gpt2_backend_threads2_batch16_r5_impact.md`
    - `target/benchmarks/review4_c20_20260317T193554_perf_close_gap_gpt2_backend_threads4_batch16_r5_impact.md`
    - `target/benchmarks/review4_c21_20260317T194630_perf_close_gap_gpt2_backend_threads2_r5_impact.md`
    - `target/benchmarks/review4_c22_20260317T195810_perf_close_gap_gpt2_backend_batch16_r5_impact.md`
  - suite summaries:
    - `target/benchmarks/review4_c19_20260317T193012_parallel_remaining_examples_suite_summary.md`
    - `target/benchmarks/review4_c20_20260317T194048_parallel_remaining_examples_suite_summary.md`
    - `target/benchmarks/review4_c21_20260317T195143_parallel_remaining_examples_suite_summary.md`
    - `target/benchmarks/review4_c22_20260317T200612_parallel_remaining_examples_suite_summary.md`

### Outcome summary for `c19..c22`

- step2-next-operator-pass:
  - `c19` `nopos`: overall token `~1.016` -> reject.
  - `c20` `nomask`: CPU mean token `~0.970`, MTL0 mean token `~1.007`, overall `~0.989` with setup inflation (`~1.747`) -> reject under conservative backend-balanced policy.
  - `c21` `headconcat`: overall token `~1.025` -> reject.
  - `c22` `statickv_off`: overall token `~1.029` -> reject.
- perf-close-cpp-gap (`r=5` medians):
  - `c19` `threads2@batch16`: CPU `~0.887`, MTL0 `~0.939` -> accept signal.
  - `c20` `threads4@batch16`: CPU `~0.958`, MTL0 `~0.891` -> accept signal in this run, but still treated as non-default due prior mixed stability.
  - `c21` `threads2@batch8`: CPU `~1.005`, MTL0 `~0.886` -> accept signal (mixed CPU near-neutral).
  - `c22` `batch16@threads2`: CPU `~0.652`, MTL0 `~0.518` -> accept signal.
- parallel-remaining-examples remained stable in every cycle:
  - `passed=3`, `failed=0`, `skipped_run_targets=13`.

### Stop condition reached

- phase2 runtime at `c22` boundary: `8202 s` (`~136.7 min`).
- cumulative loop estimate (`phase1 ~44 min` + phase2): `10842 s` (`~180.7 min`).
- stop condition satisfied: cumulative runtime `>= ~3 h` at a natural cycle boundary.

### Final policy status after `c05..c22`

- step2 defaults remain unchanged on canonical lock:
  - keep `mask_delta=true`, `position_delta=true`, `kvhead_static_precompute=true`, `head_concat_balanced=false`.
- perf tuning evidence remains strongest for `n_batch=16` and often `threads=2`; `threads=4@batch16` remains mixed/non-robust across cycles and is not promoted to default by this loop.
- suite harness remains operationally stable with unchanged summary (`passed=3`, `failed=0`, `skipped_run_targets=13`).

## Remaining-task closure pass (post 3h loop)

- Completed pending step2 operator pass (`step2-next-operator-pass-6`) with an interaction candidate:
  - variant: `--decode-stepwise-no-mask-delta --decode-stepwise-no-position-delta`
  - artifacts:
    - `target/benchmarks/review4_step2_nomask_nopos_models6_cpu_on.txt`
    - `target/benchmarks/review4_step2_nomask_nopos_models6_mtl_on.txt`
    - `target/benchmarks/review4_step2_nomask_nopos_models6_impact.md`
    - `target/benchmarks/review4_step2_nomask_nopos_models6_impact.csv`
  - result:
    - token stayed near-neutral (`overall ~0.999`),
    - setup overhead regressed heavily (`overall ~1.782`),
    - checksum parity stayed exact (`0.0` deltas).
  - decision: reject; keep defaults unchanged.

- Closed `perf-close-cpp-gap` with explicit model-exec improvement quantification:
  - baseline (`gpt2_backend`): `n_batch=8`, `threads=1`, `r=5`
    - `target/benchmarks/review4_perf_close_gap_model_exec_baseline_r5.txt`
  - optimized (`gpt2_backend`): `n_batch=16`, `threads=2`, `r=5`
    - `target/benchmarks/review4_perf_close_gap_model_exec_opt_r5.txt`
  - impact summary:
    - `target/benchmarks/review4_perf_close_gap_model_exec_opt_r5_impact.md`
    - CPU median `avg_item_ms`: `0.007759 -> 0.005056` (`34.8%` faster),
    - MTL0 median `avg_item_ms`: `0.043701 -> 0.030004` (`31.3%` faster).
  - phase2 cycle-level perf aggregation:
    - `target/benchmarks/review4_phase2_perf_summary.md`
    - best robust phase2 median candidate remained `batch16`.

- Closed pending parallel-suite synthetic/no-asset follow-up with run-arg plumbing:
  - code update in `examples/bench_upstream_suite.rs`:
    - per-target run args via `GGML_UPSTREAM_RUN_ARGS_<TARGET>`,
      where `<TARGET>` is uppercased and non-alnum chars map to `_`
      (for example `gpt-2-ctx` -> `GGML_UPSTREAM_RUN_ARGS_GPT_2_CTX`).
    - model/data-dependent targets are skipped by default unless explicit args are provided.
  - docs update:
    - `docs/ggml-rs/README.md`
    - `docs/ggml-rs/KNOWLEDGE_BASE.md`
  - validation + smoke:
    - `cargo fmt --all`
    - `cargo clippy --workspace --all-targets`
    - `cargo test --workspace`
    - `target/benchmarks/review4_parallel_remaining_examples_suite_runargs_env.txt`
    - `target/benchmarks/review4_parallel_remaining_examples_suite_runargs_env_summary.md`
      (`passed=3`, `failed=0`, `skipped_run_targets=13`).
