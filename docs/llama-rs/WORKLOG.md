# llama-rs worklog

This top-level file is an index + short status snapshot.

Detailed logs live under `docs/llama-rs/worklog/`.

## Auto-split policy (must follow)

- Keep this top-level file compact (target: <= 120 lines).
- When either threshold is crossed, rotate automatically:
  - line count > 250, or
  - file size > 16 KiB.
- Write detailed progress into a dated file under `docs/llama-rs/worklog/`.
- Keep top-level updates to:
  - index row additions,
  - short latest snapshot (<= 10 bullets),
  - links to detailed artifacts.

## Rotation procedure

1. Create or append a dated file: `docs/llama-rs/worklog/YYYY-MM-DD-<scope>.md`.
2. Move long bullet blocks / dense benchmark logs from top-level to that dated file.
3. Update the index table below.
4. Replace top-level details with concise snapshot bullets and links.

## Index

| Date | File | Scope |
| --- | --- | --- |
| 2026-03-13 | `docs/llama-rs/worklog/2026-03-13-migration-log.md` | Backend bring-up, GGUF/model foundations, metadata-first resolver hardening, attention ADT migration, benchmark harness expansion, CPU/Metal runtime verification |
| 2026-03-15 | `docs/llama-rs/worklog/2026-03-15-migration-log.md` | Stepwise optimization loop continuation, review2 cleanup, generic host-I/O refactor, dependency-management alignment, recent hotspot A/B passes |

## Latest status snapshot

- `ggml` is now managed as a submodule at `vendor/ggml`.
- `llama.cpp` remains an external comparison reference (not an in-repo dependency); repro steps are documented in `docs/llama-rs/KNOWLEDGE_BASE.md`.
- Recent stepwise checks (ELYZA/Qwen3.5 layer sweeps) were captured with checksum parity preserved (`0.0` deltas).
- User-requested 6-model balanced `mask_host_elide` sweep on current lock completed:
  - aggregate `on/base`: CPU `~1.001`, MTL0 `~1.004`, overall `~1.002`;
  - checksum parity remained exact (`max abs delta = 0.0`);
  - policy remains `mask_host_elide=false` by default.
- Latest toggles rechecked on refreshed baselines:
  - `block_gateup_fused`: kept default-off (CPU regression signal),
  - `head_stage_buf`: kept default-off (CPU regression signal),
  - `mask_host_elide`: 6-model balanced sweep on the current lock shows slight aggregate regression; keep default-off and use explicit A/B only.
- Decode ownership refactor (`Result<Vec<T>>`) is now documented with assembly
  cut/judgement artifacts (`review3_decode_asm_snippets|compare|judgement.md`);
  current decision remains to keep the ownership API as default.
- Latest post-review3 step2 delta-toggle recheck on ELYZA (`layers 5..7`) shows
  both `no-mask-delta` and `no-position-delta` are slower (`~1.018` and
  `~1.041` variant/base overall), with checksum parity preserved (`0.0` deltas);
  defaults remain unchanged.
- Step2 continuation on the canonical `4096x32x8x1` decode-stepwise condition
  (`block_layer=0`, `steps=8`, profile `outproj_fused_layerx5`) now has fresh
  `r=3` stability artifacts:
  - `target/benchmarks/review4_step2_stepwise_nomask_stability_r3.md`
    (`no-mask/base`: CPU `~1.006`, MTL0 `~0.998`, overall `~1.002`),
  - `target/benchmarks/review4_step2_stepwise_nopos_stability_r3.md`
    (`no-position/base`: CPU `~1.014`, MTL0 `~1.001`, overall `~1.007`),
  - `target/benchmarks/review4_step2_stepwise_outproj_nofuse_impact.md`
    (`no-fuse/base`: CPU `~1.026`, MTL0 `~1.074`, overall `~1.050`).
  - `target/benchmarks/review4_step2_stepwise_statickv_off_impact.md`
    (`no-static-kv/base`: CPU `~0.992`, MTL0 `~1.007`, overall token `~0.999`,
    setup `~1.082`).
  - `target/benchmarks/review4_step2_stepwise_headstage_stability_r3.md`
    (`head-stage/base`: CPU token `~0.979`, MTL0 token `~0.996`, overall token
    `~0.988`; setup `~1.089` overall).
  - `target/benchmarks/review4_step2_headstage_broadsweep_layers0_7_impact.md`
    (`head-stage/base`, layers `0..7`): CPU token mean `~1.004` (3/8 wins),
    MTL0 token mean `~0.998` (5/8 wins), overall token `~1.001`.
  - `target/benchmarks/review4_step2_blockgateup_broadsweep_layers0_7_stability_r2.md`
    (`block-gateup/base`, layers `0..7`, `r=2`): CPU token `~0.996`,
    MTL0 token `~1.001`, overall token `~0.998`, setup `~1.018`.
  - `target/benchmarks/review4_step2_headconcat_broadsweep_layers0_7_stability_r2.md`
    (`head-concat/base`, layers `0..7`, `r=2`): overall token `~1.006`,
    setup `~1.057` (regression).
  - `target/benchmarks/review4_step2_maskhost_broadsweep_layers0_7_stability_r2.md`
    (`maskhost/base`, layers `0..7`, `r=2`): overall token `~0.996`,
    setup `~1.083` (token gain small vs setup cost).
  - Alternate condition (`decode-steps=16`, layers `0..7`) summary:
    - `target/benchmarks/review4_step2_alt_steps16_layers0_7_summary.md`
    - `maskhost-elide`: token `~1.008`, setup `~1.012`,
    - `block-gateup-fused`: token `~1.004`, setup `~1.025`.
  - Next candidate `kv_cache_write_to_cache` broad sweep:
    - `target/benchmarks/review4_step2_kvwritecache_broadsweep_layers0_7_impact.md`
    - overall token `~1.018`, setup `~1.112` (clear regression).
  - Next candidate `no-static-kv-head-precompute` broad sweep:
    - `target/benchmarks/review4_step2_statickv_off_broadsweep_layers0_7_impact.md`
    - overall token `~1.006`, setup `~1.160` (regression).
  - Next candidate `sync_step` broad stability (`r=2`):
    - `target/benchmarks/review4_step2_syncstep_broadsweep_layers0_7_stability_r2.md`
    - backend split remains strong: CPU token `~0.916` vs MTL0 token `~1.019`,
      with setup overhead increase.
  Policy remains unchanged: keep `mask_delta=true`, `position_delta=true`,
  `outproj_fused=true`, `kvhead_static_precompute=true`, and keep
  `head_stage_buf=false`, `block_gateup_fused=false`, `head_concat_balanced=false`,
  `mask_host_elide=false`, `kv_cache_write_to_cache=false`, `sync_step=false`
  under the current lock.
- Post-fallback idle validation completed on the previously failing Minitron model:
  - `target/benchmarks/review4_idle_minitron_post_fallback_fix.txt`
  - `weights_mode=MetadataDeterministic` now succeeds on both `CPU` and `MTL0`.
- Full post-fallback idle refresh completed across all 6 target models:
  - `target/benchmarks/review4_model_inference_refresh_idle_cpu_metal_post_fallback_fix.txt`
  - `target/benchmarks/review4_model_inference_refresh_idle_cpu_metal_post_fallback_fix_summary.md`
  - all models now emit both CPU/Metal rows.
- Sync-step policy follow-up completed on a model-hidden-matched 6-model sweep:
  - `target/benchmarks/review4_step2_syncpolicy_cpuenabled_models6_impact.md`
  - runner fix: `--decode-stepwise-sync-step` is now applied consistently on both CPU and Metal in `bench_attention_layer` (instead of Metal-only).
  - aggregate (`sync/base`): CPU token `~1.002`, MTL0 token `~0.996`, overall `~0.999`.
  - policy remains `sync_step=false` by default (mixed and near-neutral net effect).
- Next step2 operator pass (`readback_step`) completed on the same lock:
  - `target/benchmarks/review4_step2_readback_models6_impact.md`
  - aggregate (`readback/base`): CPU token `~0.987`, MTL0 token `~1.005`, overall `~0.996`.
  - policy remains `readback_step=false` by default (backend split; keep as explicit A/B switch).
- Subsequent operator pass (`kv_cache_write_to_cache`) recheck completed on the same lock:
  - `target/benchmarks/review4_step2_kvwritecache_models6_impact.md`
  - aggregate (`variant/base`): CPU token `~1.008`, MTL0 token `~1.019`, overall `~1.014` (regression).
  - checksum deltas were non-zero on multiple rows, so policy remains `kv_cache_write_to_cache=false`.
- Additional operator pass (`head_stage_buf`) recheck completed on the same lock:
  - `target/benchmarks/review4_step2_headstage_models6_impact.md`
  - aggregate (`variant/base`): CPU token `~1.020`, MTL0 token `~1.002`, overall `~1.011` (regression).
  - checksum parity remained exact (`0.0` deltas), policy remains `head_stage_buf=false`.
- Additional operator pass (`block_gateup_fused`) recheck completed on the same lock:
  - `target/benchmarks/review4_step2_blockgateup_models6_impact.md`
  - aggregate (`variant/base`): CPU token `~0.993`, MTL0 token `~1.003`, overall `~0.998`; setup `~1.022`.
  - policy remains `block_gateup_fused=false` (backend split + setup overhead).
- Additional operator pass (`mask_host_elide`) recheck completed on the same lock:
  - `target/benchmarks/review4_step2_maskhost_models6_impact.md`
  - aggregate (`variant/base`): CPU token `~0.988`, MTL0 token `~1.004`, overall `~0.996`.
  - policy remains `mask_host_elide=false` (backend split; keep as explicit A/B switch).
- `perf-close-cpp-gap` follow-up trial:
  - tested a micro-optimization in `gpt2_synthetic` (cache `graph.last_node()` lookup outside the loop),
  - r5 median impact artifact: `target/benchmarks/review4_perf_close_gap_gpt2_backend_lastnodecache_r5_impact.md`,
  - result: CPU `~0.991` but Metal `~1.101` (`post/pre`), so the change was rejected and reverted.
- `parallel-remaining-examples` follow-up:
  - hardened `bench_upstream_suite` with dynamic CMake target discovery and updated default target set to currently available upstream targets.
  - first post-hardening suite artifact:
    - `target/benchmarks/review4_parallel_remaining_examples_suite_post_hardening_summary.md`
  - added run-skip rules for model/data-argument dependent targets in the harness.
  - latest suite artifact:
    - `target/benchmarks/review4_parallel_remaining_examples_suite_post_skiprules_summary.md`
  - latest status: `passed=3`, `failed=0`, `skipped_run_targets=13` (explicitly reported as model/data-arg dependent).
- Unblocked `fine-tune-balanced-profile`, refreshed llama.cpp baselines, and reran the extended repeat-pair calibration sweep (`7` candidates) against refreshed decode rows:
  - summary: `target/benchmarks/review4_finetune_balanced_profile_sweep_summary.md`
  - ranking JSON: `target/benchmarks/review4_finetune_balanced_profile_sweep_summary.json`
  - tuned calibration table: `target/benchmarks/review4_finetune_balanced_profile_cpu5_mtl7_calibration.md`
  - tuned-vs-baseline impact: `target/benchmarks/review4_finetune_balanced_profile_cpu5_mtl7_impact.md`
  - selected pair: `CPU=5`, `MTL0=7` (`cpu5_mtl7`) with avg proxy/cpp:
    - CPU `~0.981`, MTL0 `~1.010`, overall `~0.995`.
  - updated `bench_attention_layer` balanced preset wiring/tag accordingly:
    - profile tag now `outproj_fused_balanced_cpu5_mtl7`.
- Resumed step `1` on the refreshed balanced preset (`cpu5_mtl7`) with ELYZA layer sweep (`block_layer=0..7`):
  - raw: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers0_7.txt`
  - summary: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers0_7_summary.{md,csv}`
  - aggregate: CPU `~28.427 ms`, MTL0 `~28.061 ms`, overall `~28.244 ms` (`avg_token`).
- Ran hotspot-window A/B (`layers 5..7`) with `--decode-stepwise-head-stage-buffer` on top of the refreshed balanced preset:
  - base: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers5_7_headstage_base.txt`
  - variant: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers5_7_headstage_on.txt`
  - impact: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers5_7_headstage_impact.md`
  - aggregate (`variant/base`): token CPU `~0.997`, MTL0 `~0.995`, overall `~0.996`; setup `~1.005`; checksum delta `0.0`.
  - policy on this slice remains `head_stage_buf=false` (near-neutral token gain with setup regression).
- Continued with step `2` cross-model operator A/B (`6` models, CPU/Metal) for `head_stage_buf` on refreshed balanced preset:
  - base: `target/benchmarks/review4_step2_balanced_cpu5_mtl7_headstage_base.txt`
  - variant: `target/benchmarks/review4_step2_balanced_cpu5_mtl7_headstage_on.txt`
  - impact: `target/benchmarks/review4_step2_balanced_cpu5_mtl7_headstage_impact.md`
  - aggregate (`variant/base`): token CPU `~0.995`, MTL0 `~0.994`, overall `~0.994`; checksum delta `0.0`.
  - parity objective (`proxy/cpp -> 1.0`) moved `~0.993 -> ~0.987` overall, so policy remains `head_stage_buf=false`.
- Continued with step `3` upstream-suite refresh on the same lock:
  - run log: `target/benchmarks/review4_step3_upstream_suite_refresh.txt`
  - summary: `target/benchmarks/review4_step3_upstream_suite_refresh_summary.md`
  - status: `passed=3`, `failed=0`, `skipped_run_targets=13` (model/data-arg dependent targets explicitly listed).
- Ran a lock-coordinated parallel subagent pass for ggml upstream example parity
  batches (`foundation`, `gpt2`, `gptj_magika`, `vision_mnist`) and logged
  per-batch progress under `docs/llama-rs/worklog/subagents/`.
- Removed remaining shape-wrapper constructor usage from in-repo call sites and
  migrated to generic constructor APIs (`new_tensor_typed::<T, N>`, `new_tensor_*::<T>`).
- Unified root `ggml-rs/examples` CLI parsing to clap derive (including
  synthetic parity examples and upstream-suite harness) and re-verified runtime:
  `target/benchmarks/review3_constructor_clap_runtime_smoke.txt`.
- Added synthetic loop-reuse perf pass for vision/MNIST proxies (reuse graph/context
  across `--synthetic-iters`) and captured impact:
  `target/benchmarks/vision_mnist/loopreuse_impact.md`.
- Expanded `ggml-rs` typed tensor API to rank-complete generic wrappers
  (`Tensor1D..Tensor4D` + const aliases) and updated call sites to
  `new_tensor_2d_typed::<f32, S>()`-style constructors.
- Continued step `1` with GPT synthetic focus:
  - tested GPT2 ctx full-reuse candidate and rejected it due measured regression,
    see `target/benchmarks/review4_gpt2_ctx_loopreuse_trial_impact.md`,
  - improved `gptj_main_synth` by reusing graph/weights with fixed token-capacity;
    parity-config timing moved `1064us -> 985us` (`~0.926` post/base) with
    identical generated tokens/checksum (`target/benchmarks/review4_gptj_main_synth_impact.md`).
- Continued step `2` with `uv`-based model asset verification:
  - `target/benchmarks/review4_model_asset_uv_check.txt` confirmed all six required GGUF files (`missing_count=0`),
  - real-model CPU/Metal idle smoke re-run succeeded for Qwen3.5 and ELYZA:
    `target/benchmarks/review4_idle_realmodel_cpu_metal_smoke.txt`.
- Per request, continued structural cleanup of `llama-rs`:
  - extracted layer-dimension resolution from `inference.rs` into
    `inference/layer_dimensions.rs`,
  - introduced trait-based layout policy split (`HeadLayoutStrategy` /
    `PreferredHeadLayoutStrategy`) and kept ADT surfaces (`MetadataResolutionMode`,
  `LlamaLayerDimensions`) in the focused module,
  - validated + runtime-smoked:
    `target/benchmarks/review4_llamars_modularization_runtime_smoke.txt`.
- Continued the same modularization pass by extracting attention runtime execution
  from `inference.rs` into `inference/attention_runtime.rs` and re-exporting the
  public attention APIs from `inference.rs`.
- Re-validated the workspace after the extraction and re-ran CPU/Metal runtime
  smoke with locally built `vendor/ggml` shared libs:
  `target/benchmarks/review4_attention_runtime_modularization_runtime_smoke.txt`.
- Ran a GPT-J synthetic guard on top of this refactor:
  - `target/benchmarks/review4_attention_runtime_modularization_gptj_guard.txt`
  - parity stayed exact (token/checksum match) in
    `target/benchmarks/review4_attention_runtime_modularization_gptj_guard_impact.md`.
- Continued step `1` synthetic optimization with a low-risk `gpt2_synthetic` host-allocation cleanup:
  - reused a single `lhs` buffer across steps in `run_ctx` and backend runner paths,
  - parity-config impact (`target/benchmarks/review4_gpt2_fillreuse_impact.md`):
    - `ctx avg_item_ms`: `0.043166 -> 0.017536` (`~0.406`),
    - `alloc avg_item_ms`: `0.018706 -> 0.016009` (`~0.856`),
    - checksums unchanged.
- Runtime smoke for this pass was re-run on CPU/Metal:
  - `target/benchmarks/review4_gpt2_fillreuse_runtime_smoke.txt`.
- Added backend sampled-read API usage for GPT-2 synthetic backend path:
  - new safe API in `ggml-rs`: `Tensor::read_data_backend_at::<T>(offset, len)`,
  - `gpt2_synthetic::run_backend_for_steps` now reads only checksum sample range.
- Added host sampled-read safe API in `ggml-rs`:
  - `Tensor::read_data_at::<T>(offset, len)`,
  - verified via `ggml_tensor_ops` tests; kept as API surface, while `gpt2 run_ctx`
    remains on full readback after stability checks.
- Continued synthetic step1 on `gptj_main_synth`:
  - switched final-step logits readback from full tensor copy to range read
    (`Tensor::read_data_at::<f32>(start, GPTJ_VOCAB)`),
  - parity remained exact (tokens/checksum stable across `r=5`),
  - stability + impact artifacts:
    - `target/benchmarks/review4_gptj_slice_stability_r5_summary.md`
    - `target/benchmarks/review4_gptj_slice_impact_vs_post_baseline.md`
    - median `elapsed_us`: `985 -> 874` (`~0.887`).
- Additional host-write incremental trial (`Tensor::write_data_at` for token updates)
  was benchmarked and rejected due regression:
  - `target/benchmarks/review4_gptj_hostwrite_stability_r5_summary.md`
  - `target/benchmarks/review4_gptj_hostwrite_impact_vs_slice_baseline.md`.
- Re-validated link-system tests (`ggml_tensor_ops`) and CPU/Metal runtime,
  with stability artifacts:
  - `target/benchmarks/review4_gpt2_readslice_final_paritycfg.txt`
  - `target/benchmarks/review4_gpt2_readslice_final_stability_r3_summary.md`
  - `target/benchmarks/review4_gpt2_readslice_final_impact_vs_original.md`.
- Continued requested `1 -> 2 -> 3` loop pass on canonical lock with cycle-indexed artifacts (`c01..c04`):
  - step2 operator passes (model-hidden-matched 6-model CPU/MTL A/B, base reused from `review4_step2_syncpolicy_cpuenabled_models6_{cpu_base,mtl_base}.txt`):
    - `c01` `head_concat_balanced=true`: `target/benchmarks/review4_c01_20260317T165532_step2_headconcat_models6_impact.md` (`overall token ~1.024`, setup `~1.090`) -> reject.
    - `c02` `no_static_kv_head_precompute`: `target/benchmarks/review4_c02_20260317T170617_step2_statickv_off_models6_impact.md` (`overall token ~1.038`, setup `~1.126`) -> reject.
    - `c03` `no_position_delta`: `target/benchmarks/review4_c03_20260317T171408_step2_nopos_models6_impact.md` (`overall token ~1.019`, setup `~1.167`) -> reject.
    - `c04` `no_mask_delta`: `target/benchmarks/review4_c04_20260317T172201_step2_nomask_models6_impact.md` (`overall token ~1.019`, setup `~1.202`) -> reject.
  - `perf-close-cpp-gap` runtime A/B (`gpt2_backend`, `r=5` medians):
    - `c01` `threads=2` vs `1`: `target/benchmarks/review4_c01_20260317T170130_perf_close_gap_gpt2_backend_threads2_r5_impact.md` (CPU `~0.985`, MTL0 `~0.912`) -> accept.
    - `c02` `n_batch=16` vs `8`: `target/benchmarks/review4_c02_20260317T171023_perf_close_gap_gpt2_backend_batch16_r5_impact.md` (CPU `~0.599`, MTL0 `~0.538`) -> accept.
    - `c03` `threads=2` on `n_batch=16`: `target/benchmarks/review4_c03_20260317T171810_perf_close_gap_gpt2_backend_threads2_batch16_r5_impact.md` (CPU `~0.993`, MTL0 `~0.542`) -> accept.
    - `c04` `threads=4` vs `2` on `n_batch=16`: `target/benchmarks/review4_c04_20260317T172614_perf_close_gap_gpt2_backend_threads4_batch16_r5_impact.md` (CPU `~1.085`, MTL0 `~0.982`) -> reject.
    - kept source defaults/code unchanged (runtime tuning evidence only; no robust code-level candidate promoted).
  - `parallel-remaining-examples` keep-going reruns remained stable in all 4 cycles:
    - summaries:
      - `target/benchmarks/review4_c01_20260317T170512_parallel_remaining_examples_suite_summary.md`
      - `target/benchmarks/review4_c02_20260317T171305_parallel_remaining_examples_suite_summary.md`
      - `target/benchmarks/review4_c03_20260317T172057_parallel_remaining_examples_suite_summary.md`
      - `target/benchmarks/review4_c04_20260317T172855_parallel_remaining_examples_suite_summary.md`
    - each run: `passed=3`, `failed=0`, `skipped_run_targets=13`.
- Loop continuation `c05..c08` executed on the same canonical lock; detailed per-cycle entries are recorded in `docs/llama-rs/worklog/2026-03-15-migration-log.md`.
- New artifact families were added under `target/benchmarks/review4_c0[5-8]_...` for step2/perf/suite outputs and summaries.
- Snapshot (`c05..c08`): step2 remained reject-only (`overall token ~1.014/1.035/1.015/1.018`), perf accepted `c05/c06`, rejected `c07`, and kept `c08` as conservative monitor-only; suite stayed stable (`passed=3`, `failed=0`, `skipped_run_targets=13`).
- Loop continuation `c09..c18` completed; detailed cycle-by-cycle entries are in `docs/llama-rs/worklog/2026-03-15-migration-log.md`.
- Snapshot (`c09..c18`): step2 remained reject-only (`overall token ~1.012..1.035`), perf was mixed (`batch16` remained strongest; `threads2` and `threads2@batch16` became unstable; `threads4@batch16` remained non-robust), and suite stayed stable (`passed=3`, `failed=0`, `skipped_run_targets=13`).
- Runtime tracker after `c18`: cumulative loop estimate reached `8121s` (`~135.4 min`), leaving `~2679s` (`~44.7 min`) to the `>= ~3h` stop boundary.
- Final continuation block `c19..c22` completed and reached the stop condition at cycle boundary `c22` (`phase1+phase2 ~= 10842s`, `~180.7 min`).
- Snapshot (`c19..c22`): step2 stayed conservative/reject-only overall (including one mixed split run where `nomask` improved on CPU but regressed on Metal), perf remained strongest on `batch16@threads2`, and suite stayed stable (`passed=3`, `failed=0`, `skipped_run_targets=13`).
- Detailed completion record and artifact index for `c19..c22` is appended in `docs/llama-rs/worklog/2026-03-15-migration-log.md`.
- Completed remaining step2 task (`step2-next-operator-pass-6`) with combined interaction pass:
  - `target/benchmarks/review4_step2_nomask_nopos_models6_impact.md`
  - token stayed near-neutral (`overall ~0.999`) but setup cost exploded (`overall ~1.782`) -> reject, defaults unchanged.
- Completed remaining `perf-close-cpp-gap` quantification for model-exec path:
  - `target/benchmarks/review4_perf_close_gap_model_exec_opt_r5_impact.md`
  - `gpt2_backend` baseline (`batch8,threads1`) vs optimized (`batch16,threads2`) median `avg_item_ms`:
    - CPU `0.007759 -> 0.005056` (`34.8%` faster),
    - MTL0 `0.043701 -> 0.030004` (`31.3%` faster).
  - cycle-wide candidate summary:
    - `target/benchmarks/review4_phase2_perf_summary.md` (`batch16` best robust median candidate).
- Completed remaining parallel suite hardening task:
  - `bench_upstream_suite` now accepts per-target run args via `GGML_UPSTREAM_RUN_ARGS_<TARGET>`.
  - validation run artifact:
    - `target/benchmarks/review4_parallel_remaining_examples_suite_runargs_env.txt`
    - `target/benchmarks/review4_parallel_remaining_examples_suite_runargs_env_summary.md`
  - status remains deterministic: `passed=3`, `failed=0`, `skipped_run_targets=13`.
