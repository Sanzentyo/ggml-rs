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
