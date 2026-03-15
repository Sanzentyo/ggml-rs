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
