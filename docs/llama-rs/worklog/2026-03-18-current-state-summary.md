# 2026-03-18 Current State Summary (review4)

## 1) Baseline refresh status

- Rebuilt external `llama-bench` and refreshed:
  - `target/benchmarks/llama_cpp_baseline_all.jsonl`
  - `target/benchmarks/llama_cpp_baseline_extra.jsonl`
  - `target/benchmarks/review4_llama_cpp_baseline_refresh.log`
- Coverage check passed:
  - all 6 models present,
  - both backends (`n_gpu_layers=0` CPU / `99` Metal) present,
  - decode rows (`n_prompt=0,n_gen=128`) available for calibration.

## 2) Balanced preset retune result

- Re-ranked 7 candidates against refreshed baselines:
  - summary: `target/benchmarks/review4_finetune_balanced_profile_sweep_summary.md`
  - ranking JSON: `target/benchmarks/review4_finetune_balanced_profile_sweep_summary.json`
- New selected preset: `cpu5_mtl7`
  - calibration: `target/benchmarks/review4_finetune_balanced_profile_cpu5_mtl7_calibration.md`
  - impact vs previous (`cpu6_mtl7`): `target/benchmarks/review4_finetune_balanced_profile_cpu5_mtl7_impact.md`
  - selected averages: CPU `~0.981`, MTL0 `~1.010`, overall `~0.995`, gap `~0.029`.
- Implementation updated in `llama-rs/examples/bench_attention_layer.rs`:
  - balanced preset repeats: CPU `5`, Metal `7`
  - profile label: `outproj_fused_balanced_cpu5_mtl7`
- CPU/Metal runtime smoke revalidated:
  - `target/benchmarks/review4_finetune_balanced_profile_preset_smoke.txt`

## 3) Step1 continuation (layer/hotspot)

- Layer sweep (ELYZA, `block_layer=0..7`) on refreshed balanced preset:
  - raw: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers0_7.txt`
  - summary: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers0_7_summary.{md,csv}`
  - aggregate avg_token: CPU `~28.427 ms`, MTL0 `~28.061 ms`, overall `~28.244 ms`.
- Hotspot-window A/B (`layers 5..7`) for `head_stage_buf`:
  - base: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers5_7_headstage_base.txt`
  - on: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers5_7_headstage_on.txt`
  - impact: `target/benchmarks/review4_step1_balanced_cpu5_mtl7_elyza_layers5_7_headstage_impact.md`
  - result: token `on/base ~0.996`, setup `~1.005`, checksum delta `0.0`.

## 4) Step2 continuation (cross-model operator A/B)

- Cross-model (`6` models, CPU/Metal) `head_stage_buf` A/B:
  - base: `target/benchmarks/review4_step2_balanced_cpu5_mtl7_headstage_base.txt`
  - on: `target/benchmarks/review4_step2_balanced_cpu5_mtl7_headstage_on.txt`
  - impact: `target/benchmarks/review4_step2_balanced_cpu5_mtl7_headstage_impact.md`
- Aggregate:
  - token `on/base`: CPU `~0.995`, MTL0 `~0.994`, overall `~0.994`
  - proxy/cpp overall: `~0.993 -> ~0.987`
  - checksum delta max: `0.0`
- Policy on refreshed set: keep `head_stage_buf=false` for parity-focused default.

## 5) Step3 continuation (upstream suite)

- Refresh run:
  - log: `target/benchmarks/review4_step3_upstream_suite_refresh.txt`
  - summary: `target/benchmarks/review4_step3_upstream_suite_refresh_summary.md`
- Status: `passed=3`, `failed=0`, `skipped_run_targets=13` (model/data-arg dependent targets listed with run-arg guidance).

## 6) Validation and quality checks

- Completed after code/doc updates:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
- Runtime evidence includes CPU/Metal execution and benchmark artifacts above.

## 7) Current policy snapshot

- Balanced preset target: `cpu5_mtl7`
- Keep defaults:
  - `head_stage_buf=false`
  - `head_concat_balanced=false`
  - `position_delta=true`
- Active parity objective: keep proxy/cpp as close to `1.0` as possible while preserving checksum parity.

## 8) Next task started

- Next operator pass is now started:
  - `step2-blockgateup-crossmodel-refresh`
  - target: cross-model A/B for `--decode-stepwise-fuse-block-gate-up` on refreshed balanced preset.
