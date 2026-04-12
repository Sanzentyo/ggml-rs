# llama.cpp example parity matrix

Status legend:

- `Not started`
- `In progress`
- `Done`

Baseline readiness check:

- `llama-rs/examples/backend_smoke.rs` is passing on CPU and Metal, confirming the current scaffold can execute via `ggml-rs` safe API on both backends.
- For every migration task, preflight starts with `docs/ggml-rs/ERROR_CONTEXT_POLICY.md`.
- `llama-rs/examples/bench_matmul.rs` now provides a reproducible CPU/Metal performance gate using only `ggml-rs` safe API.
- Real-model `llama.cpp` baseline runs are captured under `target/benchmarks/llama_cpp_baseline_{all,extra}.jsonl` (`-ngl 0`/`99`, `-pg 256,0` and `-pg 0,128`).

| Example target | Status | CPU verified | Metal verified | Notes |
| --- | --- | --- | --- | --- |
| batched | In progress | Yes | Yes | `BatchedWorkload` + `batched` example で再利用可能な batched 実行基盤を追加。full token-level batching は未対応 |
| llama-bench (proxy) | In progress | Yes | Yes | `bench_matmul` + `bench_mlp_layer`（`--cases`）+ `bench_attention_layer`（`HxQxKxS` + decode mode `--decode-kv/--past` + stepwise `--decode-steps`）を実装。decode mode は projected KV cache reuse（`cache_reuse=true`）対応、stepwise mode は token-by-token growth（`stepwise=true`）対応。stepwise runner は persistent backend/context/graph 再利用で wrapper overhead を低減。real GGUF 6モデルでの `llama.cpp` baseline 取得と、metadata由来ケースでの proxy MLP/attention 実測比較（CPU/Metal）まで実施済み。`bench_compare_report` で比較レポートを自動生成可能。`outproj_fused_layerx5`/balanced の profile preset、`--decode-stepwise-{no-}static-kv-head-precompute`、`--decode-stepwise-{no-}balanced-head-concat`、`--decode-stepwise-{no-}position-delta` の A/B を追加済み。 |
| debug | Not started | No | No |  |
| embedding | In progress | N/A | N/A | `embedding_probe` で GGUF f32 tensor を embedding統計として読み出す基盤を追加。full inference は未対応 |
| eval-callback | Not started | No | No |  |
| gguf-hash | Done | N/A | N/A | `llama-rs/examples/gguf_hash` で `--all/--check/--uuid/--no-layer` を実装・検証済み |
| gguf | In progress | N/A | N/A | `gguf_inspect` + `model_catalog` を typed KV (`GgufValue`) 対応へ拡張。metadata ADT (`resolve_model_metadata` / `resolve_llama_metadata`) を追加。safe API の `GgufWriter` に `set_values/remove_key/write_data_to_file` を追加し、`llama-rs/examples/gguf` は upstream互換寄りに `w` / `r0` / `r1` / `r` モードを提供（`r1` は `--check|--no-check` 対応）して read/write parity を検証可能にした |
| idle | In progress | Yes | Yes | `llama-rs/examples/idle` を追加。real GGUF の layer-attention decode-proxy を pause schedule 付きで実行し、CPU/Metal の `avg_decode_ms` と分散を比較できる（full llama.cpp idle と同一実装ではなく proxy ベース）。Qwen系では requested layer + 層スキャンを試行し、real weights 解決不能時は metadata 由来 deterministic weights (`weights_mode=MetadataDeterministic`) にフォールバックする運用を追加。 |
| lookahead | Not started | No | No |  |
| lookup | Not started | No | No |  |
| parallel | Not started | No | No |  |
| passkey | Not started | No | No |  |
| retrieval | Not started | No | No |  |
| save-load-state | Not started | No | No |  |
| simple | In progress | Yes | Yes | `simple` + `min_infer_linear` + `min_infer_mlp` + `min_infer_mlp_layer` + `min_infer_attention_layer` まで拡張。name resolver + metadata resolver 連携で layer index 実行を自動化（`FullMetadata` / `TensorHeuristic` を可視化）。attention は ADT 化（layout/mask/rope）し multi-head + causal(CPU) 経路を追加。**Qwen3.5 E2E parity 達成**: single-token `[5328]`, multi-token 5生成 `[1088,35790,90,16,14728]`, 3-prompt+5-gen `[31,2,5,1,271]`, 5-prompt+5-gen `[6,24218,10,4838,1665]` 全 match。精度限界: prompt `[5]` の token 5 のみ diverge (23 vs 24, adjacent logits)。NeoX RoPE, causal depthwise conv, delta-net recurrence, Q/gate split 全て実装・検証済み。autoregressive decode は未実装（prefill only）|
| simple-chat | Not started | No | No |  |
| speculative | Not started | No | No |  |
| speculative-simple | Not started | No | No |  |
| gen-docs | Not started | No | No |  |
| training | Not started | No | No |  |
| diffusion | Not started | No | No |  |
| convert-llama2c-to-ggml | Not started | No | No | Built only when `GGML_BACKEND_DL=OFF` |
| sycl | Not started | No | No | Conditional target |
