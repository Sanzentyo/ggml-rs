# llama.cpp example parity matrix

Status legend:

- `Not started`
- `In progress`
- `Done`

Baseline readiness check:

- `llama-rs/examples/backend_smoke.rs` is passing on CPU and Metal, confirming the current scaffold can execute via `ggml-rs` safe API on both backends.
- For every migration task, preflight starts with `docs/ggml-rs/ERROR_CONTEXT_POLICY.md`.
- `llama-rs/examples/bench_matmul.rs` now provides a reproducible CPU/Metal performance gate using only `ggml-rs` safe API.

| Example target | Status | CPU verified | Metal verified | Notes |
| --- | --- | --- | --- | --- |
| batched | In progress | Yes | Yes | `BatchedWorkload` + `batched` example で再利用可能な batched 実行基盤を追加。full token-level batching は未対応 |
| llama-bench (proxy) | In progress | Yes | Yes | `bench_matmul` に加えて `bench_mlp_layer`（`--cases` 対応）を追加。`llama-bench` 全機能ではなく、layer-MLP 経路の性能ゲートを先行実装 |
| debug | Not started | No | No |  |
| embedding | In progress | N/A | N/A | `embedding_probe` で GGUF f32 tensor を embedding統計として読み出す基盤を追加。full inference は未対応 |
| eval-callback | Not started | No | No |  |
| gguf-hash | Done | N/A | N/A | `llama-rs/examples/gguf_hash` で `--all/--check/--uuid/--no-layer` を実装・検証済み |
| gguf | In progress | N/A | N/A | `gguf_inspect` + `model_catalog` を typed KV (`GgufValue`) 対応へ拡張。metadata ADT (`resolve_model_metadata` / `resolve_llama_metadata`) を追加。write 系は未対応 |
| idle | Not started | No | No |  |
| lookahead | Not started | No | No |  |
| lookup | Not started | No | No |  |
| parallel | Not started | No | No |  |
| passkey | Not started | No | No |  |
| retrieval | Not started | No | No |  |
| save-load-state | Not started | No | No |  |
| simple | In progress | Yes | Yes | `simple` + `min_infer_linear` + `min_infer_mlp` + `min_infer_mlp_layer` + `min_infer_attention_layer` まで拡張。name resolver + metadata resolver 連携で layer index 実行を自動化（`FullMetadata` / `TensorHeuristic` を可視化）。attention は ADT 化（layout/mask/rope）し multi-head + causal(CPU) 経路を追加。full model/token inference parity は未対応 |
| simple-chat | Not started | No | No |  |
| speculative | Not started | No | No |  |
| speculative-simple | Not started | No | No |  |
| gen-docs | Not started | No | No |  |
| training | Not started | No | No |  |
| diffusion | Not started | No | No |  |
| convert-llama2c-to-ggml | Not started | No | No | Built only when `GGML_BACKEND_DL=OFF` |
| sycl | Not started | No | No | Conditional target |
