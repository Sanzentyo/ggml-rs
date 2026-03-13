# llama.cpp example parity matrix

Status legend:

- `Not started`
- `In progress`
- `Done`

Baseline readiness check:

- `llama-rs/examples/backend_smoke.rs` is passing on CPU and Metal, confirming the current scaffold can execute via `ggml-rs` safe API on both backends.

| Example target | Status | CPU verified | Metal verified | Notes |
| --- | --- | --- | --- | --- |
| batched | Not started | No | No |  |
| debug | Not started | No | No |  |
| embedding | Not started | No | No |  |
| eval-callback | Not started | No | No |  |
| gguf-hash | Done | N/A | N/A | `llama-rs/examples/gguf_hash` で `--all/--check/--uuid/--no-layer` を実装・検証済み |
| gguf | In progress | N/A | N/A | `llama-rs/examples/gguf_inspect` で read 系を再現。write 系は未対応 |
| idle | Not started | No | No |  |
| lookahead | Not started | No | No |  |
| lookup | Not started | No | No |  |
| parallel | Not started | No | No |  |
| passkey | Not started | No | No |  |
| retrieval | Not started | No | No |  |
| save-load-state | Not started | No | No |  |
| simple | Not started | No | No |  |
| simple-chat | Not started | No | No |  |
| speculative | Not started | No | No |  |
| speculative-simple | Not started | No | No |  |
| gen-docs | Not started | No | No |  |
| training | Not started | No | No |  |
| diffusion | Not started | No | No |  |
| convert-llama2c-to-ggml | Not started | No | No | Built only when `GGML_BACKEND_DL=OFF` |
| sycl | Not started | No | No | Conditional target |
