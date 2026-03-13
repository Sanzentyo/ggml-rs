# llama-rs worklog

## Current session milestones

- Verified `ggml-rs` safe API matmul on CPU and Metal.
- Added local `llama.cpp` checkout for migration reference.
- Added structured migration docs:
  - `docs/llama-rs/README.md`
  - `docs/llama-rs/PARITY_MATRIX.md`
  - `docs/llama-rs/KNOWLEDGE_BASE.md`
- Created initial `llama-rs` crate skeleton and backend smoke example powered by `ggml-rs` safe API.
- Validated `llama-rs/examples/backend_smoke.rs` on both CPU and Metal (`[CPU] ... OK`, `[MTL0] ... OK`).
- Added safe GGUF reader API to `ggml-rs` and `llama-rs/examples/gguf_inspect.rs`.
- Verified `gguf_inspect` against a generated sample GGUF file from upstream `llama-gguf`.
- Added `llama-rs/examples/gguf_hash.rs` with layer/global hashing and manifest verification.
- Verified `gguf_hash` generation and `--check` round-trip success on sample GGUF manifest.
- Expanded `ggml-rs` safe op surface for llama runtime needs: `add`, `mul`, `silu`, `rms_norm`, `scale`, `get_rows`, `repeat`, `cpy`, `cont`, `reshape_*`, `view_*`, `permute`, `diag_mask_inf`, `soft_max(_ext)`, `rope_ext`.
- Added tensor naming (`set_name` / `name`) and backend `i32` tensor transfer helpers.
- Re-ran CPU + Metal execution checks after the op-surface expansion:
  - `ggml-rs/examples/backend_matmul.rs`
  - `llama-rs/examples/backend_smoke.rs`

## Next concrete steps

1. Expand `llama-rs` core API beyond smoke-level matrix operations.
2. Start implementing feature parity per example target from the parity matrix.
3. Mark each target as in-progress/done with CPU+Metal verification evidence.
