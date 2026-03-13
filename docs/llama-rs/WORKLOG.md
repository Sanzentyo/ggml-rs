# llama-rs worklog

This file is the index for detailed, dated worklog entries.

Detailed logs are split under `docs/llama-rs/worklog/` to keep this top-level file compact and searchable.

## Index

| Date | File | Scope |
| --- | --- | --- |
| 2026-03-13 | `docs/llama-rs/worklog/2026-03-13-migration-log.md` | Backend bring-up, GGUF/model foundations, metadata-first resolver hardening, attention ADT migration, CPU/Metal runtime verification |

## Latest summary

- Metadata-driven auto-resolution path is validated with explicit `resolution_mode` output (`FullMetadata` / `TensorHeuristic`).
- MLP and attention layer examples are verified on CPU and Metal for synthetic fixtures.
- Link-system parity tests (`mlp_cpp_parity`, `attention_parity`) pass after RoPE integration fixes.
