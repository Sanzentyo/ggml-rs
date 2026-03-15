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
