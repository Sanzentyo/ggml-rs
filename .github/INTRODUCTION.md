# Ongoing execution policy (review_1 priority)

This repository is currently following a strict execution policy to avoid losing intent across long refactor loops.

## Immediate priority

1. Apply `docs/third_reviews/review_1.md` refactor recommendations to `ggml-rs` in a dedicated `git worktree` branch.
2. Enforce no-regression performance gating (baseline vs post-change).
3. Iterate until performance is at least baseline (preferably improved).
4. Merge back to `main` only after validation and runtime checks pass.

## Safety rule

- If `unsafe` is required, include explicit safety comments proving why the usage is valid.

## Validation rule

- Always run:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
- Re-run CPU/Metal runtime smoke (`--features link-system`) for performance-sensitive paths.

## Project objective lock

- Keep `llama-rs` reproducing llama.cpp behavior on top of `ggml-rs` safe APIs.
- Improve `ggml-rs` architecture and performance in parallel.

## After review_1 completion

- Return to `llama-rs` trait/ADT continuation tasks.
