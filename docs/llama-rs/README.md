# llama-rs migration docs

## Goal

Build a new `llama-rs` crate that reproduces `llama.cpp` example behavior using the safe API from this repository's `ggml-rs` crate.

Current user direction is to target all CMake example targets, with CPU and Metal execution verification.

## Document map

- `PARITY_MATRIX.md`: target-by-target migration and verification status.
- `KNOWLEDGE_BASE.md`: implementation notes, environment findings, pitfalls.
- `WORKLOG.md`: chronological execution log for long-running migration work.

## Operating model

1. Keep all high-signal findings in docs before and during implementation.
2. Implement reusable `llama-rs` core primitives first.
3. Port examples in batches while continuously updating the parity matrix.
4. Verify each migrated example on CPU and Metal and record results.
