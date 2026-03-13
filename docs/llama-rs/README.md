# llama-rs migration docs

## Goal

Build a new `llama-rs` crate that reproduces `llama.cpp` example behavior using the safe API from this repository's `ggml-rs` crate.

Current user direction is to target all CMake example targets, with CPU and Metal execution verification.

## Mandatory preflight before implementation

Before modifying any `llama-rs` path that touches ggml-backed execution, read:

- `docs/ggml-rs/ERROR_CONTEXT_POLICY.md`

This preflight step is mandatory and must be done before coding.

## Document map

- `PARITY_MATRIX.md`: target-by-target migration and verification status.
- `KNOWLEDGE_BASE.md`: implementation notes, environment findings, pitfalls.
- `WORKLOG.md`: chronological execution log for long-running migration work.

## Operating model

1. Keep all high-signal findings in docs before and during implementation.
2. Implement reusable `llama-rs` core primitives first.
3. Port examples in batches while continuously updating the parity matrix.
4. Verify each migrated example on CPU and Metal and record results.
5. Start each implementation batch by explicitly referencing the error-context policy document above.
6. Keep a benchmark parity gate (`ggml-rs` vs `llama-rs`) for core workloads so regression is visible while feature parity expands.

## Current runtime foundations

- `model`: GGUF load + tensor lookup + payload validation, with handle-based lookup (`TensorHandle`) for repeated access and typed KV access (`kv_value`).
- `embedding`: f32 tensor summary helpers for embedding-oriented workflows.
- `batched`: backend graph reuse and batched matmul execution scaffolding.
- `metadata`: architecture-aware GGUF metadata ADTs (`ModelArchitecture`, `TransformerMetadata`, `ModelMetadata`, `LlamaModelMetadata`) for config derivation across architecture-prefixed GGUF keys.
- `inference`: minimal linear inference (`Y = W * X`), MLP-block inference, layer-index GGUF-backed MLP execution, layer-dimension auto-resolution (`resolve_llama_layer_dimensions`, with `FullMetadata` / `TensorHeuristic` resolution mode), and ADT-based attention path (multi-head + optional causal mask + optional RoPE).
- `naming`: GGUF tensor-name resolver (`blk.*` / `layers.*` / `model.layers.*`) for real-model wiring.
- Type-safety additions: feature-count newtypes and type-state builder for linear inference config, plus non-zero schedule newtypes for batched execution.
- Public error boundary: crate-level `LlamaError` / `LlamaResult<T>` for cross-module integration.
- Coverage additions: name-resolver unit tests, metadata parser unit tests, ggml-based C++ parity + CPU/Metal parity tests (`mlp_cpp_parity`), attention CPU/Metal parity + causal CPU test (`attention_parity`), and multi-case MLP-layer benchmark (`bench_mlp_layer`).
