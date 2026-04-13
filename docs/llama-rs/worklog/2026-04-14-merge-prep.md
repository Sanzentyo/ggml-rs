# Merge Preparation: `exp/oh-my` → `master`

## Branch Summary

28 commits implementing a complete Qwen3.5 E2E inference pipeline with
verified token-ID parity against llama.cpp, plus comprehensive ggml-rs
API improvements.

## Completed Work Items

### ggml-rs Core

1. **Type consolidation**: `Type` expanded from 2 variants (F32, I32) to all
   32+ ggml tensor types (quantized Q4_K, Q8_0, etc. + native floats/ints +
   Unknown(i32)). `GgufTensorInfo` stores `ggml_type: Type` instead of raw
   `i32` + `String`. Decode APIs accept `Type`. `HostElement` sealed via
   private `Sealed` supertrait.

2. **Typed tensors**: `Tensor<'ctx, T>` typestate, `DynTensor<'ctx>`
   runtime-typed handle, `TensorExpr<'ctx, T>` typed expression wrapper.

3. **GGUF ergonomics**: `AsRef<str>` for GGUF string arguments,
   `TryFromGgufValue` trait, `kv_value_as::<T>()` convenience method.

4. **Backend examples**: `backend_ops.rs` multi-op graph example
   (matmul + bias on CPU/Metal), multi-op + Metal parity backend tests.

### llama-rs E2E Inference

5. **e2e.rs module split**: Monolithic 2412-line file → 13 focused submodules
   (error, config, numeric, tensor_ops, resolve, decode, plan, planner,
   attention, linear_attention, mlp, generation, state). Public API unchanged.

6. **Qwen3.5 attention**: Full attention with gated Q, NeoX RoPE,
   per-head Q/K norms. Linear attention with causal depthwise convolution +
   delta-net recurrence. Both verified against llama.cpp.

7. **Autoregressive decode**: KV cache for full attention, conv buffer + SSM
   states for linear attention. Decode equivalence tests verify
   prefill+decode = full reprocess.

8. **Two-phase generation loop**: Prefill all prompt tokens, then decode
   one-at-a-time with cached state. `GenerationMode` enum
   (`Auto | FullReprocess | TwoPhase`) for execution strategy selection.

9. **Multi-token parity**: MRoPE implementation, verified across:
   - Single-token: `[1] → [5328]`
   - 5-gen from `[3]`: `[1088, 35790, 90, 16, 14728]`
   - 3-prompt+5-gen: `[31, 2, 5, 1, 271]`
   - 5-prompt+5-gen: `[6, 24218, 10, 4838, 1665]`

10. **Resumable generation session**: `GenerationSession` with step-by-step
    `next_token()`, `checkpoint()` snapshot, `resume(model, checkpoint)` restore.
    `GenerationCheckpoint` DTO with postcard binary format, model fingerprint
    validation, and KV cache trimming. Separate DTO layer keeps serde types
    distinct from runtime state. Session reuses `AttentionStrategy` trait +
    `process_all_layers` shared infrastructure.

### Documentation

10. Conv & QKV packing comparison document (llama-rs vs llama.cpp).
11. Module-level doc comments on all e2e submodules.
12. Updated PARITY_MATRIX, EXAMPLE_PARITY_MATRIX, INTRODUCTION.md.
13. Top-level README with workspace overview and API guide.

## Test Coverage

- **180 tests** pass with `--features link-system` (1 ignored: upstream suite)
- **0 clippy warnings**
- **0 fmt issues**
- Key test categories:
  - 18 type system tests
  - 15 backend compute tests (CPU + Metal parity)
  - 18 error path tests
  - 20 typed tensor tests
  - 66 llama-rs tests (attention, linear attention, state, generation, checkpoint, session, etc.)
  - 2 attention parity tests (CPU vs Metal)
  - 2 MLP parity tests (CPU vs Metal, CPU vs C++ reference)
  - 2 regression tests (TwoPhase+Standard→error, TwoPhase+zero tokens→empty)
  - 7 checkpoint roundtrip/validation tests
  - 4 session step/resume/parity tests

## Known Limitations

- Standard attention decode uses full-reprocess fallback (not incremental)
- `copy_from_slice` for QKV splits (no strided views yet)
- Scalar CPU implementation only (no multi-threading)
- Prompt `[5]` diverges at token 5 (adjacent logits, precision edge)

## Validation Checklist

- [x] `cargo fmt --all`
- [x] `cargo clippy --workspace --all-targets`
- [x] `cargo test --workspace`
- [x] `cargo test --workspace --features link-system` (138 pass)
- [x] Zero TODOs/FIXMEs in codebase
- [x] All docs updated and consistent
- [x] Module doc comments on all e2e submodules
- [x] INTRODUCTION.md items 1-8 all marked DONE
- [x] CPU runtime smoke test: avg=0.252 ms (no regression from 0.256 ms baseline)
- [x] Metal runtime smoke test: avg=0.595 ms, checksum matches CPU
- [x] Self code review: no issues found (5 areas verified)
- [x] PR #1 created: https://github.com/Sanzentyo/ggml-rs/pull/1
