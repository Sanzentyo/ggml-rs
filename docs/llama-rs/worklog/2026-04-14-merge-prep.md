# Merge Preparation: `exp/oh-my` → `master`

## Branch Summary

71 commits implementing a complete Qwen3.5 E2E inference pipeline with
verified token-ID parity against llama.cpp, plus comprehensive ggml-rs
API improvements (typed tensors, safe view/reshape wrappers, graph-level
projections, resumable generation sessions, chat infrastructure) and a
thorough structural DRY refactoring pass (items 34-60).

PR #2 created and reviewed by Copilot — all 6 review comments addressed.

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

11. **Detokenization + chat infrastructure**: `tokenizer.rs` byte-BPE decoding,
    `StreamingDecoder` for UTF-8 safe streaming, `encode_with_special_tokens()`,
    `chat.rs` with `ChatMessage`/`Role`/`ChatFormat` types, ChatML formatting,
    content sanitization. `simple_chat` interactive example.

### ggml-rs API Extensions

12. **Safe view/reshape wrappers**: `view_3d`, `view_4d`, `reshape_1d`, `reshape_4d`
    with Rust-side validation (contiguity, bounds, overflow). Backfilled validation
    on existing `view_1d`/`view_2d`/`reshape_2d`/`reshape_3d`. `NotContiguous` and
    `ViewOutOfBounds` error variants. 15 integration tests.

13. **Graph-level attention projections**: Full attention batches 3 matmuls (Q, K, V)
    in single graph; linear attention batches 4 (QKV, gate, alpha, beta). Output
    projections also graph-based. Decode stays host-side. Shared `project_sequence_graph`
    in `tensor_ops.rs`. Parity test confirms host vs graph within 1e-5.

### Documentation

14. Conv & QKV packing comparison document (llama-rs vs llama.cpp).
15. Module-level doc comments on all e2e submodules.
16. Updated PARITY_MATRIX, EXAMPLE_PARITY_MATRIX, INTRODUCTION.md.
17. Top-level README with workspace overview and API guide.
18. Worklog entries for all major milestones (graph projections, view wrappers,
    save-load-state, simple-chat, trait extraction, merge prep).

## Test Coverage

- **222 tests** pass with `--features link-system` (7 ignored: benchmark + upstream suite)
- **0 new clippy warnings** (pre-existing: 1 `too_many_arguments` in ggml-rs `flash_attn_ext`, 2 in test files)
- **0 fmt issues**
- Key test categories:
  - 18 type system tests
  - 15 backend compute tests (CPU + Metal parity)
  - 18 error path tests
  - 20 typed tensor tests
  - 15 llama-rs integration tests (attention, linear attention, state, generation)
  - 2 attention parity tests (CPU vs Metal)
  - 2 MLP parity tests (CPU vs Metal, CPU vs C++ reference)
  - 2 regression tests (TwoPhase+Standard→error, TwoPhase+zero tokens→empty)
  - 7 checkpoint roundtrip/validation tests
  - 4 session step/resume/parity tests
  - 20+ tokenizer/chat unit tests
  - 15 view/reshape wrapper tests (ggml-rs)
  - 1 graph projection parity test (host vs graph)
  - 7 bench graph tests (attention, linear attention, MLP, LM head, conv/QKV)

## Copilot PR Review — All 6 Issues Fixed

PR #2 received automated code review (27/71 files reviewed, 6 comments).
All issues addressed in commit `97faee0`:

1. **Overflow check** (numeric.rs): `build_causal_mask_f16_bytes` returns `Result`
   with `checked_mul` to prevent `seq_len * seq_len * 2` overflow.
2. **Tuple drop order** (mlp.rs, attention.rs, generation.rs): All 4 persistent
   builder functions swap return to `(Handle, Context)` so handle drops first.
3. **Fallback test** (session.rs): Added `#[cfg(test)] persistent_resources_disabled`
   latch; test now truly forces the fallback code path.
4. **Persistent vs fallback** (session.rs): `session_b` uses disabled latch to
   force fallback; verifies both paths independently produce expected token count.
5. **flash_attn_ext mask validation** (compute.rs): Validates mask type is f16
   before FFI call. Added `DynTensor::ggml_type()` accessor.
6. **write_bytes_backend error** (compute.rs): Uses `UnexpectedTensorByteSize`
   (byte-level) instead of `LengthMismatch` (element-level).

Additional: gated test-only `causal_depthwise_conv_graph` with `#[cfg(test)]`
to eliminate dead_code warning in lib target.

## Known Limitations

- Standard attention decode uses full-reprocess fallback (not incremental)
- Decode (seq_len=1) stays host-side for projections (graph overhead not worthwhile)
- Scalar CPU implementation only (no multi-threading)
- Prompt `[5]` diverges at token 5 (adjacent logits, precision edge)
- `too_many_arguments` clippy warnings in linear/full attention (accepted: complex state threading)

## Validation Checklist

- [x] `cargo fmt --all`
- [x] `cargo clippy --workspace --all-targets`
- [x] `cargo test --workspace`
- [x] `cargo test --workspace --features link-system` (222 pass, 7 ignored)
- [x] Zero TODOs/FIXMEs in codebase
- [x] All docs updated and consistent
- [x] Module doc comments on all e2e submodules
- [x] INTRODUCTION.md items 1-8 all marked DONE, items 34-60 documented
- [x] CPU runtime smoke test: avg=0.252 ms (no regression from 0.256 ms baseline)
- [x] Metal runtime smoke test: avg=0.595 ms, checksum matches CPU
- [x] Self code review: no issues found (5 areas verified)
