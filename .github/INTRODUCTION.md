# Ongoing execution policy

This repository is currently following a strict execution policy to avoid losing intent across long refactor loops.

**Docs-update policy**: After every significant change (API refactor, bug fix, feature addition), update this file and relevant docs under `./docs/` to reflect the new state. Read all markdown in `.github/` and `./docs/` at session start.

## Skills you should use
- rust-best-practices
And you should write rusty code(ADT, enum, type state pattern)

## Current branch

- `exp/oh-my` — dedicated branch for review_1 + review_3 refactor of `ggml-rs`.

## Immediate priority

1. ~~Close the remaining Qwen3.5 strict token-id parity gap in `llama-rs`.~~ **DONE** — parity achieved.
2. ~~Expand `Type` enum to all ggml types, seal `HostElement`, update decode APIs.~~ **DONE** — zero clippy warnings.
3. ~~Implement MRoPE for full attention layers (required for multi-token prompts).~~ **DONE** — multi-token parity achieved.
4. ~~Causal depthwise conv & QKV packing comparison.~~ **DONE** — documented in `docs/llama-rs/worklog/2026-04-13-conv-qkv-comparison.md`.
5. ~~Continue review_3 refactor items (generic inference, ND tensor, semantic wrapper dedup).~~ **12/12 DONE** — test coverage 80+ tests (zero warnings), backend examples + README updated.
6. ~~Autoregressive decode state management (prefill/decode split).~~ **DONE** — KV cache for full attention, conv buffer + SSM states for linear attention, decode equivalence tests pass.
7. ~~Two-phase generation loop (prefill + incremental decode).~~ **DONE** — generation.rs branches on layer types: all-Qwen3.5 → two-phase (prefill all prompt tokens, then decode one-at-a-time), otherwise → full-reprocess fallback.
8. ~~Backend example enhancement (review_3 item 11) + README (item 12).~~ **DONE** — `backend_ops.rs` example, fixed stale README snippets, multi-op + Metal parity tests added.
9. Merge back to `main` only after validation and runtime checks pass. **READY** — PR #1 created, all validation passed. See `docs/llama-rs/worklog/2026-04-14-merge-prep.md`.

## Completed refactor items

- `AsRef<str>` for GGUF string arguments (`find_key`, `kv_value_by_key`, `set_value`, `remove_key`).
- `TryFromGgufValue` trait and `kv_value_as::<T>()` convenience method on `GgufFile`.
- `GgufTypeMismatch` error variant for type-safe GGUF value extraction.
- `Tensor<'ctx, T>` typestate pattern and `DynTensor<'ctx>` runtime-typed handle.
- `TensorExpr<'ctx, T>` typed expression wrapper.
- `rope_ext_with_i32_positions` mixed-type RoPE helper for `f32` data + `i32` positions.
- Backend-path / ND tensor / error-path test expansion.
- `llama-rs` migration to the typed `Tensor<'ctx, T>` / `DynTensor<'ctx>` API.
- **Type consolidation**: `Type` expanded from 2 variants (F32, I32) to all 32+ ggml
  tensor types (quantized Q4_K, Q8_0, etc. + native floats/ints + Unknown(i32)).
  `GgufTensorInfo` now stores `ggml_type: Type` instead of raw `i32` + `String`.
  Decode APIs (`decode_tensor_data_to`, `tensor_element_count`) accept `Type`.
  `HostElement` sealed via private `Sealed` supertrait (eliminates `private_bounds` warning).
- **e2e.rs module split**: Monolithic 2412-line file split into 13 focused submodules
  (error, config, numeric, tensor_ops, resolve, decode, plan, planner, attention,
  linear_attention, mlp, generation, state). Public API unchanged.
- **Autoregressive decode infrastructure**: `state.rs` with `Qwen35FullAttentionState`
  (KV cache), `LinearAttentionState` (conv buffer + SSM states), `GenerationState`.
  `attention.rs` gains `qwen35_full_attention_prefill` + `decode_step`.
  `linear_attention.rs` gains `qwen35_linear_attention_prefill` + `decode_step` +
  `causal_depthwise_conv_decode_step`. RoPE `position_offset` parameter added.
  Decode equivalence tests verify prefill+decode = full reprocess.
- **Two-phase generation loop**: `generation.rs` branches on layer types: all-Qwen3.5
  → prefill + incremental decode (one token at a time with cached state); Standard
  attention present → full-reprocess fallback. Handles `max_new_tokens==0`, EOS on
  first generated token, MLP-only layers.
- **Backend examples + README** (review_3 items 11, 12): `backend_ops.rs` multi-op graph
  example (matmul + bias on CPU/Metal), fixed stale API names in README, added multi-op
  and Metal parity backend tests. Updated EXAMPLE_PARITY_MATRIX.md.
- **Generation loop refactor**: Extracted `GenerationMode` enum
  (`Auto | FullReprocess | TwoPhase`), `GenerationInputs` bundle, and
  `generate_from_plans` core loop. Integration test
  `two_phase_matches_full_reprocess_multi_layer` verifies both execution
  paths produce identical token sequences on a 3-layer synthetic model.
- **`AttentionStrategy` trait extraction**: Unified per-layer processing logic
  (norm → attention → residual → norm → MLP → residual) behind an
  `AttentionStrategy` trait with three implementations: `InferenceStrategy`
  (stateless), `PrefillStrategy` (captures state), `DecodeStrategy` (uses
  cached state). Added `process_all_layers` + `sample_next_token` shared
  helpers. Fixed crash paths: `TwoPhase + Standard` now returns
  `UnsupportedTwoPhase` error; `TwoPhase + max_new_tokens=0` returns empty.
  Two new regression tests added. 138 tests pass.
- **Iterator/chunks_exact refactoring**: Replaced procedural index loops with
  idiomatic Rust iterators across e2e modules. Extracted `SsmScratch` reusable
  buffer + `ssm_recurrence_step` helper (eliminates 60-line duplication in
  `linear_attention.rs`). Extracted `deinterleave_q_gate` helper with unified
  validation (was duplicated in prefill + decode paths). QKV split uses
  `chunks_exact` zip. Per-head norm functions use `chunks_exact_mut`.
  Conv inner loop uses `saturating_sub` for tap range. All 136 tests pass.
- **Shared projection/normalization helpers**: Extracted `project_and_prepare_qkv`
  + `PreparedAttention` in `attention.rs` and `project_linear_inputs` +
  `LinearProjections` + `split_and_norm_qk` in `linear_attention.rs`. Shared by
  core and decode_step paths. Validates dimension divisibility upfront. Decode
  path borrows `v_raw` directly from conv output (avoids extra copy).
- **Resumable generation session + state serialization** (`save-load-state`):
  `GenerationSession` (session.rs) provides step-by-step token generation via
  `new()` → `next_token()` loop, with `checkpoint()` to snapshot state and
  `resume(model, checkpoint)` to restore. `GenerationCheckpoint` (checkpoint.rs)
  uses postcard binary format with `LRCK` magic, model fingerprint validation
  (layer count, types, dims, vocab, rms_norm_eps), and KV cache trimming for
  compact serialization. Separate DTO layer (`CheckpointV1`, `LayerStateDto`,
  `ModelFingerprint`) keeps serde types distinct from runtime state. Session
  reuses `AttentionStrategy` trait + `process_all_layers` shared infrastructure.
  11 unit tests (7 checkpoint + 4 session) all passing.
- **Detokenization + chat infrastructure** (`simple-chat`):
  `tokenizer.rs` gains `decode()` / `decode_token()` (reverse GPT-2 byte-BPE),
  `encode_with_special_tokens()` (direct vocab lookup for ChatML markers),
  `special_token_id()`, and `StreamingDecoder` (buffered UTF-8 safe streaming).
  New `chat.rs` module provides `ChatMessage` / `Role` / `ChatFormat` types,
  `format_chat_prompt()` with ChatML support and content sanitization (sentinel
  rejection). `simple_chat` example: interactive multi-turn loop with streaming
  token output, `<|im_end|>` stop detection. 20+ unit tests.

- **Safe view/reshape wrappers** (`view_3d`, `view_4d`, `reshape_1d`, `reshape_4d`):
  Added missing safe API wrappers to `ggml-rs` that mirror the ggml C API for
  zero-copy tensor views. Backfilled Rust-side validation on all existing wrappers
  (`view_1d`, `view_2d`, `reshape_2d`, `reshape_3d`). Two new error variants:
  `NotContiguous` (blocks reshape of non-contiguous tensors), `ViewOutOfBounds`
  (blocks views that exceed source tensor bounds). Overflow-checked arithmetic
  prevents C-level aborts. 13 new integration tests covering aliasing, error
  paths, and OOB rejection. Enables future graph-level zero-copy QKV splits
  in llama-rs.

- **Graph-level attention projections** (full + linear attention):
  Replaced host-side scalar dot-product projections (`project_sequence`) with
  ggml `mul_mat` compute graphs for prefill/inference paths in both full and
  linear attention. Full attention batches 3 matmuls (Q, K, V) in a single graph;
  linear attention batches 4 (QKV, gate, alpha, beta). Output projections also
  use graph path. Decode (seq_len=1) stays on host-side to avoid graph overhead.
  Shared `project_sequence_graph` and `recommended_single_projection_memory`
  extracted to `tensor_ops.rs`. Backend threaded through all attention functions.
  Parity test confirms host vs graph output matches within 1e-5. 192 tests pass.

- **Example directory reorganization**: All examples reorganized from flat layouts
  into categorized subdirectories (basics/, backends/, benchmarks/, models/ for root
  crate; basics/, benchmarks/, gguf/, inference/, models/, applications/ for llama-rs).
  Added missing `save_load_state` and `simple_chat` Cargo.toml entries. All 192 tests
  pass. File history preserved via `git mv`.

- **Graph-level causal depthwise conv** (`ggml_ssm_conv`):
  Added `ssm_conv` safe wrapper to `ggml-rs` (f32-only). `causal_depthwise_conv_graph`
  in `linear_attention.rs` performs host-side transpose + left-padding then runs
  `ggml_ssm_conv` + `ggml_silu` on the backend (CPU/Metal/CUDA). Prefill path now
  uses graph-level conv; decode stays host-side (single token). 4 parity tests verify
  graph vs host-only numerical match. Original host function kept as `#[cfg(test)]`
  reference. 196 tests pass.

- **Fused projection + conv graph** (single-graph linear attention prefill):
  Merged the 4 linear projections (QKV, Z, alpha, beta) and causal depthwise
  convolution + SiLU into a single ggml compute graph: `project_and_conv_fused_graph`.
  Eliminates the host↔device round-trip between projection and conv stages.
  In-graph chain: `mul_mat → transpose → cont → concat(zeros, ...) → reshape_3d
  → ssm_conv → silu`. Pre-conv QKV still read back for `capture_conv_buffer`
  (decode state continuity). Backend buffer lifetime correctly scoped to span all
  reads. Standalone `causal_depthwise_conv_graph` demoted to `#[cfg(test)]`.
  196 tests pass.

- **Sigmoid + flash_attn_ext safe wrappers** (`ggml-rs`):
  Added `sigmoid` (elementwise σ(x)) and `flash_attn_ext` (fused multi-head
  scaled dot-product attention with optional f16 causal mask) to the safe API.
  `DynTensor::write_bytes_backend` enables raw byte writes for non-f32/i32 types
  (needed for f16 mask). 5 new integration tests: sigmoid CPU/Metal parity,
  flash_attn MHA/GQA reference match, output shape validation.

- **Fused attention scoring graph** (full attention prefill):
  Replaced the host-side O(T²·H·D) scoring loop in `qwen35_full_attention_core`
  with a single ggml compute graph: `permute → cont → flash_attn_ext → sigmoid(gate)
  → mul → reshape_2d → mul_mat(W_out)`. Flash output `[D, H, T, 1]` matches gate
  layout directly (no extra permute for gating). Causal mask built as f16 per ggml
  CPU kernel requirements. Decode path (seq_len=1) unchanged. `f32_to_f16_bits` and
  `build_causal_mask_f16_bytes` added to `numeric.rs`. 182 tests pass.

- **Fully fused single-graph full attention** (prefill):
  Merged the previous two-graph pipeline (QKV projection → host deinterleave/norm/RoPE
  → scoring) into a single ggml compute graph: `mul_mat(W_q/W_k/W_v, X) → strided
  view_3d Q/gate deinterleave → rms_norm + weight broadcast → rope_ext (NeoX mode=2)
  → permute → cont → flash_attn_ext → sigmoid(gate) → mul → reshape_2d → mul_mat(W_out)`.
  Eliminates 10 host↔device transfers and 2 graph round-trips → single transfer of
  weights/input + single compute + single readback. Post-RoPE K and raw V read back
  conditionally for KV cache capture via `build_forward_expand`. Decode path (seq_len=1)
  unchanged. 202 tests pass.

- **Layer pre-norm fusion** (attention + MLP):
  Moved `rms_norm + weight` from host-side (`process_all_layers`) into each ggml
  compute graph. Full attention (`fully_fused_attention_graph`), linear attention
  (`project_and_conv_fused_graph`), and MLP (`mlp_sequence_inference_with_weights`)
  all accept un-normed input + norm weight, applying in-graph `rms_norm(X, eps) * w`
  as the first operation. Eliminates 2× host↔device round-trips per layer (attention
  norm + MLP norm). Decode path keeps host-side norm (`DecodeStrategy` calls
  `rms_norm_with_weight` before dispatch) — single-token graph overhead not worthwhile.
  Standard attention path also keeps host-side norm. Parity tests updated: decode
  applies host-side norm to match in-graph norm, tolerance still within 1e-5
  (ggml f32 vs host f64 accumulation). 201 tests pass.

- **CPU vs Metal microbenchmarks** (`bench_graphs.rs`):
  Synthetic benchmarks at Qwen3.5 0.6B dimensions (hidden=1536, ffn=8960).
  Metal provides 2.4–3.1× speedup at seq_len=64 (MLP: 45.6→14.9ms, full attn:
  7.0→2.9ms). CPU is faster for short sequences (seq_len ≤ 4) due to Metal
  dispatch overhead — validates keeping decode path host-side. MLP is the
  throughput bottleneck (3× 1536×8960 matmuls). Linear attention Metal gain
  limited by host-side SSM recurrence. See `docs/llama-rs/worklog/2026-04-13-conv-qkv-comparison.md`
  item 12 for full table.

- **Graph-level LM head (output projection)**:
  Replaced host-side naive matmul (151936 dot products × 1536) and wasteful
  full-sequence normalization with a persistent ggml graph: `rms_norm → mul →
  reshape → mul_mat`. Weights (~935MB for Qwen3.5) uploaded once; per-step cost
  drops to ~614KB I/O (6KB hidden input + 608KB logits readback). Both
  `two_phase_loop` and `full_reprocess_loop` use the persistent graph.
  `build_lm_head_graph` + `lm_head_sample_step` in `tensor_ops.rs`;
  `graph_sample_at` convenience wrapper in `generation.rs`. No unsafe code
  (function-scoped ggml context avoids self-referential struct). Parity tests
  verify graph argmax matches host-side sampling. LM head benchmark added to
  `bench_graphs.rs`. See comparison doc item 13.

- **Decode-path QKV backend offload**:
  Both `qwen35_full_attention_decode_step` and `qwen35_linear_attention_decode_step`
  now accept `backend: &Backend` and pass `Some(backend)` to their projection helpers.
  QKV projections (3 matmuls for full, 4 for linear) and output projections are
  offloaded from host-side scalar dot products to ggml compute graphs. The
  `DecodeStrategy` in `generation.rs` forwards `backend` (was `_backend`) to both
  callsites. All 205 tests pass. See comparison doc item 14.

- **Persistent decode projections** (eliminate per-token weight upload):
  Build ggml projection graphs once per layer at decode-phase start, upload weights
  once, then reuse for every token — only ~6 KB hidden vector I/O per layer per token
  vs ~756 MB weight upload before. `PersistentDecodeProjection` enum (FullAttention /
  LinearAttention) in `tensor_ops.rs` holds persistent tensor handles, graphs, and
  backend buffers. Core decode logic extracted into `full_attention_decode_core` and
  `linear_attention_decode_core` (pure refactoring, all tests pass). `two_phase_loop`
  tries persistent path first, falls back to `DecodeStrategy` on failure (runtime
  robustness). See comparison doc item 15.
- **Decode attention scoring offload** (`flash_attn_ext`):
  Offloads the Q·K scoring + softmax + V aggregation + sigmoid gating loop to GPU
  via `flash_attn_ext` graph. `decode_scoring_gpu` in `attention.rs` builds a
  per-step temporary graph, uploads the live KV cache prefix with permutation
  `[D, Hkv, T] → [D, T, Hkv]`, runs fused attention, and reads back gated outputs.
  `full_attention_decode_core` now accepts `backend: Option<&Backend>` — GPU-first
  with silent fallback to host loop. GQA-aware parity test passes within 1e-4.
  See comparison doc item 16.
- **Decode conv/QKV/SSM analysis** (items 17–19):
  Analyzed remaining host-side decode bottlenecks. Causal depthwise conv decode
  (~5.6K FLOPs) stays on host — GPU dispatch overhead (~0.8 ms) vastly exceeds
  scalar loop cost (<1 µs). QKV routing is logically contiguous split with small
  Q/K copies for per-group normalization — already optimal. SSM recurrence
  (Delta-Net) is **incompatible** with `ggml_ssm_scan` — the delta rule feedback
  `sk = k^T·(decayed state)` creates state-dependent updates that linear selective
  scan cannot express. SIMD vectorization of inner loops identified as most
  promising near-term optimization. See comparison doc items 17–19.
- **RoPE decode + probe-once GPU failure** (items 20–21):
  RoPE decode (~6K FLOPs/layer at D=1536) stays on host — smallest per-layer
  operation, sub-microsecond scalar cost vs ~0.8 ms Metal dispatch overhead.
  Implemented probe-once GPU scoring optimization: `gpu_scoring_failed` flag in
  `Qwen35FullAttentionState` prevents repeated GPU scoring attempts after the
  first failure per sequence. Eliminates wasted dispatch overhead on CPU-only
  backends (~0.8 ms/layer/token savings). See comparison doc items 20–21.
- **Persistent KV cache design + SIMD analysis + cost model** (items 22–24):
  Designed persistent backend-resident KV cache to eliminate O(T) per-step
  upload (4 MB→4 KB at T=1000). Identified that on-device `permute+cont` O(T)
  remains even with persistent KV. Analyzed SSM recurrence SIMD vectorization:
  phases 1/3 are contiguous-access SIMD candidates; phase 4 had strided access
  pattern — **reordered** to row-major for auto-vectorization (all tests pass).
  End-to-end cost model shows KV transfer (~64 MB across 8 FA layers at T=1000)
  as dominant bottleneck, not compute. See comparison doc items 22–24.
- **Cross-context view API** (`view_Nd_of`) in `ggml-rs`:
  Added `view_1d_of` through `view_4d_of` to `Context`, enabling zero-copy
  views from tensors in a different (longer-lived) context. Safe `'src: 'ctx`
  lifetime bound replaces `'static` transmute. Unblocks persistent KV cache
  (item 22). 5 integration tests including compute graph scenario. 213 tests pass.
  See comparison doc item 25.
- **Persistent backend-resident KV cache** (`PersistentKvCache`):
  Pre-allocated K/V tensors on device with incremental O(1) per-step append
  via `write_data_backend_at`. Eliminates O(T) per-step KV upload (~64 MB/step
  at T=4000 across 8 FA layers). Uses `view_4d_of` cross-context views for
  ephemeral scoring graphs. Three-level fallback: persistent GPU → ephemeral
  GPU → host scoring. See comparison doc item 26.

## Validation checkpoints completed on this branch

- `cargo fmt --all`
- `cargo clippy --workspace --all-targets`
- `cargo test --workspace`
- `cargo test --features link-system`
- CPU perf gate: `cargo run --example bench_matmul --features link-system -- cpu -n 10`
  - current checkpoint result: `avg=0.256 ms`

## Current parity investigation status

- **Qwen3.5 strict parity: ACHIEVED** (single-token AND multi-token, up to 5-token prompts).
  - Single-token: prompt `[1]`, `max_new_tokens=1` → both produce `[5328]`.
  - Multi-token (single prompt): prompt `[3]`, `max_new_tokens=5` → both produce `[1088, 35790, 90, 16, 14728]`.
  - Multi-token (3 prompt tokens): prompt `[1,2,3]`, `max_new_tokens=5` → both produce `[31, 2, 5, 1, 271]`.
  - Multi-token (5 prompt tokens): prompt `[1,2,3,4,5]`, `max_new_tokens=5` → both produce `[6, 24218, 10, 4838, 1665]`.
  - Known precision edge: prompt `[5]` diverges at token 5 only (23 vs 24 — adjacent logits,
    numerical precision difference, not a systematic bug).
  - Bug 1 (linear attention): Head-group mapping used `head / repeat_factor` (interleaved),
    while llama.cpp's `ggml_repeat_4d` tiles block-by-block. Fixed to `head % group_count`.
  - Bug 2 (full attention): Q/gate split treated ggml's interleaved layout
    `[Q_h0(D), G_h0(D), Q_h1(D), ...]` as two flat halves with dim-major indexing.
    Fixed to head-major interleaved extraction: `head * 2D + dim` for Q, `head * 2D + D + dim` for gate.
  - `causal_depthwise_conv` verified correct (comparison documented).
  - Delta-net recurrence math verified correct.
  - NeoX RoPE for full attention implemented and verified at non-zero positions.
  - `causal_depthwise_conv` and QKV packing comparison completed — see
    `docs/llama-rs/worklog/2026-04-13-conv-qkv-comparison.md`.

## Already-implemented items (review_3 items done before this branch)

- `GgmlElement` trait with `f32`/`i32` impls.
- `GgmlType` trait mapping Rust types to `Type`.
- `BackendElement` / `HostElement` trait hierarchy.
- `with_context` / `with_no_alloc_context` scoped helpers.
- `Shape3D`, `Shape4D`, `Dims<const N>`, `new_tensor<const N>`, `rank()`, `dims()`, `shape_nd()`.
- `decode_tensor_data_to::<T>()` generic GGUF decode.
- `MaybeUninit` / `spare_capacity_mut` optimization on hot tensor readout paths.
- `define_semantic_usize!` macro for newtype dedup.
- Typed tensor wrappers (`Tensor1D`..`Tensor4D`).

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

## After parity closure

- Return to broader `llama-rs` trait/ADT continuation tasks.
