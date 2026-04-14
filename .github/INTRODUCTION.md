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
9. Merge back to `master` only after validation and runtime checks pass. **DONE** — PR #2 merged (73 commits). Copilot review 6/6 comments addressed (commit `97faee0`). See `docs/llama-rs/worklog/2026-04-14-merge-prep.md`.
10. ~~Comprehensive coverage comparison table.~~ **DONE** — 3-bucket taxonomy (Native Rust / Upstream Harness / Missing). 231 total tests, 43 examples, 8 benchmark programs. See `docs/ggml-rs/COVERAGE_TABLE.md`.

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
- **Causal depthwise conv vs QKV packing comparison** (item 27):
  Detailed architectural analysis of how Qwen3.5's two attention types
  (full + linear) differ in projection storage, convolution handling,
  state management, and GPU offloading. Covers separate vs unified QKV,
  Q+gate interleaving, sliding window buffer, fused projection+conv graph,
  and `ggml_ssm_conv` primitive usage. See comparison doc item 27.
- **Persistent MLP graphs** (item 28):
  Pre-built MLP compute graphs with weights resident on device. Eliminates
  ~165 MB weight upload per layer per token (~5.3 GB across 32 layers).
  `PersistentMlp` struct holds fixed `seq_len=1` graph (`rms_norm → mul →
  silu(gate) × up → down`). Per-step cost: ~12 KB I/O vs ~165 MB ephemeral.
  Per-layer opportunistic (failed layers fall back to ephemeral). Integrated
  into `two_phase_loop` with early-return when `remaining_decode_steps == 0`.
  See comparison doc item 28.

- **GraphAllocator safe wrapper** (item 29):
  Safe Rust wrapper for `ggml_gallocr` C API. Pre-reserves a buffer for the
  maximum graph size, then reuses it via `alloc_graph()` on each step —
  eliminating per-step Metal buffer allocation. `!Send + !Sync`, skips
  tensors with existing backend buffers. 5 integration tests.
  See comparison doc item 29.

- **PersistentScoringContext** (item 30):
  Uses GraphAllocator to eliminate per-step Metal buffer allocation in
  FA decode scoring. Extracts graph construction into `build_scoring_graph()`,
  reserves at max_tokens, reuses buffer each step. One shared context across
  all FA layers (serial execution, same dimensions). Eliminates ~800µs/step
  (8 layers × ~100µs). See comparison doc item 30.

- **Flash-friendly KV layout** (item 31):
  Changed `PersistentKvCache` from `[D, Hkv, MaxT, 1]` (head-major) to
  `[D, MaxT, Hkv, 1]` (time-major, flash-friendly). Eliminates per-step
  `permute(0,2,1,3) + cont` O(T) device copy (~136µs at T=1000). Direct
  `view_4d_of` with strided nb2 feeds flash_attn_ext; Hkv small writes
  per append (D=64, Hkv=4: negligible). See comparison doc item 31.

- **Auto-vectorization verification + decode allocation elimination** (item 32):
  Verified LLVM auto-vectorizes `ssm_recurrence_step` with NEON (129 NEON insns,
  `fmul.4s`+`fadd.4s` pairs, 8 floats/iter via `ldp`/`stp` register pairs).
  FMA rejected: changes rounding semantics, breaks decode parity.
  Added `LinearDecodeScratch` bundling `SsmScratch` + output + norm_buf, reused
  across all 24 linear layers. Eliminates 384 heap allocations per decode step
  (16/layer × 24 layers). Added `rms_norm_single_into()` and
  `rms_norm_single_in_place()` for allocation-free normalization.
  See comparison doc item 32.

- **Resumable GenerationSession + binary checkpoint** (item 33):
  `GenerationSession` wraps inference as an iterator-like API (`next_token()`)
  with mid-generation checkpointing. `GenerationCheckpoint` uses postcard
  binary format with magic bytes + version envelope. `ModelFingerprint`
  validates layer count, hidden dims, vocab, eps, and per-layer type tags
  on resume. `CheckpointV1` DTO decoupled from runtime types for format
  stability. Supports TwoPhase (Qwen3.5, performance-preserving) and
  FullReprocess (standard attention) modes. 18 tests covering roundtrip,
  validation, invariant checking, and fingerprint mismatch.
  See comparison doc item 33.

- **PersistentDecodeResources — unified session decode optimization** (item 34):
  `PersistentDecodeResources` encapsulates all GPU-resident persistent state
  (LM head, projections, KV caches, scoring context, linear scratch, MLPs)
  in a single struct with safety-critical drop ordering. Granular optionality:
  LM head always present, all other resources independently optional.
  Refactored `two_phase_loop` from ~170 lines of inline resource management
  to ~50 lines. Integrated into `GenerationSession` with lazy init after
  prefill + fallback to slow path. Eliminated duplicate LM head builds.
  See comparison doc item 34.

- **LmHeadResources extraction** (item 35):
  Extracted self-contained `LmHeadResources` struct from inline LM head fields
  in `PersistentDecodeResources`. Owns its own ggml context, buffer, tensors,
  and graph with correct drop ordering. `try_build()` + `sample_hidden()` API.
  Embedded in `PersistentDecodeResources`, also used by `full_reprocess_loop`
  (eliminates duplicate LM head construction). Removed `graph_sample_at()`
  helper. Added graceful fallback in `full_reprocess_loop` when build fails.
  See comparison doc item 35.

- **Attention dispatch helpers** (item 36):
  Added `is_standard()` predicate method to `AttentionLayerPlan` for
  boolean-query use sites. Applied in `generate_from_plans()` mode
  selection (2 sites). Dispatch match arms keep explicit variant
  patterns to preserve exhaustiveness checking.
  See comparison doc item 36.

- **Dead code cleanup** (item 37):
  Removed unused `_hidden_features` parameter from `process_all_layers`
  (8→7 args, 7 call sites). Demoted one-shot `lm_head_graph` function
  to `#[cfg(test)]` (replaced by `LmHeadResources` in production).
  See comparison doc item 37.

- **Tuple-to-struct for persistent graph builders** (item 38):
  Replaced fragile 5/12/14-element tuple returns with named structs:
  `LmHeadGraphParts`, `FullAttentionGraphParts`, `LinearAttentionGraphParts`.
  Eliminated `#[allow(clippy::type_complexity)]` on two builder functions.
  All call sites updated to use field access (e.g. `g.w_q`) instead of
  positional destructuring.
  See comparison doc item 38.

- **Inline `persistent_decode_all_layers` as method** (item 39):
  Converted 11-parameter free function into `impl PersistentDecodeResources`
  method. 5 params (`projections`, `kv_caches`, `persistent_mlps`,
  `scoring_ctx`, `linear_scratch`) now accessed via `self`, leaving 6
  explicit params. Eliminates `clippy::too_many_arguments(11/7)`.
  See comparison doc item 39.

- **`RopeParams` struct extraction** (item 40):
  Grouped 4 RoPE configuration values (`n_rot`, `freq_base`, `freq_scale`,
  `position_offset`) into `RopeParams`. `apply_neox_rope_in_place` reduced
  from 8→5 params. Eliminates `clippy::too_many_arguments(8/7)`.
  See comparison doc item 40.

- **Further clippy `too_many_arguments` reduction** (item 41):
  `project_qkv_graph` accepts `&Qwen35FullAttentionLayerPlan` instead of
  3 individual weight slices (9→8 params). `PersistentDecodeResources::try_build`
  accepts pre-built `LmHeadResources` instead of raw weight slices (8→5 params),
  returns `Self` instead of `Option<Self>`. LM head construction moved to
  callers with `Option::map` pattern. All `too_many_arguments` warnings
  eliminated from `llama-rs` crate.
  See comparison doc item 41.

- **`QkvProjections` struct** (item 42):
  Replaced `(Vec<f32>, Vec<f32>, Vec<f32>)` return type from `project_qkv_graph`
  with named struct (`q_full`, `k_proj`, `v_proj`). Eliminates
  `clippy::type_complexity` warning. Self-documenting field access at call sites.
  See comparison doc item 42.

- **`FullAttentionDims` struct** (item 43):
  Consolidated dimension validation + memory estimation from
  `fully_fused_attention_graph` into a reusable struct. `new()` validates GQA
  divisibility and derives hidden size from output weight matrix. `estimate_memory(t)`
  replaces inline calculation. Reduces function preamble from ~30 lines to 2.
  See comparison doc item 43.

- **`LinearAttentionDims` struct** (item 44):
  Consolidated dimension derivation + memory estimation for linear attention.
  `new()` reuses existing helper functions; `estimate_fused_memory(seq_len)` replaces
  ~40 lines of inline memory calculation. Three callers now use the struct instead
  of duplicated inline derivation. `project_and_conv_fused_graph` drops from 8→7
  params, removing `#[allow(clippy::too_many_arguments)]`. Spurious `#[allow]` on
  `project_linear_inputs_graph` (6 params) also removed.
  See comparison doc item 44.

- **Stale `#[allow(clippy)]` removal** (item 45):
  Removed 3 now-unnecessary `#[allow]` annotations:
  - `build_lm_head_graph`: `type_complexity` stale (returns `LmHeadGraphParts` struct)
  - `fully_fused_attention_graph`: `too_many_arguments` stale (7 params after item 43)
  - `qwen35_linear_attention_core`: `too_many_arguments` stale (7 params after item 44)
  Also fixed `build_lm_head_graph` doc comment (still said "returns tuple").

- **Projection result struct extraction** (item 46):
  Made `QkvProjections` `pub(super)` and reused it from `tensor_ops.rs` for
  `read_full_attention_projections` (was `(Vec<f32>, Vec<f32>, Vec<f32>)`).
  Added `RawLinearProjections { qkv, z, alpha, beta }` in `tensor_ops.rs` for
  `read_linear_attention_projections` (was `(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)`).
  All `#[allow(clippy::type_complexity)]` eliminated from llama-rs.
  Only remaining `#[allow]` is `ssm_recurrence_step` (10-param math kernel, deliberate).
  See comparison doc items 45-46.

- **`sum_matmul_memories` DRY helper** (item 47):
  Extracted common memory estimation pattern from `recommended_persistent_full_attention_memory`
  and `recommended_persistent_linear_attention_memory`. Takes a declarative slice of
  `(weight_shape, input_shape, label)` tuples, uses `try_fold` + `checked_add`.
  Both callers reduced from ~35 lines to ~10 lines.
  See comparison doc item 47.

- **`upload_weight` DRY helper** (item 48):
  Extracted `write_data_backend` + `map_err` pattern from `build_one_persistent_full`
  and `build_one_persistent_linear`. Each weight upload reduced from 3 lines to 1 line,
  preserving per-weight error labels for debuggability.
  See comparison doc item 48.

- **`OutputProjectionGraph` sub-struct extraction** (item 49):
  Extracted shared output projection fields (`w`, `x`, `y`, `graph`) from
  `FullAttentionGraphParts` and `LinearAttentionGraphParts` into `OutputProjectionGraph`.
  Both graph builder functions now call `build_output_projection_graph` instead of
  duplicating ~15 lines of tensor creation + graph wiring. `PersistentDecodeProjection`
  enum variants also compose with the sub-struct, simplifying `project_output`.
  See comparison doc item 49.

- **Hoist `upload_weight` to `tensor_ops` + apply across all modules** (item 50):
  Moved `upload_weight` from `generation.rs` to `tensor_ops.rs` as `pub(super)`.
  Applied across `attention.rs` (23→4 occurrences), `linear_attention.rs` (15→0),
  `mlp.rs` (10→0), eliminating 66 lines of `write_data_backend` + `map_err`
  boilerplate while preserving per-weight error labels.
  See comparison doc item 50.

- **Tail-only `qkv_pre_conv` readback** (item 51):
  `project_and_conv_fused_graph` previously read back the FULL pre-conv QKV tensor
  from GPU. Now reads only the last `kernel_size - 1` rows via `read_data_backend_at`
  (~680x reduction for 2048-token prompts). When no decode state is needed, skips
  the readback entirely.
  See comparison doc item 51.

- **Direct SSM recurrence into persistent state** (item 52):
  `qwen35_linear_attention_core` now writes SSM recurrence directly into
  `state.ssm_states` instead of allocating a temporary buffer + memcpy.
  Removed dead `capture_ssm_states` method. Added `debug_assert_eq` for size invariant.
  See comparison doc item 52.

- **Linear attention prefill benchmark** (item 53):
  Added `bench_e2e_graphs_linear_attention_prefill` to `bench_graphs.rs`.
  Prefill-with-state is comparable or faster than stateless inference,
  confirming items 51-52 eliminated state capture overhead. Metal achieves
  2.86x speedup at seq_len=256. CPU faster below seq_len~16 (dispatch overhead).
  See comparison doc item 53.

- **Phase-breakdown comparison benchmark** (item 54):
  Added `bench_linear_attention_phases()` decomposing linear attention into
  4 timed phases. At seq_len=256 on Metal: SSM recurrence 45.7% (4.8ms),
  Proj+Conv 35.9%, OutProj 17.9%, QK split 0.5%. SSM runs at identical
  speed on CPU and Metal (pure scalar work). See comparison doc item 54.

- **SSM loop optimization** (item 55):
  Hoisted `state_size`/`time_step_rank`/`group_count`/`state_size_sq` out of
  inner loop. Replaced `checked_mul` offset with `chunks_exact_mut().enumerate()`.
  Hoisted `token_rank_base`. Result: ~6% SSM recurrence improvement across all
  backends (4.794→4.491ms Metal seq=256). See comparison doc item 55.

- **Batch projection helper** (item 56):
  Added `ProjectionSpec` / `BuiltProjection` / `build_batch_projections()` to
  `tensor_ops.rs`. Refactored 3 sites (persistent full attention, persistent
  linear attention, one-shot QKV) to use the shared helper. Eliminates ~50
  lines of repeated `new_tensor_2d` + `mul_mat` boilerplate. Named struct
  preserves diagnostic labels. See comparison doc item 56.

- **Conv vs QKV packing micro-benchmark** (item 57):
  Quantitative comparison of packed QKV (single matmul) vs separate Q/K/V
  (3 matmuls), host conv vs graph conv (ssm_conv), and layout prep overhead.
  Key findings: packed QKV is NOT faster than separate (separate is 15% faster
  on Metal at seq_len=1024); graph conv wins at seq_len≥256; layout prep
  (transpose+cont+pad) costs as much as conv itself. See comparison doc item 57.

- **MLP graph topology extraction** (item 58):
  Extracted shared `MlpGraphParts` struct and `build_mlp_graph` builder that
  encapsulates the 7-op MLP chain (rms_norm → scale → gate matmul → silu →
  up matmul → mul → down matmul). Both `mlp_sequence_inference_with_weights`
  (one-shot prefill) and `build_persistent_mlp` (decode) now delegate to
  `build_mlp_graph`, eliminating ~40 lines of duplicated topology code.
  See comparison doc item 58.

- **Fully-fused attention QKV via batch projections** (item 59):
  Refactored `fully_fused_attention_graph` QKV creation (3 separate
  `new_tensor_2d` + `mul_mat` blocks) to use `build_batch_projections`.
  Reduces ~15 lines of boilerplate while preserving all error labels.
  See comparison doc item 59.

- **Linear attention projections via batch helper** (item 60):
  Extracted `linear_projection_specs` returning the 4-projection spec array
  (QKV, Z, alpha, beta). Both `project_linear_inputs_graph` and
  `project_and_conv_fused_graph` now delegate to `build_batch_projections`
  via this shared spec, eliminating ~30 lines of duplicated tensor creation
  and matmul boilerplate. See comparison doc item 60.

- **Incremental standard-attention decode** (item 61):
  Standard attention previously forced FullReprocess mode (entire sequence
  recomputed per token). Now supports TwoPhase: `StandardAttentionState` in
  `state.rs` manages KV cache (post-RoPE K + raw V per head, with
  `append_batch` and per-head accessors). `standard_attention_prefill()` uses
  a fused ggml graph with KV readback via `build_forward_expand` for
  intermediate tensors. `standard_attention_decode_step()` performs host-side
  dot-product scoring with cached KV (matching Qwen35Full decode pattern).
  `standard_attention_inference()` provides a stateless wrapper. Wired into
  all three strategies (Inference/Prefill/Decode) in `generation.rs`. Removed
  `has_standard → FullReprocess` guard. Persistent projections skip Standard
  layers gracefully (continue, not return None). Checkpoint format bumped
  V1→V2 with KV cache fields. Removed unused `is_standard()` and
  `attention_inference_with_weights_on_backend_repeats_with_length`. 123 tests
  pass, zero clippy warnings. Commit `e0e3675`.

- **Extract shared flash-attention pipeline** (item 62):
  `fully_fused_attention_graph` and `standard_attention_graph` shared ~70%
  code for the flash-attention pipeline (reshape_4d → permute → cont →
  flash_attn_ext → optional gating → output projection). Extracted two helpers:
  `run_flash_attention_pipeline` (QKV tuple + optional mask/gate → output
  tensor) and `apply_optional_per_head_norm` (conditional rms_norm + weight
  scaling). `FlashAttentionConfig` struct carries dimensional/scalar params.
  Net ~75 line reduction, zero clippy warnings, 229 tests pass.
- **Split attention.rs into coherent submodules** (item 63):
  Split the 2576-line monolithic `attention.rs` into 5 focused submodules:
  `shared.rs` (RoPE + flash-attention helpers), `persistent.rs` (KV cache +
  GPU scoring), `projection.rs` (QKV projection/deinterleaving),
  `qwen35_full.rs` (gated attention fused graph + decode), `standard.rs`
  (standard attention fused graph + decode). Module root retains `pub(super)`
  re-exports for stable consumer import paths. Tests stay in root module.
  Zero clippy warnings, 229 tests pass.
- **Split linear_attention.rs into coherent submodules** (item 64):
  Split the 1767-line monolithic `linear_attention.rs` into 3 focused
  submodules: `projection.rs` (LinearProjections, FusedLinearOutputs,
  LinearAttentionDims, graph + host fallback projections), `conv.rs`
  (causal depthwise convolution: host reference, graph-accelerated, fused
  projection+conv, decode step), `ssm.rs` (SsmScratch, LinearDecodeScratch,
  ssm_recurrence_step, split_and_norm_qk). Root retains entry points
  (inference/prefill/decode), core logic, bench utilities, tests.
  Zero clippy warnings, 229 tests pass.
- **Split generation.rs into coherent submodules** (item 65):
  Split the 1810-line monolithic `generation.rs` into 2 focused submodules:
  `strategy.rs` (AttentionStrategy trait, InferenceStrategy, PrefillStrategy,
  DecodeStrategy — pure dispatch pattern), `resources.rs` (LmHeadResources,
  PersistentDecodeResources, persistent projection/KV cache/MLP builders,
  unsafe transmute + drop-order safety logic). Root retains GenerationMode,
  GenerationInputs, GenerationOutput, process_all_layers, generation loops,
  public entry points, tests. Zero clippy warnings, 229 tests pass.

- **Split tensor_ops.rs into 5 thematic submodules** (item 66):
  Split the 1460-line `tensor_ops.rs` into 5 focused submodules:
  `normalization.rs` (rms_norm_with_weight, rms_norm_single[_into],
  per_head_{rms,l2}_norm), `host_ops.rs` (add_in_place, project_sequence,
  head_slice[_mut], gather_embeddings), `projection.rs` (ProjectionSpec,
  BuiltProjection, OutputProjectionGraph, batch/sequence projection builders,
  upload_weight, sum_matmul_memories), `lm_head.rs` (LmHeadGraphParts,
  build_lm_head_graph, argmax_token_id, lm_head_sample_step),
  `persistent_decode.rs` (PersistentDecodeProjection, FullAttentionGraphParts,
  LinearAttentionGraphParts, RawLinearProjections, persistent graph builders).
  Visibility: pub(in crate::e2e) for e2e consumers, pub(super) for internal,
  module-private where sole user is same file. Zero clippy warnings, 229 tests pass.

- **Fix Standard-attention checkpoint validation bug** (item 67):
  `validate_invariants()` was checking `total_sequence_length * kv_features` for
  Standard-attention caches, but `From<&LayerAttentionState>` serializes trimmed
  caches (`cached_len * kv_features`). Fixed to check `cached_len * kv_features`,
  matching Qwen35Full branch. Added `standard_capture_roundtrip_via_from_impl` test.

- **Split checkpoint.rs into dto + runtime submodules** (item 68):
  Split the 1030-line `checkpoint.rs` into 3 files:
  `checkpoint.rs` root (~90 lines): module declarations, GenerationCheckpoint
  public API (save_to, load_from, accessors), facade roundtrip tests.
  `checkpoint/dto.rs` (~320 lines): CheckpointV1, ModelFingerprint, LayerTypeTag,
  LayerStateDto, CHECKPOINT_VERSION, from_plans, validate_against,
  validate_invariants + 24 validation tests.
  `checkpoint/runtime.rs` (~270 lines): From<&LayerAttentionState>, into_runtime_state,
  CaptureInput, capture, restore_state + 7 conversion/capture tests.
  Total test count increased from 230 to 244. Zero clippy warnings.

- **Split session.rs into init + runtime submodules** (item 69):
  Split the 831-line `session.rs` into 3 files:
  `session.rs` root (~339 lines): struct definition, checkpoint(), accessors,
  mod declarations, re-exports, 9 tests (including 2 facade tests).
  `session/init.rs` (~280 lines): new(), resume(), shared ResolvedModel helper
  that deduplicates ~80 lines of model resolution between constructors.
  `session/runtime.rs` (~215 lines): next_token(), step_two_phase(),
  step_full_reprocess(), ensure_persistent_resources() (pub(super) for test
  access), emit_token(), sample_next().
  Total test count increased from 244 to 246. Zero clippy warnings.

- **Split bench_graphs.rs into 4 submodules** (item 70):
  Split the 1064-line test-only benchmark module into 4 focused submodules:
  `bench_graphs.rs` root (~23 lines): module doc + `#[cfg(test)]` mod decls.
  `bench_graphs/helpers.rs` (~154 lines): bench_fn, BenchResult, print_results,
  available_backends, synthetic plan/input builders.
  `bench_graphs/inference.rs` (~392 lines): 6 inference graph benchmarks
  (full attention, linear attention ×3, MLP, combined).
  `bench_graphs/lm_head.rs` (~110 lines): LM head host/cold/warm graph bench.
  `bench_graphs/micro.rs` (~380 lines): conv vs QKV packing micro-benchmarks.
  8 ignored bench tests preserved, 246 total tests, zero clippy warnings.

71. **Split `generation.rs` root into api + loops submodules** — DONE (commit `0477f13`)
  Root was 980 lines with 2 existing submodules (resources, strategy).
  Extracted:
  `generation/api.rs` (~166 lines): public API entry points
    (generate_from_path/model, resolve_eos_token_id, tokenize_prompt_text).
  `generation/loops.rs` (~255 lines): core generation loops
    (generate_from_plans, full_reprocess_loop, two_phase_loop).
  Root now ~180 lines (excl. tests): types, process_all_layers, sampling.
  Submodule count: 2 → 4 (api, loops, resources, strategy).
  246 tests pass, zero clippy warnings.

72. **Split `linear_attention.rs` root into decode + sequence + bench submodules** — DONE (commit `5f7bbee`)
  Root was 912 lines with 3 existing submodules (conv, projection, ssm).
  Extracted:
  `linear_attention/decode.rs` (~140 lines): `linear_attention_decode_core` + `qwen35_linear_attention_decode_step`.
  `linear_attention/sequence.rs` (~170 lines): `qwen35_linear_attention_core` (full-sequence SSM recurrence).
  `linear_attention/bench.rs` (~160 lines, `#[cfg(test)]`): `LinearAttentionPhaseTimings` + `bench_linear_attention_phases`.
  Root now ~450 lines (incl. tests): thin wrappers, utilities, re-exports, 7 tests.
  Submodule count: 3 → 6 (bench, conv, decode, projection, sequence, ssm).
  246 tests pass, zero clippy warnings.

73. **Unit tests for resolve.rs + GgufModel test stub** — DONE (commit `fae4281`)
  Added `#[cfg(test)]` stub constructors to `GgufModel` (`stub` with configurable
  tensor sizes + KV entries, `stub_from_names` convenience). Enforces name uniqueness.
  17 tests covering:
  - Pure candidate generators (global, layer, SSM patterns)
  - Resolution precedence (earliest candidate wins, not model order)
  - Error payloads (role, layer, tried candidates in error messages)
  - Optional vs required tensor resolution
  - Alternative naming conventions (HuggingFace-style)
  263 tests pass, zero clippy warnings.

74. **Unit tests for planner.rs** — DONE (commit `8da9ba8`)
  10 tests covering layer plan construction:
  - `required_transformer_usize`: Some→Ok, None→error
  - `build_layer_mlp_plan`: success, invalid gate shape, hidden_features=0
  - Policy tests: Strict errors on missing attention (qwen35 arch), propagates
    malformed attention (llama arch); SkipUnsupportedAttention builds MLP-only
  - Standard attention end-to-end smoke test with minimal llama model
  Test helpers: stub_model (element-count sizing), MLP/attention tensor builders,
  llama/qwen35 KV metadata builders.
  273 tests pass, zero clippy warnings.

75. **Comprehensive unit tests for numeric.rs** — DONE (commit `bb47cca`)
  32 new tests (expanding 1 existing) covering all pure math functions:
  - checked_mul: success + overflow
  - validate_token_id: valid/negative/boundary/out-of-range
  - value_to_i32: all GgufValue variants, overflow, fractional, float out-of-range
  - dot: basic product + empty
  - softmax_prefix: uniform, dominated, prefix-only semantics, len=0
  - sigmoid/silu/softplus: known values, boundary at 20.0 threshold
  - f32_to_f16_bits: zero, neg zero, ±1, ±Inf, NaN, overflow, subnormal boundary
  - build_causal_mask_f16_bytes: seq=1, seq=3 lower-triangular, overflow
  305 tests pass, zero clippy warnings.

76. **GgmlResultExt extension trait** — DONE (commit `3fd472b`)
  Added `GgmlResultExt<T>` trait on `Result<T, ggml_rs::Error>` with `.ggml_ctx(context)`
  method, replacing `.map_err(|source| E2eError::ggml(..., source))` across 14 files,
  262 call sites. Net reduction: 206 lines. No behavior change.
  305 tests pass, zero clippy warnings.

77. **AttentionLayerPlan lightweight helpers** — DONE (commit `9853c0a`)
  Added `is_standard()`, `kv_head_count()`, `head_dimension()` on `AttentionLayerPlan`
  and `conv_channels()` on `Qwen35LinearAttentionLayerPlan`. Updated callers in
  dto.rs, state.rs, session/init.rs. Net -7 lines across 4 files.
  305 tests pass, zero clippy warnings.

78. **LayerPassConfig parameter bundling** — DONE (commit `5926418`)
  Introduced `LayerPassConfig` struct bundling `layer_plans`, `rms_norm_eps`, `backend`.
  `process_all_layers` reduced from 7 params to 5. `decode_step` also simplified.
  Updated 7 call sites across loops.rs, resources.rs, runtime.rs.
  305 tests pass, zero clippy warnings.

79. **execute_batch_projections dedup** — DONE (commit `73d36f9`)
  Extracted `execute_batch_projections()` helper in `tensor_ops/projection.rs` to eliminate
  ~101 lines of duplication between `project_qkv_graph` and `project_linear_inputs_graph`.
  Both callers now delegate to the shared helper while preserving typed per-module wrappers.
  Removed unused `BuiltProjection` re-export from `tensor_ops.rs`.
  325 tests pass, zero clippy warnings.

80. **greedy_sample_at_index + LayerPassConfig hoist** — DONE (commit `f25b0ed`)
  Extracted `greedy_sample_at_index()` in generation.rs to unify sampling
  fallback logic between `graph_sample_fallback` and `sample_next`.
  Uses `rms_norm_single` instead of full-buffer `rms_norm_with_weight`.
  Hoisted duplicate `LayerPassConfig` construction in `step_two_phase`.
  325 tests pass, zero clippy warnings.

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
