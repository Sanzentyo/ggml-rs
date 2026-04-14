# ggml-rs Coverage Comparison Table

Comprehensive comparison of ggml-rs test/example/benchmark coverage against
upstream ggml C/C++ source-tree targets.

## Taxonomy

| Symbol | Meaning |
|--------|---------|
| ✅ | **Native Rust** — Equivalent test/example implemented in pure Rust |
| ✅ partial | **Partial native Rust** — Subset of upstream functionality covered |
| ⚠️ | **Upstream binary harness** — C/C++ binary invoked via `ggml_upstream_suite.rs`¹ |
| ❌ | **Missing** — No coverage in ggml-rs |
| N/A | Not applicable to Rust |

¹ Harness coverage is **opt-in**: requires `--features link-system`, `--ignored`,
and pre-built upstream binaries (`GGML_UPSTREAM_BUILD_DIR`). Not part of default
`cargo test`.

---

## 1. Upstream ggml Test Targets (21 source-tree targets²)

All 21 upstream test targets are listed in the binary harness (`ggml_upstream_suite.rs`).
Some also have additional native Rust parity tests.

| Upstream Target | Harness | Native Rust | Notes |
|---|---|---|---|
| `test-backend-ops` | ⚠️ | ✅ partial | `ggml_backend_compute.rs` (20 tests): matmul, sigmoid, flash_attn_ext (MHA/GQA), multi-op, roundtrip. Upstream tests hundreds of ops; Rust covers key subset. |
| `test-opt` | ⚠️ | ❌ | No native Rust optimizer tests |
| `test-quantize-fns` | ⚠️ | ❌ | No native Rust quantization function tests |
| `test-quantize-perf` | ⚠️ | ❌ | No native Rust quantization perf tests |
| `test-pool` | ⚠️ | ❌ | |
| `test-arange` | ⚠️ | ❌ | |
| `test-timestep_embedding` | ⚠️ | ❌ | |
| `test-pad-reflect-1d` | ⚠️ | ❌ | |
| `test-roll` | ⚠️ | ❌ | |
| `test-conv-transpose` | ⚠️ | ❌ | |
| `test-conv-transpose-1d` | ⚠️ | ❌ | |
| `test-dup` | ⚠️ | ❌ | |
| `test-rel-pos` | ⚠️ | ❌ | |
| `test-customop` | ⚠️ | ❌ | |
| `test-conv1d` | ⚠️ | ❌ | ggml-rs wraps `ssm_conv` but no dedicated conv1d test |
| `test-conv1d-dw-c1` | ⚠️ | ❌ | |
| `test-conv1d-dw-c2` | ⚠️ | ❌ | |
| `test-conv2d` | ⚠️ | ❌ | |
| `test-conv2d-dw` | ⚠️ | ❌ | |
| `test-cont` | ⚠️ | ✅ partial | `ggml_test_cont.rs`: host-path F32 only (upstream tests backend + F16) |
| `test-interpolate` | ⚠️ | ❌ | |

**Summary**: 21/21 listed in harness; **2/21 also have native Rust tests** (both partial).

² Upstream test availability is build-config-dependent: 20/21 are behind
`if (NOT GGML_BACKEND_DL)` in CMakeLists.txt.

---

## 2. Upstream ggml Examples (17 build targets + 1 standalone fixture)

| Upstream Example | Rust Counterpart | Status | Location |
|---|---|---|---|
| `simple-ctx` | `simple_ctx` | ✅ Direct | `examples/basics/simple_ctx.rs` (also: native parity test `ggml_simple_ctx.rs`) |
| `simple-backend` | `backend_matmul` | ✅ Direct | `examples/backends/backend_matmul.rs` |
| `perf-metal` | `perf_metal` | ✅ Direct | `examples/backends/perf_metal.rs` |
| `gpt-2-ctx` | `gpt2_ctx` | ✅ Synthetic | `llama-rs/examples/models/gpt2_ctx.rs` |
| `gpt-2-alloc` | `gpt2_alloc` | ✅ Synthetic | `llama-rs/examples/models/gpt2_alloc.rs` |
| `gpt-2-backend` | `gpt2_backend` | ✅ Synthetic | `llama-rs/examples/models/gpt2_backend.rs` |
| `gpt-2-sched` | `gpt2_sched` | ✅ Synthetic | `llama-rs/examples/models/gpt2_sched.rs` |
| `gpt-2-batched` | `gpt2_batched` | ✅ Synthetic | `llama-rs/examples/models/gpt2_batched.rs` |
| `gpt-2-quantize` | `gpt2_quantize` | ✅ Synthetic | `llama-rs/examples/models/gpt2_quantize.rs` |
| `gpt-j` | `gptj_main_synth` | ✅ Synthetic | `examples/models/gptj_main_synth.rs` |
| `gpt-j-quantize` | `gptj_quantize_synth` | ✅ Synthetic | `examples/models/gptj_quantize_synth.rs` |
| `magika` | `magika_main_synth` | ✅ Synthetic | `examples/models/magika_main_synth.rs` |
| `mnist-eval` | `mnist_eval` | ✅ Synthetic | `examples/models/mnist_eval.rs` |
| `mnist-train` | `mnist_train` | ✅ Synthetic | `examples/models/mnist_train.rs` |
| `sam` | `sam` | ✅ Synthetic | `examples/models/sam.rs` |
| `yolov3-tiny` | `yolov3_tiny` | ✅ Synthetic | `examples/models/yolov3_tiny.rs` |
| `mnist` (WASM) | — | ❌ | Emscripten/WASM build; no Rust equivalent |
| `test-cmake`³ | — | N/A | Standalone CMake integration fixture |

**Summary**: **16/17 covered** (✅); 1 missing (`mnist` WASM).

³ `test-cmake` is a standalone build-system fixture (not added by the parent
`examples/CMakeLists.txt`) and is excluded from the example count.

---

## 3. Upstream Benchmarks

Upstream ggml has no dedicated `benchmarks/` directory. Performance testing is
integrated into test targets and examples.

| Upstream Target | Rust Counterpart | Status | Notes |
|---|---|---|---|
| `test-quantize-perf` | — | ⚠️ Harness only | Upstream test target, no native Rust benchmark |
| `perf-metal` | `perf_metal` | ✅ Direct | `examples/backends/perf_metal.rs` |

---

## 4. Rust-Only Tests (no upstream counterpart)

These tests exercise the Rust API layer, typed wrappers, and llama-rs E2E
inference pipeline. **No upstream equivalent exists.**

### Counting rule

Total workspace `#[test]` functions: **231**.
Upstream-mapped tests (excluded from Rust-only count):
- `ggml_backend_compute.rs` (20 tests — partial `test-backend-ops` coverage)
- `ggml_test_cont.rs` (1 test — partial `test-cont` coverage)
- `ggml_simple_ctx.rs` (1 test — `simple-ctx` parity)
- `ggml_upstream_suite.rs` (1 test — harness runner)

**Rust-only**: 231 − 23 = **208 tests**.

### ggml-rs Core Tests

| Test File | Count | What It Tests |
|---|---|---|
| `src/types.rs` (inline) | 18 | Type enum classification, display, hashing, roundtrip |
| `tests/ggml_error_paths.rs` | 18 | Error variants, bounds checking, overflow, mismatch |
| `tests/ggml_tensor_nd_extended.rs` | 20 | Typed tensor API, DynTensor, rank/shape/dims |
| `tests/ggml_tensor_ops.rs` | 20 | View/reshape wrappers, cross-context views, aliasing |
| `tests/ggml_graph_allocator.rs` | 5 | GraphAllocator lifecycle, reuse, pre-alloc skip |
| `tests/gguf_roundtrip.rs` | 2 | GGUF writer roundtrip, kv_value_as extraction |
| **Subtotal** | **83** | |

### llama-rs E2E Tests (inline `#[cfg(test)]`)

| Test Module | Count | What It Tests |
|---|---|---|
| `e2e/attention.rs` | 9 | Q/gate split, RoPE, prefill+decode parity, GPU scoring |
| `e2e/linear_attention.rs` | 7 | Head group mapping, conv decode, conv graph parity |
| `e2e/generation.rs` | 5 | Greedy sampler, two-phase/full-reprocess, error cases |
| `e2e/session.rs` | 8 | Step-by-step generation, checkpoint, persistent resources |
| `e2e/checkpoint.rs` | 14 | Roundtrip, magic, fingerprint, invariant validation |
| `e2e/tensor_ops.rs` | 9 | RMS norm, projection parity, argmax, LM head, persistent proj |
| `e2e/state.rs` | 4 | KV cache, conv buffer, generation state |
| `e2e/numeric.rs` | 1 | GGUF numeric conversion |
| `e2e/bench_graphs.rs` | 8 | Microbenchmarks (full attn, linear attn, MLP, LM head, conv/QKV)⁴ |
| **Subtotal** | **65** | |

⁴ Bench functions are `#[test] #[ignore]` — counted as tests, run manually.

### llama-rs Library Tests (inline `#[cfg(test)]`)

| Test Module | Count | What It Tests |
|---|---|---|
| `naming.rs` | 7 | Tensor name resolution (blk/hf style) |
| `metadata.rs` | 6 | Model metadata parsing (llama, qwen3.5, RoPE) |
| `inference.rs` | 7 | Attention layout, causal mask, config, decode cache |
| `idle.rs` | 2 | Pause schedule, idle config validation |
| `bench_report.rs` | 6 | Benchmark report parsing and rendering |
| `tokenizer.rs` | 18 | BPE encode/decode, streaming, special tokens |
| `chat.rs` | 10 | ChatML formatting, sanitization, role display |
| **Subtotal** | **56** | |

### llama-rs Integration Tests

| Test File | Count | What It Tests |
|---|---|---|
| `tests/attention_parity.rs` | 2 | CPU vs Metal attention parity |
| `tests/mlp_cpp_parity.rs` | 2 | CPU vs C++ reference MLP, CPU vs Metal MLP |
| **Subtotal** | **4** | |

**Rust-only grand total**: **208 tests**

---

## 5. Rust-Only Examples (no upstream counterpart)

### ggml-rs Core (4 programs)

| Example | Purpose |
|---|---|
| `arithmetic_expr` | Expression-style arithmetic API demo |
| `backend_ops` | Multi-op backend graph (matmul + bias) |
| `bench_matmul` | Context-path matmul benchmarking |
| `bench_upstream_suite` | Upstream C test suite runner |

### llama-rs (23 programs)

| Example | Category | Purpose |
|---|---|---|
| `backend_smoke` | Basics | Backend functionality smoke test |
| `simple` | Basics | Minimal llama-rs usage |
| `bench_llama_matmul` | Benchmarks | LLaMA-specific matmul benchmark |
| `bench_mlp_layer` | Benchmarks | MLP layer performance |
| `bench_attention_layer` | Benchmarks | Attention layer performance |
| `bench_attention_decode_cpp_compare` | Benchmarks | Compare with C++ reference |
| `bench_compare_report` | Benchmarks | Comparative benchmark reports |
| `gguf_inspect` | GGUF | Inspect GGUF file contents |
| `gguf` | GGUF | General GGUF operations |
| `gguf_hash` | GGUF | GGUF file hashing |
| `model_catalog` | GGUF | Model catalog browser |
| `resolve_tensor_names` | GGUF | Tensor name resolution |
| `embedding_probe` | Inference | Probe model embeddings |
| `min_infer_linear` | Inference | Minimal linear inference |
| `min_infer_mlp` | Inference | Minimal MLP inference |
| `min_infer_mlp_layer` | Inference | MLP layer with custom weights |
| `min_infer_attention_layer` | Inference | Minimal attention inference |
| `e2e_generate_tokens` | Inference | End-to-end token generation |
| `e2e_parity_harness` | Inference | Parity testing harness |
| `batched` | Applications | Batched inference |
| `idle` | Applications | Idle token cache management |
| `save_load_state` | Applications | Checkpoint save/load |
| `simple_chat` | Applications | Interactive chat |

**Total Rust-only examples**: **27** (4 root + 23 llama-rs)

---

## 6. Rust-Only Benchmark Programs

| Program | Location | Purpose |
|---|---|---|
| `bench_matmul` | `examples/benchmarks/bench_matmul.rs` | Context-path matmul |
| `bench_upstream_suite` | `examples/benchmarks/bench_upstream_suite.rs` | Upstream suite runner |
| `bench_llama_matmul` | `llama-rs/examples/benchmarks/bench_llama_matmul.rs` | LLaMA matmul |
| `bench_mlp_layer` | `llama-rs/examples/benchmarks/bench_mlp_layer.rs` | MLP layer |
| `bench_attention_layer` | `llama-rs/examples/benchmarks/bench_attention_layer.rs` | Attention layer |
| `bench_attention_decode_cpp_compare` | `llama-rs/examples/benchmarks/bench_attention_decode_cpp_compare.rs` | C++ comparison |
| `bench_compare_report` | `llama-rs/examples/benchmarks/bench_compare_report.rs` | Report generation |

**Total**: **7 Rust-only** benchmark programs + 1 upstream-mapped (`perf_metal`) = **8 total**.

Additionally, `e2e/bench_graphs.rs` contains **8 inline `#[test] #[ignore]` benchmark functions**
(full attention, linear attention, MLP, LM head, conv vs QKV, batch projections, layout prep).

---

## 7. Summary Statistics

### Upstream Coverage

| Category | Source Targets | Covered (✅) | Harness (⚠️) | Missing (❌) |
|---|---|---|---|---|
| Tests | 21 | 2 partial native | 21 in harness¹ | 0 |
| Examples | 17 | 16 | 0 | 1 (WASM) |
| Benchmarks | 2 | 1 (perf-metal) | 1 (quantize-perf) | 0 |

### Rust-Only Additions

| Category | Count |
|---|---|
| Tests | 208 |
| Examples | 27 |
| Benchmark programs | 7 |
| Benchmark functions | 8 (inline `#[ignore]`) |

### Grand Totals

| Metric | Count |
|---|---|
| Total `#[test]` functions | 231 (222 run by default + 1 harness + 8 bench) |
| Total example programs | 43 (16 upstream-mapped + 27 Rust-only) |
| Total benchmark programs | 8 (1 upstream-mapped + 7 Rust-only) |

---

## 8. Cross-Reference

- **Example parity details**: See [`EXAMPLE_PARITY_MATRIX.md`](./EXAMPLE_PARITY_MATRIX.md)
  for per-target result parity, performance snapshots, and asset blockers.
- **Conv/QKV comparison**: See [`docs/llama-rs/worklog/2026-04-13-conv-qkv-comparison.md`](../llama-rs/worklog/2026-04-13-conv-qkv-comparison.md)
  for detailed causal depthwise conv and QKV packing analysis.
- **Upstream suite harness**: See `tests/ggml_upstream_suite.rs` for configuration
  (env vars, target filtering, timeout).
