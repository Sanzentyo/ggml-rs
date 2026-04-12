# 2026-04-12 Qwen3 parity follow-up

## Scope

- Continue the true-E2E parity closure work described in:
  - `docs/llama-rs/worklog/2026-03-18-e2e-inference-comparison.md`
  - `docs/llama-rs/worklog/2026-03-15-migration-log.md`
- Target the remaining prompt-sensitive Qwen3 mismatch
  (`"Hello "`: `llama-rs [25]` vs `llama.cpp [17]`) by removing
  implementation gaps that are visible from current docs and upstream model code.

## Implemented follow-up

- Added GGUF metadata parsing for:
  - `{arch}.attention.layer_norm_rms_epsilon`
  - `{arch}.attention.scale`
  - `{arch}.rope.scaling.original_context_length`
- Threaded those values into llama-rs attention/runtime configuration:
  - `AttentionInferenceConfig` now carries `attention_scale` and `rms_norm_eps`
  - RoPE config now preserves `original_context`
- Extended layer-name resolution with optional attention head norms:
  - `attn_q_norm`
  - `attn_k_norm`
- Extended decoded attention weights to load optional q/k norm tensors.
- Applied optional head-wise RMSNorm to Q/K projections before RoPE and score matmul.
- Updated E2E RMSNorm to stop using a fixed `1e-5` epsilon and instead use
  the GGUF metadata value when present.

## Why this pass

- Qwen3 upstream attention applies:
  - `q_norm(q_proj(x))`
  - `k_norm(k_proj(x))`
  before RoPE.
- Qwen3 config also uses `rms_norm_eps = 1e-6`.
- llama.cpp GGUF constants expose the same metadata/tensor surface:
  - `attention.layer_norm_rms_epsilon`
  - `attention.scale`
  - `rope.scaling.original_context_length`
  - `attn_q_norm` / `attn_k_norm`
- Before this pass, llama-rs E2E did not consume those pieces, so the
  remaining parity mismatch had an obvious implementation-side explanation.

## Validation

- `cargo fmt --all`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`

All passed on this checkpoint.

## Asset/bootstrap follow-up

- Prepared an external `llama.cpp` checkout at:
  - `/tmp/llama.cpp`
- Built the local comparison helpers used by the docs flow:
  - `/tmp/llama.cpp/build/bin/llama-bench`
  - `/tmp/llama.cpp/build/bin/llama-gguf`
  - `/tmp/llama.cpp/build/bin/llama-simple`
- Added a uv-managed asset fetch helper:
  - `scripts/fetch_model_assets.py`
- Updated `docs/llama-rs/KNOWLEDGE_BASE.md` so the canonical model bootstrap path is:
  - `uv run scripts/fetch_model_assets.py`
- The helper downloads into HF cache, then creates stable symlinks under:
  - `target/models/...`
  to avoid brittle `local_dir` symlink layouts.
- Extended the strict parity harness/tooling to support explicit prompt token IDs:
  - `llama-rs/examples/e2e_parity_harness.rs` now accepts `--prompt-tokens`
  - `llama-rs/tools/llama_simple_token_ids.cpp` now accepts `--prompt-tokens`
  This opens strict llama.cpp comparisons for models that are not yet reachable
  through the current text-tokenizer path.

## Strict parity rerun

- Rebuilt local `ggml` shared libraries under:
  - `vendor/ggml/build`
- Re-ran `llama-rs/examples/e2e_parity_harness.rs` with:
  - `target/models/qwen3_8b_q4_k_m/Qwen3-8B-Q4_K_M.gguf`
  - `/tmp/llama.cpp/build/bin/llama-simple`
  - `/tmp/llama.cpp/build/bin/llama-simple-token-ids`
- New evidence artifacts:
  - `target/benchmarks/review4_e2e_parity_harness_qwen3_8b_cpu_metal_v4_hello.txt`
  - `target/benchmarks/review4_e2e_parity_harness_qwen3_8b_cpu_metal_v4_hello_space.txt`

## Result

- Prompt `"Hello"` remains parity-clean:
  - CPU: `llama-rs [82]` vs `llama.cpp [82]`
  - MTL0: `llama-rs [82]` vs `llama.cpp [82]`
- Prompt `"Hello "` mismatch is now closed:
  - prompt token IDs still match on both sides: `[9707, 220]`
  - CPU: `llama-rs [17]` vs `llama.cpp [17]`
  - MTL0: `llama-rs [17]` vs `llama.cpp [17]`
- This confirms the q/k head-norm + metadata-epsilon/scale/original-context
  follow-up was sufficient to eliminate the remaining prompt-sensitive Qwen3
  strict token-id parity gap covered by this harness.

## Post-Qwen3 expansion

- ELYZA strict parity is now measurable through the new `--prompt-tokens` path:
  - artifact:
    - `target/benchmarks/review4_e2e_parity_harness_elyza_cpu_metal_v1_prompt1.txt`
  - prompt token IDs matched: `[1]`
  - `llama-rs` generated `[29295]` on both CPU and MTL0
  - `llama.cpp` generated `[1811]` on CPU but `[29295]` on MTL0
  - implication: this case is currently not a pure llama-rs mismatch; the
    upstream comparison surface itself splits by backend for the same prompt.
- Minitron strict parity no longer sits in a vague "not yet run" state:
  - first rerun artifact:
    - `target/benchmarks/review4_e2e_parity_harness_minitron_cpu_metal_v1_prompt1.txt`
  - initial blocker was:
    - `InvalidRopeDimensions { rope_dimensions: 128, head_dimension: 96 }`
  - model metadata exposed the real issue:
    - `embedding_length=3072`
    - `attention.head_count=32`
    - `attention.key_length=value_length=128`
    - `rope.dimension_count=128`
  - fixed by threading metadata-derived projection head width through:
    - `TransformerMetadata` / `LlamaModelMetadata`
    - `resolve_llama_layer_dimensions`
    - `AttentionLayout::from_projection_dimensions`
  - post-fix strict parity artifact:
    - `target/benchmarks/review4_e2e_parity_harness_minitron_cpu_metal_v2_prompt1.txt`
  - result after the fix:
    - prompt token IDs match: `[1]`
    - CPU: `llama-rs [726]` vs `llama.cpp [726]`
    - MTL0: `llama-rs [726]` vs `llama.cpp [726]`
- Qwen3.5 strict path is no longer blocked at layer-role resolution:
  - artifact:
    - `target/benchmarks/review4_e2e_parity_harness_qwen35_cpu_metal_v2_prompt1.txt`
  - follow-up implementation added:
    - metadata parsing for `qwen35.full_attention_interval` and `qwen35.ssm.*`,
    - qwen35-specific layer planning for both full-attention blocks and
      `attn_qkv + ssm_*` linear-delta blocks,
    - direct Rust execution for qwen35 full-attention gating (`attn_q` split into
      `q + gate`) and linear-delta recurrence,
    - strict planning now reaches `attention_layers=32`, `mlp_only_layers=0`.
  - current result is still mismatch on prompt `[1]`:
    - CPU: `llama-rs [198]` vs `llama.cpp [5328]`
    - MTL0: `llama-rs [198]` vs `llama.cpp [5328]`
  - implication:
    - the remaining blocker has moved from naming/planning into qwen35 block
      math parity itself (delta-net / fused-q-gate / exact operator semantics).

## Progress checkpoint after review_3 closure

- The branch work that was interleaved with this parity pass is now complete:
  - `ggml-rs` review_1 / review_3 refactor items landed on `exp/oh-my`
  - typed tensor migration is complete:
    - `Tensor<'ctx, T>`
    - `DynTensor<'ctx>`
    - typed `TensorExpr<'ctx, T>`
  - GGUF ergonomics landed:
    - `AsRef<str>` key APIs
    - `TryFromGgufValue`
    - `kv_value_as::<T>()`
  - additional validation/test coverage landed:
    - backend compute path tests
    - ND tensor tests
    - error-path / boundary tests

## Validation checkpoint on `exp/oh-my`

- `cargo fmt --all`
- `cargo clippy --workspace --all-targets`
- `cargo test --workspace`
- `cargo test --features link-system`
- CPU perf gate:
  - command:
    - `cargo run --example bench_matmul --features link-system -- cpu -n 10`
  - current result:
    - `[CPU] matmul 256x256 · 256x256 avg=0.256 ms, checksum=956.435547`
- Metal backend path is covered by the new link-system tests, but the current
  local benchmark invocation still needs the matching `ggml-metal` shared-lib
  link path before a clean bench rerun can be recorded.

## Qwen3.5 parity narrowing after upstream comparison

- Compared the Rust qwen3.5 implementation against upstream references in:
  - `/tmp/llama.cpp/src/models/qwen35.cpp`
  - `/tmp/llama.cpp/src/models/delta-net-base.cpp`
  - `/tmp/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu`
  - `/tmp/llama.cpp/vendor/ggml/src/ggml-cpu/ops.cpp`
  - `/tmp/llama.cpp/vendor/ggml/src/ggml-cuda/ssm-scan.cu`
- Current conclusion:
  - the Rust gated delta-net recurrence is not the leading mismatch candidate
    for the scalar-gated Qwen3.5 path
  - rewriting the recurrence into the CUDA-style update order is expected to be
    algebraically equivalent and is unlikely to flip the generated token by
    itself
- The leading remaining suspects have therefore narrowed to pre-recurrence tensor
  preparation:
  - `causal_depthwise_conv` layout / boundary semantics
  - Q/K/V packing, group repetition, or head reshaping before the delta-net block

## Head-group mapping comparison (causal_depthwise_conv + QKV packing)

### Investigation scope

Compared llama-rs vs llama.cpp on all tensor-preparation steps in the qwen3.5
linear-attention block:
- post-conv channel ordering
- split points for `q`, `k`, `v`
- head/group packing layout before L2 norm and recurrence

### causal_depthwise_conv — verified correct

- Conv weight layout: `weight[channel * kernel_size + tap]` matches ggml's
  `src1` layout of `[d_conv, d_inner]` (fastest dim = tap).
- Boundary semantics for initial prompt (zero conv states): llama-rs zero-pads
  missing taps, which is equivalent to ggml_ssm_conv prepending zero conv_states.
- SiLU is applied inside the conv function in llama-rs, separately after
  `ggml_ssm_conv` in llama.cpp — mathematically equivalent.

### QKV split — verified correct

- Per token, conv output is split as:
  - Q: first `num_k_heads * head_k_dim` elements
  - K: next `num_k_heads * head_k_dim` elements
  - V: remaining `num_v_heads * head_v_dim` elements
- This matches llama.cpp `ggml_view_4d` offsets in `build_layer_attn_linear`.

### L2 norm — verified correct

- `per_head_l2_norm` normalizes along `head_dimension` per head,
  matching `ggml_l2_norm` along dimension 0.

### Head-group mapping — **BUG FOUND AND FIXED**

- Qwen3.5 8B has `num_k_heads=16`, `num_v_heads=32`.
  Q/K are projected with 16 heads; V has 32 heads.
  Q/K must be "repeated" to 32 for the per-head delta-net recurrence.

- **llama.cpp** uses `ggml_repeat_4d` which tiles block-by-block:
  ```
  [g0,g1,...,g15, g0,g1,...,g15]
  ```
  (Implementation: outer loop over repeat blocks, inner loop over source elements.)

- **llama-rs (before fix)** used `head / repeat_factor`:
  ```
  [g0,g0, g1,g1, ..., g15,g15]
  ```
  (Interleaved: consecutive V-heads shared the same Q/K group.)

- **Fix**: changed to `head % attention.group_count`:
  ```
  [g0,g1,...,g15, g0,g1,...,g15]
  ```
  This produces the same tiled pattern as `ggml_repeat_4d`.

- Added shape invariant assertions:
  - `time_step_rank % group_count == 0`
  - `inner_size == time_step_rank * state_size`

- Added regression test `qwen35_linear_head_group_mapping_is_tiled` covering:
  - Successful execution with `group_count < time_step_rank`
  - Error on indivisible `group_count`

### Validation

- `cargo fmt --all`
- `cargo clippy --workspace --all-targets`
- `cargo test --workspace`

All passed.

### Next

- Strict parity rerun on Qwen3.5 with the head-mapping fix.
- If parity closes, update INTRODUCTION.md status to resolved.
- If residual delta remains, investigate non-initial-prompt conv state handling.
