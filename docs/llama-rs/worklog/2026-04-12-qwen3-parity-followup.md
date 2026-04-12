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
