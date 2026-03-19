# 2026-03-18 End-to-End Inference Comparison Status

## Request coverage

- Goal requested: run end-to-end inference for all models, compare against original implementation (llama.cpp), verify output parity, then benchmark.
- Current checkpoint update:
  - token-id based true E2E generation path is implemented in `llama-rs` for transformer metadata (`general.architecture`) models that expose LLaMA-like layer tensor roles (safe API only),
  - GGUF tokenizer-backed text prompt path is implemented for `tokenizer.ggml.model=gpt2` models,
  - CPU/Metal runtime execution has been validated on:
    - `Llama-3-ELYZA-JP-8B-q4_k_m.gguf`,
    - `Qwen3-8B-Q4_K_M.gguf`,
    with matching generated token output between backends.
- Remaining gap for the original ask:
  - mixed-layer architecture support is currently fallback-only (`--mixed-layer-policy skip-unsupported-attention`),
  - strict mixed-layer execution (`qwen35` `attn_qkv` + `ssm_*` with full attention/SSM semantics) is still missing,
  - strict parity vs `llama.cpp` is now measurable with direct token-id extraction, but prompt-sensitive mismatches remain.

## A) Available measurable comparison today (decode path)

- Original implementation reference: `target/benchmarks/llama_cpp_baseline_{all,extra}.jsonl` decode rows (`n_prompt=0,n_gen=128`).
- Current llama-rs tuned preset reference: `target/benchmarks/review4_step2_balanced_cpu5_mtl7_blockgate_on.txt`.

| Model | Case | llama.cpp CPU ms/token | llama.cpp MTL0 ms/token | llama-rs proxy CPU ms/token | llama-rs proxy MTL0 ms/token | CPU proxy/cpp | MTL0 proxy/cpp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3-ELYZA-JP-8B-q4_k_m.gguf | `4096x32x8x1` | 24.106 | 22.830 | 28.294 | 28.043 | 1.174 | 1.228 |
| KaLM-Embedding-Gemma3-12B-2511.Q2_K.gguf | `3840x16x8x1` | 46.941 | 44.211 | 28.919 | 27.436 | 0.616 | 0.621 |
| InternVL3-8B-Q4_K_M.gguf | `3584x28x4x1` | 23.071 | 22.052 | 30.874 | 29.445 | 1.338 | 1.335 |
| Llama-3.1-Minitron-4B-Width-Base-Q4_0.gguf | `3072x32x8x1` | 13.406 | 13.248 | 14.559 | 16.136 | 1.086 | 1.218 |
| Qwen3.5-4B-Q4_K_M.gguf | `2560x16x4x1` | 17.488 | 20.896 | 11.882 | 12.936 | 0.679 | 0.619 |
| Qwen3-8B-Q4_K_M.gguf | `4096x32x8x1` | 24.768 | 24.187 | 24.971 | 25.328 | 1.008 | 1.047 |

## B) True end-to-end generation support matrix (prompt -> token generation)

| Model | llama.cpp E2E | llama-rs E2E | Output parity check vs llama.cpp | Benchmark status | Notes |
| --- | --- | --- | --- | --- | --- |
| Llama-3-ELYZA-JP-8B-q4_k_m.gguf | Supported | **Supported (token-id prompt path, greedy)** | CPU vs MTL0 output parity confirmed (`[1811]` from prompt `[1]`); llama.cpp strict parity pending | Partial | Evidence: `target/benchmarks/review4_e2e_tokenid_elyza_cpu_metal.{txt,md}` |
| Llama-3.1-Minitron-4B-Width-Base-Q4_0.gguf | Supported | **Implementation path available (LLaMA arch)** | Not run this checkpoint | Partial | Same code path as ELYZA (token-id prompt) |
| KaLM-Embedding-Gemma3-12B-2511.Q2_K.gguf | Supported | **Not supported yet** | N/A | Blocked | Architecture key not yet supported in true-E2E loader path |
| InternVL3-8B-Q4_K_M.gguf | Supported | **Not supported yet** | N/A | Blocked | Architecture key not yet supported in true-E2E loader path |
| Qwen3.5-4B-Q4_K_M.gguf | Supported | **Supported (fallback mode)** | CPU vs MTL0 output parity confirmed in fallback (`generated_token_ids=[220]` from prompt `[1]`); strict mode still fails on layer-role resolution | Partial | `--mixed-layer-policy skip-unsupported-attention` currently executes `mlp_only_layers=32` (no full attention layers yet); strict failure evidence remains for missing `attn_q` at layer `0` |
| Qwen3-8B-Q4_K_M.gguf | Supported | **Supported (token-id prompt path, greedy)** | strict llama.cpp token-id parity now runnable: prompt `"Hello"` matches on CPU/MTL0 (`[82]`), while prompt `"Hello "` still mismatches (`llama-rs [25]` vs `llama.cpp [17]`) even with prompt tokenization parity (`[9707,220]`) | Partial | Evidence: `target/benchmarks/review4_e2e_transformer_unblock_qwen3_8b_cpu_metal*.{txt,md}`, `target/benchmarks/review4_e2e_parity_harness_qwen3_8b_cpu_metal_v3_hello.txt`, `target/benchmarks/review4_e2e_parity_harness_qwen3_8b_cpu_metal_v3_hello_space.txt` |

## C) Blocking implementation gap for true E2E parity benchmark

- Newly implemented in this checkpoint:
  - full-model forward orchestration (embedding + transformer blocks + output projection) for transformer metadata models with LLaMA-like attention/MLP tensor roles,
  - mixed-layer fallback policy (`Strict` / `SkipUnsupportedAttention`) for per-layer planning,
  - GGUF tokenizer-backed text prompt path (`gpt2` byte-level BPE),
  - autoregressive token-id generation loop with greedy sampling,
  - active-window E2E execution (attention/MLP now consume `current_token_count` tokens each step instead of future-padded sequence),
  - strict llama.cpp parity harness hardening: direct token-id capture via helper binary (`llama-simple-token-ids`) to avoid re-tokenize boundary ambiguity.
- Remaining blockers for full ask completion:
  - tokenizer coverage beyond current `gpt2` model family,
  - strict mixed-block model execution support beyond fallback MLP-only path (`qwen35`, others),
  - strict llama.cpp parity closure across prompts/models (Qwen3 currently prompt-sensitive: `"Hello"` parity pass, `"Hello "` mismatch; prompt tokenization itself matches),
  - runtime optimization (backend reuse removed most per-layer backend init churn, but true persistent KV-cache/runtime reuse across generation steps is not yet in place).

## D) Conclusion at this checkpoint

- Original implementation (`llama.cpp`) end-to-end decode performance for all target models is available and measured.
- `llama-rs` now has a working true-E2E **token-id** generation path for transformer metadata models with LLaMA-like layer roles, with CPU/Metal execution evidence on ELYZA and Qwen3-8B.
- Strict parity tooling is now in place (direct llama.cpp token-id extraction) and confirmed for Qwen3 prompt `"Hello"` on CPU/Metal; next task is to close remaining prompt-sensitive generation mismatches (e.g. `"Hello "` where prompt token IDs already match), then expand parity+benchmark coverage across all target models.
