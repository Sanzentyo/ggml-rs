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
  - mixed-layer architecture support is no longer resolver-blocked,
  - qwen35 strict execution now reaches full layer planning/execution, but token
    parity is still missing,
  - strict parity vs `llama.cpp` is now measurable with direct token-id extraction, but broader model/prompt coverage still remains.

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
| Llama-3-ELYZA-JP-8B-q4_k_m.gguf | Supported | **Supported (token-id prompt path, greedy)** | strict parity is now measurable with `--prompt-tokens`: prompt `[1]` gives `llama-rs=[29295]` on CPU/MTL0, while `llama.cpp` splits by backend (`CPU=[1811]`, `MTL0=[29295]`) | Partial | Evidence: `target/benchmarks/review4_e2e_tokenid_elyza_cpu_metal.{txt,md}`, `target/benchmarks/review4_e2e_parity_harness_elyza_cpu_metal_v1_prompt1.txt` |
| Llama-3.1-Minitron-4B-Width-Base-Q4_0.gguf | Supported | **Supported (token-id prompt path, greedy)** | strict llama.cpp token-id parity confirmed on CPU/MTL0 for prompt `[1]` (`[726]`) after threading metadata-derived projection head width (`attention.key_length=128`) into the attention config | Partial | Evidence: `target/benchmarks/review4_e2e_parity_harness_minitron_cpu_metal_v2_prompt1.txt` |
| KaLM-Embedding-Gemma3-12B-2511.Q2_K.gguf | Supported | **Not supported yet** | N/A | Blocked | Architecture key not yet supported in true-E2E loader path |
| InternVL3-8B-Q4_K_M.gguf | Supported | **Not supported yet** | N/A | Blocked | Architecture key not yet supported in true-E2E loader path |
| Qwen3.5-4B-Q4_K_M.gguf | Supported | **Supported (fallback + strict WIP)** | fallback remains parity-clean at `[220]`; strict prompt `[1]` now executes all 32 attention layers on CPU/MTL0 but currently mismatches (`llama-rs=[198]`, `llama.cpp=[5328]`) | Partial | strict planning/execution is now implemented for both full-attention and `attn_qkv + ssm_*` layers; remaining gap is qwen35 math parity rather than missing tensor-role resolution |
| Qwen3-8B-Q4_K_M.gguf | Supported | **Supported (token-id prompt path, greedy)** | strict llama.cpp token-id parity confirmed on CPU/MTL0 for prompt `"Hello"` (`[82]`) and prompt `"Hello "` (`[17]`); prompt tokenization also matches (`[9707,220]`) | Partial | Evidence: `target/benchmarks/review4_e2e_transformer_unblock_qwen3_8b_cpu_metal*.{txt,md}`, `target/benchmarks/review4_e2e_parity_harness_qwen3_8b_cpu_metal_v4_hello.txt`, `target/benchmarks/review4_e2e_parity_harness_qwen3_8b_cpu_metal_v4_hello_space.txt` |

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
  - strict mixed-block math parity beyond the current qwen35 first-pass implementation,
  - strict llama.cpp parity expansion across more prompts/models beyond the current Qwen3 smoke cases,
  - runtime optimization (backend reuse removed most per-layer backend init churn, but true persistent KV-cache/runtime reuse across generation steps is not yet in place).

## D) Conclusion at this checkpoint

- Original implementation (`llama.cpp`) end-to-end decode performance for all target models is available and measured.
- `llama-rs` now has a working true-E2E **token-id** generation path for transformer metadata models with LLaMA-like layer roles, with CPU/Metal execution evidence on ELYZA and Qwen3-8B.
- Strict parity tooling now also supports explicit prompt token IDs, which made two follow-ups concrete:
  - ELYZA exposes an upstream backend split on prompt `[1]` (`llama.cpp` CPU vs MTL0 differ),
  - Minitron is now unblocked by metadata-derived projection head width support and matches `llama.cpp` on prompt `[1]`.
- Qwen3.5 is no longer stuck at `MissingLayerTensor attn_q`; strict mode now
  executes its full 32-layer hybrid stack, but the generated token still
  mismatches on prompt `[1]`, so the open problem is exact block math parity.
- Next task is to continue parity+benchmark expansion across additional prompts and models, while keeping an eye on backend-sensitive upstream cases such as ELYZA prompt `[1]`.
