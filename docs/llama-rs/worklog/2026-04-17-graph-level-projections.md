# Graph-Level Attention Projections

## Date: 2026-04-17

## Summary

Replaced host-side scalar dot-product projections with ggml `mul_mat` compute
graphs for prefill/inference paths in both full and linear attention modules.

## Changes

### Full Attention (`attention.rs`)
- `project_qkv_graph()`: Batches 3 matmuls (Q, K, V) sharing input X into a
  single ggml graph. Used by `project_and_prepare_qkv()` when backend is available.
- `recommended_qkv_projection_memory()`: Sums per-matmul memory estimates + 4MB slack.
- Output projection in `qwen35_full_attention_core()` now uses `project_sequence_graph()`.
- `project_and_prepare_qkv()` dispatches graph vs host based on `Option<&Backend>`.
- Backend threaded through: `qwen35_full_attention_inference`, `_prefill`, `_core`.
- Decode path (`_decode_step`) passes `None` — stays host-side to avoid graph overhead.

### Linear Attention (`linear_attention.rs`)
- `project_linear_inputs_graph()`: Batches 4 matmuls (QKV, gate, alpha, beta) into
  a single ggml graph.
- `recommended_linear_projection_memory()`: Sums 4 matmul estimates + slack.
- `project_linear_inputs()` dispatches graph vs host based on `Option<&Backend>`.
- Output projection uses `project_sequence_graph()`.
- Backend threaded through: `qwen35_linear_attention_inference`, `_prefill`, `_core`.
- Decode path unchanged — stays host-side.

### Shared (`tensor_ops.rs`)
- `project_sequence_graph()`: General-purpose single-matmul graph function.
- `recommended_single_projection_memory()`: Memory estimate for a single matmul.
- `PROJECTION_SLACK_BYTES`: Shared 4MB slack constant.
- Parity test: `project_sequence_graph_matches_host_projection` confirms
  host vs graph output matches within 1e-5.

### Generation (`generation.rs`)
- `InferenceStrategy`: Passes `backend` to full and linear attention inference.
- `PrefillStrategy`: Passes `backend` to full and linear attention prefill.
- `DecodeStrategy`: Unchanged — decode functions don't take backend.

## Design Decisions

1. **Decode stays host-side**: For seq_len=1, ggml graph overhead (context
   allocation, tensor creation, backend sync) exceeds benefit of BLAS kernels
   for small matmuls.
2. **Single graph per function**: Each QKV/projection call creates its own
   ggml context and graph. No cross-call context sharing — simpler lifetime
   management, context is dropped after results are read back.
3. **Backend as parameter, not struct field**: `Backend` doesn't implement
   `Clone`, so plan structs stay copyable. Backend is threaded through as `&Backend`.
4. **Conservative memory estimates**: Input tensor counted N times (once per
   matmul estimate) — overestimates but never underestimates.

## Test Results

- 192 tests pass (was 191, +1 parity test)
- Zero clippy warnings (aside from expected `too_many_arguments`)
- Host vs graph parity verified within 1e-5 tolerance
