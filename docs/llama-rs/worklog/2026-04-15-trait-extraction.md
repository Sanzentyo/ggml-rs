# AttentionStrategy Trait Extraction

## Summary

Refactored `generation.rs` to eliminate ~120 lines of duplicated per-layer
processing logic by introducing an `AttentionStrategy` trait with static
dispatch.

## Problem

Three code paths in `generation.rs` duplicated the identical per-layer body:

1. `full_reprocess_loop` — stateless, processes all tokens each step
2. `two_phase_loop` prefill — captures state while processing prompt
3. `two_phase_loop` decode — uses cached state for single-token steps

Each path had its own copy of:
```
if attention { normalize → dispatch → residual }
normalize → MLP → residual
```

Additionally, `TwoPhase` mode had crash paths:
- `Standard` attention triggered `unreachable!()` panic
- `max_new_tokens == 0` could sample past buffer bounds

## Solution

### AttentionStrategy trait

```rust
trait AttentionStrategy {
    fn process_attention(
        &mut self, layer_idx, attention, normalized_input,
        seq_len, rms_norm_eps, backend,
    ) -> Result<Vec<f32>, E2eError>;
}
```

Three implementations:
- `InferenceStrategy` — ZST, dispatches to `*_inference` functions
- `PrefillStrategy<'a>` — holds `&mut GenerationState`, calls `*_prefill`
- `DecodeStrategy<'a>` — holds `&mut GenerationState`, calls `*_decode_step`

### Shared helpers

- `process_all_layers()` — the unified per-layer body
- `sample_next_token()` — normalize output + greedy sampling
- `AttentionLayerPlan::norm_values()` — common accessor on the enum

### Bug fixes

- `TwoPhase + Standard` attention now returns `E2eError::UnsupportedTwoPhase`
- `TwoPhase + max_new_tokens == 0` returns empty output cleanly

## Verification

- 138 tests pass (136 existing + 2 new regression tests)
- Zero clippy warnings
- Floating-point parity preserved (same functions called in same order)

# Shared Projection/Normalization Helper Extraction

## Summary

Extracted shared projection and normalization helpers from `attention.rs` and
`linear_attention.rs`, eliminating ~80 lines of duplicated code between
core (prefill/inference) and decode_step functions.

## Changes

### attention.rs
- **`PreparedAttention` struct**: Groups Q, K, V, gate, and dimension metadata
- **`project_and_prepare_qkv()`**: Shared helper for projection → deinterleave →
  per-head RMS norm. Validates `hidden_features` divisibility upfront (catches
  malformed weights early instead of silently truncating via integer division).
- Both `qwen35_full_attention_core` and `decode_step` now call this helper, then
  apply RoPE with their own `position_offset`.

### linear_attention.rs
- **`LinearProjections` struct**: Groups qkv, z, alpha, beta + dimension metadata
- **`project_linear_inputs()`**: Shared 4-projection helper with divisibility
  validation.
- **`split_and_norm_qk()`**: Splits conv output into Q/K regions and L2-normalizes.
  Decode path borrows `v_raw` directly from conv output (avoids extra copy).

## Verification

- 138 tests pass, zero clippy warnings
- No behavioral changes — pure refactoring
