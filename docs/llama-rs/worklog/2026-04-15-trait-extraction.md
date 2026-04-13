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
