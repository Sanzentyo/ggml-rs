# Save-Load-State Implementation

## Summary

Implemented resumable generation sessions with binary checkpoint serialization
for the `save-load-state` parity matrix target.

## Architecture

### GenerationSession (session.rs)

Step-by-step token generation wrapping the same inference logic as
`generate_token_ids_from_model`, exposed via an iterator-like API.

```
GenerationSession
├── new(model, config) → Session
├── next_token() → Option<i32>       // prefill on first call, then decode
├── checkpoint() → GenerationCheckpoint
├── resume(model, backend, policy, checkpoint) → Session
├── generated_tokens() → &[i32]
├── all_tokens() → &[i32]
├── generated_count() → usize
└── is_finished() → bool
```

Execution modes:
- **TwoPhase** (Qwen3.5): Prefill captures KV/conv/SSM state, decode uses
  cached state. Checkpoints are performance-preserving.
- **FullReprocess** (Standard): All tokens reprocessed each step. Checkpoints
  save token IDs and position but state is recomputed on resume.

### GenerationCheckpoint (checkpoint.rs)

Serializable snapshot using a separate DTO layer decoupled from runtime types.

- Binary format: `LRCK` magic + postcard (compact, stable wire format)
- Model fingerprint validation: layer count, types, dims, vocab, rms_norm_eps
- KV cache trimming: only serializes populated portion (not full pre-allocated)
- Invariant validation: checks all structural invariants on load to prevent
  panics from malformed data

### API Design Decisions

1. **Separate DTO layer** — `CheckpointV1`, `LayerStateDto`, `ModelFingerprint`
   are serde types distinct from `GenerationState` / `LayerAttentionState`.
   Internal refactoring won't break serialized checkpoints.

2. **`resume()` takes minimal args** — only `backend` and `mixed_layer_policy`
   from the caller. All other parameters (prompt, max_new_tokens, EOS, pad)
   restored from the checkpoint. Prevents confusing misuse of a full config.

3. **postcard over bincode** — RUSTSEC-2025-0141 marks bincode as unmaintained.
   postcard 1.0 has a compact wire format and stable serde support.

4. **Cross-backend resume** — Allowed but documented as not guaranteed to
   produce identical tokens (FP precision differences in greedy argmax).

## Test Coverage

- 14 checkpoint tests (roundtrip, bad magic, fingerprint mismatches, KV trim,
  7 invariant validation tests)
- 4 session tests (step generation, checkpoint roundtrip, one-shot parity,
  zero-token edge case)
- save_load_state example binary with --verify mode

## Dependencies Added

- `serde = { version = "1.0", features = ["derive"] }`
- `postcard = { version = "1.0", features = ["use-std"] }`
