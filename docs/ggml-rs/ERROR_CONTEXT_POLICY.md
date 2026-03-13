# Error context policy (ggml-rs / llama-rs)

This document defines the required error-handling style for low-level and
FFI-adjacent code paths.

## Goals

- Preserve root-cause information (`source`) for debugging.
- Attach actionable context strings for where a failure occurred.
- Avoid success-path overhead from context construction.

## Required rules

1. Do not thread location info via dedicated `site` parameters.
2. Add context at the call site using `map_err` / `ok_or_else`.
3. Build context strings only on error paths.
4. Keep `source` errors intact (do not erase inner errors).

## Canonical patterns

Null pointer from FFI:

```rust
let raw = NonNull::new(raw_ptr)
    .ok_or_else(|| Error::null_pointer("ggml_new_tensor_2d"))?;
```

Integer conversion:

```rust
let ne0 = ne0
    .try_into_checked()
    .map_err(|source| Error::int_conversion("reshape_3d.ne0", source))?;
```

Wrap helper with extra context:

```rust
self.wrap_tensor(raw)
    .map_err(|error| error.with_context("ggml_mul_mat"))?;
```

## Why this style

- `ok_or_else` / `map_err` closures run only when an error happens.
- Context remains close to the failing operation.
- Resulting errors are easier to trace in logs and bug reports.

## Scope

- Mandatory for `ggml-rs` compute/GGUF layers.
- Mandatory for `llama-rs` layers that orchestrate ggml-backed operations.
- Continue applying this policy as new examples and runtime paths are added.
