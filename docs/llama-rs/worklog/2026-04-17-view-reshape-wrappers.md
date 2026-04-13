# Safe view/reshape wrappers for ggml-rs

**Date**: 2026-04-17
**Status**: DONE

## Summary

Added missing safe API wrappers to ggml-rs for zero-copy tensor views and
reshaping, completing the set needed for graph-level QKV split optimization
in llama-rs.

## New wrappers

| Function      | ggml C equivalent    | Notes                          |
|---------------|----------------------|--------------------------------|
| `view_3d`     | `ggml_view_3d`       | 3D strided view with offset    |
| `view_4d`     | `ggml_view_4d`       | 4D strided view with offset    |
| `reshape_1d`  | `ggml_reshape_1d`    | Flatten to 1D                  |
| `reshape_4d`  | `ggml_reshape_4d`    | Reshape to 4D                  |

## Validation backfill

All existing view/reshape wrappers (`view_1d`, `view_2d`, `reshape_2d`,
`reshape_3d`) gained Rust-side validation that was previously missing:

- **Reshape validation** (`validate_reshape_source`):
  - Checks `ggml_is_contiguous()` — non-contiguous tensors (from
    transpose/permute) must call `cont()` first
  - Verifies target element count == source element count
  - Returns `Error::NotContiguous` or `Error::LengthMismatch`

- **View extent validation** (`validate_view_extent`):
  - Computes max addressed byte with overflow-checked arithmetic
  - Verifies extent ≤ source `nbytes()`
  - Returns `Error::ViewOutOfBounds` or `Error::Overflow`

## Error variants added

```rust
Error::NotContiguous
Error::ViewOutOfBounds { offset: usize, extent: usize, source_size: usize }
```

## Tests added (13)

- `reshape_1d_flattens_2d` — basic reshape_1d
- `reshape_4d_smoke` — basic reshape_4d
- `reshape_element_count_mismatch_returns_error` — invalid element count
- `reshape_non_contiguous_returns_error` — transpose then reshape
- `view_2d_strided_with_offset` — non-trivial stride and offset
- `view_3d_smoke` — contiguous 3D view
- `view_4d_smoke` — contiguous 4D view
- `view_aliases_base_tensor` — write through view modifies base
- `reshape_aliases_base_tensor` — write through reshape modifies base
- `view_1d_out_of_bounds_returns_error` — extent and offset OOB
- `view_3d_out_of_bounds_returns_error` — 3D extent exceeds source

Total: 191 tests pass, zero clippy warnings.

## Future work

- Use `view_3d`/`view_4d` in llama-rs to push QKV splits into the ggml
  compute graph (requires pipeline restructuring to keep data in ggml tensors
  longer, eliminating `copy_from_slice` host-side copies)
