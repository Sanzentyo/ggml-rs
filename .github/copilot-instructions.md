# Rust implementation policy (llama-rs / ggml-rs)

When modifying Rust code in this repository, follow these defaults:

- Prefer ADT-centered APIs over long procedural `run_*` entrypoints.
- Build execution flows with typed plans/builders and type-state where required fields exist.
- Prefer static dispatch (`enum`/generics/traits) over dynamic dispatch when behavior is known at compile time.
- Keep unsafe surface minimal; do not add `unsafe` unless strictly necessary and justified.
- Keep modules focused: split oversized files into coherent submodules.
- Preserve safe API ergonomics from `src/lib.rs` re-exports; avoid leaking FFI details.
- After behavior or structure changes, always run:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
- For performance-sensitive paths (decode/bench), re-run CPU/Metal runtime checks and compare against recent benchmark artifacts to avoid regressions.

Use `rust-best-practices` skill guidance as the coding baseline.
