//! Microbenchmarks for fused e2e compute graphs (CPU vs Metal).
//!
//! These benchmarks measure the wall-clock time of each fused graph type at
//! two scales: a small "test" size for quick iteration, and a larger "realistic"
//! size matching Qwen3.5 (0.6B) model dimensions.
//!
//! Run with:
//! ```sh
//! GGML_RS_LIB_DIR=... DYLD_FALLBACK_LIBRARY_PATH=... \
//!   cargo test --workspace --features link-system --release \
//!   -- bench_e2e_graphs --nocapture --ignored
//! ```
//!
//! All benchmarks are `#[ignore]`d to keep normal `cargo test` fast.

#[cfg(test)]
mod helpers;
#[cfg(test)]
mod inference;
#[cfg(test)]
mod lm_head;
#[cfg(test)]
mod micro;
