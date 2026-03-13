# Upstream suite operations runbook

This runbook documents practical operation of the upstream-suite harnesses:

- `tests/ggml_upstream_suite.rs`
- `examples/bench_upstream_suite.rs`

## Goals

- Run full upstream targets when needed.
- Allow narrow target scopes for quick iteration.
- Keep behavior reproducible in CI and local environments.

## Core commands

```bash
# Full test-target suite (ignored test)
cargo test --features link-system --test ggml_upstream_suite -- --ignored

# Bench-target suite
cargo run --example bench_upstream_suite --features link-system
```

## Common environment controls

- `GGML_UPSTREAM_BUILD_DIR=<path>`  
  Override upstream build root (default: `target/vendor/ggml/build`).

- `GGML_UPSTREAM_SKIP_BUILD=1`  
  Skip `cmake --build` and run already-built binaries.

- `GGML_UPSTREAM_BUILD_JOBS=<n>`  
  Set explicit parallel build jobs for `cmake --parallel`.

- `GGML_UPSTREAM_LIST_ONLY=1`  
  Print selected targets and exit without build/run.

- `GGML_UPSTREAM_EXCLUDE_TARGETS=a,b,c`  
  Comma-separated targets removed from the selected set.

- `GGML_UPSTREAM_SUMMARY_PATH=<file>`  
  Write end-of-run summary (selected targets, pass/fail counts, elapsed time).

## Test-suite controls

- `GGML_UPSTREAM_TEST_TARGETS=test-cont,test-opt`  
  Explicit target list for `ggml_upstream_suite`.

- `GGML_UPSTREAM_KEEP_GOING=1`  
  Continue after per-target failures and report aggregate summary.

## Bench-suite controls

- `GGML_UPSTREAM_BENCH_TARGETS=test-backend-ops,test-quantize-perf`  
  Explicit target list for `bench_upstream_suite`.

- CLI flags:
  - `--skip-build`
  - `--list-only`
  - `--keep-going`
  - `--fail-fast`

- CLI target args override env/default target selection:

```bash
cargo run --example bench_upstream_suite --features link-system -- test-backend-ops
```

## Practical workflows

Quick local smoke:

```bash
GGML_UPSTREAM_TEST_TARGETS=test-cont \
cargo test --features link-system --test ggml_upstream_suite -- --ignored
```

List selected targets before execution:

```bash
GGML_UPSTREAM_LIST_ONLY=1 \
GGML_UPSTREAM_TEST_TARGETS=test-cont,test-opt \
cargo test --features link-system --test ggml_upstream_suite -- --ignored --nocapture
```

Collect bench summary to a file:

```bash
GGML_UPSTREAM_SUMMARY_PATH=target/upstream-bench-summary.txt \
GGML_UPSTREAM_BENCH_TARGETS=test-backend-ops \
cargo run --example bench_upstream_suite --features link-system
```
