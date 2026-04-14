# ggml upstream example parity matrix

Status legend:

- `Done (direct)`: upstream C++ target and Rust counterpart both executed directly.
- `Done (synthetic)`: deterministic synthetic C++/Rust parity completed; real upstream run blocked by missing assets.
- `Blocked`: missing required assets for real upstream execution.

## Current matrix (`2026-03-15` parallel subagent pass)

| Upstream target | Rust counterpart | Status | Result parity | Perf snapshot (Rust/C++) | Real upstream blocker |
| --- | --- | --- | --- | --- | --- |
| `simple-ctx` | `examples/basics/simple_ctx.rs` | Done (direct) | checksum delta `0.0` | N/A | none |
| `simple-backend` | `examples/backends/backend_matmul.rs` | Done (direct) | checksum delta `0.0` (CPU/Metal) | N/A | none |
| `perf-metal` | `examples/backends/perf_metal.rs` | Done (direct) | checksum delta `0.0` | `~0.98x` (`n_op=512`, `n_iter=64`) | none |
| `gpt-2-ctx` | `llama-rs/examples/models/gpt2_ctx.rs` | Done (synthetic) | checksum delta `~2.18e-5` | `~9.60x` | `models/gpt-2-117M/ggml-model.bin` missing |
| `gpt-2-alloc` | `llama-rs/examples/models/gpt2_alloc.rs` | Done (synthetic) | checksum delta `~2.18e-5` | `~14.81x` | same |
| `gpt-2-backend` | `llama-rs/examples/models/gpt2_backend.rs` | Done (synthetic) | checksum delta `~2.18e-5` | `~12.21x` | same |
| `gpt-2-sched` | `llama-rs/examples/models/gpt2_sched.rs` | Done (synthetic) | checksum delta `~2.18e-5` | `~11.92x` | same |
| `gpt-2-batched` | `llama-rs/examples/models/gpt2_batched.rs` | Done (synthetic) | checksum delta `~2.18e-5` | `~15.84x` | same |
| `gpt-2-quantize` | `llama-rs/examples/models/gpt2_quantize.rs` | Done (synthetic) | checksum delta `~4.8e-7`, RMSE delta `~9e-11` | `~32.57x` | same |
| `gpt-j` | `examples/models/gptj_main_synth.rs` | Done (synthetic) | PASS | `~1.39x` | GPT-J model asset missing |
| `gpt-j-quantize` | `examples/models/gptj_quantize_synth.rs` | Done (synthetic) | PASS | `~12.11x` | GPT-J model asset missing |
| `magika` | `examples/models/magika_main_synth.rs` | Done (synthetic) | PASS | `~1.44x` | Magika model asset missing |
| `mnist-eval` | `examples/models/mnist_eval.rs` | Done (synthetic) | PASS | `~7.91x` | MNIST model/dataset missing |
| `mnist-train` | `examples/models/mnist_train.rs` | Done (synthetic) | PASS | `~56.55x` | MNIST model/dataset missing |
| `sam` | `examples/models/sam.rs` | Done (synthetic) | PASS | `~16.70x` | SAM model missing (example image present) |
| `yolov3-tiny` | `examples/models/yolov3_tiny.rs` | Done (synthetic) | PASS | `~36.21x` | YOLO model/input/labels missing |
## Additional examples (no upstream counterpart)

| Example | Purpose |
| --- | --- |
| `examples/backends/backend_ops.rs` | Multi-op backend graph (matmul + bias add) on CPU/Metal |
| `examples/basics/arithmetic_expr.rs` | Expression-style arithmetic API demo |
| `examples/benchmarks/bench_matmul.rs` | Context-path matmul benchmarking |
| `examples/benchmarks/bench_upstream_suite.rs` | Upstream benchmark suite runner |

See also: [`COVERAGE_TABLE.md`](./COVERAGE_TABLE.md) for comprehensive 3-bucket
coverage comparison (Native Rust / Upstream Harness / Missing).

## Artifacts

- Foundation direct parity:
  - `target/benchmarks/foundation_parity_report.md`
  - `target/benchmarks/foundation_parity_report.json`
- GPT-2 synthetic parity + upstream real-run blockers:
  - `target/benchmarks/gpt2_parity_summary.txt`
  - `target/benchmarks/gpt2_parity_summary.json`
  - `target/benchmarks/gpt2_upstream_attempts.log`
- GPT-J / Magika synthetic parity:
  - `target/benchmarks/gptj_magika_synth_parity_summary.txt`
  - `target/benchmarks/gptj_magika_synth_perf_summary.txt`
- Vision/MNIST synthetic parity + asset check:
  - `target/benchmarks/vision_mnist/parity_perf_summary.md`
  - `target/benchmarks/vision_mnist/parity_perf.json`
  - `target/benchmarks/vision_mnist/loopreuse_impact.md`
