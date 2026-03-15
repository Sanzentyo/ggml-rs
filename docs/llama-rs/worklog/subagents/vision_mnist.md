# vision/mnist batch log

- scope: `mnist-eval`, `mnist-train`, `sam`, `yolov3-tiny`
- owner: parallel subagent

## 2026-03-15 Step 1 (inspection)
- reviewed upstream scope files: `mnist-eval.cpp`, `mnist-train.cpp`, `sam.cpp`, `yolov3-tiny.cpp`
- audited safe `ggml-rs` API surface in `src/compute.rs` and identified available ops (`mul_mat`, `add`, `soft_max`, tensor/view/reshape)
- confirmed full upstream assets are not guaranteed; planned deterministic synthetic parity path for MNIST eval/train, SAM mask summary, and YOLO detection summary

## 2026-03-15 Step 2 (implementation)
- added deterministic synthetic mode + check flags to upstream C++ targets:
  - `vendor/ggml/examples/mnist/mnist-eval.cpp`
  - `vendor/ggml/examples/mnist/mnist-train.cpp`
  - `vendor/ggml/examples/sam/sam.cpp`
  - `vendor/ggml/examples/yolo/yolov3-tiny.cpp`
- added idiomatic Rust example counterparts using safe `ggml-rs` APIs:
  - `examples/mnist_eval.rs`
  - `examples/mnist_train.rs`
  - `examples/sam.rs`
  - `examples/yolov3_tiny.rs`
- wired new examples in `Cargo.toml` with `required-features = ["link-system"]`

## 2026-03-15 Step 3 (validation + parity/perf)
- rebuilt upstream C++ targets with lock: `mnist-eval`, `mnist-train`, `sam`, `yolov3-tiny`
- validated Rust side with lock:
  - `cargo fmt --all`
  - `cargo clippy --features link-system --example mnist_eval --example mnist_train --example sam --example yolov3_tiny -- -D warnings`
  - `cargo test --features link-system --test ggml_simple_ctx --test ggml_test_cont --test ggml_tensor_ops`
- executed synthetic parity/perf runs (bench lock) for C++ and Rust counterparts and generated artifacts under `target/benchmarks/vision_mnist/`
- recorded missing real assets and blocked real-asset upstream commands; synthetic mode used as deterministic fallback
