# ggml-rs

Safe Rust bindings for [ggml](https://github.com/ggml-org/ggml), plus a
higher-level `llama-rs` inference crate built on top.

## Workspace crates

| Crate | Description |
|---|---|
| `ggml-rs` | Safe wrapper around ggml's C API — context, tensor, graph, backend, GGUF I/O |
| `llama-rs` | Model-level inference on GGUF models — tokenizer, metadata, E2E generation |

## Quick start

### Prerequisites

Initialize the ggml submodule and build the shared libraries:

```bash
git submodule update --init --recursive

cmake -S vendor/ggml -B vendor/ggml/build \
  -DGGML_METAL=ON -DGGML_CPU=ON \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/ggml/build -j
```

### Build and test

```bash
export GGML_RS_LIB_DIR=$(pwd)/vendor/ggml/build/src

# Build everything
cargo build --workspace --all-targets

# Run tests
DYLD_FALLBACK_LIBRARY_PATH=$GGML_RS_LIB_DIR cargo test --workspace
```

### Run examples

```bash
export GGML_RS_LIB_DIR=$(pwd)/vendor/ggml/build/src
export DYLD_FALLBACK_LIBRARY_PATH=$GGML_RS_LIB_DIR

# ggml-rs: simple context matmul
cargo run --example simple_ctx --features link-system

# ggml-rs: backend compute (CPU/Metal)
cargo run --example backend_matmul --features link-system

# ggml-rs: multi-op backend graph (matmul + bias)
cargo run --example backend_ops --features link-system -- cpu

# ggml-rs: expression-style arithmetic
cargo run --example arithmetic_expr --features link-system

# llama-rs: generate tokens from a GGUF model
cargo run -p llama-rs --example e2e_generate_tokens --features link-system -- \
  --model path/to/model.gguf --prompt "Hello" --max-new-tokens 32
```

## ggml-rs API overview

```rust
use ggml_rs::prelude::*;
```

### Scoped context helpers

```rust
let result = with_context(Bytes::new(64 * 1024 * 1024), |ctx| {
    let a = ctx.new_tensor_2d::<f32>(Shape2D::new(64, 64))?;
    let b = ctx.new_tensor_2d::<f32>(Shape2D::new(64, 64))?;
    let c = ctx.mul_mat(&a, &b)?;
    // tensors cannot escape this scope
    Ok(c.shape_nd()?)
})?;
```

### Typed tensors and shapes

```rust
// Generic construction: 1D through 4D
let t1 = ctx.new_tensor_1d::<f32>(Length::new(128))?;
let t3 = ctx.new_tensor_3d::<f32>(Shape3D::new(64, 32, 8))?;

// Runtime-typed (any ggml type including quantized)
let dyn_t = ctx.new_tensor::<2>(Type::Q4_K, Dims::new([64, 64]))?;

// Introspection
let rank = t3.rank()?;        // 3
let dims = t3.shape_nd()?;    // [64, 32, 8]
```

### Expression API

```rust
let expr = ((ctx.expr(a) + ctx.expr(b))? * ctx.expr(c))? / ctx.expr(d);
let result = expr?.into_tensor();
```

### Backend compute (CPU / Metal)

```rust
let backend = Backend::new(BackendKind::Cpu)?;

// No-alloc context: tensors are graph placeholders until backend allocates storage.
let ctx = Context::new_no_alloc_bytes(Bytes::new(64 * 1024 * 1024))?;
let a = ctx.new_tensor_2d::<f32>(shape_a)?;
let b = ctx.new_tensor_2d::<f32>(shape_b)?;
let c = ctx.mul_mat(&a, &b)?;

let mut graph = ctx.new_graph()?;
graph.build_forward_expand(&c);

// Allocate backend storage, transfer data, compute, read results.
let _buffer = ctx.allocate_tensors(&backend)?;
a.write_data_backend(&input_a)?;
b.write_data_backend(&input_b)?;
backend.compute(&mut graph)?;
let output: Vec<f32> = c.read_data_backend()?;
```

### GGUF file inspection

```rust
let file = GgufFile::open("model.gguf")?;
let kv_count = file.kv_count()?;
for i in 0..kv_count {
    let key = file.kv_key(i)?;
    let value = file.kv_value(i)?;
    println!("{key}: {value:?}");
}
let tensor_count = file.tensor_count()?;
for i in 0..tensor_count {
    let info = file.tensor_info(i)?;
    println!("{}: {:?}", info.name, info.ggml_type);
}
```

### Type-safe GGUF value extraction

```rust
// Extract typed values using TryFromGgufValue
let vocab_size: Option<u32> = file.kv_value_as("llama.vocab_size")?;
let head_count: Option<u32> = file.kv_value_as("llama.attention.head_count")?;
```

### Generic GGUF decode

```rust
// Decode quantized tensor data to f32
let values: Vec<f32> = decode_tensor_data_to(tensor_type, raw_data, element_count)?;
```

## llama-rs overview

`llama-rs` provides architecture-aware inference built on `ggml-rs`:

- **Tokenizer**: BPE tokenization from GGUF vocabulary
- **Metadata**: Typed extraction of model hyperparameters (`TransformerMetadata`)
- **E2E inference**: Full transformer generation (Qwen3.5, standard attention, MLP)
  - Full attention with NeoX RoPE (position offset for autoregressive decode)
  - Linear attention with causal depthwise convolution + delta-net recurrence
  - Two-phase generation: prefill all prompt tokens, then decode one-at-a-time
  - Verified token-ID parity with llama.cpp
- **Type system**: `Type` covers all 32+ ggml tensor types (quantized, float, integer)
- **Modular e2e**: 13 focused submodules (attention, linear_attention, state, generation, etc.)

```bash
# Parity harness — verify llama-rs matches llama.cpp output
cargo run -p llama-rs --example e2e_parity_harness --features link-system -- \
  --model target/models/qwen3_5_4b_q4_k_m/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt-tokens 3 --max-new-tokens 3 \
  --llama-simple-bin /tmp/llama.cpp/build/bin/llama-simple \
  --llama-token-ids-bin /tmp/llama.cpp/build/bin/llama-simple-token-ids \
  --skip-metal
```

## Linking configuration

| Environment variable | Purpose |
|---|---|
| `GGML_RS_LIB_DIR` | Single directory for ggml shared libraries |
| `GGML_RS_LIB_DIRS` | Colon-separated list of library directories |
| `GGML_RS_LIBS` | Comma-separated list of library names to link |
| `GGML_RS_GGML_INCLUDE_DIR` | Override header search path for bindgen |

## Documentation

| Document | Location |
|---|---|
| ggml-rs API details | `docs/ggml-rs/README.md` |
| Error handling policy | `docs/ggml-rs/ERROR_CONTEXT_POLICY.md` |
| Example parity matrix | `docs/ggml-rs/EXAMPLE_PARITY_MATRIX.md` |
| Test suite operations | `docs/ggml-rs/SUITE_OPERATIONS.md` |
| llama-rs knowledge base | `docs/llama-rs/KNOWLEDGE_BASE.md` |
| llama-rs parity matrix | `docs/llama-rs/PARITY_MATRIX.md` |
| Development worklog | `docs/llama-rs/WORKLOG.md` |
| Conv & QKV comparison | `docs/llama-rs/worklog/2026-04-13-conv-qkv-comparison.md` |

## License

See individual crate files for license terms.
