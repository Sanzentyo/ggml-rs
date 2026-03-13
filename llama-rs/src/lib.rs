pub mod backend;
pub mod bench;
pub mod gguf;
pub mod gguf_hash;
pub mod simple;
pub mod smoke;

pub use backend::LlamaBackend;
pub use bench::{BenchError, MatmulBenchConfig, MatmulBenchReport, run_backend_matmul_bench};
pub use gguf::{GgufKvEntry, GgufReport, inspect_gguf};
pub use gguf_hash::{GgufHashError, HashAlgorithm, HashOptions, HashRecord, hash_file};
pub use simple::{SimpleError, SimpleReport, run_simple_ctx};
pub use smoke::{SmokeError, SmokeReport, run_backend_smoke};
