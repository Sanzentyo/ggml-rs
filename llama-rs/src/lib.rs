pub mod backend;
pub mod batched;
pub mod bench;
pub mod embedding;
pub mod error;
pub mod gguf;
pub mod gguf_hash;
pub mod inference;
pub mod metadata;
pub mod model;
pub mod naming;
pub mod simple;
pub mod smoke;

pub use backend::{LlamaBackend, ParseBackendError};
pub use batched::{
    BatchSize, BatchedConfig, BatchedError, BatchedReport, BatchedWorkload, ReadbackEvery,
    RepeatCount, run_batched_matmul, run_batched_matmul_with_workload,
};
pub use bench::{BenchError, MatmulBenchConfig, MatmulBenchReport, run_backend_matmul_bench};
pub use embedding::{EmbeddingError, EmbeddingStats, summarize_embedding_tensor};
pub use error::{LlamaError, LlamaResult};
pub use ggml_rs::{GgufArrayValue, GgufValue};
pub use gguf::{GgufKvEntry, GgufReport, inspect_gguf};
pub use gguf_hash::{GgufHashError, HashAlgorithm, HashOptions, HashRecord, hash_file};
pub use inference::{
    AttentionHeadCount, AttentionHeadDimension, AttentionInferenceConfig, AttentionInferenceReport,
    AttentionLayout, AttentionMaskPolicy, AttentionWeights, FfnFeatures, HiddenFeatures,
    InFeatures, InferenceError, LinearInferenceConfig, LinearInferenceReport, LinearWeights,
    LlamaLayerDimensions, MetadataResolutionMode, MlpInferenceConfig, MlpInferenceReport,
    MlpWeights, OutFeatures, RopeConfig, RotaryEmbedding, resolve_attention_weights_for_layer,
    resolve_attention_weights_for_layer_auto, resolve_llama_layer_dimensions,
    resolve_mlp_weights_for_layer, resolve_mlp_weights_for_layer_auto,
    run_attention_inference_for_layer, run_attention_inference_for_layer_auto,
    run_attention_inference_for_layer_auto_repeats, run_attention_inference_for_layer_repeats,
    run_attention_inference_with_weights, run_attention_inference_with_weights_repeats,
    run_linear_inference, run_linear_inference_with_weights,
    run_linear_inference_with_weights_repeats, run_mlp_inference, run_mlp_inference_for_layer,
    run_mlp_inference_for_layer_repeats, run_mlp_inference_with_weights,
    run_mlp_inference_with_weights_repeats,
};
pub use metadata::{
    LlamaModelMetadata, MetadataError, ModelArchitecture, ModelMetadata, TransformerMetadata,
    resolve_llama_metadata, resolve_llama_metadata_from_kv, resolve_model_metadata,
    resolve_model_metadata_from_kv, resolve_transformer_metadata,
    resolve_transformer_metadata_from_kv,
};
pub use model::{GgufModel, ModelError, TensorHandle};
pub use naming::{
    LlamaLayerTensorNames, LlamaTensorNames, NamingError, detect_layer_indices,
    detect_layer_indices_from_names, resolve_llama_layer_tensor_names,
    resolve_llama_layer_tensor_names_from_names, resolve_llama_tensor_names,
    resolve_llama_tensor_names_from_names,
};
pub use simple::{SimpleError, SimpleReport, run_simple_ctx};
pub use smoke::{SmokeError, SmokeReport, run_backend_smoke};
