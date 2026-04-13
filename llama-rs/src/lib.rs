//! `llama-rs` — architecture-aware inference on GGUF models, built on `ggml-rs`.
//!
//! This crate provides tokenization, metadata extraction, and end-to-end token
//! generation for transformer architectures (Qwen3.5 full/linear attention,
//! standard attention, MLP).  The [`e2e`] module implements a two-phase
//! generation loop (prefill + autoregressive decode) with verified token-ID
//! parity against llama.cpp.

pub mod backend;
pub mod batched;
pub mod bench;
pub mod bench_report;
pub mod e2e;
pub mod embedding;
pub mod error;
pub mod gguf;
pub mod gguf_hash;
pub mod gpt2_synthetic;
pub mod idle;
pub mod inference;
pub mod metadata;
pub mod model;
pub mod naming;
pub mod simple;
pub mod smoke;
pub mod tokenizer;

pub use backend::{LlamaBackend, ParseBackendError};
pub use batched::{
    BatchSize, BatchedConfig, BatchedError, BatchedReport, BatchedWorkload, ReadbackEvery,
    RepeatCount, batched_matmul, batched_matmul_with_workload,
};
pub use bench::{BenchError, MatmulBenchConfig, MatmulBenchReport, backend_matmul_bench};
pub use bench_report::{
    AttentionBenchRow, BenchBackend, BenchReportError, LlamaCppBenchRow, MlpBenchRow,
    parse_attention_bench_output, parse_llama_cpp_jsonl, parse_mlp_bench_output,
    render_markdown_summary,
};
pub use e2e::{
    E2eError, E2eGenerationConfig, E2eGenerationReport, GenerationCheckpoint, GenerationSession,
    MixedLayerPolicy, generate_token_ids_from_model, generate_token_ids_from_path,
    resolve_eos_token_id, tokenize_prompt_text,
};
pub use embedding::{EmbeddingError, EmbeddingStats, summarize_embedding_tensor};
pub use error::{LlamaError, LlamaResult};
pub use ggml_rs::{GgufArrayValue, GgufValue};
pub use gguf::{GgufKvEntry, GgufReport, inspect_gguf};
pub use gguf_hash::{GgufHashError, HashAlgorithm, HashOptions, HashRecord, hash_file};
pub use gpt2_synthetic::{
    QuantizeConfig, QuantizeReport, SyntheticConfig, SyntheticError, SyntheticReport, parse_kv_f64,
    run_alloc, run_backend, run_batched, run_ctx, run_quantize, run_sched,
};
pub use idle::{
    IdleConfig, IdleError, IdlePauseReport, IdlePauseSchedule, IdleReport, IdleWeightsMode,
    PauseScheduleEmpty, PauseScheduleReady, idle_decode_proxy, idle_decode_proxy_from_path,
};
pub use inference::{
    AttentionDecodeCache, AttentionDecodeCacheInput, AttentionDecodePlan,
    AttentionDecodePlanBuilder, AttentionDecodeProxyReport, AttentionDecodeSource,
    AttentionDecodeStepwiseBenchReport, AttentionDecodeStepwiseBenchSweepReport,
    AttentionDecodeStepwiseConfig, AttentionDecodeStepwiseReport, AttentionDecodeWeightsInput,
    AttentionHeadCount, AttentionHeadDimension, AttentionInferenceConfig, AttentionInferenceReport,
    AttentionLayout, AttentionMaskPolicy, AttentionWeights, DecodeStepBenchSet, DecodeStepPlan,
    DecodeStepPlanBuilder, FfnFeatures, HiddenFeatures, InFeatures, InferenceError,
    LinearInferenceConfig, LinearInferenceReport, LinearWeights, LlamaLayerDimensions,
    MetadataResolutionMode, MlpInferenceConfig, MlpInferenceReport, MlpWeights, OutFeatures,
    RopeConfig, RotaryEmbedding, attention_inference_for_layer, attention_inference_for_layer_auto,
    attention_inference_for_layer_auto_repeats, attention_inference_for_layer_repeats,
    attention_inference_with_weights, attention_inference_with_weights_repeats,
    build_attention_decode_cache, linear_inference, linear_inference_with_weights,
    linear_inference_with_weights_repeats, mlp_inference, mlp_inference_for_layer,
    mlp_inference_for_layer_repeats, mlp_inference_with_weights,
    mlp_inference_with_weights_repeats, resolve_attention_weights_for_layer,
    resolve_attention_weights_for_layer_auto, resolve_llama_layer_dimensions,
    resolve_mlp_weights_for_layer, resolve_mlp_weights_for_layer_auto,
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
pub use simple::{SimpleError, SimpleReport, simple_ctx};
pub use smoke::{SmokeError, SmokeReport, backend_smoke};
pub use tokenizer::{GgufTokenizer, TokenizerError, TokenizerModel, tokenize_text_prompt};
