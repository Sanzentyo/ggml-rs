//! Token-id based end-to-end generation helpers for transformer-style GGUF models.
//!
//! This module supports both direct token-id prompts and GGUF tokenizer-backed
//! prompt text (currently for `tokenizer.ggml.model=gpt2`).
//!
//! # Module structure
//!
//! - [`error`] — `E2eError` enum and error helpers
//! - [`config`] — `E2eGenerationConfig`, `E2eGenerationReport`, `MixedLayerPolicy`
//! - [`plan`] — Layer plan ADTs (`LayerPlan`, `AttentionLayerPlan`, etc.)
//! - [`planner`] — Plan builders (`build_layer_plans`, etc.)
//! - [`resolve`] — Tensor name resolution and candidate generators
//! - [`decode`] — Tensor decode helpers
//! - [`numeric`] — Scalar math primitives (dot, softmax, sigmoid, silu, etc.)
//! - [`tensor_ops`] — Sequence-level tensor operations (RMS norm, projection, head slicing)
//! - [`attention`] — Full attention inference (Qwen3.5) + NeoX RoPE
//! - [`linear_attention`] — Linear attention inference (delta-net) + causal depthwise conv
//! - [`mlp`] — MLP sequence inference on backend
//! - [`generation`] — Token generation loop and public API entry points
//! - [`checkpoint`] — Serializable checkpoint DTOs and save/load
//! - [`session`] — Resumable step-by-step generation session

mod attention;
#[cfg(test)]
mod bench_graphs;
mod checkpoint;
mod config;
mod decode;
mod error;
mod generation;
mod linear_attention;
mod mlp;
mod numeric;
mod plan;
mod planner;
mod resolve;
mod session;
mod state;
mod tensor_ops;

pub use checkpoint::GenerationCheckpoint;
pub use config::{E2eGenerationConfig, E2eGenerationReport, MixedLayerPolicy};
pub use error::E2eError;
pub use generation::{
    generate_token_ids_from_model, generate_token_ids_from_path, resolve_eos_token_id,
    tokenize_prompt_text,
};
pub use session::GenerationSession;
