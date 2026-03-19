//! Unified public error boundary for `llama-rs`.

use crate::backend::ParseBackendError;
use crate::batched::BatchedError;
use crate::bench::BenchError;
use crate::bench_report::BenchReportError;
use crate::e2e::E2eError;
use crate::embedding::EmbeddingError;
use crate::gguf_hash::GgufHashError;
use crate::idle::IdleError;
use crate::inference::InferenceError;
use crate::metadata::MetadataError;
use crate::model::ModelError;
use crate::naming::NamingError;
use crate::simple::SimpleError;
use crate::smoke::SmokeError;
use crate::tokenizer::TokenizerError;
use std::error::Error as StdError;
use std::fmt;

/// Crate-wide result type with a unified error boundary.
pub type LlamaResult<T> = Result<T, LlamaError>;

#[derive(Debug)]
/// Top-level error type for applications that use multiple `llama-rs` modules.
pub enum LlamaError {
    ParseBackend(ParseBackendError),
    Ggml(ggml_rs::Error),
    Bench(BenchError),
    BenchReport(BenchReportError),
    Batched(BatchedError),
    Model(ModelError),
    Embedding(EmbeddingError),
    E2e(E2eError),
    Inference(InferenceError),
    Metadata(MetadataError),
    Naming(NamingError),
    Simple(SimpleError),
    Smoke(SmokeError),
    GgufHash(GgufHashError),
    Idle(IdleError),
    Tokenizer(TokenizerError),
}

impl fmt::Display for LlamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseBackend(source) => write!(f, "backend parse error: {source}"),
            Self::Ggml(source) => write!(f, "ggml error: {source}"),
            Self::Bench(source) => write!(f, "bench error: {source}"),
            Self::BenchReport(source) => write!(f, "bench report error: {source}"),
            Self::Batched(source) => write!(f, "batched error: {source}"),
            Self::Model(source) => write!(f, "model error: {source}"),
            Self::Embedding(source) => write!(f, "embedding error: {source}"),
            Self::E2e(source) => write!(f, "e2e error: {source}"),
            Self::Inference(source) => write!(f, "inference error: {source}"),
            Self::Metadata(source) => write!(f, "metadata error: {source}"),
            Self::Naming(source) => write!(f, "naming error: {source}"),
            Self::Simple(source) => write!(f, "simple error: {source}"),
            Self::Smoke(source) => write!(f, "smoke error: {source}"),
            Self::GgufHash(source) => write!(f, "gguf_hash error: {source}"),
            Self::Idle(source) => write!(f, "idle error: {source}"),
            Self::Tokenizer(source) => write!(f, "tokenizer error: {source}"),
        }
    }
}

impl StdError for LlamaError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::ParseBackend(source) => Some(source),
            Self::Ggml(source) => Some(source),
            Self::Bench(source) => Some(source),
            Self::BenchReport(source) => Some(source),
            Self::Batched(source) => Some(source),
            Self::Model(source) => Some(source),
            Self::Embedding(source) => Some(source),
            Self::E2e(source) => Some(source),
            Self::Inference(source) => Some(source),
            Self::Metadata(source) => Some(source),
            Self::Naming(source) => Some(source),
            Self::Simple(source) => Some(source),
            Self::Smoke(source) => Some(source),
            Self::GgufHash(source) => Some(source),
            Self::Idle(source) => Some(source),
            Self::Tokenizer(source) => Some(source),
        }
    }
}

impl From<ParseBackendError> for LlamaError {
    fn from(value: ParseBackendError) -> Self {
        Self::ParseBackend(value)
    }
}

impl From<ggml_rs::Error> for LlamaError {
    fn from(value: ggml_rs::Error) -> Self {
        Self::Ggml(value)
    }
}

impl From<BenchError> for LlamaError {
    fn from(value: BenchError) -> Self {
        Self::Bench(value)
    }
}

impl From<BenchReportError> for LlamaError {
    fn from(value: BenchReportError) -> Self {
        Self::BenchReport(value)
    }
}

impl From<BatchedError> for LlamaError {
    fn from(value: BatchedError) -> Self {
        Self::Batched(value)
    }
}

impl From<ModelError> for LlamaError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

impl From<EmbeddingError> for LlamaError {
    fn from(value: EmbeddingError) -> Self {
        Self::Embedding(value)
    }
}

impl From<E2eError> for LlamaError {
    fn from(value: E2eError) -> Self {
        Self::E2e(value)
    }
}

impl From<InferenceError> for LlamaError {
    fn from(value: InferenceError) -> Self {
        Self::Inference(value)
    }
}

impl From<MetadataError> for LlamaError {
    fn from(value: MetadataError) -> Self {
        Self::Metadata(value)
    }
}

impl From<NamingError> for LlamaError {
    fn from(value: NamingError) -> Self {
        Self::Naming(value)
    }
}

impl From<SimpleError> for LlamaError {
    fn from(value: SimpleError) -> Self {
        Self::Simple(value)
    }
}

impl From<SmokeError> for LlamaError {
    fn from(value: SmokeError) -> Self {
        Self::Smoke(value)
    }
}

impl From<GgufHashError> for LlamaError {
    fn from(value: GgufHashError) -> Self {
        Self::GgufHash(value)
    }
}

impl From<IdleError> for LlamaError {
    fn from(value: IdleError) -> Self {
        Self::Idle(value)
    }
}

impl From<TokenizerError> for LlamaError {
    fn from(value: TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}
