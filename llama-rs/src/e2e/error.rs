use crate::inference::InferenceError;
use crate::metadata::MetadataError;
use crate::model::ModelError;
use crate::naming::NamingError;
use crate::tokenizer::TokenizerError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum E2eError {
    #[error("{context}: {source}")]
    Model {
        context: &'static str,
        #[source]
        source: ModelError,
    },
    #[error("{context}: {source}")]
    Metadata {
        context: &'static str,
        #[source]
        source: MetadataError,
    },
    #[error("{context}: {source}")]
    Naming {
        context: &'static str,
        #[source]
        source: NamingError,
    },
    #[error("{context}: {source}")]
    Inference {
        context: &'static str,
        #[source]
        source: InferenceError,
    },
    #[error("{context}: {source}")]
    Tokenizer {
        context: &'static str,
        #[source]
        source: TokenizerError,
    },
    #[error("{context}: {source}")]
    Ggml {
        context: &'static str,
        #[source]
        source: ggml_rs::Error,
    },
    #[error("prompt_token_ids must not be empty")]
    EmptyPrompt,
    #[error("invalid token id {token_id}; valid range is [0, {vocab_size})")]
    InvalidTokenId { token_id: i32, vocab_size: usize },
    #[error(
        "token embedding tensor `{tensor_name}` has incompatible shape: hidden_features={hidden_features}, tensor_len={tensor_len}"
    )]
    InvalidTokenEmbeddingShape {
        tensor_name: String,
        hidden_features: usize,
        tensor_len: usize,
    },
    #[error(
        "output projection tensor `{tensor_name}` length mismatch: expected {expected}, got {actual}"
    )]
    OutputWeightLengthMismatch {
        tensor_name: String,
        expected: usize,
        actual: usize,
    },
    #[error("norm tensor `{tensor_name}` length mismatch: expected {expected}, got {actual}")]
    NormWeightLengthMismatch {
        tensor_name: String,
        expected: usize,
        actual: usize,
    },
    #[error("hidden feature mismatch at layer {layer}: expected {expected}, got {actual}")]
    HiddenFeatureMismatch {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "requested total sequence length {requested} exceeds model context length {context_length}"
    )]
    SequenceTooLong {
        requested: usize,
        context_length: usize,
    },
    #[error(
        "MLP gate tensor `{tensor_name}` has incompatible shape for hidden_features={hidden_features}: tensor_len={tensor_len}"
    )]
    InvalidMlpGateShape {
        tensor_name: String,
        hidden_features: usize,
        tensor_len: usize,
    },
    #[error("buffer length mismatch: expected {expected}, got {actual}")]
    BufferLengthMismatch { expected: usize, actual: usize },
    #[error(
        "invalid GQA head configuration: head_count={head_count}, kv_head_count={kv_head_count} (must be positive and head_count must be a multiple of kv_head_count)"
    )]
    InvalidGqaHeadConfig {
        head_count: usize,
        kv_head_count: usize,
    },
    #[error("memory size overflow while building generation graph")]
    MemorySizeOverflow,
    #[error(
        "invalid RoPE config: rope_n_dims={rope_n_dims} must be even and <= head_dimension={head_dimension}"
    )]
    RopeConfigInvalid {
        rope_n_dims: usize,
        head_dimension: usize,
    },
    #[error("TwoPhase mode requires all attention layers to be Qwen3.5 (Full or Linear)")]
    UnsupportedTwoPhase,
    #[error("checkpoint version mismatch: file has v{file_version}, expected v{expected_version}")]
    CheckpointVersionMismatch {
        file_version: u32,
        expected_version: u32,
    },
    #[error("checkpoint is incompatible with current model: {reason}")]
    CheckpointModelMismatch { reason: String },
    #[error("checkpoint I/O error: {0}")]
    CheckpointIo(#[from] std::io::Error),
    #[error("checkpoint deserialization failed: {0}")]
    CheckpointDeserialize(String),
}

impl E2eError {
    pub(super) fn model(context: &'static str, source: ModelError) -> Self {
        Self::Model { context, source }
    }

    pub(super) fn metadata(context: &'static str, source: MetadataError) -> Self {
        Self::Metadata { context, source }
    }

    pub(super) fn naming(context: &'static str, source: NamingError) -> Self {
        Self::Naming { context, source }
    }

    pub(super) fn inference(context: &'static str, source: InferenceError) -> Self {
        Self::Inference { context, source }
    }

    pub(super) fn tokenizer(context: &'static str, source: TokenizerError) -> Self {
        Self::Tokenizer { context, source }
    }

    pub(super) fn ggml(context: &'static str, source: ggml_rs::Error) -> Self {
        Self::Ggml { context, source }
    }
}

/// Extension trait to convert `Result<T, ggml_rs::Error>` into `Result<T, E2eError>`
/// with a static context label.
pub(super) trait GgmlResultExt<T> {
    fn ggml_ctx(self, context: &'static str) -> Result<T, E2eError>;
}

impl<T> GgmlResultExt<T> for Result<T, ggml_rs::Error> {
    fn ggml_ctx(self, context: &'static str) -> Result<T, E2eError> {
        self.map_err(|source| E2eError::ggml(context, source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constructor helpers ─────────────────────────────────────
    #[test]
    fn model_helper_builds_variant() {
        let source = ModelError::MissingTensor { name: "x".into() };
        let err = E2eError::model("ctx", source);
        assert!(matches!(err, E2eError::Model { context: "ctx", .. }));
    }

    #[test]
    fn metadata_helper_builds_variant() {
        let source = MetadataError::MissingRequiredKey { key: "k".into() };
        let err = E2eError::metadata("ctx", source);
        assert!(matches!(err, E2eError::Metadata { context: "ctx", .. }));
    }

    #[test]
    fn naming_helper_builds_variant() {
        let source = NamingError::NoLayersDetected;
        let err = E2eError::naming("ctx", source);
        assert!(matches!(err, E2eError::Naming { context: "ctx", .. }));
    }

    #[test]
    fn inference_helper_builds_variant() {
        let source = InferenceError::InvalidAttentionLayout {
            hidden_features: 0,
            query_head_count: 0,
            kv_head_count: 0,
        };
        let err = E2eError::inference("ctx", source);
        assert!(matches!(err, E2eError::Inference { context: "ctx", .. }));
    }

    #[test]
    fn tokenizer_helper_builds_variant() {
        let source = TokenizerError::MissingMetadata { key: "model" };
        let err = E2eError::tokenizer("ctx", source);
        assert!(matches!(err, E2eError::Tokenizer { context: "ctx", .. }));
    }

    #[test]
    fn ggml_helper_builds_variant() {
        let source = ggml_rs::Error::ZeroMemorySize;
        let err = E2eError::ggml("ctx", source);
        assert!(matches!(err, E2eError::Ggml { context: "ctx", .. }));
    }

    // ── GgmlResultExt ───────────────────────────────────────────
    #[test]
    fn ggml_result_ext_ok_passes_through() {
        let r: Result<i32, ggml_rs::Error> = Ok(42);
        assert_eq!(r.ggml_ctx("ctx").unwrap(), 42);
    }

    #[test]
    fn ggml_result_ext_err_wraps() {
        let r: Result<i32, ggml_rs::Error> = Err(ggml_rs::Error::ZeroMemorySize);
        let err = r.ggml_ctx("conv").unwrap_err();
        assert!(matches!(
            err,
            E2eError::Ggml {
                context: "conv",
                ..
            }
        ));
    }

    // ── Display ─────────────────────────────────────────────────
    #[test]
    fn display_includes_context() {
        let source = ModelError::MissingTensor { name: "foo".into() };
        let err = E2eError::model("loading weights", source);
        let msg = format!("{err}");
        assert!(
            msg.contains("loading weights"),
            "display should include context: {msg}"
        );
    }
}
