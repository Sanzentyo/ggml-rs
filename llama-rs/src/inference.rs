//! Minimal inference-path helpers.
//!
//! The current scope is a single linear layer execution path backed by ggml
//! safe APIs. This acts as the first inference building block before full
//! transformer graph assembly.

use crate::backend::LlamaBackend;
use crate::metadata::MetadataError;
use crate::model::{GgufModel, ModelError};
use crate::naming::{LlamaLayerTensorNames, NamingError, resolve_llama_layer_tensor_names};
use ggml_rs::{Backend, Context, GgmlElement, Length, Shape2D};
use num_traits::NumCast;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use thiserror::Error;

mod attention_ops;
mod attention_runtime;
mod backend_runtime;
mod decode_proxy_plan;
mod layer_dimensions;
mod projection_ops;
mod stepwise_decode;
mod stepwise_plan;
use attention_ops::{LlamaRotaryApplier, RotaryApplier};
#[cfg(test)]
pub(crate) use attention_runtime::build_causal_mask_values;
pub(crate) use attention_runtime::{
    attention_decode_proxy_with_cache_repeats_inner,
    attention_inference_with_weights_on_backend_repeats_with_length, fill_causal_mask_values,
    recommended_attention_backend_memory_bytes_for_lengths,
};
pub use attention_runtime::{
    attention_inference_for_layer, attention_inference_for_layer_auto,
    attention_inference_for_layer_auto_repeats, attention_inference_for_layer_repeats,
    attention_inference_with_weights, attention_inference_with_weights_repeats,
    build_attention_decode_cache, resolve_attention_weights_for_layer,
    resolve_attention_weights_for_layer_auto,
};
use backend_runtime::{BackendRuntimeBuilder, DefaultBackendRuntimeBuilder};
pub use decode_proxy_plan::{
    AttentionDecodeCacheInput, AttentionDecodePlan, AttentionDecodePlanBuilder,
    AttentionDecodeSource, AttentionDecodeWeightsInput,
};
#[cfg(test)]
pub(crate) use layer_dimensions::infer_attention_layout_from_features;
pub use layer_dimensions::{
    LlamaLayerDimensions, MetadataResolutionMode, resolve_llama_layer_dimensions,
};
use projection_ops::{DecodeCacheBuilder, F32MatmulProjector, StandardDecodeCacheBuilder};
pub use stepwise_decode::{
    AttentionDecodeStepwiseBenchReport, AttentionDecodeStepwiseBenchSweepReport,
    AttentionDecodeStepwiseConfig, AttentionDecodeStepwiseReport,
};
pub use stepwise_plan::{DecodeStepBenchSet, DecodeStepPlan, DecodeStepPlanBuilder};

macro_rules! define_non_zero_count {
    ($(#[$meta:meta])* $name:ident, $error_ctor:expr) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        $(#[$meta])*
        pub struct $name(NonZeroUsize);

        impl $name {
            pub fn new(value: usize) -> Result<Self, InferenceError> {
                NonZeroUsize::new(value)
                    .map(Self)
                    .ok_or_else(|| ($error_ctor)(value))
            }

            pub const fn get(self) -> usize {
                self.0.get()
            }
        }
    };
}

define_non_zero_count!(
    /// Strongly typed non-zero input feature count.
    InFeatures,
    |value| InferenceError::InvalidLinearShape {
        in_features: value,
        out_features: 0,
    }
);

define_non_zero_count!(
    /// Strongly typed non-zero output feature count.
    OutFeatures,
    |value| InferenceError::InvalidLinearShape {
        in_features: 0,
        out_features: value,
    }
);

/// Marker: required builder field is not set yet.
pub struct Missing;
/// Marker: required builder field is set.
pub struct Present;

/// Type-state builder for [`LinearInferenceConfig`].
pub struct LinearInferenceConfigBuilder<InState, OutState> {
    in_features: Option<InFeatures>,
    out_features: Option<OutFeatures>,
    _state: PhantomData<(InState, OutState)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Configuration for a single linear projection.
pub struct LinearInferenceConfig {
    /// Input feature width.
    pub in_features: InFeatures,
    /// Output feature width.
    pub out_features: OutFeatures,
}

impl LinearInferenceConfig {
    /// Creates a config from linear-layer feature dimensions.
    pub fn new(in_features: usize, out_features: usize) -> Result<Self, InferenceError> {
        Ok(Self {
            in_features: InFeatures::new(in_features)?,
            out_features: OutFeatures::new(out_features)?,
        })
    }

    /// Starts a type-state builder that enforces required dimensions at compile time.
    pub fn builder() -> LinearInferenceConfigBuilder<Missing, Missing> {
        LinearInferenceConfigBuilder {
            in_features: None,
            out_features: None,
            _state: PhantomData,
        }
    }

    pub const fn in_features(self) -> usize {
        self.in_features.get()
    }

    pub const fn out_features(self) -> usize {
        self.out_features.get()
    }

    pub const fn expected_weight_len(self) -> usize {
        self.in_features.get() * self.out_features.get()
    }
}

impl<OutState> LinearInferenceConfigBuilder<Missing, OutState> {
    /// Sets input feature width and advances the builder state.
    pub fn in_features(
        self,
        value: usize,
    ) -> Result<LinearInferenceConfigBuilder<Present, OutState>, InferenceError> {
        Ok(LinearInferenceConfigBuilder {
            in_features: Some(InFeatures::new(value)?),
            out_features: self.out_features,
            _state: PhantomData,
        })
    }
}

impl<InState> LinearInferenceConfigBuilder<InState, Missing> {
    /// Sets output feature width and advances the builder state.
    pub fn out_features(
        self,
        value: usize,
    ) -> Result<LinearInferenceConfigBuilder<InState, Present>, InferenceError> {
        Ok(LinearInferenceConfigBuilder {
            in_features: self.in_features,
            out_features: Some(OutFeatures::new(value)?),
            _state: PhantomData,
        })
    }
}

impl LinearInferenceConfigBuilder<Present, Present> {
    /// Builds a fully specified linear config.
    pub fn build(self) -> LinearInferenceConfig {
        LinearInferenceConfig {
            in_features: self
                .in_features
                .expect("builder invariant: in_features is present"),
            out_features: self
                .out_features
                .expect("builder invariant: out_features is present"),
        }
    }
}

#[derive(Debug, Clone)]
/// Output payload and execution metadata.
pub struct LinearInferenceReport<T = f32> {
    pub backend_name: String,
    pub in_features: usize,
    pub out_features: usize,
    pub repeats: usize,
    pub output: Vec<T>,
}

#[derive(Debug, Clone)]
/// Pre-decoded linear weights for repeated inference calls.
pub struct LinearWeights<T = f32> {
    pub tensor_name: String,
    pub in_features: usize,
    pub out_features: usize,
    values: Vec<T>,
}

impl<T> LinearWeights<T>
where
    T: GgmlElement + NumCast,
{
    /// Decodes a GGUF tensor into reusable linear weights.
    pub fn from_model(
        model: &GgufModel,
        tensor_name: impl AsRef<str>,
        config: LinearInferenceConfig,
    ) -> Result<Self, InferenceError> {
        let tensor_name = tensor_name.as_ref();
        let expected_weights = config.expected_weight_len();
        let values = model
            .tensor_values::<T>(tensor_name)
            .map_err(|source| InferenceError::model("GgufModel::tensor_values", source))?;
        if values.len() != expected_weights {
            return Err(InferenceError::InvalidWeightLength {
                tensor_name: tensor_name.to_string(),
                expected: expected_weights,
                actual: values.len(),
            });
        }

        Ok(Self {
            tensor_name: tensor_name.to_string(),
            in_features: config.in_features(),
            out_features: config.out_features(),
            values,
        })
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }
}

define_non_zero_count!(
    /// Strongly typed non-zero hidden feature count for MLP-style block inference.
    HiddenFeatures,
    |value| InferenceError::InvalidMlpShape {
        hidden_features: value,
        ffn_features: 0,
    }
);

define_non_zero_count!(
    /// Strongly typed non-zero FFN feature count for MLP-style block inference.
    FfnFeatures,
    |value| InferenceError::InvalidMlpShape {
        hidden_features: 0,
        ffn_features: value,
    }
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Configuration for a minimal MLP block (`down(silu(gate(x)) * up(x))`).
pub struct MlpInferenceConfig {
    pub hidden_features: HiddenFeatures,
    pub ffn_features: FfnFeatures,
}

impl MlpInferenceConfig {
    pub fn new(hidden_features: usize, ffn_features: usize) -> Result<Self, InferenceError> {
        Ok(Self {
            hidden_features: HiddenFeatures::new(hidden_features)?,
            ffn_features: FfnFeatures::new(ffn_features)?,
        })
    }

    pub const fn hidden_features(self) -> usize {
        self.hidden_features.get()
    }

    pub const fn ffn_features(self) -> usize {
        self.ffn_features.get()
    }

    pub const fn expected_gate_weight_len(self) -> usize {
        self.hidden_features.get() * self.ffn_features.get()
    }

    pub const fn expected_up_weight_len(self) -> usize {
        self.expected_gate_weight_len()
    }

    pub const fn expected_down_weight_len(self) -> usize {
        self.hidden_features.get() * self.ffn_features.get()
    }
}

#[derive(Debug, Clone)]
/// Pre-decoded MLP weights for repeated backend executions.
pub struct MlpWeights<T = f32> {
    pub gate_tensor_name: String,
    pub up_tensor_name: String,
    pub down_tensor_name: String,
    pub hidden_features: usize,
    pub ffn_features: usize,
    gate_values: Vec<T>,
    up_values: Vec<T>,
    down_values: Vec<T>,
}

impl MlpWeights<f32> {
    /// Builds deterministic synthetic MLP weights for runtime smoke checks.
    pub fn deterministic(config: MlpInferenceConfig) -> Self {
        let gate_values = (0..config.expected_gate_weight_len())
            .map(|index| (index % 17) as f32 * 0.03125)
            .collect();
        let up_values = (0..config.expected_up_weight_len())
            .map(|index| ((index + 11) % 23) as f32 * 0.015625)
            .collect();
        let down_values = (0..config.expected_down_weight_len())
            .map(|index| ((index + 7) % 29) as f32 * 0.0078125)
            .collect();
        Self {
            gate_tensor_name: "<deterministic-gate>".to_string(),
            up_tensor_name: "<deterministic-up>".to_string(),
            down_tensor_name: "<deterministic-down>".to_string(),
            hidden_features: config.hidden_features(),
            ffn_features: config.ffn_features(),
            gate_values,
            up_values,
            down_values,
        }
    }
}

impl<T> MlpWeights<T>
where
    T: GgmlElement + NumCast,
{
    /// Loads MLP weights from three GGUF tensors (`gate`, `up`, `down`).
    pub fn from_model(
        model: &GgufModel,
        gate_tensor_name: impl AsRef<str>,
        up_tensor_name: impl AsRef<str>,
        down_tensor_name: impl AsRef<str>,
        config: MlpInferenceConfig,
    ) -> Result<Self, InferenceError> {
        let gate_tensor_name = gate_tensor_name.as_ref();
        let up_tensor_name = up_tensor_name.as_ref();
        let down_tensor_name = down_tensor_name.as_ref();
        let gate_values = model
            .tensor_values::<T>(gate_tensor_name)
            .map_err(|source| InferenceError::model("GgufModel::tensor_values(gate)", source))?;
        let up_values = model
            .tensor_values::<T>(up_tensor_name)
            .map_err(|source| InferenceError::model("GgufModel::tensor_values(up)", source))?;
        let down_values = model
            .tensor_values::<T>(down_tensor_name)
            .map_err(|source| InferenceError::model("GgufModel::tensor_values(down)", source))?;

        Self::from_raw(
            gate_tensor_name.to_string(),
            up_tensor_name.to_string(),
            down_tensor_name.to_string(),
            gate_values,
            up_values,
            down_values,
            config,
        )
    }

    /// Loads MLP weights for a resolved transformer layer and infers FFN width
    /// from `gate` tensor size and caller-provided hidden feature width.
    pub fn from_model_layer(
        model: &GgufModel,
        layer: &LlamaLayerTensorNames,
        hidden_features: usize,
    ) -> Result<Self, InferenceError> {
        let gate_values = model
            .tensor_values::<T>(&layer.ffn_gate)
            .map_err(|source| {
                InferenceError::model("GgufModel::tensor_values(ffn_gate)", source)
            })?;
        if hidden_features == 0 || gate_values.len() % hidden_features != 0 {
            return Err(InferenceError::InvalidMlpWeightShape {
                tensor_name: layer.ffn_gate.clone(),
                hidden_features,
                weight_len: gate_values.len(),
            });
        }
        let ffn_features = gate_values.len() / hidden_features;
        let config = MlpInferenceConfig::new(hidden_features, ffn_features)?;
        let up_values = model
            .tensor_values::<T>(&layer.ffn_up)
            .map_err(|source| InferenceError::model("GgufModel::tensor_values(ffn_up)", source))?;
        let down_values = model
            .tensor_values::<T>(&layer.ffn_down)
            .map_err(|source| {
                InferenceError::model("GgufModel::tensor_values(ffn_down)", source)
            })?;

        Self::from_raw(
            layer.ffn_gate.clone(),
            layer.ffn_up.clone(),
            layer.ffn_down.clone(),
            gate_values,
            up_values,
            down_values,
            config,
        )
    }

    /// Builds MLP weights from caller-owned tensors with shape validation.
    pub fn from_raw(
        gate_tensor_name: String,
        up_tensor_name: String,
        down_tensor_name: String,
        gate_values: Vec<T>,
        up_values: Vec<T>,
        down_values: Vec<T>,
        config: MlpInferenceConfig,
    ) -> Result<Self, InferenceError> {
        let expected_gate = config.expected_gate_weight_len();
        if gate_values.len() != expected_gate {
            return Err(InferenceError::InvalidWeightLength {
                tensor_name: gate_tensor_name,
                expected: expected_gate,
                actual: gate_values.len(),
            });
        }
        let expected_up = config.expected_up_weight_len();
        if up_values.len() != expected_up {
            return Err(InferenceError::InvalidWeightLength {
                tensor_name: up_tensor_name,
                expected: expected_up,
                actual: up_values.len(),
            });
        }
        let expected_down = config.expected_down_weight_len();
        if down_values.len() != expected_down {
            return Err(InferenceError::InvalidWeightLength {
                tensor_name: down_tensor_name,
                expected: expected_down,
                actual: down_values.len(),
            });
        }

        Ok(Self {
            gate_tensor_name,
            up_tensor_name,
            down_tensor_name,
            hidden_features: config.hidden_features(),
            ffn_features: config.ffn_features(),
            gate_values,
            up_values,
            down_values,
        })
    }

    pub fn gate_values(&self) -> &[T] {
        &self.gate_values
    }

    pub fn up_values(&self) -> &[T] {
        &self.up_values
    }

    pub fn down_values(&self) -> &[T] {
        &self.down_values
    }
}

#[derive(Debug, Clone)]
/// Output payload and execution metadata for MLP block inference.
pub struct MlpInferenceReport<T = f32> {
    pub backend_name: String,
    pub hidden_features: usize,
    pub ffn_features: usize,
    pub repeats: usize,
    pub output: Vec<T>,
}

#[derive(Debug, Error)]
/// Errors surfaced by inference helpers.
pub enum InferenceError {
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
    Ggml {
        context: &'static str,
        #[source]
        source: ggml_rs::Error,
    },
    #[error("weight tensor `{tensor_name}` length mismatch: expected {expected}, got {actual}")]
    InvalidWeightLength {
        tensor_name: String,
        expected: usize,
        actual: usize,
    },
    #[error("input length mismatch: expected {expected}, got {actual}")]
    InvalidInputLength { expected: usize, actual: usize },
    #[error("invalid linear shape: in_features={in_features}, out_features={out_features}")]
    InvalidLinearShape {
        in_features: usize,
        out_features: usize,
    },
    #[error("invalid MLP shape: hidden_features={hidden_features}, ffn_features={ffn_features}")]
    InvalidMlpShape {
        hidden_features: usize,
        ffn_features: usize,
    },
    #[error(
        "cannot infer MLP shape from tensor `{tensor_name}`: hidden_features={hidden_features}, weight_len={weight_len}"
    )]
    InvalidMlpWeightShape {
        tensor_name: String,
        hidden_features: usize,
        weight_len: usize,
    },
    #[error(
        "invalid attention shape: hidden_features={hidden_features}, sequence_length={sequence_length}"
    )]
    InvalidAttentionShape {
        hidden_features: usize,
        sequence_length: usize,
    },
    #[error(
        "invalid attention layout: hidden_features={hidden_features}, query_head_count={query_head_count}, kv_head_count={kv_head_count}"
    )]
    InvalidAttentionLayout {
        hidden_features: usize,
        query_head_count: usize,
        kv_head_count: usize,
    },
    #[error(
        "invalid attention weight tensor `{tensor_name}` length mismatch: expected {expected}, got {actual}"
    )]
    InvalidAttentionWeightShape {
        tensor_name: String,
        expected: usize,
        actual: usize,
    },
    #[error(
        "invalid rope dimensions: rope_dimensions={rope_dimensions}, head_dimension={head_dimension}"
    )]
    InvalidRopeDimensions {
        rope_dimensions: usize,
        head_dimension: usize,
    },
    #[error("layer not found in resolved catalog: {layer}")]
    LayerNotFound { layer: usize },
    #[error("memory size overflow while building inference graph")]
    MemorySizeOverflow,
    #[error("repeats must be greater than zero")]
    InvalidRepeats,
}

impl InferenceError {
    fn model(context: &'static str, source: ModelError) -> Self {
        Self::Model { context, source }
    }

    fn metadata(context: &'static str, source: MetadataError) -> Self {
        Self::Metadata { context, source }
    }

    fn naming(context: &'static str, source: NamingError) -> Self {
        Self::Naming { context, source }
    }

    fn ggml(context: &'static str, source: ggml_rs::Error) -> Self {
        Self::Ggml { context, source }
    }
}

/// Runs one linear projection `Y = W * X` using backend execution.
///
/// `W` is loaded from a GGUF tensor and interpreted as shape
/// `[in_features, out_features]` in ggml's `(cols, rows)` convention.
pub fn linear_inference<T>(
    model: &GgufModel,
    weight_tensor_name: impl AsRef<str>,
    input: impl AsRef<[T]>,
    config: LinearInferenceConfig,
    backend_kind: LlamaBackend,
) -> Result<LinearInferenceReport<T>, InferenceError>
where
    T: GgmlElement + NumCast,
{
    let input = input.as_ref();
    let weights = LinearWeights::from_model(model, weight_tensor_name, config)?;
    linear_inference_with_weights(&weights, input, backend_kind)
}

/// Runs one linear projection using pre-decoded reusable weights.
pub fn linear_inference_with_weights<T>(
    weights: &LinearWeights<T>,
    input: impl AsRef<[T]>,
    backend_kind: LlamaBackend,
) -> Result<LinearInferenceReport<T>, InferenceError>
where
    T: GgmlElement + NumCast,
{
    linear_inference_with_weights_repeats(weights, input, backend_kind, 1)
}

/// Runs one linear projection using pre-decoded reusable weights and repeated
/// backend execution on the same graph.
pub fn linear_inference_with_weights_repeats<T>(
    weights: &LinearWeights<T>,
    input: impl AsRef<[T]>,
    backend_kind: LlamaBackend,
    repeats: usize,
) -> Result<LinearInferenceReport<T>, InferenceError>
where
    T: GgmlElement + NumCast,
{
    let input = input.as_ref();
    if input.len() != weights.in_features {
        return Err(InferenceError::InvalidInputLength {
            expected: weights.in_features,
            actual: input.len(),
        });
    }
    if repeats == 0 {
        return Err(InferenceError::InvalidRepeats);
    }

    let weight_shape = Shape2D::new(weights.in_features, weights.out_features);
    let input_shape = Shape2D::new(weights.in_features, 1);
    let ctx_size = Context::recommended_backend_matmul_memory::<T>(weight_shape, input_shape)
        .map_err(|source| {
            InferenceError::ggml("Context::recommended_backend_matmul_memory", source)
        })?;
    let runtime = DefaultBackendRuntimeBuilder.build_runtime(backend_kind, ctx_size)?;
    let backend = runtime.backend;
    let backend_name = runtime.backend_name;
    let ctx = runtime.ctx;

    let w = ctx
        .new_tensor_2d::<T>(weight_shape)
        .map_err(|source| InferenceError::ggml("Context::new_tensor_2d_shape<W>", source))?;
    let x = ctx
        .new_tensor_2d::<T>(input_shape)
        .map_err(|source| InferenceError::ggml("Context::new_tensor_2d_shape<X>", source))?;
    let y = ctx
        .mul_mat(&w, &x)
        .map_err(|source| InferenceError::ggml("Context::mul_mat", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| InferenceError::ggml("Context::new_graph", source))?;
    graph.build_forward_expand(&y);
    let _buffer = ctx
        .allocate_tensors(&backend)
        .map_err(|source| InferenceError::ggml("Context::allocate_tensors", source))?;

    w.write_data_backend(weights.values())
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W>", source))?;
    x.write_data_backend(input)
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<X>", source))?;
    for _ in 0..repeats {
        backend
            .compute(&mut graph)
            .map_err(|source| InferenceError::ggml("Backend::compute", source))?;
    }

    let output = graph
        .last_node()
        .map_err(|source| InferenceError::ggml("Graph::last_node", source))?
        .read_data_backend::<T>()
        .map_err(|source| InferenceError::ggml("Tensor::read_data_backend", source))?;

    Ok(LinearInferenceReport {
        backend_name,
        in_features: weights.in_features,
        out_features: weights.out_features,
        repeats,
        output,
    })
}

/// Runs a minimal MLP block `down(silu(gate(x)) * up(x))` from GGUF tensors.
pub fn mlp_inference<T>(
    model: &GgufModel,
    gate_tensor_name: impl AsRef<str>,
    up_tensor_name: impl AsRef<str>,
    down_tensor_name: impl AsRef<str>,
    input: impl AsRef<[T]>,
    config: MlpInferenceConfig,
    backend_kind: LlamaBackend,
) -> Result<MlpInferenceReport<T>, InferenceError>
where
    T: GgmlElement + NumCast,
{
    let input = input.as_ref();
    let weights = MlpWeights::from_model(
        model,
        gate_tensor_name,
        up_tensor_name,
        down_tensor_name,
        config,
    )?;
    mlp_inference_with_weights(&weights, input, backend_kind)
}

/// Resolves a transformer layer by index from GGUF tensor names and runs the
/// minimal MLP block for that layer.
pub fn mlp_inference_for_layer<T>(
    model: &GgufModel,
    layer: usize,
    input: impl AsRef<[T]>,
    backend_kind: LlamaBackend,
) -> Result<MlpInferenceReport<T>, InferenceError>
where
    T: GgmlElement + NumCast,
{
    mlp_inference_for_layer_repeats(model, layer, input, backend_kind, 1)
}

/// Resolves and decodes reusable MLP weights for one transformer layer.
pub fn resolve_mlp_weights_for_layer<T>(
    model: &GgufModel,
    layer: usize,
    hidden_features: usize,
) -> Result<MlpWeights<T>, InferenceError>
where
    T: GgmlElement + NumCast,
{
    let layer_names = resolve_llama_layer_tensor_names(model, layer)
        .map_err(|source| InferenceError::naming("resolve_llama_layer_tensor_names", source))?;
    MlpWeights::from_model_layer(model, &layer_names, hidden_features)
}

/// Resolves and decodes reusable MLP weights using GGUF metadata-derived hidden width.
pub fn resolve_mlp_weights_for_layer_auto<T>(
    model: &GgufModel,
    layer: usize,
) -> Result<MlpWeights<T>, InferenceError>
where
    T: GgmlElement + NumCast,
{
    let dimensions = resolve_llama_layer_dimensions(model, layer)?;
    resolve_mlp_weights_for_layer(model, layer, dimensions.hidden_features)
}

/// Same as [`mlp_inference_for_layer`] with explicit repeated backend runs.
pub fn mlp_inference_for_layer_repeats<T>(
    model: &GgufModel,
    layer: usize,
    input: impl AsRef<[T]>,
    backend_kind: LlamaBackend,
    repeats: usize,
) -> Result<MlpInferenceReport<T>, InferenceError>
where
    T: GgmlElement + NumCast,
{
    let input = input.as_ref();
    let weights = resolve_mlp_weights_for_layer_auto(model, layer)?;
    mlp_inference_with_weights_repeats(&weights, input, backend_kind, repeats)
}

/// Runs a minimal MLP block using pre-decoded reusable weights.
pub fn mlp_inference_with_weights<T>(
    weights: &MlpWeights<T>,
    input: impl AsRef<[T]>,
    backend_kind: LlamaBackend,
) -> Result<MlpInferenceReport<T>, InferenceError>
where
    T: GgmlElement + NumCast,
{
    mlp_inference_with_weights_repeats(weights, input, backend_kind, 1)
}

/// Runs a minimal MLP block using pre-decoded reusable weights and repeated
/// backend execution on a single graph.
pub fn mlp_inference_with_weights_repeats<T>(
    weights: &MlpWeights<T>,
    input: impl AsRef<[T]>,
    backend_kind: LlamaBackend,
    repeats: usize,
) -> Result<MlpInferenceReport<T>, InferenceError>
where
    T: GgmlElement + NumCast,
{
    let input = input.as_ref();
    if input.len() != weights.hidden_features {
        return Err(InferenceError::InvalidInputLength {
            expected: weights.hidden_features,
            actual: input.len(),
        });
    }
    if repeats == 0 {
        return Err(InferenceError::InvalidRepeats);
    }

    let hidden = weights.hidden_features;
    let ffn = weights.ffn_features;
    let ctx_size = recommended_mlp_backend_memory_bytes::<T>(hidden, ffn)?;
    let runtime = DefaultBackendRuntimeBuilder.build_runtime(backend_kind, ctx_size)?;
    let backend = runtime.backend;
    let backend_name = runtime.backend_name;
    let ctx = runtime.ctx;

    let gate_shape = Shape2D::new(hidden, ffn);
    let up_shape = Shape2D::new(hidden, ffn);
    let down_shape = Shape2D::new(ffn, hidden);
    let input_shape = Shape2D::new(hidden, 1);

    let w_gate = ctx
        .new_tensor_2d::<T>(gate_shape)
        .map_err(|source| InferenceError::ggml("Context::new_tensor_2d_shape<W_GATE>", source))?;
    let w_up = ctx
        .new_tensor_2d::<T>(up_shape)
        .map_err(|source| InferenceError::ggml("Context::new_tensor_2d_shape<W_UP>", source))?;
    let w_down = ctx
        .new_tensor_2d::<T>(down_shape)
        .map_err(|source| InferenceError::ggml("Context::new_tensor_2d_shape<W_DOWN>", source))?;
    let x = ctx
        .new_tensor_2d::<T>(input_shape)
        .map_err(|source| InferenceError::ggml("Context::new_tensor_2d_shape<X>", source))?;

    let gate = ctx
        .mul_mat(&w_gate, &x)
        .map_err(|source| InferenceError::ggml("Context::mul_mat(gate)", source))?;
    let up = ctx
        .mul_mat(&w_up, &x)
        .map_err(|source| InferenceError::ggml("Context::mul_mat(up)", source))?;
    let activated = ctx
        .silu(&gate)
        .map_err(|source| InferenceError::ggml("Context::silu", source))?;
    let fused = ctx
        .mul(&activated, &up)
        .map_err(|source| InferenceError::ggml("Context::mul(gated_up)", source))?;
    let y = ctx
        .mul_mat(&w_down, &fused)
        .map_err(|source| InferenceError::ggml("Context::mul_mat(down)", source))?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| InferenceError::ggml("Context::new_graph", source))?;
    graph.build_forward_expand(&y);
    let _buffer = ctx
        .allocate_tensors(&backend)
        .map_err(|source| InferenceError::ggml("Context::allocate_tensors", source))?;

    w_gate
        .write_data_backend(weights.gate_values())
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W_GATE>", source))?;
    w_up.write_data_backend(weights.up_values())
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W_UP>", source))?;
    w_down
        .write_data_backend(weights.down_values())
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<W_DOWN>", source))?;
    x.write_data_backend(input)
        .map_err(|source| InferenceError::ggml("Tensor::write_data_backend<X>", source))?;

    for _ in 0..repeats {
        backend
            .compute(&mut graph)
            .map_err(|source| InferenceError::ggml("Backend::compute", source))?;
    }

    let output = graph
        .last_node()
        .map_err(|source| InferenceError::ggml("Graph::last_node", source))?
        .read_data_backend::<T>()
        .map_err(|source| InferenceError::ggml("Tensor::read_data_backend", source))?;

    Ok(MlpInferenceReport {
        backend_name,
        hidden_features: hidden,
        ffn_features: ffn,
        repeats,
        output,
    })
}

fn recommended_mlp_backend_memory_bytes<T: GgmlElement>(
    hidden_features: usize,
    ffn_features: usize,
) -> Result<ggml_rs::Bytes, InferenceError> {
    let gate_matmul = Context::recommended_backend_matmul_memory::<T>(
        Shape2D::new(hidden_features, ffn_features),
        Shape2D::new(hidden_features, 1),
    )
    .map_err(|source| {
        InferenceError::ggml("Context::recommended_backend_matmul_memory(gate)", source)
    })?;
    let down_matmul = Context::recommended_backend_matmul_memory::<T>(
        Shape2D::new(ffn_features, hidden_features),
        Shape2D::new(ffn_features, 1),
    )
    .map_err(|source| {
        InferenceError::ggml("Context::recommended_backend_matmul_memory(down)", source)
    })?;

    let total = gate_matmul
        .get()
        .checked_add(down_matmul.get())
        .and_then(|value| value.checked_add(1024 * 1024))
        .ok_or(InferenceError::MemorySizeOverflow)?;
    Ok(ggml_rs::Bytes::new(total))
}

define_non_zero_count!(
    /// Strongly typed non-zero attention head count.
    AttentionHeadCount,
    |value| InferenceError::InvalidAttentionLayout {
        hidden_features: 0,
        query_head_count: value,
        kv_head_count: value,
    }
);

define_non_zero_count!(
    /// Strongly typed non-zero attention head dimension.
    AttentionHeadDimension,
    |value| InferenceError::InvalidAttentionShape {
        hidden_features: value,
        sequence_length: 0,
    }
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Attention topology expressed with explicit query/KV head counts.
pub struct AttentionLayout {
    hidden_features: NonZeroUsize,
    query_head_count: AttentionHeadCount,
    kv_head_count: AttentionHeadCount,
    head_dimension: AttentionHeadDimension,
}

impl AttentionLayout {
    pub fn from_hidden_features(
        hidden_features: usize,
        query_head_count: usize,
        kv_head_count: usize,
    ) -> Result<Self, InferenceError> {
        if hidden_features == 0
            || query_head_count == 0
            || !hidden_features.is_multiple_of(query_head_count)
        {
            return Err(InferenceError::InvalidAttentionLayout {
                hidden_features,
                query_head_count,
                kv_head_count,
            });
        }
        Self::from_projection_dimensions(
            hidden_features,
            query_head_count,
            kv_head_count,
            hidden_features / query_head_count,
        )
    }

    pub fn from_projection_dimensions(
        hidden_features: usize,
        query_head_count: usize,
        kv_head_count: usize,
        head_dimension: usize,
    ) -> Result<Self, InferenceError> {
        if hidden_features == 0
            || query_head_count == 0
            || kv_head_count == 0
            || !query_head_count.is_multiple_of(kv_head_count)
        {
            return Err(InferenceError::InvalidAttentionLayout {
                hidden_features,
                query_head_count,
                kv_head_count,
            });
        }

        Ok(Self {
            hidden_features: NonZeroUsize::new(hidden_features).expect("validated non-zero"),
            query_head_count: AttentionHeadCount::new(query_head_count)?,
            kv_head_count: AttentionHeadCount::new(kv_head_count)?,
            head_dimension: AttentionHeadDimension::new(head_dimension)?,
        })
    }

    pub const fn query_head_count(self) -> usize {
        self.query_head_count.get()
    }

    pub const fn kv_head_count(self) -> usize {
        self.kv_head_count.get()
    }

    pub const fn head_dimension(self) -> usize {
        self.head_dimension.get()
    }

    pub const fn hidden_features(self) -> usize {
        self.hidden_features.get()
    }

    pub const fn query_features(self) -> usize {
        self.query_head_count.get() * self.head_dimension.get()
    }

    pub const fn kv_features(self) -> usize {
        self.kv_head_count.get() * self.head_dimension.get()
    }

    pub const fn kv_group_size(self) -> usize {
        self.query_head_count.get() / self.kv_head_count.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Attention mask policy.
pub enum AttentionMaskPolicy {
    None,
    Causal { past_tokens: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Rotary embedding parameters for LLaMA-style RoPE.
pub struct RopeConfig {
    pub dimensions: AttentionHeadDimension,
    pub base: f32,
    pub scale: f32,
    pub original_context: Option<NonZeroUsize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Rotary embedding policy.
pub enum RotaryEmbedding {
    Disabled,
    Llama(RopeConfig),
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Configuration for multi-head self-attention inference.
pub struct AttentionInferenceConfig {
    pub layout: AttentionLayout,
    pub sequence_length: NonZeroUsize,
    pub mask: AttentionMaskPolicy,
    pub rotary: RotaryEmbedding,
    pub attention_scale: f32,
    pub rms_norm_eps: f32,
}

impl AttentionInferenceConfig {
    /// Backward-compatible constructor using a single attention head.
    pub fn new(hidden_features: usize, sequence_length: usize) -> Result<Self, InferenceError> {
        let layout = AttentionLayout::from_hidden_features(hidden_features, 1, 1)?;
        Self::from_layout(layout, sequence_length)
    }

    pub fn from_layout(
        layout: AttentionLayout,
        sequence_length: usize,
    ) -> Result<Self, InferenceError> {
        let sequence_length =
            NonZeroUsize::new(sequence_length).ok_or(InferenceError::InvalidAttentionShape {
                hidden_features: layout.hidden_features(),
                sequence_length,
            })?;
        Ok(Self {
            layout,
            sequence_length,
            mask: AttentionMaskPolicy::None,
            rotary: RotaryEmbedding::Disabled,
            attention_scale: 1.0 / (layout.head_dimension() as f32).sqrt(),
            rms_norm_eps: 1e-5,
        })
    }

    pub fn from_layer_dimensions(
        dimensions: LlamaLayerDimensions,
        sequence_length: usize,
    ) -> Result<Self, InferenceError> {
        let layout = AttentionLayout::from_projection_dimensions(
            dimensions.hidden_features,
            dimensions.query_head_count,
            dimensions.kv_head_count,
            dimensions.head_dimension,
        )?;
        let mut config = Self::from_layout(layout, sequence_length)?;
        if let Some(rope_dimensions) = dimensions.rope_dimension_count {
            let rope_dimensions = AttentionHeadDimension::new(rope_dimensions)?;
            config = config.with_rotary(RotaryEmbedding::Llama(RopeConfig {
                dimensions: rope_dimensions,
                base: dimensions.rope_freq_base,
                scale: dimensions.rope_freq_scale,
                original_context: dimensions
                    .rope_original_context_length
                    .and_then(NonZeroUsize::new),
            }));
        }
        let config = if let Some(attention_scale) = dimensions.attention_scale {
            config.with_attention_scale(attention_scale)
        } else {
            config
        };
        Ok(config.with_rms_norm_eps(dimensions.attention_layer_norm_rms_epsilon))
    }

    pub const fn with_mask(mut self, mask: AttentionMaskPolicy) -> Self {
        self.mask = mask;
        self
    }

    pub const fn with_rotary(mut self, rotary: RotaryEmbedding) -> Self {
        self.rotary = rotary;
        self
    }

    pub const fn with_attention_scale(mut self, attention_scale: f32) -> Self {
        self.attention_scale = attention_scale;
        self
    }

    pub const fn with_rms_norm_eps(mut self, rms_norm_eps: f32) -> Self {
        self.rms_norm_eps = rms_norm_eps;
        self
    }

    pub const fn hidden_features(self) -> usize {
        self.layout.hidden_features()
    }

    pub const fn query_features(self) -> usize {
        self.layout.query_features()
    }

    pub const fn kv_features(self) -> usize {
        self.layout.kv_features()
    }

    pub const fn sequence_length(self) -> usize {
        self.sequence_length.get()
    }

    pub const fn attention_scale(self) -> f32 {
        self.attention_scale
    }

    pub const fn rms_norm_eps(self) -> f32 {
        self.rms_norm_eps
    }

    pub const fn query_head_count(self) -> usize {
        self.layout.query_head_count()
    }

    pub const fn kv_head_count(self) -> usize {
        self.layout.kv_head_count()
    }

    pub const fn head_dimension(self) -> usize {
        self.layout.head_dimension()
    }

    pub const fn expected_q_weight_len(self) -> usize {
        self.hidden_features() * self.query_features()
    }

    pub const fn expected_kv_weight_len(self) -> usize {
        self.hidden_features() * self.kv_features()
    }

    pub const fn expected_o_weight_len(self) -> usize {
        self.query_features() * self.hidden_features()
    }
}

#[derive(Debug, Clone)]
/// Reusable attention projection weights (`q`, `k`, `v`, `o`).
pub struct AttentionWeights<T = f32> {
    pub q_tensor_name: String,
    pub k_tensor_name: String,
    pub v_tensor_name: String,
    pub o_tensor_name: String,
    pub q_norm_tensor_name: Option<String>,
    pub k_norm_tensor_name: Option<String>,
    pub config: AttentionInferenceConfig,
    q_values: Vec<T>,
    k_values: Vec<T>,
    v_values: Vec<T>,
    o_values: Vec<T>,
    q_norm_values: Option<Vec<T>>,
    k_norm_values: Option<Vec<T>>,
}

impl AttentionWeights<f32> {
    /// Builds deterministic synthetic attention weights.
    pub fn deterministic(config: AttentionInferenceConfig) -> Self {
        let q_values = (0..config.expected_q_weight_len())
            .map(|index| (index % 31) as f32 * 0.01)
            .collect();
        let k_values = (0..config.expected_kv_weight_len())
            .map(|index| ((index + 7) % 29) as f32 * 0.011)
            .collect();
        let v_values = (0..config.expected_kv_weight_len())
            .map(|index| ((index + 13) % 23) as f32 * 0.013)
            .collect();
        let o_values = (0..config.expected_o_weight_len())
            .map(|index| ((index + 17) % 19) as f32 * 0.009)
            .collect();
        Self {
            q_tensor_name: "<deterministic-attn-q>".to_string(),
            k_tensor_name: "<deterministic-attn-k>".to_string(),
            v_tensor_name: "<deterministic-attn-v>".to_string(),
            o_tensor_name: "<deterministic-attn-o>".to_string(),
            q_norm_tensor_name: None,
            k_norm_tensor_name: None,
            config,
            q_values,
            k_values,
            v_values,
            o_values,
            q_norm_values: None,
            k_norm_values: None,
        }
    }
}

impl<T> AttentionWeights<T>
where
    T: GgmlElement + NumCast,
{
    /// Loads attention projection weights for one resolved transformer layer.
    pub fn from_model_layer(
        model: &GgufModel,
        layer: &LlamaLayerTensorNames,
        config: AttentionInferenceConfig,
    ) -> Result<Self, InferenceError> {
        let q_values = model
            .tensor_values::<T>(&layer.attn_q)
            .map_err(|source| InferenceError::model("GgufModel::tensor_values(attn_q)", source))?;
        let k_values = model
            .tensor_values::<T>(&layer.attn_k)
            .map_err(|source| InferenceError::model("GgufModel::tensor_values(attn_k)", source))?;
        let v_values = model
            .tensor_values::<T>(&layer.attn_v)
            .map_err(|source| InferenceError::model("GgufModel::tensor_values(attn_v)", source))?;
        let o_values = model
            .tensor_values::<T>(&layer.attn_output)
            .map_err(|source| {
                InferenceError::model("GgufModel::tensor_values(attn_output)", source)
            })?;
        let q_norm_values = layer
            .attn_q_norm
            .as_deref()
            .map(|tensor_name| {
                model.tensor_values::<T>(tensor_name).map_err(|source| {
                    InferenceError::model("GgufModel::tensor_values(attn_q_norm)", source)
                })
            })
            .transpose()?;
        let k_norm_values = layer
            .attn_k_norm
            .as_deref()
            .map(|tensor_name| {
                model.tensor_values::<T>(tensor_name).map_err(|source| {
                    InferenceError::model("GgufModel::tensor_values(attn_k_norm)", source)
                })
            })
            .transpose()?;

        for (name, expected, actual) in [
            (
                &layer.attn_q,
                config.expected_q_weight_len(),
                q_values.len(),
            ),
            (
                &layer.attn_k,
                config.expected_kv_weight_len(),
                k_values.len(),
            ),
            (
                &layer.attn_v,
                config.expected_kv_weight_len(),
                v_values.len(),
            ),
            (
                &layer.attn_output,
                config.expected_o_weight_len(),
                o_values.len(),
            ),
        ] {
            if expected != actual {
                return Err(InferenceError::InvalidAttentionWeightShape {
                    tensor_name: name.clone(),
                    expected,
                    actual,
                });
            }
        }
        for (name, values) in [
            (layer.attn_q_norm.as_deref(), q_norm_values.as_ref()),
            (layer.attn_k_norm.as_deref(), k_norm_values.as_ref()),
        ] {
            if let (Some(name), Some(values)) = (name, values)
                && values.len() != config.head_dimension()
            {
                return Err(InferenceError::InvalidAttentionWeightShape {
                    tensor_name: name.to_string(),
                    expected: config.head_dimension(),
                    actual: values.len(),
                });
            }
        }

        Ok(Self {
            q_tensor_name: layer.attn_q.clone(),
            k_tensor_name: layer.attn_k.clone(),
            v_tensor_name: layer.attn_v.clone(),
            o_tensor_name: layer.attn_output.clone(),
            q_norm_tensor_name: layer.attn_q_norm.clone(),
            k_norm_tensor_name: layer.attn_k_norm.clone(),
            config,
            q_values,
            k_values,
            v_values,
            o_values,
            q_norm_values,
            k_norm_values,
        })
    }

    fn q_values(&self) -> &[T] {
        &self.q_values
    }

    fn k_values(&self) -> &[T] {
        &self.k_values
    }

    fn v_values(&self) -> &[T] {
        &self.v_values
    }

    fn o_values(&self) -> &[T] {
        &self.o_values
    }

    fn q_norm_values(&self) -> Option<&[T]> {
        self.q_norm_values.as_deref()
    }

    fn k_norm_values(&self) -> Option<&[T]> {
        self.k_norm_values.as_deref()
    }
}

#[derive(Debug, Clone)]
/// Output payload and execution metadata for attention inference.
pub struct AttentionInferenceReport<T = f32> {
    pub backend_name: String,
    pub hidden_features: usize,
    pub sequence_length: usize,
    pub repeats: usize,
    pub output: Vec<T>,
}

#[derive(Debug, Clone)]
/// Output payload and execution metadata for decode-like attention proxy inference.
pub struct AttentionDecodeProxyReport<T = f32> {
    pub backend_name: String,
    pub hidden_features: usize,
    pub query_length: usize,
    pub key_value_length: usize,
    pub repeats: usize,
    pub output: Vec<T>,
}

#[derive(Debug, Clone)]
/// Reusable projected KV cache for decode-like attention proxy runs.
pub struct AttentionDecodeCache<T = f32> {
    key_value_length: usize,
    kv_features: usize,
    projected_k_values: Vec<T>,
    projected_v_values: Vec<T>,
}

impl<T: Clone> AttentionDecodeCache<T> {
    pub const fn key_value_length(&self) -> usize {
        self.key_value_length
    }

    pub const fn kv_features(&self) -> usize {
        self.kv_features
    }

    pub fn prefix(&self, key_value_length: usize) -> Result<Self, InferenceError> {
        if key_value_length == 0 || key_value_length > self.key_value_length {
            return Err(InferenceError::InvalidInputLength {
                expected: self.key_value_length,
                actual: key_value_length,
            });
        }
        let projected_len = self
            .kv_features
            .checked_mul(key_value_length)
            .ok_or(InferenceError::MemorySizeOverflow)?;
        Ok(Self {
            key_value_length,
            kv_features: self.kv_features,
            projected_k_values: self.projected_k_values[..projected_len].to_vec(),
            projected_v_values: self.projected_v_values[..projected_len].to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionDecodeCache, AttentionInferenceConfig, AttentionLayout, InferenceError,
        build_causal_mask_values, infer_attention_layout_from_features,
    };

    #[test]
    fn attention_layout_validates_grouped_heads() {
        let layout = AttentionLayout::from_hidden_features(64, 8, 2)
            .expect("layout should accept grouped kv heads");
        assert_eq!(layout.head_dimension(), 8);
        assert_eq!(layout.query_head_count(), 8);
        assert_eq!(layout.kv_head_count(), 2);

        let error = AttentionLayout::from_hidden_features(64, 7, 2)
            .expect_err("head count must divide hidden features");
        assert!(matches!(
            error,
            InferenceError::InvalidAttentionLayout { .. }
        ));
    }

    #[test]
    fn attention_layout_supports_projection_width_larger_than_hidden_per_head() {
        let layout = AttentionLayout::from_projection_dimensions(3072, 32, 8, 128)
            .expect("layout should accept metadata-derived projection width");
        assert_eq!(layout.hidden_features(), 3072);
        assert_eq!(layout.head_dimension(), 128);
        assert_eq!(layout.query_features(), 4096);
        assert_eq!(layout.kv_features(), 1024);
    }

    #[test]
    fn causal_mask_blocks_future_tokens() {
        let mask = build_causal_mask_values(3, 3, 0);
        assert_eq!(
            mask,
            vec![0.0, -1.0e9, -1.0e9, 0.0, 0.0, -1.0e9, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn causal_mask_supports_decode_like_rectangular_shape() {
        let mask = build_causal_mask_values(1, 8, 7);
        assert_eq!(mask, vec![0.0; 8]);
    }

    #[test]
    fn config_uses_single_head_compat_constructor() {
        let config = AttentionInferenceConfig::new(16, 5).expect("single head config should build");
        assert_eq!(config.query_head_count(), 1);
        assert_eq!(config.kv_head_count(), 1);
        assert_eq!(config.head_dimension(), 16);
        assert_eq!(config.sequence_length(), 5);
        assert_eq!(config.attention_scale(), 0.25);
    }

    #[test]
    fn infers_grouped_layout_without_metadata() {
        let inferred =
            infer_attention_layout_from_features(4096, 1024).expect("layout should be inferred");
        assert_eq!(inferred.query_head_count, 32);
        assert_eq!(inferred.kv_head_count, 8);
    }

    #[test]
    fn decode_cache_prefix_truncates_projected_values() {
        let cache = AttentionDecodeCache {
            key_value_length: 4,
            kv_features: 3,
            projected_k_values: (0..12).map(|value| value as f32).collect(),
            projected_v_values: (100..112).map(|value| value as f32).collect(),
        };
        let prefix = cache.prefix(2).expect("prefix cache");
        assert_eq!(prefix.key_value_length(), 2);
        assert_eq!(prefix.kv_features(), 3);
        assert_eq!(
            prefix.projected_k_values,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
        assert_eq!(
            prefix.projected_v_values,
            vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        );
    }
}
