//! Minimal inference-path helpers.
//!
//! The current scope is a single linear layer execution path backed by ggml
//! safe APIs. This acts as the first inference building block before full
//! transformer graph assembly.

use crate::backend::LlamaBackend;
use crate::metadata::{LlamaModelMetadata, MetadataError, resolve_llama_metadata};
use crate::model::{GgufModel, ModelError};
use crate::naming::{LlamaLayerTensorNames, NamingError, resolve_llama_layer_tensor_names};
use ggml_rs::{Backend, Context, RopeExtParams, Shape2D};
use std::error::Error as StdError;
use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Strongly typed non-zero input feature count.
pub struct InFeatures(NonZeroUsize);

impl InFeatures {
    pub fn new(value: usize) -> Result<Self, InferenceError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(InferenceError::InvalidLinearShape {
                in_features: value,
                out_features: 0,
            })
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Strongly typed non-zero output feature count.
pub struct OutFeatures(NonZeroUsize);

impl OutFeatures {
    pub fn new(value: usize) -> Result<Self, InferenceError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(InferenceError::InvalidLinearShape {
                in_features: 0,
                out_features: value,
            })
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

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
pub struct LinearInferenceReport {
    pub backend_name: String,
    pub in_features: usize,
    pub out_features: usize,
    pub repeats: usize,
    pub output: Vec<f32>,
}

#[derive(Debug, Clone)]
/// Pre-decoded linear weights for repeated inference calls.
pub struct LinearWeights {
    pub tensor_name: String,
    pub in_features: usize,
    pub out_features: usize,
    values: Vec<f32>,
}

impl LinearWeights {
    /// Decodes a GGUF tensor into reusable linear weights.
    pub fn from_model(
        model: &GgufModel,
        tensor_name: &str,
        config: LinearInferenceConfig,
    ) -> Result<Self, InferenceError> {
        let expected_weights = config.expected_weight_len();
        let values = model
            .tensor_f32_values(tensor_name)
            .map_err(|source| InferenceError::model("GgufModel::tensor_f32_values", source))?;
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

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Strongly typed non-zero hidden feature count for MLP-style block inference.
pub struct HiddenFeatures(NonZeroUsize);

impl HiddenFeatures {
    pub fn new(value: usize) -> Result<Self, InferenceError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(InferenceError::InvalidMlpShape {
                hidden_features: value,
                ffn_features: 0,
            })
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Strongly typed non-zero FFN feature count for MLP-style block inference.
pub struct FfnFeatures(NonZeroUsize);

impl FfnFeatures {
    pub fn new(value: usize) -> Result<Self, InferenceError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(InferenceError::InvalidMlpShape {
                hidden_features: 0,
                ffn_features: value,
            })
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

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
pub struct MlpWeights {
    pub gate_tensor_name: String,
    pub up_tensor_name: String,
    pub down_tensor_name: String,
    pub hidden_features: usize,
    pub ffn_features: usize,
    gate_values: Vec<f32>,
    up_values: Vec<f32>,
    down_values: Vec<f32>,
}

impl MlpWeights {
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

    /// Loads MLP weights from three GGUF tensors (`gate`, `up`, `down`).
    pub fn from_model(
        model: &GgufModel,
        gate_tensor_name: &str,
        up_tensor_name: &str,
        down_tensor_name: &str,
        config: MlpInferenceConfig,
    ) -> Result<Self, InferenceError> {
        let gate_values = model
            .tensor_f32_values(gate_tensor_name)
            .map_err(|source| {
                InferenceError::model("GgufModel::tensor_f32_values(gate)", source)
            })?;
        let up_values = model
            .tensor_f32_values(up_tensor_name)
            .map_err(|source| InferenceError::model("GgufModel::tensor_f32_values(up)", source))?;
        let down_values = model
            .tensor_f32_values(down_tensor_name)
            .map_err(|source| {
                InferenceError::model("GgufModel::tensor_f32_values(down)", source)
            })?;

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
        let gate_values = model.tensor_f32_values(&layer.ffn_gate).map_err(|source| {
            InferenceError::model("GgufModel::tensor_f32_values(ffn_gate)", source)
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
        let up_values = model.tensor_f32_values(&layer.ffn_up).map_err(|source| {
            InferenceError::model("GgufModel::tensor_f32_values(ffn_up)", source)
        })?;
        let down_values = model.tensor_f32_values(&layer.ffn_down).map_err(|source| {
            InferenceError::model("GgufModel::tensor_f32_values(ffn_down)", source)
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
        gate_values: Vec<f32>,
        up_values: Vec<f32>,
        down_values: Vec<f32>,
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

    pub fn gate_values(&self) -> &[f32] {
        &self.gate_values
    }

    pub fn up_values(&self) -> &[f32] {
        &self.up_values
    }

    pub fn down_values(&self) -> &[f32] {
        &self.down_values
    }
}

#[derive(Debug, Clone)]
/// Output payload and execution metadata for MLP block inference.
pub struct MlpInferenceReport {
    pub backend_name: String,
    pub hidden_features: usize,
    pub ffn_features: usize,
    pub repeats: usize,
    pub output: Vec<f32>,
}

#[derive(Debug)]
/// Errors surfaced by inference helpers.
pub enum InferenceError {
    Model {
        context: &'static str,
        source: ModelError,
    },
    Metadata {
        context: &'static str,
        source: MetadataError,
    },
    Naming {
        context: &'static str,
        source: NamingError,
    },
    Ggml {
        context: &'static str,
        source: ggml_rs::Error,
    },
    InvalidWeightLength {
        tensor_name: String,
        expected: usize,
        actual: usize,
    },
    InvalidInputLength {
        expected: usize,
        actual: usize,
    },
    InvalidLinearShape {
        in_features: usize,
        out_features: usize,
    },
    InvalidMlpShape {
        hidden_features: usize,
        ffn_features: usize,
    },
    InvalidMlpWeightShape {
        tensor_name: String,
        hidden_features: usize,
        weight_len: usize,
    },
    InvalidAttentionShape {
        hidden_features: usize,
        sequence_length: usize,
    },
    InvalidAttentionLayout {
        hidden_features: usize,
        query_head_count: usize,
        kv_head_count: usize,
    },
    InvalidAttentionWeightShape {
        tensor_name: String,
        expected: usize,
        actual: usize,
    },
    InvalidRopeDimensions {
        rope_dimensions: usize,
        head_dimension: usize,
    },
    LayerNotFound {
        layer: usize,
    },
    MemorySizeOverflow,
    InvalidRepeats,
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model { context, source } => write!(f, "{context}: {source}"),
            Self::Metadata { context, source } => write!(f, "{context}: {source}"),
            Self::Naming { context, source } => write!(f, "{context}: {source}"),
            Self::Ggml { context, source } => write!(f, "{context}: {source}"),
            Self::InvalidWeightLength {
                tensor_name,
                expected,
                actual,
            } => write!(
                f,
                "weight tensor `{tensor_name}` length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidInputLength { expected, actual } => write!(
                f,
                "input length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidLinearShape {
                in_features,
                out_features,
            } => write!(
                f,
                "invalid linear shape: in_features={in_features}, out_features={out_features}"
            ),
            Self::InvalidMlpShape {
                hidden_features,
                ffn_features,
            } => write!(
                f,
                "invalid MLP shape: hidden_features={hidden_features}, ffn_features={ffn_features}"
            ),
            Self::InvalidMlpWeightShape {
                tensor_name,
                hidden_features,
                weight_len,
            } => write!(
                f,
                "cannot infer MLP shape from tensor `{tensor_name}`: hidden_features={hidden_features}, weight_len={weight_len}"
            ),
            Self::InvalidAttentionShape {
                hidden_features,
                sequence_length,
            } => write!(
                f,
                "invalid attention shape: hidden_features={hidden_features}, sequence_length={sequence_length}"
            ),
            Self::InvalidAttentionLayout {
                hidden_features,
                query_head_count,
                kv_head_count,
            } => write!(
                f,
                "invalid attention layout: hidden_features={hidden_features}, query_head_count={query_head_count}, kv_head_count={kv_head_count}"
            ),
            Self::InvalidAttentionWeightShape {
                tensor_name,
                expected,
                actual,
            } => write!(
                f,
                "invalid attention weight tensor `{tensor_name}` length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidRopeDimensions {
                rope_dimensions,
                head_dimension,
            } => write!(
                f,
                "invalid rope dimensions: rope_dimensions={rope_dimensions}, head_dimension={head_dimension}"
            ),
            Self::LayerNotFound { layer } => {
                write!(f, "layer not found in resolved catalog: {layer}")
            }
            Self::MemorySizeOverflow => {
                write!(f, "memory size overflow while building inference graph")
            }
            Self::InvalidRepeats => write!(f, "repeats must be greater than zero"),
        }
    }
}

impl StdError for InferenceError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Model { source, .. } => Some(source),
            Self::Metadata { source, .. } => Some(source),
            Self::Naming { source, .. } => Some(source),
            Self::Ggml { source, .. } => Some(source),
            Self::InvalidWeightLength { .. }
            | Self::InvalidInputLength { .. }
            | Self::InvalidLinearShape { .. }
            | Self::InvalidMlpShape { .. }
            | Self::InvalidMlpWeightShape { .. }
            | Self::InvalidAttentionShape { .. }
            | Self::InvalidAttentionLayout { .. }
            | Self::InvalidAttentionWeightShape { .. }
            | Self::InvalidRopeDimensions { .. }
            | Self::LayerNotFound { .. }
            | Self::MemorySizeOverflow
            | Self::InvalidRepeats => None,
        }
    }
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
pub fn run_linear_inference(
    model: &GgufModel,
    weight_tensor_name: &str,
    input: &[f32],
    config: LinearInferenceConfig,
    backend_kind: LlamaBackend,
) -> Result<LinearInferenceReport, InferenceError> {
    let weights = LinearWeights::from_model(model, weight_tensor_name, config)?;
    run_linear_inference_with_weights(&weights, input, backend_kind)
}

/// Runs one linear projection using pre-decoded reusable weights.
pub fn run_linear_inference_with_weights(
    weights: &LinearWeights,
    input: &[f32],
    backend_kind: LlamaBackend,
) -> Result<LinearInferenceReport, InferenceError> {
    run_linear_inference_with_weights_repeats(weights, input, backend_kind, 1)
}

/// Runs one linear projection using pre-decoded reusable weights and repeated
/// backend execution on the same graph.
pub fn run_linear_inference_with_weights_repeats(
    weights: &LinearWeights,
    input: &[f32],
    backend_kind: LlamaBackend,
    repeats: usize,
) -> Result<LinearInferenceReport, InferenceError> {
    if input.len() != weights.in_features {
        return Err(InferenceError::InvalidInputLength {
            expected: weights.in_features,
            actual: input.len(),
        });
    }
    if repeats == 0 {
        return Err(InferenceError::InvalidRepeats);
    }

    Backend::load_all();
    let backend = Backend::new(backend_kind.into())
        .map_err(|source| InferenceError::ggml("Backend::new", source))?;
    let backend_name = backend
        .name()
        .map_err(|source| InferenceError::ggml("Backend::name", source))?
        .to_string();

    let weight_shape = Shape2D::new(weights.in_features, weights.out_features);
    let input_shape = Shape2D::new(weights.in_features, 1);
    let ctx_size =
        Context::recommended_backend_matmul_memory_f32_shapes_bytes(weight_shape, input_shape)
            .map_err(|source| {
                InferenceError::ggml(
                    "Context::recommended_backend_matmul_memory_f32_shapes_bytes",
                    source,
                )
            })?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| InferenceError::ggml("Context::new_no_alloc_bytes", source))?;

    let w = ctx
        .new_f32_tensor_2d_shape(weight_shape)
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W>", source))?;
    let x = ctx
        .new_f32_tensor_2d_shape(input_shape)
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<X>", source))?;
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

    w.set_f32_backend(weights.values())
        .map_err(|source| InferenceError::ggml("Tensor::set_f32_backend<W>", source))?;
    x.set_f32_backend(input)
        .map_err(|source| InferenceError::ggml("Tensor::set_f32_backend<X>", source))?;
    for _ in 0..repeats {
        backend
            .compute(&mut graph)
            .map_err(|source| InferenceError::ggml("Backend::compute", source))?;
    }

    let output = graph
        .last_node()
        .map_err(|source| InferenceError::ggml("Graph::last_node", source))?
        .to_vec_f32_backend()
        .map_err(|source| InferenceError::ggml("Tensor::to_vec_f32_backend", source))?;

    Ok(LinearInferenceReport {
        backend_name,
        in_features: weights.in_features,
        out_features: weights.out_features,
        repeats,
        output,
    })
}

/// Runs a minimal MLP block `down(silu(gate(x)) * up(x))` from GGUF tensors.
pub fn run_mlp_inference(
    model: &GgufModel,
    gate_tensor_name: &str,
    up_tensor_name: &str,
    down_tensor_name: &str,
    input: &[f32],
    config: MlpInferenceConfig,
    backend_kind: LlamaBackend,
) -> Result<MlpInferenceReport, InferenceError> {
    let weights = MlpWeights::from_model(
        model,
        gate_tensor_name,
        up_tensor_name,
        down_tensor_name,
        config,
    )?;
    run_mlp_inference_with_weights(&weights, input, backend_kind)
}

/// Resolves a transformer layer by index from GGUF tensor names and runs the
/// minimal MLP block for that layer.
pub fn run_mlp_inference_for_layer(
    model: &GgufModel,
    layer: usize,
    input: &[f32],
    backend_kind: LlamaBackend,
) -> Result<MlpInferenceReport, InferenceError> {
    run_mlp_inference_for_layer_repeats(model, layer, input, backend_kind, 1)
}

/// Resolves and decodes reusable MLP weights for one transformer layer.
pub fn resolve_mlp_weights_for_layer(
    model: &GgufModel,
    layer: usize,
    hidden_features: usize,
) -> Result<MlpWeights, InferenceError> {
    let layer_names = resolve_llama_layer_tensor_names(model, layer)
        .map_err(|source| InferenceError::naming("resolve_llama_layer_tensor_names", source))?;
    MlpWeights::from_model_layer(model, &layer_names, hidden_features)
}

/// Resolves and decodes reusable MLP weights using GGUF metadata-derived hidden width.
pub fn resolve_mlp_weights_for_layer_auto(
    model: &GgufModel,
    layer: usize,
) -> Result<MlpWeights, InferenceError> {
    let dimensions = resolve_llama_layer_dimensions(model, layer)?;
    resolve_mlp_weights_for_layer(model, layer, dimensions.hidden_features)
}

/// Same as [`run_mlp_inference_for_layer`] with explicit repeated backend runs.
pub fn run_mlp_inference_for_layer_repeats(
    model: &GgufModel,
    layer: usize,
    input: &[f32],
    backend_kind: LlamaBackend,
    repeats: usize,
) -> Result<MlpInferenceReport, InferenceError> {
    let weights = resolve_mlp_weights_for_layer_auto(model, layer)?;
    run_mlp_inference_with_weights_repeats(&weights, input, backend_kind, repeats)
}

/// Runs a minimal MLP block using pre-decoded reusable weights.
pub fn run_mlp_inference_with_weights(
    weights: &MlpWeights,
    input: &[f32],
    backend_kind: LlamaBackend,
) -> Result<MlpInferenceReport, InferenceError> {
    run_mlp_inference_with_weights_repeats(weights, input, backend_kind, 1)
}

/// Runs a minimal MLP block using pre-decoded reusable weights and repeated
/// backend execution on a single graph.
pub fn run_mlp_inference_with_weights_repeats(
    weights: &MlpWeights,
    input: &[f32],
    backend_kind: LlamaBackend,
    repeats: usize,
) -> Result<MlpInferenceReport, InferenceError> {
    if input.len() != weights.hidden_features {
        return Err(InferenceError::InvalidInputLength {
            expected: weights.hidden_features,
            actual: input.len(),
        });
    }
    if repeats == 0 {
        return Err(InferenceError::InvalidRepeats);
    }

    Backend::load_all();
    let backend = Backend::new(backend_kind.into())
        .map_err(|source| InferenceError::ggml("Backend::new", source))?;
    let backend_name = backend
        .name()
        .map_err(|source| InferenceError::ggml("Backend::name", source))?
        .to_string();

    let hidden = weights.hidden_features;
    let ffn = weights.ffn_features;
    let ctx_size = recommended_mlp_backend_memory_bytes(hidden, ffn)?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| InferenceError::ggml("Context::new_no_alloc_bytes", source))?;

    let gate_shape = Shape2D::new(hidden, ffn);
    let up_shape = Shape2D::new(hidden, ffn);
    let down_shape = Shape2D::new(ffn, hidden);
    let input_shape = Shape2D::new(hidden, 1);

    let w_gate = ctx.new_f32_tensor_2d_shape(gate_shape).map_err(|source| {
        InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_GATE>", source)
    })?;
    let w_up = ctx
        .new_f32_tensor_2d_shape(up_shape)
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_UP>", source))?;
    let w_down = ctx.new_f32_tensor_2d_shape(down_shape).map_err(|source| {
        InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_DOWN>", source)
    })?;
    let x = ctx
        .new_f32_tensor_2d_shape(input_shape)
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<X>", source))?;

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
        .set_f32_backend(weights.gate_values())
        .map_err(|source| InferenceError::ggml("Tensor::set_f32_backend<W_GATE>", source))?;
    w_up.set_f32_backend(weights.up_values())
        .map_err(|source| InferenceError::ggml("Tensor::set_f32_backend<W_UP>", source))?;
    w_down
        .set_f32_backend(weights.down_values())
        .map_err(|source| InferenceError::ggml("Tensor::set_f32_backend<W_DOWN>", source))?;
    x.set_f32_backend(input)
        .map_err(|source| InferenceError::ggml("Tensor::set_f32_backend<X>", source))?;

    for _ in 0..repeats {
        backend
            .compute(&mut graph)
            .map_err(|source| InferenceError::ggml("Backend::compute", source))?;
    }

    let output = graph
        .last_node()
        .map_err(|source| InferenceError::ggml("Graph::last_node", source))?
        .to_vec_f32_backend()
        .map_err(|source| InferenceError::ggml("Tensor::to_vec_f32_backend", source))?;

    Ok(MlpInferenceReport {
        backend_name,
        hidden_features: hidden,
        ffn_features: ffn,
        repeats,
        output,
    })
}

fn recommended_mlp_backend_memory_bytes(
    hidden_features: usize,
    ffn_features: usize,
) -> Result<ggml_rs::Bytes, InferenceError> {
    let gate_matmul = Context::recommended_backend_matmul_memory_f32_shapes_bytes(
        Shape2D::new(hidden_features, ffn_features),
        Shape2D::new(hidden_features, 1),
    )
    .map_err(|source| {
        InferenceError::ggml(
            "Context::recommended_backend_matmul_memory_f32_shapes_bytes(gate)",
            source,
        )
    })?;
    let down_matmul = Context::recommended_backend_matmul_memory_f32_shapes_bytes(
        Shape2D::new(ffn_features, hidden_features),
        Shape2D::new(ffn_features, 1),
    )
    .map_err(|source| {
        InferenceError::ggml(
            "Context::recommended_backend_matmul_memory_f32_shapes_bytes(down)",
            source,
        )
    })?;

    let total = gate_matmul
        .get()
        .checked_add(down_matmul.get())
        .and_then(|value| value.checked_add(1024 * 1024))
        .ok_or(InferenceError::MemorySizeOverflow)?;
    Ok(ggml_rs::Bytes::new(total))
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Indicates whether layer dimensions were resolved from full metadata or heuristics.
pub enum MetadataResolutionMode {
    FullMetadata,
    TensorHeuristic,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Auto-resolved per-layer dimensions derived from GGUF metadata + tensor sizes.
pub struct LlamaLayerDimensions {
    pub resolution_mode: MetadataResolutionMode,
    pub hidden_features: usize,
    pub ffn_features: usize,
    pub query_head_count: usize,
    pub kv_head_count: usize,
    pub head_dimension: usize,
    pub rope_dimension_count: Option<usize>,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
}

/// Resolves per-layer dimensions from GGUF metadata and tensor lengths.
pub fn resolve_llama_layer_dimensions(
    model: &GgufModel,
    layer: usize,
) -> Result<LlamaLayerDimensions, InferenceError> {
    let layer_names = resolve_llama_layer_tensor_names(model, layer)
        .map_err(|source| InferenceError::naming("resolve_llama_layer_tensor_names", source))?;
    resolve_llama_layer_dimensions_from_names(model, &layer_names)
}

fn resolve_llama_layer_dimensions_from_names(
    model: &GgufModel,
    layer_names: &LlamaLayerTensorNames,
) -> Result<LlamaLayerDimensions, InferenceError> {
    let metadata = match resolve_llama_metadata(model) {
        Ok(metadata) => Some(metadata),
        Err(MetadataError::MissingRequiredKey { .. }) | Err(MetadataError::MissingArchitecture) => {
            None
        }
        Err(source) => {
            return Err(InferenceError::metadata("resolve_llama_metadata", source));
        }
    };
    let resolution_mode = if metadata.is_some() {
        MetadataResolutionMode::FullMetadata
    } else {
        MetadataResolutionMode::TensorHeuristic
    };

    let hidden_features = metadata.map_or_else(
        || infer_hidden_features_from_query_weight(model, layer_names),
        |metadata| Ok(metadata.embedding_length()),
    )?;
    let inferred_kv_features =
        infer_projection_output_features(model, &layer_names.attn_k, hidden_features)?;
    let inferred_layout =
        infer_attention_layout_from_features(hidden_features, inferred_kv_features)?;
    let query_head_count = metadata.map_or(inferred_layout.query_head_count, |metadata| {
        metadata.attention_head_count()
    });
    let kv_head_count = metadata.map_or(inferred_layout.kv_head_count, |metadata| {
        metadata.attention_head_count_kv()
    });
    if hidden_features == 0
        || query_head_count == 0
        || kv_head_count == 0
        || !hidden_features.is_multiple_of(query_head_count)
        || !query_head_count.is_multiple_of(kv_head_count)
    {
        return Err(InferenceError::InvalidAttentionLayout {
            hidden_features,
            query_head_count,
            kv_head_count,
        });
    }
    let head_dimension = hidden_features / query_head_count;
    let rope_dimension_count = metadata.and_then(LlamaModelMetadata::rope_dimension_count);
    if let Some(rope_dimensions) = rope_dimension_count
        && rope_dimensions > head_dimension
    {
        return Err(InferenceError::InvalidRopeDimensions {
            rope_dimensions,
            head_dimension,
        });
    }

    let ffn_features = infer_ffn_features(
        model,
        layer_names,
        hidden_features,
        metadata.and_then(LlamaModelMetadata::feed_forward_length),
    )?;
    validate_attention_projection_lengths(
        model,
        layer_names,
        hidden_features,
        query_head_count,
        kv_head_count,
        head_dimension,
    )?;

    Ok(LlamaLayerDimensions {
        resolution_mode,
        hidden_features,
        ffn_features,
        query_head_count,
        kv_head_count,
        head_dimension,
        rope_dimension_count,
        rope_freq_base: metadata.map_or(10_000.0, LlamaModelMetadata::rope_freq_base),
        rope_freq_scale: metadata.map_or(1.0, LlamaModelMetadata::rope_freq_scale),
    })
}

fn infer_hidden_features_from_query_weight(
    model: &GgufModel,
    layer_names: &LlamaLayerTensorNames,
) -> Result<usize, InferenceError> {
    let q_len = model
        .tensor_f32_len(&layer_names.attn_q)
        .map_err(|source| InferenceError::model("GgufModel::tensor_f32_len(attn_q)", source))?;
    let hidden = (q_len as f64).sqrt() as usize;
    if hidden == 0 || hidden * hidden != q_len {
        return Err(InferenceError::InvalidAttentionWeightShape {
            tensor_name: layer_names.attn_q.clone(),
            expected: hidden * hidden,
            actual: q_len,
        });
    }
    Ok(hidden)
}

fn infer_projection_output_features(
    model: &GgufModel,
    tensor_name: &str,
    hidden_features: usize,
) -> Result<usize, InferenceError> {
    let weight_len = model
        .tensor_f32_len(tensor_name)
        .map_err(|source| InferenceError::model("GgufModel::tensor_f32_len(attn_proj)", source))?;
    if hidden_features == 0 || !weight_len.is_multiple_of(hidden_features) {
        return Err(InferenceError::InvalidAttentionWeightShape {
            tensor_name: tensor_name.to_string(),
            expected: hidden_features.saturating_mul(weight_len / hidden_features.max(1)),
            actual: weight_len,
        });
    }
    Ok(weight_len / hidden_features)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InferredAttentionLayout {
    query_head_count: usize,
    kv_head_count: usize,
}

fn infer_attention_layout_from_features(
    hidden_features: usize,
    kv_features: usize,
) -> Result<InferredAttentionLayout, InferenceError> {
    const PREFERRED_HEAD_DIMENSIONS: &[usize] = &[128, 96, 80, 64, 48, 40, 32, 24, 16, 8, 4, 2, 1];

    for &head_dimension in PREFERRED_HEAD_DIMENSIONS {
        if !hidden_features.is_multiple_of(head_dimension)
            || !kv_features.is_multiple_of(head_dimension)
        {
            continue;
        }
        let query_head_count = hidden_features / head_dimension;
        let kv_head_count = kv_features / head_dimension;
        if query_head_count >= kv_head_count && query_head_count.is_multiple_of(kv_head_count) {
            return Ok(InferredAttentionLayout {
                query_head_count,
                kv_head_count,
            });
        }
    }

    let head_dimension = gcd(hidden_features, kv_features);
    if head_dimension == 0 {
        return Err(InferenceError::InvalidAttentionLayout {
            hidden_features,
            query_head_count: 0,
            kv_head_count: 0,
        });
    }
    let query_head_count = hidden_features / head_dimension;
    let kv_head_count = kv_features / head_dimension;
    if query_head_count >= kv_head_count && query_head_count.is_multiple_of(kv_head_count) {
        Ok(InferredAttentionLayout {
            query_head_count,
            kv_head_count,
        })
    } else {
        Err(InferenceError::InvalidAttentionLayout {
            hidden_features,
            query_head_count,
            kv_head_count,
        })
    }
}

fn gcd(mut lhs: usize, mut rhs: usize) -> usize {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}

fn infer_ffn_features(
    model: &GgufModel,
    layer_names: &LlamaLayerTensorNames,
    hidden_features: usize,
    metadata_ffn_features: Option<usize>,
) -> Result<usize, InferenceError> {
    let gate_len = model
        .tensor_f32_len(&layer_names.ffn_gate)
        .map_err(|source| InferenceError::model("GgufModel::tensor_f32_len(ffn_gate)", source))?;
    if gate_len % hidden_features != 0 {
        return Err(InferenceError::InvalidMlpWeightShape {
            tensor_name: layer_names.ffn_gate.clone(),
            hidden_features,
            weight_len: gate_len,
        });
    }
    let inferred_ffn = gate_len / hidden_features;
    let ffn_features = metadata_ffn_features.unwrap_or(inferred_ffn);
    let expected = hidden_features * ffn_features;
    if gate_len != expected {
        return Err(InferenceError::InvalidWeightLength {
            tensor_name: layer_names.ffn_gate.clone(),
            expected,
            actual: gate_len,
        });
    }
    for tensor_name in [&layer_names.ffn_up, &layer_names.ffn_down] {
        let weight_len = model
            .tensor_f32_len(tensor_name)
            .map_err(|source| InferenceError::model("GgufModel::tensor_f32_len(ffn)", source))?;
        if weight_len != expected {
            return Err(InferenceError::InvalidWeightLength {
                tensor_name: tensor_name.clone(),
                expected,
                actual: weight_len,
            });
        }
    }
    Ok(ffn_features)
}

fn validate_attention_projection_lengths(
    model: &GgufModel,
    layer_names: &LlamaLayerTensorNames,
    hidden_features: usize,
    query_head_count: usize,
    kv_head_count: usize,
    head_dimension: usize,
) -> Result<(), InferenceError> {
    let query_features = query_head_count
        .checked_mul(head_dimension)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let kv_features = kv_head_count
        .checked_mul(head_dimension)
        .ok_or(InferenceError::MemorySizeOverflow)?;

    let expected_q = hidden_features
        .checked_mul(query_features)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let expected_kv = hidden_features
        .checked_mul(kv_features)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let expected_o = query_features
        .checked_mul(hidden_features)
        .ok_or(InferenceError::MemorySizeOverflow)?;

    let q_len = model
        .tensor_f32_len(&layer_names.attn_q)
        .map_err(|source| InferenceError::model("GgufModel::tensor_f32_len(attn_q)", source))?;
    if q_len != expected_q {
        return Err(InferenceError::InvalidAttentionWeightShape {
            tensor_name: layer_names.attn_q.clone(),
            expected: expected_q,
            actual: q_len,
        });
    }
    let k_len = model
        .tensor_f32_len(&layer_names.attn_k)
        .map_err(|source| InferenceError::model("GgufModel::tensor_f32_len(attn_k)", source))?;
    if k_len != expected_kv {
        return Err(InferenceError::InvalidAttentionWeightShape {
            tensor_name: layer_names.attn_k.clone(),
            expected: expected_kv,
            actual: k_len,
        });
    }
    let v_len = model
        .tensor_f32_len(&layer_names.attn_v)
        .map_err(|source| InferenceError::model("GgufModel::tensor_f32_len(attn_v)", source))?;
    if v_len != expected_kv {
        return Err(InferenceError::InvalidAttentionWeightShape {
            tensor_name: layer_names.attn_v.clone(),
            expected: expected_kv,
            actual: v_len,
        });
    }
    let o_len = model
        .tensor_f32_len(&layer_names.attn_output)
        .map_err(|source| {
            InferenceError::model("GgufModel::tensor_f32_len(attn_output)", source)
        })?;
    if o_len != expected_o {
        return Err(InferenceError::InvalidAttentionWeightShape {
            tensor_name: layer_names.attn_output.clone(),
            expected: expected_o,
            actual: o_len,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Strongly typed non-zero attention head count.
pub struct AttentionHeadCount(NonZeroUsize);

impl AttentionHeadCount {
    pub fn new(value: usize) -> Result<Self, InferenceError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(InferenceError::InvalidAttentionLayout {
                hidden_features: 0,
                query_head_count: value,
                kv_head_count: value,
            })
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Strongly typed non-zero attention head dimension.
pub struct AttentionHeadDimension(NonZeroUsize);

impl AttentionHeadDimension {
    pub fn new(value: usize) -> Result<Self, InferenceError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(InferenceError::InvalidAttentionShape {
                hidden_features: value,
                sequence_length: 0,
            })
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Attention topology expressed with explicit query/KV head counts.
pub struct AttentionLayout {
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
            || kv_head_count == 0
            || !hidden_features.is_multiple_of(query_head_count)
            || !query_head_count.is_multiple_of(kv_head_count)
        {
            return Err(InferenceError::InvalidAttentionLayout {
                hidden_features,
                query_head_count,
                kv_head_count,
            });
        }

        let head_dimension = hidden_features / query_head_count;
        Ok(Self {
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
        self.query_head_count.get() * self.head_dimension.get()
    }

    pub const fn query_features(self) -> usize {
        self.hidden_features()
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
        })
    }

    pub fn from_layer_dimensions(
        dimensions: LlamaLayerDimensions,
        sequence_length: usize,
    ) -> Result<Self, InferenceError> {
        let layout = AttentionLayout::from_hidden_features(
            dimensions.hidden_features,
            dimensions.query_head_count,
            dimensions.kv_head_count,
        )?;
        let mut config = Self::from_layout(layout, sequence_length)?;
        if let Some(rope_dimensions) = dimensions.rope_dimension_count {
            let rope_dimensions = AttentionHeadDimension::new(rope_dimensions)?;
            config = config.with_rotary(RotaryEmbedding::Llama(RopeConfig {
                dimensions: rope_dimensions,
                base: dimensions.rope_freq_base,
                scale: dimensions.rope_freq_scale,
                original_context: None,
            }));
        }
        Ok(config)
    }

    pub const fn with_mask(mut self, mask: AttentionMaskPolicy) -> Self {
        self.mask = mask;
        self
    }

    pub const fn with_rotary(mut self, rotary: RotaryEmbedding) -> Self {
        self.rotary = rotary;
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
pub struct AttentionWeights {
    pub q_tensor_name: String,
    pub k_tensor_name: String,
    pub v_tensor_name: String,
    pub o_tensor_name: String,
    pub config: AttentionInferenceConfig,
    q_values: Vec<f32>,
    k_values: Vec<f32>,
    v_values: Vec<f32>,
    o_values: Vec<f32>,
}

impl AttentionWeights {
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
            config,
            q_values,
            k_values,
            v_values,
            o_values,
        }
    }

    /// Loads attention projection weights for one resolved transformer layer.
    pub fn from_model_layer(
        model: &GgufModel,
        layer: &LlamaLayerTensorNames,
        config: AttentionInferenceConfig,
    ) -> Result<Self, InferenceError> {
        let q_values = model.tensor_f32_values(&layer.attn_q).map_err(|source| {
            InferenceError::model("GgufModel::tensor_f32_values(attn_q)", source)
        })?;
        let k_values = model.tensor_f32_values(&layer.attn_k).map_err(|source| {
            InferenceError::model("GgufModel::tensor_f32_values(attn_k)", source)
        })?;
        let v_values = model.tensor_f32_values(&layer.attn_v).map_err(|source| {
            InferenceError::model("GgufModel::tensor_f32_values(attn_v)", source)
        })?;
        let o_values = model
            .tensor_f32_values(&layer.attn_output)
            .map_err(|source| {
                InferenceError::model("GgufModel::tensor_f32_values(attn_output)", source)
            })?;

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

        Ok(Self {
            q_tensor_name: layer.attn_q.clone(),
            k_tensor_name: layer.attn_k.clone(),
            v_tensor_name: layer.attn_v.clone(),
            o_tensor_name: layer.attn_output.clone(),
            config,
            q_values,
            k_values,
            v_values,
            o_values,
        })
    }

    fn q_values(&self) -> &[f32] {
        &self.q_values
    }

    fn k_values(&self) -> &[f32] {
        &self.k_values
    }

    fn v_values(&self) -> &[f32] {
        &self.v_values
    }

    fn o_values(&self) -> &[f32] {
        &self.o_values
    }
}

#[derive(Debug, Clone)]
/// Output payload and execution metadata for attention inference.
pub struct AttentionInferenceReport {
    pub backend_name: String,
    pub hidden_features: usize,
    pub sequence_length: usize,
    pub repeats: usize,
    pub output: Vec<f32>,
}

/// Resolves and decodes reusable attention weights for one transformer layer.
pub fn resolve_attention_weights_for_layer(
    model: &GgufModel,
    layer: usize,
    config: AttentionInferenceConfig,
) -> Result<AttentionWeights, InferenceError> {
    let layer_names = resolve_llama_layer_tensor_names(model, layer)
        .map_err(|source| InferenceError::naming("resolve_llama_layer_tensor_names", source))?;
    AttentionWeights::from_model_layer(model, &layer_names, config)
}

/// Resolves attention weights with auto-derived topology and optional RoPE metadata.
pub fn resolve_attention_weights_for_layer_auto(
    model: &GgufModel,
    layer: usize,
    sequence_length: usize,
) -> Result<AttentionWeights, InferenceError> {
    let dimensions = resolve_llama_layer_dimensions(model, layer)?;
    let config = AttentionInferenceConfig::from_layer_dimensions(dimensions, sequence_length)?;
    resolve_attention_weights_for_layer(model, layer, config)
}

/// Runs minimal self-attention from resolved GGUF layer tensors.
pub fn run_attention_inference_for_layer(
    model: &GgufModel,
    layer: usize,
    input: &[f32],
    config: AttentionInferenceConfig,
    backend_kind: LlamaBackend,
) -> Result<AttentionInferenceReport, InferenceError> {
    run_attention_inference_for_layer_repeats(model, layer, input, config, backend_kind, 1)
}

/// Runs minimal self-attention from resolved GGUF layer tensors with explicit repeats.
pub fn run_attention_inference_for_layer_repeats(
    model: &GgufModel,
    layer: usize,
    input: &[f32],
    config: AttentionInferenceConfig,
    backend_kind: LlamaBackend,
    repeats: usize,
) -> Result<AttentionInferenceReport, InferenceError> {
    let weights = resolve_attention_weights_for_layer(model, layer, config)?;
    run_attention_inference_with_weights_repeats(&weights, input, backend_kind, repeats)
}

/// Runs minimal self-attention with auto-derived topology and RoPE metadata.
pub fn run_attention_inference_for_layer_auto(
    model: &GgufModel,
    layer: usize,
    input: &[f32],
    sequence_length: usize,
    backend_kind: LlamaBackend,
) -> Result<AttentionInferenceReport, InferenceError> {
    run_attention_inference_for_layer_auto_repeats(
        model,
        layer,
        input,
        sequence_length,
        backend_kind,
        1,
    )
}

/// Runs minimal self-attention with auto-derived topology and explicit repeats.
pub fn run_attention_inference_for_layer_auto_repeats(
    model: &GgufModel,
    layer: usize,
    input: &[f32],
    sequence_length: usize,
    backend_kind: LlamaBackend,
    repeats: usize,
) -> Result<AttentionInferenceReport, InferenceError> {
    let weights = resolve_attention_weights_for_layer_auto(model, layer, sequence_length)?;
    run_attention_inference_with_weights_repeats(&weights, input, backend_kind, repeats)
}

/// Runs minimal self-attention with reusable decoded weights.
pub fn run_attention_inference_with_weights(
    weights: &AttentionWeights,
    input: &[f32],
    backend_kind: LlamaBackend,
) -> Result<AttentionInferenceReport, InferenceError> {
    run_attention_inference_with_weights_repeats(weights, input, backend_kind, 1)
}

/// Runs minimal self-attention with reusable decoded weights and explicit repeats.
pub fn run_attention_inference_with_weights_repeats(
    weights: &AttentionWeights,
    input: &[f32],
    backend_kind: LlamaBackend,
    repeats: usize,
) -> Result<AttentionInferenceReport, InferenceError> {
    let config = weights.config;
    let hidden_features = config.hidden_features();
    let sequence_length = config.sequence_length();
    let expected_input_len = hidden_features
        .checked_mul(sequence_length)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    if input.len() != expected_input_len {
        return Err(InferenceError::InvalidInputLength {
            expected: expected_input_len,
            actual: input.len(),
        });
    }
    if repeats == 0 {
        return Err(InferenceError::InvalidRepeats);
    }

    Backend::load_all();
    let backend = Backend::new(backend_kind.into())
        .map_err(|source| InferenceError::ggml("Backend::new", source))?;
    let backend_name = backend
        .name()
        .map_err(|source| InferenceError::ggml("Backend::name", source))?
        .to_string();

    let ctx_size = recommended_attention_backend_memory_bytes(config)?;
    let ctx = Context::new_no_alloc_bytes(ctx_size)
        .map_err(|source| InferenceError::ggml("Context::new_no_alloc_bytes", source))?;

    let query_features = config.query_features();
    let kv_features = config.kv_features();

    let w_q = ctx
        .new_f32_tensor_2d_shape(Shape2D::new(hidden_features, query_features))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_Q>", source))?;
    let w_k = ctx
        .new_f32_tensor_2d_shape(Shape2D::new(hidden_features, kv_features))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_K>", source))?;
    let w_v = ctx
        .new_f32_tensor_2d_shape(Shape2D::new(hidden_features, kv_features))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_V>", source))?;
    let w_o = ctx
        .new_f32_tensor_2d_shape(Shape2D::new(query_features, hidden_features))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<W_O>", source))?;
    let x = ctx
        .new_f32_tensor_2d_shape(Shape2D::new(hidden_features, sequence_length))
        .map_err(|source| InferenceError::ggml("Context::new_f32_tensor_2d_shape<X>", source))?;

    let q = ctx
        .mul_mat(&w_q, &x)
        .map_err(|source| InferenceError::ggml("Context::mul_mat(Q)", source))?;
    let k = ctx
        .mul_mat(&w_k, &x)
        .map_err(|source| InferenceError::ggml("Context::mul_mat(K)", source))?;
    let v = ctx
        .mul_mat(&w_v, &x)
        .map_err(|source| InferenceError::ggml("Context::mul_mat(V)", source))?;

    let positions = if matches!(config.rotary, RotaryEmbedding::Llama(_)) {
        Some(
            ctx.new_i32_tensor_1d(sequence_length)
                .map_err(|source| InferenceError::ggml("Context::new_i32_tensor_1d", source))?,
        )
    } else {
        None
    };
    let mask = if matches!(config.mask, AttentionMaskPolicy::Causal { .. }) {
        Some(
            ctx.new_f32_tensor_2d_shape(Shape2D::new(sequence_length, sequence_length))
                .map_err(|source| {
                    InferenceError::ggml("Context::new_f32_tensor_2d_shape<CAUSAL_MASK>", source)
                })?,
        )
    } else {
        None
    };

    let mut output_projection = None;
    let bytes_per_element = std::mem::size_of::<f32>();
    let q_row_stride = query_features
        .checked_mul(bytes_per_element)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let kv_row_stride = kv_features
        .checked_mul(bytes_per_element)
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let o_row_stride = q_row_stride;
    let attention_scale = 1.0 / (config.head_dimension() as f32).sqrt();

    for head in 0..config.query_head_count() {
        let query_offset = head
            .checked_mul(config.head_dimension())
            .and_then(|value| value.checked_mul(bytes_per_element))
            .ok_or(InferenceError::MemorySizeOverflow)?;
        let kv_head = head / config.layout.kv_group_size();
        let kv_offset = kv_head
            .checked_mul(config.head_dimension())
            .and_then(|value| value.checked_mul(bytes_per_element))
            .ok_or(InferenceError::MemorySizeOverflow)?;

        let q_head = ctx
            .view_2d(
                &q,
                config.head_dimension(),
                sequence_length,
                q_row_stride,
                query_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(Q_HEAD)", source))?;
        let k_head = ctx
            .view_2d(
                &k,
                config.head_dimension(),
                sequence_length,
                kv_row_stride,
                kv_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(K_HEAD)", source))?;
        let v_head = ctx
            .view_2d(
                &v,
                config.head_dimension(),
                sequence_length,
                kv_row_stride,
                kv_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(V_HEAD)", source))?;

        let q_head = apply_rotary_if_enabled(&ctx, &q_head, positions.as_ref(), config)?;
        let k_head = apply_rotary_if_enabled(&ctx, &k_head, positions.as_ref(), config)?;

        let scores = ctx
            .mul_mat(&k_head, &q_head)
            .map_err(|source| InferenceError::ggml("Context::mul_mat(K_HEAD*Q_HEAD)", source))?;

        let probabilities = match mask.as_ref() {
            Some(mask) => ctx
                .soft_max_ext(&scores, Some(mask), attention_scale, 0.0)
                .map_err(|source| InferenceError::ggml("Context::soft_max_ext(causal)", source))?,
            None => {
                let scaled = ctx
                    .scale(&scores, attention_scale)
                    .map_err(|source| InferenceError::ggml("Context::scale(scores)", source))?;
                ctx.soft_max(&scaled)
                    .map_err(|source| InferenceError::ggml("Context::soft_max", source))?
            }
        };
        let v_t = ctx
            .transpose(&v_head)
            .map_err(|source| InferenceError::ggml("Context::transpose(V_HEAD)", source))?;
        let v_t = ctx
            .cont(&v_t)
            .map_err(|source| InferenceError::ggml("Context::cont(V_HEAD)", source))?;
        let head_output = ctx
            .mul_mat(&v_t, &probabilities)
            .map_err(|source| InferenceError::ggml("Context::mul_mat(VT*P)", source))?;
        let w_o_head = ctx
            .view_2d(
                &w_o,
                config.head_dimension(),
                hidden_features,
                o_row_stride,
                query_offset,
            )
            .map_err(|source| InferenceError::ggml("Context::view_2d(W_O_HEAD)", source))?;
        let projected = ctx
            .mul_mat(&w_o_head, &head_output)
            .map_err(|source| InferenceError::ggml("Context::mul_mat(W_O_HEAD*HEAD)", source))?;

        output_projection = Some(if let Some(acc) = output_projection {
            ctx.add(&acc, &projected)
                .map_err(|source| InferenceError::ggml("Context::add(head_acc)", source))?
        } else {
            projected
        });
    }

    let y = output_projection.ok_or(InferenceError::InvalidAttentionLayout {
        hidden_features,
        query_head_count: config.query_head_count(),
        kv_head_count: config.kv_head_count(),
    })?;

    let mut graph = ctx
        .new_graph()
        .map_err(|source| InferenceError::ggml("Context::new_graph", source))?;
    graph.build_forward_expand(&y);
    let _buffer = ctx
        .allocate_tensors(&backend)
        .map_err(|source| InferenceError::ggml("Context::allocate_tensors", source))?;

    w_q.set_f32_backend(weights.q_values())
        .map_err(|source| InferenceError::ggml("Tensor::set_f32_backend<W_Q>", source))?;
    w_k.set_f32_backend(weights.k_values())
        .map_err(|source| InferenceError::ggml("Tensor::set_f32_backend<W_K>", source))?;
    w_v.set_f32_backend(weights.v_values())
        .map_err(|source| InferenceError::ggml("Tensor::set_f32_backend<W_V>", source))?;
    w_o.set_f32_backend(weights.o_values())
        .map_err(|source| InferenceError::ggml("Tensor::set_f32_backend<W_O>", source))?;
    x.set_f32_backend(input)
        .map_err(|source| InferenceError::ggml("Tensor::set_f32_backend<X>", source))?;

    if let Some(positions) = positions {
        let positions_values: Result<Vec<i32>, InferenceError> = (0..sequence_length)
            .map(|index| i32::try_from(index).map_err(|_| InferenceError::MemorySizeOverflow))
            .collect();
        positions
            .set_i32_backend(&positions_values?)
            .map_err(|source| InferenceError::ggml("Tensor::set_i32_backend<POSITIONS>", source))?;
    }
    if let Some(mask) = mask {
        let mask_values = build_causal_mask_values(
            sequence_length,
            match config.mask {
                AttentionMaskPolicy::None => 0,
                AttentionMaskPolicy::Causal { past_tokens } => past_tokens,
            },
        );
        mask.set_f32_backend(&mask_values).map_err(|source| {
            InferenceError::ggml("Tensor::set_f32_backend<CAUSAL_MASK>", source)
        })?;
    }

    for _ in 0..repeats {
        backend
            .compute(&mut graph)
            .map_err(|source| InferenceError::ggml("Backend::compute", source))?;
    }

    let output = graph
        .last_node()
        .map_err(|source| InferenceError::ggml("Graph::last_node", source))?
        .to_vec_f32_backend()
        .map_err(|source| InferenceError::ggml("Tensor::to_vec_f32_backend", source))?;

    Ok(AttentionInferenceReport {
        backend_name,
        hidden_features,
        sequence_length,
        repeats,
        output,
    })
}

fn apply_rotary_if_enabled<'ctx>(
    ctx: &'ctx Context,
    tensor: &ggml_rs::Tensor<'ctx>,
    positions: Option<&ggml_rs::Tensor<'ctx>>,
    config: AttentionInferenceConfig,
) -> Result<ggml_rs::Tensor<'ctx>, InferenceError> {
    match (config.rotary, positions) {
        (RotaryEmbedding::Disabled, _) => Ok(*tensor),
        (RotaryEmbedding::Llama(params), Some(positions)) => {
            let rope_dimensions = params.dimensions.get();
            if rope_dimensions > config.head_dimension() {
                return Err(InferenceError::InvalidRopeDimensions {
                    rope_dimensions,
                    head_dimension: config.head_dimension(),
                });
            }
            let n_dims =
                i32::try_from(rope_dimensions).map_err(|_| InferenceError::MemorySizeOverflow)?;
            let n_ctx_orig = params
                .original_context
                .and_then(|value| i32::try_from(value.get()).ok())
                .unwrap_or(0);
            let rope_params = RopeExtParams {
                n_dims,
                n_ctx_orig,
                freq_base: params.base,
                freq_scale: params.scale,
                ..RopeExtParams::default()
            };
            let contiguous = ctx
                .cont(tensor)
                .map_err(|source| InferenceError::ggml("Context::cont(rope)", source))?;
            let reshaped = ctx
                .reshape_3d(
                    &contiguous,
                    config.head_dimension(),
                    1,
                    config.sequence_length(),
                )
                .map_err(|source| InferenceError::ggml("Context::reshape_3d(rope)", source))?;
            let rotated = ctx
                .rope_ext(&reshaped, positions, None, rope_params)
                .map_err(|source| InferenceError::ggml("Context::rope_ext", source))?;
            ctx.reshape_2d(&rotated, config.head_dimension(), config.sequence_length())
                .map_err(|source| InferenceError::ggml("Context::reshape_2d(rope)", source))
        }
        (RotaryEmbedding::Llama(_), None) => Err(InferenceError::InvalidAttentionShape {
            hidden_features: config.hidden_features(),
            sequence_length: config.sequence_length(),
        }),
    }
}

fn build_causal_mask_values(sequence_length: usize, past_tokens: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(sequence_length * sequence_length);
    for query in 0..sequence_length {
        for key in 0..sequence_length {
            let allowed = key + past_tokens <= query + past_tokens;
            values.push(if allowed { 0.0 } else { -1.0e9 });
        }
    }
    values
}

fn recommended_attention_backend_memory_bytes(
    config: AttentionInferenceConfig,
) -> Result<ggml_rs::Bytes, InferenceError> {
    let hidden_features = config.hidden_features();
    let sequence_length = config.sequence_length();
    let query_features = config.query_features();
    let kv_features = config.kv_features();
    let head_dimension = config.head_dimension();

    let q_projection = Context::recommended_backend_matmul_memory_f32_shapes_bytes(
        Shape2D::new(hidden_features, query_features),
        Shape2D::new(hidden_features, sequence_length),
    )
    .map_err(|source| {
        InferenceError::ggml(
            "Context::recommended_backend_matmul_memory_f32_shapes_bytes(q_projection)",
            source,
        )
    })?;
    let kv_projection = Context::recommended_backend_matmul_memory_f32_shapes_bytes(
        Shape2D::new(hidden_features, kv_features),
        Shape2D::new(hidden_features, sequence_length),
    )
    .map_err(|source| {
        InferenceError::ggml(
            "Context::recommended_backend_matmul_memory_f32_shapes_bytes(kv_projection)",
            source,
        )
    })?;
    let score_matmul = Context::recommended_backend_matmul_memory_f32_shapes_bytes(
        Shape2D::new(head_dimension, sequence_length),
        Shape2D::new(head_dimension, sequence_length),
    )
    .map_err(|source| {
        InferenceError::ggml(
            "Context::recommended_backend_matmul_memory_f32_shapes_bytes(score)",
            source,
        )
    })?;
    let value_matmul = Context::recommended_backend_matmul_memory_f32_shapes_bytes(
        Shape2D::new(sequence_length, head_dimension),
        Shape2D::new(sequence_length, sequence_length),
    )
    .map_err(|source| {
        InferenceError::ggml(
            "Context::recommended_backend_matmul_memory_f32_shapes_bytes(value)",
            source,
        )
    })?;
    let output_projection = Context::recommended_backend_matmul_memory_f32_shapes_bytes(
        Shape2D::new(head_dimension, hidden_features),
        Shape2D::new(head_dimension, sequence_length),
    )
    .map_err(|source| {
        InferenceError::ggml(
            "Context::recommended_backend_matmul_memory_f32_shapes_bytes(output_projection)",
            source,
        )
    })?;

    let head_count = config.query_head_count();
    let head_contrib = score_matmul
        .get()
        .checked_add(value_matmul.get())
        .and_then(|value| value.checked_add(output_projection.get()))
        .ok_or(InferenceError::MemorySizeOverflow)?;
    let total = q_projection
        .get()
        .checked_add(kv_projection.get())
        .and_then(|value| value.checked_add(kv_projection.get()))
        .and_then(|value| {
            head_contrib
                .checked_mul(head_count)
                .and_then(|head| value.checked_add(head))
        })
        .and_then(|value| value.checked_add(4 * 1024 * 1024))
        .ok_or(InferenceError::MemorySizeOverflow)?;
    Ok(ggml_rs::Bytes::new(total))
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionInferenceConfig, AttentionLayout, InferenceError, build_causal_mask_values,
        infer_attention_layout_from_features,
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
    fn causal_mask_blocks_future_tokens() {
        let mask = build_causal_mask_values(3, 0);
        assert_eq!(
            mask,
            vec![0.0, -1.0e9, -1.0e9, 0.0, 0.0, -1.0e9, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn config_uses_single_head_compat_constructor() {
        let config = AttentionInferenceConfig::new(16, 5).expect("single head config should build");
        assert_eq!(config.query_head_count(), 1);
        assert_eq!(config.kv_head_count(), 1);
        assert_eq!(config.head_dimension(), 16);
        assert_eq!(config.sequence_length(), 5);
    }

    #[test]
    fn infers_grouped_layout_without_metadata() {
        let inferred =
            infer_attention_layout_from_features(4096, 1024).expect("layout should be inferred");
        assert_eq!(inferred.query_head_count, 32);
        assert_eq!(inferred.kv_head_count, 8);
    }
}
