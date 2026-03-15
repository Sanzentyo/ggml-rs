use crate::backend::LlamaBackend;
use crate::inference::{
    AttentionDecodeCacheInput, AttentionDecodePlan, AttentionInferenceConfig, AttentionLayout,
    AttentionMaskPolicy, AttentionWeights, InferenceError, RotaryEmbedding,
    build_attention_decode_cache, resolve_attention_weights_for_layer,
    resolve_llama_layer_dimensions,
};
use crate::metadata::{ModelArchitecture, resolve_transformer_metadata};
use crate::model::GgufModel;
use crate::naming::detect_layer_indices;
use std::error::Error as StdError;
use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum IdleError {
    Model {
        context: &'static str,
        source: crate::model::ModelError,
    },
    Inference {
        context: &'static str,
        source: InferenceError,
    },
    EmptyPauseSchedule,
    InvalidKeyValueLength(usize),
    InvalidIterations(usize),
    InvalidPastTokens {
        past_tokens: usize,
        key_value_length: usize,
    },
    NoAttentionLayerFound {
        requested_layer: usize,
        tried_layers: Vec<usize>,
    },
}

impl fmt::Display for IdleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model { context, source } => write!(f, "{context}: {source}"),
            Self::Inference { context, source } => write!(f, "{context}: {source}"),
            Self::EmptyPauseSchedule => {
                write!(f, "idle pause schedule must contain at least one value")
            }
            Self::InvalidKeyValueLength(value) => {
                write!(f, "key/value length must be > 0, got {value}")
            }
            Self::InvalidIterations(value) => write!(f, "iterations must be > 0, got {value}"),
            Self::InvalidPastTokens {
                past_tokens,
                key_value_length,
            } => write!(
                f,
                "past tokens must be smaller than key/value length: past={past_tokens}, kv={key_value_length}"
            ),
            Self::NoAttentionLayerFound {
                requested_layer,
                tried_layers,
            } => write!(
                f,
                "failed to resolve an attention-capable layer for requested layer {requested_layer}; tried: {}",
                tried_layers
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

impl StdError for IdleError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Model { source, .. } => Some(source),
            Self::Inference { source, .. } => Some(source),
            Self::EmptyPauseSchedule
            | Self::InvalidKeyValueLength(_)
            | Self::InvalidIterations(_)
            | Self::InvalidPastTokens { .. }
            | Self::NoAttentionLayerFound { .. } => None,
        }
    }
}

impl IdleError {
    fn model(context: &'static str, source: crate::model::ModelError) -> Self {
        Self::Model { context, source }
    }

    fn inference(context: &'static str, source: InferenceError) -> Self {
        Self::Inference { context, source }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PauseScheduleEmpty;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PauseScheduleReady;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdlePauseSchedule<State> {
    values_ms: Vec<u64>,
    _state: PhantomData<State>,
}

impl IdlePauseSchedule<PauseScheduleEmpty> {
    pub fn new() -> Self {
        Self {
            values_ms: Vec::new(),
            _state: PhantomData,
        }
    }

    pub fn push(mut self, pause_ms: u64) -> IdlePauseSchedule<PauseScheduleReady> {
        self.values_ms.push(pause_ms);
        IdlePauseSchedule {
            values_ms: self.values_ms,
            _state: PhantomData,
        }
    }

    pub fn from_vec(
        values_ms: Vec<u64>,
    ) -> Result<IdlePauseSchedule<PauseScheduleReady>, IdleError> {
        let mut values = values_ms.into_iter();
        let Some(first) = values.next() else {
            return Err(IdleError::EmptyPauseSchedule);
        };
        let mut schedule = Self::new().push(first);
        for pause_ms in values {
            schedule = schedule.push(pause_ms);
        }
        Ok(schedule)
    }

    pub fn default_profile() -> IdlePauseSchedule<PauseScheduleReady> {
        Self::new()
            .push(0)
            .push(800)
            .push(1_600)
            .push(2_400)
            .push(3_200)
            .push(4_000)
    }
}

impl Default for IdlePauseSchedule<PauseScheduleEmpty> {
    fn default() -> Self {
        Self::new()
    }
}

impl IdlePauseSchedule<PauseScheduleReady> {
    pub fn push(mut self, pause_ms: u64) -> Self {
        self.values_ms.push(pause_ms);
        self
    }

    pub fn values(&self) -> &[u64] {
        &self.values_ms
    }
}

#[derive(Debug, Clone)]
pub struct IdleConfig {
    pub layer: usize,
    pub key_value_length: NonZeroUsize,
    pub past_tokens: usize,
    pub iterations: NonZeroUsize,
    pub pauses: IdlePauseSchedule<PauseScheduleReady>,
}

impl IdleConfig {
    pub fn new(
        layer: usize,
        key_value_length: usize,
        past_tokens: usize,
        iterations: usize,
        pauses: IdlePauseSchedule<PauseScheduleReady>,
    ) -> Result<Self, IdleError> {
        let key_value_length = NonZeroUsize::new(key_value_length)
            .ok_or(IdleError::InvalidKeyValueLength(key_value_length))?;
        let iterations =
            NonZeroUsize::new(iterations).ok_or(IdleError::InvalidIterations(iterations))?;
        if past_tokens >= key_value_length.get() {
            return Err(IdleError::InvalidPastTokens {
                past_tokens,
                key_value_length: key_value_length.get(),
            });
        }
        Ok(Self {
            layer,
            key_value_length,
            past_tokens,
            iterations,
            pauses,
        })
    }
}

#[derive(Debug, Clone)]
pub struct IdlePauseReport {
    pub pause_ms: u64,
    pub average_decode_ms: f64,
    pub stddev_decode_ms: f64,
    pub samples_ms: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct IdleReport {
    pub backend_name: String,
    pub requested_layer: usize,
    pub layer: usize,
    pub weights_mode: IdleWeightsMode,
    pub key_value_length: usize,
    pub past_tokens: usize,
    pub iterations: usize,
    pub hidden_features: usize,
    pub checksum: f32,
    pub pauses: Vec<IdlePauseReport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdleWeightsMode {
    ModelLayer,
    MetadataDeterministic,
}

pub fn idle_decode_proxy(
    model: &GgufModel,
    backend_kind: LlamaBackend,
    config: IdleConfig,
) -> Result<IdleReport, IdleError> {
    let (resolved_layer, weights_mode, weights) =
        resolve_idle_attention_weights(model, config.layer, config.past_tokens)?;
    let hidden_features = weights.config.hidden_features();
    let key_value_input = build_key_value_input(hidden_features, config.key_value_length.get());
    let cache =
        build_attention_decode_cache(&weights, &key_value_input, config.key_value_length.get())
            .map_err(|source| IdleError::inference("build_attention_decode_cache", source))?;
    let query_input = build_query_input(hidden_features);
    let decode_plan = AttentionDecodePlan::builder()
        .backend(backend_kind)
        .repeats(1)
        .past_tokens(config.past_tokens)
        .build()
        .map_err(|source| IdleError::inference("AttentionDecodePlan::build", source))?;
    let decode_source = AttentionDecodeCacheInput::new(&weights, &query_input, &cache);

    let warmup = decode_plan
        .execute(decode_source)
        .map_err(|source| IdleError::inference("AttentionDecodePlan::execute(warmup)", source))?;

    let mut backend_name = warmup.backend_name;
    let mut checksum = output_checksum(&warmup.output);
    let mut pause_reports = Vec::with_capacity(config.pauses.values().len());

    for &pause_ms in config.pauses.values() {
        let mut samples_ms = Vec::with_capacity(config.iterations.get());
        for _ in 0..config.iterations.get() {
            if pause_ms > 0 {
                thread::sleep(Duration::from_millis(pause_ms));
            }

            let start = Instant::now();
            let report = decode_plan
                .execute(decode_source)
                .map_err(|source| IdleError::inference("AttentionDecodePlan::execute", source))?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            backend_name = report.backend_name;
            checksum = output_checksum(&report.output);
            samples_ms.push(elapsed_ms);
        }

        let (average_decode_ms, stddev_decode_ms) = average_and_stddev(&samples_ms);
        pause_reports.push(IdlePauseReport {
            pause_ms,
            average_decode_ms,
            stddev_decode_ms,
            samples_ms,
        });
    }

    Ok(IdleReport {
        backend_name,
        requested_layer: config.layer,
        layer: resolved_layer,
        weights_mode,
        key_value_length: config.key_value_length.get(),
        past_tokens: config.past_tokens,
        iterations: config.iterations.get(),
        hidden_features,
        checksum,
        pauses: pause_reports,
    })
}

fn resolve_idle_attention_weights(
    model: &GgufModel,
    requested_layer: usize,
    past_tokens: usize,
) -> Result<(usize, IdleWeightsMode, AttentionWeights), IdleError> {
    let mut candidates = vec![requested_layer];
    let discovered_layers = detect_layer_indices(model);
    if discovered_layers.is_empty() {
        if let Ok(metadata) = resolve_transformer_metadata(model) {
            candidates
                .extend((0..metadata.block_count()).filter(|layer| *layer != requested_layer));
        }
    } else {
        candidates.extend(
            discovered_layers
                .into_iter()
                .filter(|layer| *layer != requested_layer),
        );
    }

    let mut tried_layers = Vec::new();
    for layer in candidates {
        tried_layers.push(layer);
        let attention_config = match build_idle_attention_config(model, layer, past_tokens) {
            Ok(config) => config,
            Err(_) => continue,
        };
        match resolve_attention_weights_for_layer(model, layer, attention_config) {
            Ok(weights) => return Ok((layer, IdleWeightsMode::ModelLayer, weights)),
            Err(_) => continue,
        }
    }

    if let Ok(metadata) = resolve_transformer_metadata(model)
        && matches!(metadata.architecture(), ModelArchitecture::Other(_))
    {
        let layout = AttentionLayout::from_hidden_features(
            metadata.embedding_length(),
            metadata.attention_head_count(),
            metadata.attention_head_count_kv(),
        )
        .map_err(|source| IdleError::inference("AttentionLayout::from_hidden_features", source))?;
        let config = AttentionInferenceConfig::from_layout(layout, 1)
            .map_err(|source| {
                IdleError::inference("AttentionInferenceConfig::from_layout", source)
            })?
            .with_mask(AttentionMaskPolicy::Causal { past_tokens })
            .with_rotary(RotaryEmbedding::Disabled);
        return Ok((
            requested_layer,
            IdleWeightsMode::MetadataDeterministic,
            AttentionWeights::deterministic(config),
        ));
    }

    Err(IdleError::NoAttentionLayerFound {
        requested_layer,
        tried_layers,
    })
}

fn build_idle_attention_config(
    model: &GgufModel,
    layer: usize,
    past_tokens: usize,
) -> Result<AttentionInferenceConfig, IdleError> {
    let dimensions = resolve_llama_layer_dimensions(model, layer)
        .map_err(|source| IdleError::inference("resolve_llama_layer_dimensions", source))?;
    let layout = AttentionLayout::from_hidden_features(
        dimensions.hidden_features,
        dimensions.query_head_count,
        dimensions.kv_head_count,
    )
    .map_err(|source| IdleError::inference("AttentionLayout::from_hidden_features", source))?;
    AttentionInferenceConfig::from_layout(layout, 1)
        .map_err(|source| IdleError::inference("AttentionInferenceConfig::from_layout", source))
        .map(|config| {
            config
                .with_mask(AttentionMaskPolicy::Causal { past_tokens })
                .with_rotary(RotaryEmbedding::Disabled)
        })
}

pub fn idle_decode_proxy_from_path<P: AsRef<Path>>(
    model_path: P,
    backend_kind: LlamaBackend,
    config: IdleConfig,
) -> Result<IdleReport, IdleError> {
    let model = GgufModel::open(model_path)
        .map_err(|source| IdleError::model("GgufModel::open", source))?;
    idle_decode_proxy(&model, backend_kind, config)
}

fn build_query_input(hidden_features: usize) -> Vec<f32> {
    (0..hidden_features)
        .map(|index| ((index % 23) as f32 - 11.0) / 11.0)
        .collect()
}

fn build_key_value_input(hidden_features: usize, key_value_length: usize) -> Vec<f32> {
    let total = hidden_features.saturating_mul(key_value_length);
    (0..total)
        .map(|index| ((index % 29) as f32 - 14.0) / 14.0)
        .collect()
}

fn output_checksum(values: &[f32]) -> f32 {
    values.iter().copied().sum()
}

fn average_and_stddev(samples: &[f64]) -> (f64, f64) {
    let sample_count = samples.len();
    if sample_count == 0 {
        return (0.0, 0.0);
    }

    let sample_count_f64 = sample_count as f64;
    let average = samples.iter().copied().sum::<f64>() / sample_count_f64;
    if sample_count == 1 {
        return (average, 0.0);
    }

    let variance = samples
        .iter()
        .copied()
        .map(|sample| {
            let delta = sample - average;
            delta * delta
        })
        .sum::<f64>()
        / ((sample_count - 1) as f64);
    (average, variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::{IdleConfig, IdlePauseSchedule, PauseScheduleEmpty};

    #[test]
    fn pause_schedule_rejects_empty_input() {
        let schedule = IdlePauseSchedule::<PauseScheduleEmpty>::from_vec(vec![]);
        assert!(schedule.is_err());
    }

    #[test]
    fn idle_config_rejects_past_outside_kv_window() {
        let pauses = IdlePauseSchedule::<PauseScheduleEmpty>::default_profile();
        let config = IdleConfig::new(0, 16, 16, 1, pauses);
        assert!(config.is_err());
    }
}
