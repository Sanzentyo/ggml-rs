//! Tensor-name resolution helpers for real GGUF model layouts.

use crate::model::GgufModel;
use std::collections::{BTreeSet, HashSet};
use std::error::Error as StdError;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Resolved tensor names for one transformer layer.
pub struct LlamaLayerTensorNames {
    pub layer: usize,
    pub attn_norm: String,
    pub attn_q_norm: Option<String>,
    pub attn_k_norm: Option<String>,
    pub attn_q: String,
    pub attn_k: String,
    pub attn_v: String,
    pub attn_output: String,
    pub ffn_norm: String,
    pub ffn_gate: String,
    pub ffn_up: String,
    pub ffn_down: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Resolved high-level tensor-name catalog for a LLaMA-like GGUF model.
pub struct LlamaTensorNames {
    pub token_embedding: String,
    pub output_norm: String,
    pub output: Option<String>,
    pub layers: Vec<LlamaLayerTensorNames>,
}

#[derive(Debug)]
/// Errors surfaced while resolving canonical tensor roles to GGUF tensor names.
pub enum NamingError {
    NoLayersDetected,
    LayerNotFound {
        layer: usize,
        available: Vec<usize>,
    },
    MissingGlobalTensor {
        role: &'static str,
        tried: Vec<String>,
    },
    MissingLayerTensor {
        layer: usize,
        role: &'static str,
        tried: Vec<String>,
    },
}

impl fmt::Display for NamingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLayersDetected => write!(
                f,
                "no transformer layer indices detected in tensor names (tried blk./layers./model.layers. prefixes)"
            ),
            Self::LayerNotFound { layer, available } => write!(
                f,
                "requested layer {layer} not found; available layers: {}",
                if available.is_empty() {
                    "<none>".to_string()
                } else {
                    available
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            ),
            Self::MissingGlobalTensor { role, tried } => write!(
                f,
                "missing global tensor for role `{role}`; tried: {}",
                tried.join(", ")
            ),
            Self::MissingLayerTensor { layer, role, tried } => write!(
                f,
                "missing layer tensor for role `{role}` at layer {layer}; tried: {}",
                tried.join(", ")
            ),
        }
    }
}

impl StdError for NamingError {}

/// Detects layer indices from common GGUF tensor-name layouts.
pub fn detect_layer_indices(model: &GgufModel) -> Vec<usize> {
    detect_layer_indices_from_names(model.tensor_names())
}

/// Detects layer indices from arbitrary tensor-name iterators.
pub fn detect_layer_indices_from_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<usize> {
    let names: HashSet<&str> = names.into_iter().collect();
    detect_layer_indices_from_name_set(&names)
}

/// Resolves canonical LLaMA tensor roles into concrete names from an arbitrary
/// tensor-name iterator.
pub fn resolve_llama_tensor_names_from_names<'a>(
    names_iter: impl IntoIterator<Item = &'a str>,
) -> Result<LlamaTensorNames, NamingError> {
    let names: HashSet<&str> = names_iter.into_iter().collect();
    let layers = detect_layer_indices_from_name_set(&names);
    if layers.is_empty() {
        return Err(NamingError::NoLayersDetected);
    }

    let token_embedding = resolve_required_global(
        &names,
        "token_embedding",
        vec![
            "token_embd.weight".to_string(),
            "tok_embeddings.weight".to_string(),
            "model.embed_tokens.weight".to_string(),
        ],
    )?;
    let output_norm = resolve_required_global(
        &names,
        "output_norm",
        vec![
            "output_norm.weight".to_string(),
            "norm.weight".to_string(),
            "model.norm.weight".to_string(),
        ],
    )?;
    let output = resolve_optional_global(
        &names,
        vec![
            "output.weight".to_string(),
            "lm_head.weight".to_string(),
            "model.lm_head.weight".to_string(),
        ],
    );

    let mut resolved_layers = Vec::with_capacity(layers.len());
    for layer in layers {
        resolved_layers.push(resolve_layer(&names, layer)?);
    }

    Ok(LlamaTensorNames {
        token_embedding,
        output_norm,
        output,
        layers: resolved_layers,
    })
}

/// Resolves canonical LLaMA tensor roles into concrete GGUF tensor names.
pub fn resolve_llama_tensor_names(model: &GgufModel) -> Result<LlamaTensorNames, NamingError> {
    resolve_llama_tensor_names_from_names(model.tensor_names())
}

/// Resolves tensor names for one specific layer index.
pub fn resolve_llama_layer_tensor_names_from_names<'a>(
    names: impl IntoIterator<Item = &'a str>,
    layer: usize,
) -> Result<LlamaLayerTensorNames, NamingError> {
    let names: HashSet<&str> = names.into_iter().collect();
    let available = detect_layer_indices_from_name_set(&names);
    if available.is_empty() {
        return Err(NamingError::NoLayersDetected);
    }
    if !available.contains(&layer) {
        return Err(NamingError::LayerNotFound { layer, available });
    }
    resolve_layer(&names, layer)
}

/// Resolves tensor names for one specific layer index.
pub fn resolve_llama_layer_tensor_names(
    model: &GgufModel,
    layer: usize,
) -> Result<LlamaLayerTensorNames, NamingError> {
    resolve_llama_layer_tensor_names_from_names(model.tensor_names(), layer)
}

fn detect_layer_indices_from_name_set(names: &HashSet<&str>) -> Vec<usize> {
    let mut layer_indices = BTreeSet::new();
    for &name in names {
        if let Some(index) = parse_layer_index(name, "blk.") {
            layer_indices.insert(index);
        }
        if let Some(index) = parse_layer_index(name, "layers.") {
            layer_indices.insert(index);
        }
        if let Some(index) = parse_layer_index(name, "model.layers.") {
            layer_indices.insert(index);
        }
    }
    layer_indices.into_iter().collect()
}

fn resolve_layer(
    names: &HashSet<&str>,
    layer: usize,
) -> Result<LlamaLayerTensorNames, NamingError> {
    let attn_norm = resolve_required_layer(
        names,
        layer,
        "attn_norm",
        vec![
            format!("blk.{layer}.attn_norm.weight"),
            format!("layers.{layer}.attention_norm.weight"),
            format!("model.layers.{layer}.input_layernorm.weight"),
        ],
    )?;
    let attn_q = resolve_required_layer(
        names,
        layer,
        "attn_q",
        vec![
            format!("blk.{layer}.attn_q.weight"),
            format!("layers.{layer}.attention.wq.weight"),
            format!("model.layers.{layer}.self_attn.q_proj.weight"),
        ],
    )?;
    let attn_q_norm = resolve_optional_layer(
        names,
        vec![
            format!("blk.{layer}.attn_q_norm.weight"),
            format!("layers.{layer}.attention.q_norm.weight"),
            format!("model.layers.{layer}.self_attn.q_norm.weight"),
        ],
    );
    let attn_k = resolve_required_layer(
        names,
        layer,
        "attn_k",
        vec![
            format!("blk.{layer}.attn_k.weight"),
            format!("layers.{layer}.attention.wk.weight"),
            format!("model.layers.{layer}.self_attn.k_proj.weight"),
        ],
    )?;
    let attn_k_norm = resolve_optional_layer(
        names,
        vec![
            format!("blk.{layer}.attn_k_norm.weight"),
            format!("layers.{layer}.attention.k_norm.weight"),
            format!("model.layers.{layer}.self_attn.k_norm.weight"),
        ],
    );
    let attn_v = resolve_required_layer(
        names,
        layer,
        "attn_v",
        vec![
            format!("blk.{layer}.attn_v.weight"),
            format!("layers.{layer}.attention.wv.weight"),
            format!("model.layers.{layer}.self_attn.v_proj.weight"),
        ],
    )?;
    let attn_output = resolve_required_layer(
        names,
        layer,
        "attn_output",
        vec![
            format!("blk.{layer}.attn_output.weight"),
            format!("layers.{layer}.attention.wo.weight"),
            format!("model.layers.{layer}.self_attn.o_proj.weight"),
        ],
    )?;
    let ffn_norm = resolve_required_layer(
        names,
        layer,
        "ffn_norm",
        vec![
            format!("blk.{layer}.ffn_norm.weight"),
            format!("blk.{layer}.post_attention_norm.weight"),
            format!("layers.{layer}.ffn_norm.weight"),
            format!("model.layers.{layer}.post_attention_layernorm.weight"),
            format!("model.layers.{layer}.post_attention_norm.weight"),
        ],
    )?;
    let ffn_gate = resolve_required_layer(
        names,
        layer,
        "ffn_gate",
        vec![
            format!("blk.{layer}.ffn_gate.weight"),
            format!("layers.{layer}.feed_forward.w1.weight"),
            format!("model.layers.{layer}.mlp.gate_proj.weight"),
        ],
    )?;
    let ffn_up = resolve_required_layer(
        names,
        layer,
        "ffn_up",
        vec![
            format!("blk.{layer}.ffn_up.weight"),
            format!("layers.{layer}.feed_forward.w3.weight"),
            format!("model.layers.{layer}.mlp.up_proj.weight"),
        ],
    )?;
    let ffn_down = resolve_required_layer(
        names,
        layer,
        "ffn_down",
        vec![
            format!("blk.{layer}.ffn_down.weight"),
            format!("layers.{layer}.feed_forward.w2.weight"),
            format!("model.layers.{layer}.mlp.down_proj.weight"),
        ],
    )?;

    Ok(LlamaLayerTensorNames {
        layer,
        attn_norm,
        attn_q_norm,
        attn_k_norm,
        attn_q,
        attn_k,
        attn_v,
        attn_output,
        ffn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
    })
}

fn parse_layer_index(name: &str, prefix: &str) -> Option<usize> {
    let rest = name.strip_prefix(prefix)?;
    let (layer, _) = rest.split_once('.')?;
    layer.parse::<usize>().ok()
}

fn resolve_required_global(
    names: &HashSet<&str>,
    role: &'static str,
    candidates: Vec<String>,
) -> Result<String, NamingError> {
    if let Some(name) = resolve_first(names, &candidates) {
        return Ok(name);
    }
    Err(NamingError::MissingGlobalTensor {
        role,
        tried: candidates,
    })
}

fn resolve_optional_global(names: &HashSet<&str>, candidates: Vec<String>) -> Option<String> {
    resolve_first(names, &candidates)
}

fn resolve_required_layer(
    names: &HashSet<&str>,
    layer: usize,
    role: &'static str,
    candidates: Vec<String>,
) -> Result<String, NamingError> {
    if let Some(name) = resolve_first(names, &candidates) {
        return Ok(name);
    }
    Err(NamingError::MissingLayerTensor {
        layer,
        role,
        tried: candidates,
    })
}

fn resolve_optional_layer(names: &HashSet<&str>, candidates: Vec<String>) -> Option<String> {
    resolve_first(names, &candidates)
}

fn resolve_first(names: &HashSet<&str>, candidates: &[String]) -> Option<String> {
    candidates
        .iter()
        .find(|name| names.contains(name.as_str()))
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::{
        NamingError, detect_layer_indices_from_names, resolve_llama_layer_tensor_names_from_names,
        resolve_llama_tensor_names_from_names,
    };

    #[test]
    fn resolves_blk_style_names() {
        let names = [
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];

        let resolved = resolve_llama_tensor_names_from_names(names)
            .expect("resolver should accept blk.* naming");
        assert_eq!(resolved.token_embedding, "token_embd.weight");
        assert_eq!(resolved.layers.len(), 1);
        assert_eq!(resolved.layers[0].layer, 0);
        assert_eq!(resolved.layers[0].attn_q_norm, None);
        assert_eq!(resolved.layers[0].ffn_gate, "blk.0.ffn_gate.weight");
    }

    #[test]
    fn resolves_hf_style_names() {
        let names = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.lm_head.weight",
            "model.layers.3.input_layernorm.weight",
            "model.layers.3.self_attn.q_norm.weight",
            "model.layers.3.self_attn.k_norm.weight",
            "model.layers.3.self_attn.q_proj.weight",
            "model.layers.3.self_attn.k_proj.weight",
            "model.layers.3.self_attn.v_proj.weight",
            "model.layers.3.self_attn.o_proj.weight",
            "model.layers.3.post_attention_layernorm.weight",
            "model.layers.3.mlp.gate_proj.weight",
            "model.layers.3.mlp.up_proj.weight",
            "model.layers.3.mlp.down_proj.weight",
        ];

        let resolved = resolve_llama_tensor_names_from_names(names)
            .expect("resolver should accept model.layers.* naming");
        assert_eq!(resolved.token_embedding, "model.embed_tokens.weight");
        assert_eq!(resolved.output_norm, "model.norm.weight");
        assert_eq!(resolved.layers.len(), 1);
        assert_eq!(resolved.layers[0].layer, 3);
        assert_eq!(
            resolved.layers[0].attn_q_norm.as_deref(),
            Some("model.layers.3.self_attn.q_norm.weight")
        );
        assert_eq!(
            resolved.layers[0].attn_k_norm.as_deref(),
            Some("model.layers.3.self_attn.k_norm.weight")
        );
    }

    #[test]
    fn reports_missing_layer_role() {
        let names = [
            "token_embd.weight",
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            // missing blk.0.attn_q.weight
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];

        let error = resolve_llama_tensor_names_from_names(names)
            .expect_err("resolver should fail when a required role is missing");
        match error {
            NamingError::MissingLayerTensor { layer, role, .. } => {
                assert_eq!(layer, 0);
                assert_eq!(role, "attn_q");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn detects_layers_sorted_and_unique() {
        let names = [
            "blk.10.attn_q.weight",
            "blk.2.attn_q.weight",
            "model.layers.7.self_attn.q_proj.weight",
            "layers.2.attention.wq.weight",
        ];

        let layers = detect_layer_indices_from_names(names);
        assert_eq!(layers, vec![2, 7, 10]);
    }

    #[test]
    fn resolves_multiple_layers_in_sorted_order() {
        let names = [
            "token_embd.weight",
            "output_norm.weight",
            "blk.1.attn_norm.weight",
            "blk.1.attn_q.weight",
            "blk.1.attn_k.weight",
            "blk.1.attn_v.weight",
            "blk.1.attn_output.weight",
            "blk.1.ffn_norm.weight",
            "blk.1.ffn_gate.weight",
            "blk.1.ffn_up.weight",
            "blk.1.ffn_down.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];

        let resolved = resolve_llama_tensor_names_from_names(names)
            .expect("resolver should support multiple blk layers");
        let layer_ids: Vec<usize> = resolved.layers.iter().map(|layer| layer.layer).collect();
        assert_eq!(layer_ids, vec![0, 1]);
    }

    #[test]
    fn reports_not_found_layer_with_available_indices() {
        let names = [
            "token_embd.weight",
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];

        let error = resolve_llama_layer_tensor_names_from_names(names, 5)
            .expect_err("missing layer should produce LayerNotFound");

        match error {
            NamingError::LayerNotFound { layer, available } => {
                assert_eq!(layer, 5);
                assert_eq!(available, vec![0]);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn resolves_requested_layer_even_when_other_layers_are_incomplete() {
        let names = [
            "token_embd.weight",
            "output_norm.weight",
            // layer 0 intentionally incomplete for llama-style attention roles
            "blk.0.attn_norm.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            // layer 3 has a complete llama-style attention role set
            "blk.3.attn_norm.weight",
            "blk.3.attn_q.weight",
            "blk.3.attn_k.weight",
            "blk.3.attn_v.weight",
            "blk.3.attn_output.weight",
            "blk.3.ffn_norm.weight",
            "blk.3.ffn_gate.weight",
            "blk.3.ffn_up.weight",
            "blk.3.ffn_down.weight",
        ];

        let resolved = resolve_llama_layer_tensor_names_from_names(names, 3)
            .expect("layer-scoped resolution should not fail on unrelated incomplete layers");
        assert_eq!(resolved.layer, 3);
        assert_eq!(resolved.attn_q, "blk.3.attn_q.weight");
        assert_eq!(resolved.attn_output, "blk.3.attn_output.weight");
    }
}
