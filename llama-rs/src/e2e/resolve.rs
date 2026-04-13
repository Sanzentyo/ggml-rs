use super::error::E2eError;
use crate::model::GgufModel;
use crate::naming::NamingError;

#[derive(Debug, Clone)]
pub(super) struct GlobalTensorNames {
    pub token_embedding: String,
    pub output_norm: String,
    pub output: Option<String>,
}

pub(super) fn resolve_global_tensor_names(
    model: &GgufModel,
) -> Result<GlobalTensorNames, E2eError> {
    let token_embedding = resolve_required_global_tensor_name(
        model,
        "token_embedding",
        global_token_embedding_candidates(),
    )?;
    let output_norm =
        resolve_required_global_tensor_name(model, "output_norm", global_output_norm_candidates())?;
    let output = resolve_optional_global_tensor_name(model, global_output_projection_candidates());

    Ok(GlobalTensorNames {
        token_embedding,
        output_norm,
        output,
    })
}

pub(super) fn resolve_required_global_tensor_name(
    model: &GgufModel,
    role: &'static str,
    candidates: Vec<String>,
) -> Result<String, E2eError> {
    if let Some(name) = resolve_first_tensor_name(model, &candidates) {
        return Ok(name);
    }
    Err(E2eError::naming(
        "resolve_required_global_tensor_name",
        NamingError::MissingGlobalTensor {
            role,
            tried: candidates,
        },
    ))
}

fn resolve_optional_global_tensor_name(
    model: &GgufModel,
    candidates: Vec<String>,
) -> Option<String> {
    resolve_first_tensor_name(model, &candidates)
}

pub(super) fn resolve_required_layer_tensor_name(
    model: &GgufModel,
    layer: usize,
    role: &'static str,
    candidates: Vec<String>,
) -> Result<String, E2eError> {
    if let Some(name) = resolve_first_tensor_name(model, &candidates) {
        return Ok(name);
    }
    Err(E2eError::naming(
        "resolve_required_layer_tensor_name",
        NamingError::MissingLayerTensor {
            layer,
            role,
            tried: candidates,
        },
    ))
}

pub(super) fn resolve_first_tensor_name(
    model: &GgufModel,
    candidates: &[String],
) -> Option<String> {
    candidates
        .iter()
        .find(|name| model.find_tensor(name.as_str()).is_some())
        .cloned()
}

fn global_token_embedding_candidates() -> Vec<String> {
    vec![
        "token_embd.weight".to_string(),
        "tok_embeddings.weight".to_string(),
        "model.embed_tokens.weight".to_string(),
    ]
}

fn global_output_norm_candidates() -> Vec<String> {
    vec![
        "output_norm.weight".to_string(),
        "norm.weight".to_string(),
        "model.norm.weight".to_string(),
    ]
}

fn global_output_projection_candidates() -> Vec<String> {
    vec![
        "output.weight".to_string(),
        "lm_head.weight".to_string(),
        "model.lm_head.weight".to_string(),
    ]
}

pub(super) fn layer_ffn_norm_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.ffn_norm.weight"),
        format!("blk.{layer}.post_attention_norm.weight"),
        format!("layers.{layer}.ffn_norm.weight"),
        format!("model.layers.{layer}.post_attention_layernorm.weight"),
        format!("model.layers.{layer}.post_attention_norm.weight"),
    ]
}

pub(super) fn layer_ffn_gate_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.ffn_gate.weight"),
        format!("layers.{layer}.feed_forward.w1.weight"),
        format!("model.layers.{layer}.mlp.gate_proj.weight"),
    ]
}

pub(super) fn layer_ffn_up_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.ffn_up.weight"),
        format!("layers.{layer}.feed_forward.w3.weight"),
        format!("model.layers.{layer}.mlp.up_proj.weight"),
    ]
}

pub(super) fn layer_ffn_down_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.ffn_down.weight"),
        format!("layers.{layer}.feed_forward.w2.weight"),
        format!("model.layers.{layer}.mlp.down_proj.weight"),
    ]
}

pub(super) fn layer_attn_norm_candidates(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.attn_norm.weight"),
        format!("layers.{layer}.attention_norm.weight"),
        format!("model.layers.{layer}.input_layernorm.weight"),
    ]
}

pub(super) fn layer_attn_qkv_candidates(layer: usize) -> Vec<String> {
    vec![format!("blk.{layer}.attn_qkv.weight")]
}

pub(super) fn layer_attn_gate_candidates(layer: usize) -> Vec<String> {
    vec![format!("blk.{layer}.attn_gate.weight")]
}

pub(super) fn layer_ssm_alpha_candidates(layer: usize) -> Vec<String> {
    vec![format!("blk.{layer}.ssm_alpha.weight")]
}

pub(super) fn layer_ssm_beta_candidates(layer: usize) -> Vec<String> {
    vec![format!("blk.{layer}.ssm_beta.weight")]
}

pub(super) fn layer_ssm_conv1d_candidates(layer: usize) -> Vec<String> {
    vec![format!("blk.{layer}.ssm_conv1d.weight")]
}

pub(super) fn layer_ssm_dt_candidates(layer: usize) -> Vec<String> {
    vec![format!("blk.{layer}.ssm_dt.bias")]
}

pub(super) fn layer_ssm_a_candidates(layer: usize) -> Vec<String> {
    vec![format!("blk.{layer}.ssm_a")]
}

pub(super) fn layer_ssm_norm_candidates(layer: usize) -> Vec<String> {
    vec![format!("blk.{layer}.ssm_norm.weight")]
}

pub(super) fn layer_ssm_out_candidates(layer: usize) -> Vec<String> {
    vec![format!("blk.{layer}.ssm_out.weight")]
}
