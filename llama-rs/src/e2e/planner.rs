use super::config::MixedLayerPolicy;
use super::decode::{decode_exact_tensor, decode_matrix_tensor, decode_norm_tensor};
use super::error::E2eError;
use super::numeric::checked_mul;
use super::plan::{
    AttentionLayerPlan, LayerPlan, MlpLayerPlan, Qwen35FullAttentionLayerPlan,
    Qwen35LinearAttentionLayerPlan, StandardAttentionLayerPlan,
};
use super::resolve::{
    layer_attn_gate_candidates, layer_attn_norm_candidates, layer_attn_qkv_candidates,
    layer_ffn_down_candidates, layer_ffn_gate_candidates, layer_ffn_norm_candidates,
    layer_ffn_up_candidates, layer_ssm_a_candidates, layer_ssm_alpha_candidates,
    layer_ssm_beta_candidates, layer_ssm_conv1d_candidates, layer_ssm_dt_candidates,
    layer_ssm_norm_candidates, layer_ssm_out_candidates, resolve_first_tensor_name,
    resolve_required_layer_tensor_name,
};
use crate::inference::{
    AttentionInferenceConfig, AttentionMaskPolicy, AttentionWeights, MlpInferenceConfig,
    MlpWeights, resolve_llama_layer_dimensions,
};
use crate::metadata::{MetadataError, TransformerMetadata};
use crate::model::GgufModel;
use crate::naming::{LlamaLayerTensorNames, NamingError, resolve_llama_layer_tensor_names};

pub(super) fn build_layer_plans(
    model: &GgufModel,
    metadata: &TransformerMetadata,
    hidden_features: usize,
    total_sequence_length: usize,
    mixed_layer_policy: MixedLayerPolicy,
) -> Result<Vec<LayerPlan>, E2eError> {
    let block_count = metadata.block_count();
    let mut layer_plans = Vec::with_capacity(block_count);
    for layer in 0..block_count {
        match try_build_attention_layer_plan(
            model,
            metadata,
            layer,
            hidden_features,
            total_sequence_length,
        ) {
            Ok(Some(attention)) => layer_plans.push(LayerPlan {
                attention: Some(attention),
                mlp: build_layer_mlp_plan(model, layer, hidden_features)?,
            }),
            Ok(None) => match mixed_layer_policy {
                MixedLayerPolicy::Strict => {
                    return Err(E2eError::naming(
                        "try_build_attention_layer_plan",
                        NamingError::MissingLayerTensor {
                            layer,
                            role: "attention",
                            tried: vec![
                                format!("blk.{layer}.attn_q.weight"),
                                format!("blk.{layer}.attn_qkv.weight"),
                            ],
                        },
                    ));
                }
                MixedLayerPolicy::SkipUnsupportedAttention => {
                    layer_plans.push(build_mlp_only_layer_plan(model, layer, hidden_features)?);
                }
            },
            Err(source) => match mixed_layer_policy {
                MixedLayerPolicy::Strict => return Err(source),
                MixedLayerPolicy::SkipUnsupportedAttention => {
                    layer_plans.push(build_mlp_only_layer_plan(model, layer, hidden_features)?);
                }
            },
        }
    }
    Ok(layer_plans)
}

fn try_build_attention_layer_plan(
    model: &GgufModel,
    metadata: &TransformerMetadata,
    layer: usize,
    hidden_features: usize,
    total_sequence_length: usize,
) -> Result<Option<AttentionLayerPlan>, E2eError> {
    match resolve_llama_layer_tensor_names(model, layer) {
        Ok(names) => {
            if metadata.architecture().as_str() == "qwen35" {
                build_qwen35_full_attention_layer_plan(model, metadata, names, hidden_features)
                    .map(Some)
            } else {
                build_standard_attention_layer_plan(
                    model,
                    names,
                    hidden_features,
                    total_sequence_length,
                )
                .map(Some)
            }
        }
        Err(source)
            if metadata.architecture().as_str() == "qwen35"
                && matches!(
                    source,
                    NamingError::MissingLayerTensor { role: "attn_q", .. }
                ) =>
        {
            if resolve_first_tensor_name(model, &layer_attn_qkv_candidates(layer)).is_none() {
                return Ok(None);
            }
            build_qwen35_linear_attention_layer_plan(model, metadata, layer, hidden_features)
                .map(Some)
        }
        Err(source) => Err(E2eError::naming("resolve_llama_layer_tensor_names", source)),
    }
}

fn build_standard_attention_layer_plan(
    model: &GgufModel,
    names: LlamaLayerTensorNames,
    hidden_features: usize,
    total_sequence_length: usize,
) -> Result<AttentionLayerPlan, E2eError> {
    let layer = names.layer;
    let dimensions = resolve_llama_layer_dimensions(model, layer)
        .map_err(|source| E2eError::inference("resolve_llama_layer_dimensions", source))?;
    if dimensions.hidden_features != hidden_features {
        return Err(E2eError::HiddenFeatureMismatch {
            layer,
            expected: hidden_features,
            actual: dimensions.hidden_features,
        });
    }
    let attention_config =
        AttentionInferenceConfig::from_layer_dimensions(dimensions, total_sequence_length)
            .map_err(|source| {
                E2eError::inference("AttentionInferenceConfig::from_layer_dimensions", source)
            })?
            .with_mask(AttentionMaskPolicy::Causal { past_tokens: 0 });
    let attention_weights = AttentionWeights::from_model_layer(model, &names, attention_config)
        .map_err(|source| E2eError::inference("AttentionWeights::from_model_layer", source))?;
    let attn_norm_values =
        decode_norm_tensor(model, &names.attn_norm, hidden_features, "attn_norm")?;

    Ok(AttentionLayerPlan::Standard(StandardAttentionLayerPlan {
        weights: attention_weights,
        norm_values: attn_norm_values,
    }))
}

fn build_layer_mlp_plan(
    model: &GgufModel,
    layer: usize,
    hidden_features: usize,
) -> Result<MlpLayerPlan, E2eError> {
    let ffn_norm = resolve_required_layer_tensor_name(
        model,
        layer,
        "ffn_norm",
        layer_ffn_norm_candidates(layer),
    )?;
    let ffn_gate = resolve_required_layer_tensor_name(
        model,
        layer,
        "ffn_gate",
        layer_ffn_gate_candidates(layer),
    )?;
    let ffn_up =
        resolve_required_layer_tensor_name(model, layer, "ffn_up", layer_ffn_up_candidates(layer))?;
    let ffn_down = resolve_required_layer_tensor_name(
        model,
        layer,
        "ffn_down",
        layer_ffn_down_candidates(layer),
    )?;

    let gate_len = model
        .tensor_len(&ffn_gate)
        .map_err(|source| E2eError::model("GgufModel::tensor_len(ffn_gate)", source))?;
    if hidden_features == 0 || !gate_len.is_multiple_of(hidden_features) {
        return Err(E2eError::InvalidMlpGateShape {
            tensor_name: ffn_gate,
            hidden_features,
            tensor_len: gate_len,
        });
    }
    let ffn_features = gate_len / hidden_features;
    let mlp_config = MlpInferenceConfig::new(hidden_features, ffn_features)
        .map_err(|source| E2eError::inference("MlpInferenceConfig::new", source))?;
    let mlp_weights = MlpWeights::from_model(model, &ffn_gate, &ffn_up, &ffn_down, mlp_config)
        .map_err(|source| E2eError::inference("MlpWeights::from_model", source))?;
    let ffn_norm_values = decode_norm_tensor(model, &ffn_norm, hidden_features, "ffn_norm")?;

    Ok(MlpLayerPlan {
        weights: mlp_weights,
        norm_values: ffn_norm_values,
    })
}

fn build_qwen35_full_attention_layer_plan(
    model: &GgufModel,
    metadata: &TransformerMetadata,
    names: LlamaLayerTensorNames,
    hidden_features: usize,
) -> Result<AttentionLayerPlan, E2eError> {
    let head_dimension = required_transformer_usize(
        metadata.attention_key_length(),
        "qwen35.attention.key_length",
    )?;
    let head_count = metadata.attention_head_count();
    let kv_head_count = metadata.attention_head_count_kv();
    let q_norm_name = names.attn_q_norm.as_deref().ok_or_else(|| {
        E2eError::naming(
            "build_qwen35_full_attention_layer_plan",
            NamingError::MissingLayerTensor {
                layer: names.layer,
                role: "attn_q_norm",
                tried: vec![format!("blk.{}.attn_q_norm.weight", names.layer)],
            },
        )
    })?;
    let k_norm_name = names.attn_k_norm.as_deref().ok_or_else(|| {
        E2eError::naming(
            "build_qwen35_full_attention_layer_plan",
            NamingError::MissingLayerTensor {
                layer: names.layer,
                role: "attn_k_norm",
                tried: vec![format!("blk.{}.attn_k_norm.weight", names.layer)],
            },
        )
    })?;

    let rope_n_dims = metadata.rope_dimension_count().unwrap_or(head_dimension);
    if rope_n_dims > head_dimension || !rope_n_dims.is_multiple_of(2) {
        return Err(E2eError::RopeConfigInvalid {
            rope_n_dims,
            head_dimension,
        });
    }

    Ok(AttentionLayerPlan::Qwen35Full(
        Qwen35FullAttentionLayerPlan {
            norm_values: decode_norm_tensor(model, &names.attn_norm, hidden_features, "attn_norm")?,
            q_norm_values: decode_exact_tensor(model, q_norm_name, head_dimension, "attn_q_norm")?,
            k_norm_values: decode_exact_tensor(model, k_norm_name, head_dimension, "attn_k_norm")?,
            q_weight_values: decode_matrix_tensor(
                model,
                &names.attn_q,
                hidden_features,
                checked_mul(head_count, checked_mul(head_dimension, 2)?)?,
                "attn_q",
            )?,
            k_weight_values: decode_matrix_tensor(
                model,
                &names.attn_k,
                hidden_features,
                checked_mul(kv_head_count, head_dimension)?,
                "attn_k",
            )?,
            v_weight_values: decode_matrix_tensor(
                model,
                &names.attn_v,
                hidden_features,
                checked_mul(kv_head_count, head_dimension)?,
                "attn_v",
            )?,
            output_weight_values: decode_matrix_tensor(
                model,
                &names.attn_output,
                checked_mul(head_count, head_dimension)?,
                hidden_features,
                "attn_output",
            )?,
            head_count,
            kv_head_count,
            head_dimension,
            attention_scale: metadata
                .attention_scale()
                .unwrap_or(1.0 / (head_dimension as f32).sqrt()),
            rope_n_dims,
            rope_freq_base: metadata.rope_freq_base(),
            rope_freq_scale: metadata.rope_freq_scale(),
        },
    ))
}

fn build_qwen35_linear_attention_layer_plan(
    model: &GgufModel,
    metadata: &TransformerMetadata,
    layer: usize,
    hidden_features: usize,
) -> Result<AttentionLayerPlan, E2eError> {
    let state_size =
        required_transformer_usize(metadata.ssm_state_size(), "qwen35.ssm.state_size")?;
    let group_count =
        required_transformer_usize(metadata.ssm_group_count(), "qwen35.ssm.group_count")?;
    let time_step_rank =
        required_transformer_usize(metadata.ssm_time_step_rank(), "qwen35.ssm.time_step_rank")?;
    let inner_size =
        required_transformer_usize(metadata.ssm_inner_size(), "qwen35.ssm.inner_size")?;
    let conv_kernel =
        required_transformer_usize(metadata.ssm_conv_kernel(), "qwen35.ssm.conv_kernel")?;
    let conv_channels = inner_size + checked_mul(checked_mul(group_count, state_size)?, 2)?;
    let attn_qkv = resolve_required_layer_tensor_name(
        model,
        layer,
        "attn_qkv",
        layer_attn_qkv_candidates(layer),
    )?;
    let attn_gate = resolve_required_layer_tensor_name(
        model,
        layer,
        "attn_gate",
        layer_attn_gate_candidates(layer),
    )?;
    let attn_norm = resolve_required_layer_tensor_name(
        model,
        layer,
        "attn_norm",
        layer_attn_norm_candidates(layer),
    )?;
    let ssm_alpha = resolve_required_layer_tensor_name(
        model,
        layer,
        "ssm_alpha",
        layer_ssm_alpha_candidates(layer),
    )?;
    let ssm_beta = resolve_required_layer_tensor_name(
        model,
        layer,
        "ssm_beta",
        layer_ssm_beta_candidates(layer),
    )?;
    let ssm_conv1d = resolve_required_layer_tensor_name(
        model,
        layer,
        "ssm_conv1d",
        layer_ssm_conv1d_candidates(layer),
    )?;
    let ssm_dt =
        resolve_required_layer_tensor_name(model, layer, "ssm_dt", layer_ssm_dt_candidates(layer))?;
    let ssm_a =
        resolve_required_layer_tensor_name(model, layer, "ssm_a", layer_ssm_a_candidates(layer))?;
    let ssm_norm = resolve_required_layer_tensor_name(
        model,
        layer,
        "ssm_norm",
        layer_ssm_norm_candidates(layer),
    )?;
    let ssm_out = resolve_required_layer_tensor_name(
        model,
        layer,
        "ssm_out",
        layer_ssm_out_candidates(layer),
    )?;

    Ok(AttentionLayerPlan::Qwen35Linear(
        Qwen35LinearAttentionLayerPlan {
            norm_values: decode_norm_tensor(model, &attn_norm, hidden_features, "attn_norm")?,
            qkv_weight_values: decode_matrix_tensor(
                model,
                &attn_qkv,
                hidden_features,
                conv_channels,
                "attn_qkv",
            )?,
            gate_weight_values: decode_matrix_tensor(
                model,
                &attn_gate,
                hidden_features,
                inner_size,
                "attn_gate",
            )?,
            alpha_weight_values: decode_matrix_tensor(
                model,
                &ssm_alpha,
                hidden_features,
                time_step_rank,
                "ssm_alpha",
            )?,
            beta_weight_values: decode_matrix_tensor(
                model,
                &ssm_beta,
                hidden_features,
                time_step_rank,
                "ssm_beta",
            )?,
            conv_weight_values: decode_matrix_tensor(
                model,
                &ssm_conv1d,
                conv_kernel,
                conv_channels,
                "ssm_conv1d",
            )?,
            dt_bias_values: decode_exact_tensor(model, &ssm_dt, time_step_rank, "ssm_dt")?,
            ssm_a_values: decode_exact_tensor(model, &ssm_a, time_step_rank, "ssm_a")?,
            ssm_norm_values: decode_exact_tensor(model, &ssm_norm, state_size, "ssm_norm")?,
            ssm_out_weight_values: decode_matrix_tensor(
                model,
                &ssm_out,
                inner_size,
                hidden_features,
                "ssm_out",
            )?,
            state_size,
            group_count,
            time_step_rank,
            inner_size,
            conv_kernel,
        },
    ))
}

fn required_transformer_usize(value: Option<usize>, key: &str) -> Result<usize, E2eError> {
    value.ok_or_else(|| {
        E2eError::metadata(
            "required_transformer_usize",
            MetadataError::MissingRequiredKey {
                key: key.to_string(),
            },
        )
    })
}

fn build_mlp_only_layer_plan(
    model: &GgufModel,
    layer: usize,
    hidden_features: usize,
) -> Result<LayerPlan, E2eError> {
    Ok(LayerPlan {
        attention: None,
        mlp: build_layer_mlp_plan(model, layer, hidden_features)?,
    })
}
