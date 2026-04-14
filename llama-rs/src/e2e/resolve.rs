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

#[cfg(test)]
mod tests {
    use super::*;

    // ── Candidate generator tests ──────────────────────────────────────

    #[test]
    fn global_token_embedding_candidates_has_expected_patterns() {
        let c = global_token_embedding_candidates();
        assert_eq!(c.len(), 3);
        assert!(c.contains(&"token_embd.weight".to_string()));
        assert!(c.contains(&"tok_embeddings.weight".to_string()));
        assert!(c.contains(&"model.embed_tokens.weight".to_string()));
    }

    #[test]
    fn global_output_norm_candidates_has_expected_patterns() {
        let c = global_output_norm_candidates();
        assert_eq!(c.len(), 3);
        assert!(c.contains(&"output_norm.weight".to_string()));
        assert!(c.contains(&"norm.weight".to_string()));
        assert!(c.contains(&"model.norm.weight".to_string()));
    }

    #[test]
    fn global_output_projection_candidates_has_expected_patterns() {
        let c = global_output_projection_candidates();
        assert_eq!(c.len(), 3);
        assert!(c.contains(&"output.weight".to_string()));
        assert!(c.contains(&"lm_head.weight".to_string()));
        assert!(c.contains(&"model.lm_head.weight".to_string()));
    }

    #[test]
    fn layer_candidates_format_layer_number_correctly() {
        let layer = 42;
        assert!(
            layer_ffn_norm_candidates(layer)
                .iter()
                .any(|s| s.contains("blk.42."))
        );
        assert!(
            layer_ffn_gate_candidates(layer)
                .iter()
                .any(|s| s.contains("blk.42."))
        );
        assert!(
            layer_ffn_up_candidates(layer)
                .iter()
                .any(|s| s.contains("blk.42."))
        );
        assert!(
            layer_ffn_down_candidates(layer)
                .iter()
                .any(|s| s.contains("blk.42."))
        );
        assert!(
            layer_attn_norm_candidates(layer)
                .iter()
                .any(|s| s.contains("blk.42."))
        );
    }

    #[test]
    fn layer_ssm_candidates_produce_single_pattern_each() {
        let layer = 7;
        assert_eq!(layer_ssm_alpha_candidates(layer).len(), 1);
        assert_eq!(layer_ssm_beta_candidates(layer).len(), 1);
        assert_eq!(layer_ssm_conv1d_candidates(layer).len(), 1);
        assert_eq!(layer_ssm_dt_candidates(layer).len(), 1);
        assert_eq!(layer_ssm_a_candidates(layer).len(), 1);
        assert_eq!(layer_ssm_norm_candidates(layer).len(), 1);
        assert_eq!(layer_ssm_out_candidates(layer).len(), 1);

        assert_eq!(
            layer_ssm_alpha_candidates(layer)[0],
            "blk.7.ssm_alpha.weight"
        );
        assert_eq!(layer_ssm_dt_candidates(layer)[0], "blk.7.ssm_dt.bias");
        assert_eq!(layer_ssm_a_candidates(layer)[0], "blk.7.ssm_a");
    }

    #[test]
    fn layer_candidates_at_zero() {
        assert!(
            layer_ffn_norm_candidates(0)
                .iter()
                .any(|s| s.contains("blk.0."))
        );
        assert_eq!(layer_attn_qkv_candidates(0)[0], "blk.0.attn_qkv.weight");
    }

    // ── Resolution logic tests ─────────────────────────────────────────

    #[test]
    fn resolve_first_tensor_name_returns_first_candidate_match() {
        let model = GgufModel::stub_from_names(&["b_tensor", "a_tensor"]);
        let candidates = vec!["a_tensor".to_string(), "b_tensor".to_string()];
        // "a_tensor" is the first *candidate*, even though "b_tensor" appears first in model.
        let result = resolve_first_tensor_name(&model, &candidates);
        assert_eq!(result, Some("a_tensor".to_string()));
    }

    #[test]
    fn resolve_first_tensor_name_returns_none_on_no_match() {
        let model = GgufModel::stub_from_names(&["x_tensor"]);
        let candidates = vec!["a_tensor".to_string(), "b_tensor".to_string()];
        assert_eq!(resolve_first_tensor_name(&model, &candidates), None);
    }

    #[test]
    fn resolve_first_tensor_name_empty_candidates_returns_none() {
        let model = GgufModel::stub_from_names(&["a_tensor"]);
        assert_eq!(resolve_first_tensor_name(&model, &[]), None);
    }

    #[test]
    fn resolve_required_global_returns_ok_on_match() {
        let model = GgufModel::stub_from_names(&["token_embd.weight"]);
        let result = resolve_required_global_tensor_name(
            &model,
            "token_embedding",
            global_token_embedding_candidates(),
        );
        assert_eq!(result.unwrap(), "token_embd.weight");
    }

    #[test]
    fn resolve_required_global_returns_error_with_tried_candidates() {
        let model = GgufModel::stub_from_names(&[]);
        let candidates = global_token_embedding_candidates();
        let expected_tried = candidates.clone();
        let result = resolve_required_global_tensor_name(&model, "token_embedding", candidates);
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("token_embedding"), "error should mention role");
        for tried in &expected_tried {
            assert!(msg.contains(tried), "error should list tried name: {tried}");
        }
    }

    #[test]
    fn resolve_required_layer_returns_ok_on_match() {
        let model = GgufModel::stub_from_names(&["blk.3.ffn_norm.weight"]);
        let result =
            resolve_required_layer_tensor_name(&model, 3, "ffn_norm", layer_ffn_norm_candidates(3));
        assert_eq!(result.unwrap(), "blk.3.ffn_norm.weight");
    }

    #[test]
    fn resolve_required_layer_error_includes_layer_number() {
        let model = GgufModel::stub_from_names(&[]);
        let result =
            resolve_required_layer_tensor_name(&model, 5, "ffn_norm", layer_ffn_norm_candidates(5));
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("5"), "error should mention layer number");
        assert!(msg.contains("ffn_norm"), "error should mention role");
    }

    // ── Global tensor orchestration tests ──────────────────────────────

    #[test]
    fn resolve_global_tensor_names_all_found() {
        let model = GgufModel::stub_from_names(&[
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
        ]);
        let names = resolve_global_tensor_names(&model).unwrap();
        assert_eq!(names.token_embedding, "token_embd.weight");
        assert_eq!(names.output_norm, "output_norm.weight");
        assert_eq!(names.output, Some("output.weight".to_string()));
    }

    #[test]
    fn resolve_global_tensor_names_output_optional_missing() {
        let model = GgufModel::stub_from_names(&["token_embd.weight", "output_norm.weight"]);
        let names = resolve_global_tensor_names(&model).unwrap();
        assert_eq!(names.output, None);
    }

    #[test]
    fn resolve_global_tensor_names_fails_on_missing_required() {
        // Missing token_embedding — should fail.
        let model = GgufModel::stub_from_names(&["output_norm.weight"]);
        let result = resolve_global_tensor_names(&model);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_global_prefers_alternative_naming_conventions() {
        // Use HuggingFace-style names instead of ggml-style.
        let model = GgufModel::stub_from_names(&[
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.lm_head.weight",
        ]);
        let names = resolve_global_tensor_names(&model).unwrap();
        assert_eq!(names.token_embedding, "model.embed_tokens.weight");
        assert_eq!(names.output_norm, "model.norm.weight");
        assert_eq!(names.output, Some("model.lm_head.weight".to_string()));
    }
}
