use crate::model::GgufModel;
use ggml_rs::{GgufArrayValue, GgufValue};
use regex::Regex;
use std::collections::HashMap;
use std::num::TryFromIntError;
use std::sync::OnceLock;
use thiserror::Error;

const TOKENIZER_MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENIZER_PRE_KEY: &str = "tokenizer.ggml.pre";
const TOKENIZER_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const TOKENIZER_MERGES_KEY: &str = "tokenizer.ggml.merges";
const TOKENIZER_ADD_BOS_KEY: &str = "tokenizer.ggml.add_bos_token";
const TOKENIZER_BOS_ID_KEY: &str = "tokenizer.ggml.bos_token_id";

const GPT2_PIECE_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerModel {
    Gpt2,
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("missing tokenizer metadata key `{key}`")]
    MissingMetadata { key: &'static str },
    #[error("invalid tokenizer metadata type for key `{key}`: expected {expected}, got {actual}")]
    InvalidMetadataType {
        key: &'static str,
        expected: &'static str,
        actual: &'static str,
    },
    #[error("unsupported tokenizer model `{model}` (currently only `gpt2` is supported)")]
    UnsupportedTokenizerModel { model: String },
    #[error("invalid tokenizer merge rule at index {index}: `{rule}`")]
    InvalidMergeRule { index: usize, rule: String },
    #[error("tokenizer merge split failed for rule `{rule}`")]
    InvalidMergePair { rule: String },
    #[error("token `{token}` is missing in tokenizer vocabulary")]
    MissingTokenInVocabulary { token: String },
    #[error("tokenizer BOS id is enabled but missing (`tokenizer.ggml.bos_token_id`)")]
    MissingBosTokenId,
    #[error("tokenizer token index overflows i32: {index}")]
    TokenIdOverflow { index: usize },
}

#[derive(Debug, Clone)]
pub struct GgufTokenizer {
    model: TokenizerModel,
    pre: Option<String>,
    vocab: HashMap<String, i32>,
    merge_ranks: HashMap<String, usize>,
    add_bos_token: bool,
    bos_token_id: Option<i32>,
}

impl GgufTokenizer {
    pub fn from_model(model: &GgufModel) -> Result<Self, TokenizerError> {
        let model_name = required_string(model, TOKENIZER_MODEL_KEY)?;
        let tokenizer_model = parse_tokenizer_model(model_name)?;
        let pre = optional_string(model, TOKENIZER_PRE_KEY).map(ToOwned::to_owned);
        let tokens = required_string_array(model, TOKENIZER_TOKENS_KEY)?;
        let merges = required_string_array(model, TOKENIZER_MERGES_KEY)?;
        let add_bos_token = optional_bool(model, TOKENIZER_ADD_BOS_KEY).unwrap_or(false);
        let bos_token_id = optional_i32(model, TOKENIZER_BOS_ID_KEY);
        if add_bos_token && bos_token_id.is_none() {
            return Err(TokenizerError::MissingBosTokenId);
        }

        let vocab = build_vocab(tokens)?;
        let merge_ranks = build_merge_ranks(merges)?;

        Ok(Self {
            model: tokenizer_model,
            pre,
            vocab,
            merge_ranks,
            add_bos_token,
            bos_token_id,
        })
    }

    pub fn model(&self) -> TokenizerModel {
        self.model
    }

    pub fn pre(&self) -> Option<&str> {
        self.pre.as_deref()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>, TokenizerError> {
        let mut token_ids = Vec::new();
        for piece in gpt2_piece_regex().find_iter(text).map(|m| m.as_str()) {
            let encoded_piece = byte_encode(piece);
            if encoded_piece.is_empty() {
                continue;
            }
            for token in self.bpe(&encoded_piece) {
                let token_id = self
                    .vocab
                    .get(&token)
                    .copied()
                    .ok_or(TokenizerError::MissingTokenInVocabulary { token })?;
                token_ids.push(token_id);
            }
        }

        if self.add_bos_token
            && let Some(bos_token_id) = self.bos_token_id
        {
            token_ids.insert(0, bos_token_id);
        }

        Ok(token_ids)
    }

    fn bpe(&self, piece: &str) -> Vec<String> {
        let mut symbols: Vec<String> = piece.chars().map(|ch| ch.to_string()).collect();
        if symbols.len() < 2 {
            return symbols;
        }

        loop {
            let mut best: Option<(usize, String, String)> = None;
            for pair_index in 0..(symbols.len() - 1) {
                let lhs = &symbols[pair_index];
                let rhs = &symbols[pair_index + 1];
                let pair_key = format!("{lhs} {rhs}");
                if let Some(&rank) = self.merge_ranks.get(&pair_key) {
                    match best {
                        Some((best_rank, _, _)) if rank >= best_rank => {}
                        _ => {
                            best = Some((rank, lhs.clone(), rhs.clone()));
                        }
                    }
                }
            }

            let Some((_, best_lhs, best_rhs)) = best else {
                break;
            };

            let mut merged = Vec::with_capacity(symbols.len());
            let mut cursor = 0usize;
            while cursor < symbols.len() {
                if cursor + 1 < symbols.len()
                    && symbols[cursor] == best_lhs
                    && symbols[cursor + 1] == best_rhs
                {
                    merged.push(format!("{}{}", symbols[cursor], symbols[cursor + 1]));
                    cursor += 2;
                } else {
                    merged.push(symbols[cursor].clone());
                    cursor += 1;
                }
            }
            symbols = merged;
            if symbols.len() < 2 {
                break;
            }
        }

        symbols
    }
}

pub fn tokenize_text_prompt(model: &GgufModel, text: &str) -> Result<Vec<i32>, TokenizerError> {
    GgufTokenizer::from_model(model)?.encode(text)
}

fn parse_tokenizer_model(value: &str) -> Result<TokenizerModel, TokenizerError> {
    match value {
        "gpt2" => Ok(TokenizerModel::Gpt2),
        other => Err(TokenizerError::UnsupportedTokenizerModel {
            model: other.to_owned(),
        }),
    }
}

fn build_vocab(tokens: Vec<String>) -> Result<HashMap<String, i32>, TokenizerError> {
    let mut vocab = HashMap::with_capacity(tokens.len());
    for (index, token) in tokens.into_iter().enumerate() {
        let token_id = i32::try_from(index)
            .map_err(|_: TryFromIntError| TokenizerError::TokenIdOverflow { index })?;
        vocab.insert(token, token_id);
    }
    Ok(vocab)
}

fn build_merge_ranks(merges: Vec<String>) -> Result<HashMap<String, usize>, TokenizerError> {
    let mut ranks = HashMap::with_capacity(merges.len());
    for (index, merge_rule) in merges.into_iter().enumerate() {
        if !merge_rule.contains(' ') {
            return Err(TokenizerError::InvalidMergeRule {
                index,
                rule: merge_rule,
            });
        }
        let (lhs, rhs) =
            merge_rule
                .split_once(' ')
                .ok_or_else(|| TokenizerError::InvalidMergePair {
                    rule: merge_rule.clone(),
                })?;
        if lhs.is_empty() || rhs.is_empty() {
            return Err(TokenizerError::InvalidMergePair { rule: merge_rule });
        }
        ranks.insert(merge_rule, index);
    }
    Ok(ranks)
}

fn required_string<'a>(model: &'a GgufModel, key: &'static str) -> Result<&'a str, TokenizerError> {
    match model.kv_value(key) {
        Some(GgufValue::String(value)) => Ok(value.as_str()),
        Some(other) => Err(TokenizerError::InvalidMetadataType {
            key,
            expected: "string",
            actual: other.type_name(),
        }),
        None => Err(TokenizerError::MissingMetadata { key }),
    }
}

fn optional_string<'a>(model: &'a GgufModel, key: &'static str) -> Option<&'a str> {
    match model.kv_value(key) {
        Some(GgufValue::String(value)) => Some(value.as_str()),
        _ => None,
    }
}

fn required_string_array(
    model: &GgufModel,
    key: &'static str,
) -> Result<Vec<String>, TokenizerError> {
    match model.kv_value(key) {
        Some(GgufValue::Array(GgufArrayValue::String(values))) => Ok(values.clone()),
        Some(other) => Err(TokenizerError::InvalidMetadataType {
            key,
            expected: "array<string>",
            actual: other.type_name(),
        }),
        None => Err(TokenizerError::MissingMetadata { key }),
    }
}

fn optional_bool(model: &GgufModel, key: &'static str) -> Option<bool> {
    match model.kv_value(key) {
        Some(GgufValue::Bool(value)) => Some(*value),
        _ => None,
    }
}

fn optional_i32(model: &GgufModel, key: &'static str) -> Option<i32> {
    model.kv_value(key).and_then(gguf_value_to_i32)
}

fn gguf_value_to_i32(value: &GgufValue) -> Option<i32> {
    match value {
        GgufValue::U8(value) => Some(i32::from(*value)),
        GgufValue::I8(value) => Some(i32::from(*value)),
        GgufValue::U16(value) => Some(i32::from(*value)),
        GgufValue::I16(value) => Some(i32::from(*value)),
        GgufValue::U32(value) => i32::try_from(*value).ok(),
        GgufValue::I32(value) => Some(*value),
        GgufValue::U64(value) => i32::try_from(*value).ok(),
        GgufValue::I64(value) => i32::try_from(*value).ok(),
        GgufValue::F32(value) if value.fract() == 0.0 => i32::try_from(*value as i64).ok(),
        GgufValue::F64(value) if value.fract() == 0.0 => i32::try_from(*value as i64).ok(),
        _ => None,
    }
}

fn gpt2_piece_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(GPT2_PIECE_PATTERN).expect("valid GPT-2 tokenizer regex"))
}

fn byte_encode(text: &str) -> String {
    let table = byte_to_unicode_table();
    text.as_bytes()
        .iter()
        .map(|byte| table[*byte as usize])
        .collect()
}

fn byte_to_unicode_table() -> &'static [char; 256] {
    static TABLE: OnceLock<[char; 256]> = OnceLock::new();
    TABLE.get_or_init(build_byte_to_unicode_table)
}

fn build_byte_to_unicode_table() -> [char; 256] {
    let mut table = ['\0'; 256];

    let mut bs: Vec<u16> = (33u16..=126u16).collect();
    bs.extend(161u16..=172u16);
    bs.extend(174u16..=255u16);

    let mut cs: Vec<u32> = bs.iter().map(|&value| u32::from(value)).collect();
    let mut present = [false; 256];
    for &value in &bs {
        present[value as usize] = true;
    }

    let mut extra = 0u32;
    for byte in 0u16..=255u16 {
        if !present[byte as usize] {
            bs.push(byte);
            cs.push(256 + extra);
            extra += 1;
        }
    }

    for (byte, codepoint) in bs.into_iter().zip(cs.into_iter()) {
        table[byte as usize] = char::from_u32(codepoint).expect("valid codepoint");
    }

    table
}

#[cfg(test)]
mod tests {
    use super::{GgufTokenizer, build_byte_to_unicode_table, gpt2_piece_regex};
    use std::collections::HashMap;

    fn tokenizer_for_test(merge_rules: &[&str]) -> GgufTokenizer {
        let mut vocab = HashMap::new();
        for (index, token) in ["a", "b", "c", "ab", "bc", "abc"].iter().enumerate() {
            vocab.insert((*token).to_owned(), index as i32);
        }
        let mut merge_ranks = HashMap::new();
        for (index, rule) in merge_rules.iter().enumerate() {
            merge_ranks.insert((*rule).to_owned(), index);
        }
        GgufTokenizer {
            model: super::TokenizerModel::Gpt2,
            pre: Some("gpt2".to_owned()),
            vocab,
            merge_ranks,
            add_bos_token: false,
            bos_token_id: None,
        }
    }

    #[test]
    fn byte_to_unicode_maps_space_to_glyph() {
        let table = build_byte_to_unicode_table();
        assert_eq!(table[32], 'Ġ');
        assert_eq!(table[65], 'A');
    }

    #[test]
    fn bpe_prefers_lowest_rank_pair() {
        let tokenizer = tokenizer_for_test(&["b c", "a b"]);
        let merged = tokenizer.bpe("abc");
        assert_eq!(merged, vec!["a".to_string(), "bc".to_string()]);
    }

    #[test]
    fn bpe_applies_recursive_merges() {
        let tokenizer = tokenizer_for_test(&["a b", "ab c"]);
        let merged = tokenizer.bpe("abc");
        assert_eq!(merged, vec!["abc".to_string()]);
    }

    #[test]
    fn regex_splits_text_into_gpt2_like_pieces() {
        let pieces: Vec<&str> = gpt2_piece_regex()
            .find_iter("Hello, world!")
            .map(|capture| capture.as_str())
            .collect();
        assert_eq!(pieces, vec!["Hello", ",", " world", "!"]);
    }
}
