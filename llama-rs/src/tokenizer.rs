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
    #[error("unknown token id {token_id} not in vocabulary")]
    UnknownTokenId { token_id: i32 },
    #[error("decoded bytes are not valid UTF-8")]
    InvalidUtf8,
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
    reverse_vocab: HashMap<i32, String>,
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
        let reverse_vocab: HashMap<i32, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();
        let merge_ranks = build_merge_ranks(merges)?;

        Ok(Self {
            model: tokenizer_model,
            pre,
            vocab,
            reverse_vocab,
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
        let mut token_ids = self.encode_raw(text)?;

        if self.add_bos_token
            && let Some(bos_token_id) = self.bos_token_id
        {
            token_ids.insert(0, bos_token_id);
        }

        Ok(token_ids)
    }

    /// Decode a sequence of token IDs back into text.
    ///
    /// This reverses the GPT-2 byte-level BPE encoding. Special tokens
    /// (BOS, EOS) are included verbatim unless filtered by the caller.
    pub fn decode(&self, token_ids: &[i32]) -> Result<String, TokenizerError> {
        let mut unicode_pieces = String::new();
        for &token_id in token_ids {
            let piece = self
                .reverse_vocab
                .get(&token_id)
                .ok_or(TokenizerError::UnknownTokenId { token_id })?;
            unicode_pieces.push_str(piece);
        }
        byte_decode(&unicode_pieces).ok_or(TokenizerError::InvalidUtf8)
    }

    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, token_id: i32) -> Result<String, TokenizerError> {
        let piece = self
            .reverse_vocab
            .get(&token_id)
            .ok_or(TokenizerError::UnknownTokenId { token_id })?;
        byte_decode(piece).ok_or(TokenizerError::InvalidUtf8)
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// BOS token ID, if configured.
    pub fn bos_token_id(&self) -> Option<i32> {
        self.bos_token_id
    }

    /// Look up a special token (e.g. `<|im_start|>`) directly in the vocabulary.
    ///
    /// Special tokens are stored verbatim in the GGUF vocab and must not go
    /// through BPE. Returns `None` if the token is not in the vocabulary.
    pub fn special_token_id(&self, token: &str) -> Option<i32> {
        self.vocab.get(token).copied()
    }

    /// Encode a chat prompt that contains special token markers.
    ///
    /// Splits the input at each occurrence of the given `special_tokens`,
    /// encodes normal text through BPE and special tokens via direct lookup.
    /// This avoids the regex pre-tokenizer mangling special token syntax.
    pub fn encode_with_special_tokens(
        &self,
        text: &str,
        special_tokens: &[&str],
    ) -> Result<Vec<i32>, TokenizerError> {
        let segments = split_on_special_tokens(text, special_tokens);
        let mut token_ids = Vec::new();
        for segment in segments {
            if special_tokens.contains(&segment.as_str()) {
                let id = self.vocab.get(segment.as_str()).copied().ok_or(
                    TokenizerError::MissingTokenInVocabulary {
                        token: segment.to_string(),
                    },
                )?;
                token_ids.push(id);
            } else if !segment.is_empty() {
                let mut encoded = self.encode_raw(&segment)?;
                token_ids.append(&mut encoded);
            }
        }

        if self.add_bos_token
            && let Some(bos_token_id) = self.bos_token_id
        {
            token_ids.insert(0, bos_token_id);
        }

        Ok(token_ids)
    }

    /// Create a streaming decoder that buffers tokens for UTF-8 safe output.
    pub fn streaming_decoder(&self) -> StreamingDecoder<'_> {
        StreamingDecoder::new(self)
    }

    /// Raw BPE encoding of text (no BOS prepend). Used by both `encode` and
    /// `encode_with_special_tokens`.
    fn encode_raw(&self, text: &str) -> Result<Vec<i32>, TokenizerError> {
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

/// Reverse of `byte_encode`: maps GPT-2 Unicode chars back to raw bytes.
fn byte_decode(encoded: &str) -> Option<String> {
    let bytes = byte_decode_to_bytes(encoded);
    String::from_utf8(bytes).ok()
}

/// Decode a GPT-2 BPE unicode piece to raw bytes.
///
/// Unlike `byte_decode`, this returns the raw byte sequence without requiring
/// valid UTF-8, which is needed for the streaming decoder to handle cross-token
/// multi-byte sequences.
fn byte_decode_to_bytes(encoded: &str) -> Vec<u8> {
    let table = unicode_to_byte_table();
    encoded
        .chars()
        .filter_map(|ch| table.get(&ch).copied())
        .collect()
}

fn unicode_to_byte_table() -> &'static HashMap<char, u8> {
    static TABLE: OnceLock<HashMap<char, u8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let forward = byte_to_unicode_table();
        forward
            .iter()
            .enumerate()
            .map(|(byte, &ch)| (ch, byte as u8))
            .collect()
    })
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

// ---------------------------------------------------------------------------
// Special token splitting
// ---------------------------------------------------------------------------

/// Split text into segments, separating special tokens from normal text.
///
/// Returns segments in order. Special tokens appear as their own segments.
fn split_on_special_tokens(text: &str, special_tokens: &[&str]) -> Vec<String> {
    if special_tokens.is_empty() {
        return vec![text.to_string()];
    }

    let mut segments = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        // Find the earliest special token match
        let earliest = special_tokens
            .iter()
            .filter_map(|&st| remaining.find(st).map(|pos| (pos, st)))
            .min_by_key(|&(pos, _)| pos);

        match earliest {
            Some((pos, token)) => {
                if pos > 0 {
                    segments.push(remaining[..pos].to_string());
                }
                segments.push(token.to_string());
                remaining = &remaining[pos + token.len()..];
            }
            None => {
                segments.push(remaining.to_string());
                break;
            }
        }
    }
    segments
}

// ---------------------------------------------------------------------------
// Streaming decoder
// ---------------------------------------------------------------------------

/// Incrementally decodes token IDs into UTF-8 text.
///
/// GPT-2 byte-level BPE tokens can represent partial UTF-8 byte sequences.
/// This decoder buffers token IDs and only yields text when the accumulated
/// tokens decode to valid UTF-8, preventing garbled or errored output
/// during streaming generation.
#[derive(Debug)]
pub struct StreamingDecoder<'t> {
    tokenizer: &'t GgufTokenizer,
    /// Undecoded byte suffix from previous tokens (incomplete UTF-8 tail).
    pending_bytes: Vec<u8>,
    /// Total number of tokens fed so far.
    count: usize,
}

impl<'t> StreamingDecoder<'t> {
    fn new(tokenizer: &'t GgufTokenizer) -> Self {
        Self {
            tokenizer,
            pending_bytes: Vec::new(),
            count: 0,
        }
    }

    /// Feed a new token and return any newly decodable text.
    ///
    /// Returns `Ok(Some(text))` when new text can be emitted, `Ok(None)` when
    /// the token is buffered but not yet decodable, or an error if the token
    /// ID is unknown.
    pub fn next_token(&mut self, token_id: i32) -> Result<Option<String>, TokenizerError> {
        let piece = self
            .tokenizer
            .reverse_vocab
            .get(&token_id)
            .ok_or(TokenizerError::UnknownTokenId { token_id })?;
        self.count += 1;

        // Decode the token's BPE piece to raw bytes
        let new_bytes = byte_decode_to_bytes(piece);
        self.pending_bytes.extend_from_slice(&new_bytes);

        // Find the longest valid UTF-8 prefix
        match std::str::from_utf8(&self.pending_bytes) {
            Ok(s) => {
                let text = s.to_string();
                self.pending_bytes.clear();
                if text.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(text))
                }
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if valid_up_to == 0 {
                    // No valid UTF-8 yet — keep buffering
                    Ok(None)
                } else {
                    let text = std::str::from_utf8(&self.pending_bytes[..valid_up_to])
                        .expect("valid_up_to guarantees valid UTF-8")
                        .to_string();
                    let remaining = self.pending_bytes[valid_up_to..].to_vec();
                    self.pending_bytes = remaining;
                    Ok(Some(text))
                }
            }
        }
    }

    /// Flush any remaining buffered bytes, returning the final text.
    pub fn flush(&mut self) -> Result<Option<String>, TokenizerError> {
        if self.pending_bytes.is_empty() {
            return Ok(None);
        }
        match std::str::from_utf8(&self.pending_bytes) {
            Ok(s) => {
                let text = s.to_string();
                self.pending_bytes.clear();
                if text.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(text))
                }
            }
            // Remaining bytes don't form valid UTF-8 — discard
            Err(_) => {
                self.pending_bytes.clear();
                Ok(None)
            }
        }
    }

    /// Total number of tokens fed so far.
    pub fn token_count(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GgufTokenizer, StreamingDecoder, build_byte_to_unicode_table, gpt2_piece_regex,
        split_on_special_tokens,
    };
    use std::collections::HashMap;

    fn tokenizer_for_test(merge_rules: &[&str]) -> GgufTokenizer {
        let mut vocab = HashMap::new();
        for (index, token) in ["a", "b", "c", "ab", "bc", "abc"].iter().enumerate() {
            vocab.insert((*token).to_owned(), index as i32);
        }
        let reverse_vocab: HashMap<i32, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();
        let mut merge_ranks = HashMap::new();
        for (index, rule) in merge_rules.iter().enumerate() {
            merge_ranks.insert((*rule).to_owned(), index);
        }
        GgufTokenizer {
            model: super::TokenizerModel::Gpt2,
            pre: Some("gpt2".to_owned()),
            vocab,
            reverse_vocab,
            merge_ranks,
            add_bos_token: false,
            bos_token_id: None,
        }
    }

    fn tokenizer_with_special_tokens() -> GgufTokenizer {
        let tokens = [
            "a",
            "b",
            "c",
            "ab",
            "bc",
            "abc",
            "<|im_start|>",
            "<|im_end|>",
            "\n",
        ];
        let mut vocab = HashMap::new();
        for (index, token) in tokens.iter().enumerate() {
            vocab.insert((*token).to_owned(), index as i32);
        }
        let reverse_vocab: HashMap<i32, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();
        GgufTokenizer {
            model: super::TokenizerModel::Gpt2,
            pre: Some("gpt2".to_owned()),
            vocab,
            reverse_vocab,
            merge_ranks: HashMap::new(),
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

    #[test]
    fn decode_reverses_known_ids() {
        let tokenizer = tokenizer_for_test(&["a b", "ab c"]);
        // "a" = 0, "b" = 1, "c" = 2
        let decoded = tokenizer.decode(&[0, 1, 2]).unwrap();
        assert_eq!(decoded, "abc");
    }

    #[test]
    fn decode_unknown_token_id_returns_error() {
        let tokenizer = tokenizer_for_test(&[]);
        let result = tokenizer.decode(&[999]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_token_returns_single_piece() {
        let tokenizer = tokenizer_for_test(&[]);
        assert_eq!(tokenizer.decode_token(3).unwrap(), "ab");
        assert_eq!(tokenizer.decode_token(5).unwrap(), "abc");
    }

    #[test]
    fn byte_encode_decode_roundtrip() {
        let original = "Hello, world! 日本語";
        let encoded = super::byte_encode(original);
        let decoded = super::byte_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn vocab_size_matches_token_count() {
        let tokenizer = tokenizer_for_test(&[]);
        assert_eq!(tokenizer.vocab_size(), 6);
    }

    // --- Special token tests ---

    #[test]
    fn special_token_id_lookup() {
        let tokenizer = tokenizer_with_special_tokens();
        assert_eq!(tokenizer.special_token_id("<|im_start|>"), Some(6));
        assert_eq!(tokenizer.special_token_id("<|im_end|>"), Some(7));
        assert_eq!(tokenizer.special_token_id("<|nonexistent|>"), None);
    }

    #[test]
    fn split_on_special_tokens_basic() {
        let segments =
            split_on_special_tokens("<|im_start|>abc<|im_end|>", &["<|im_start|>", "<|im_end|>"]);
        assert_eq!(segments, vec!["<|im_start|>", "abc", "<|im_end|>"]);
    }

    #[test]
    fn split_on_special_tokens_no_specials() {
        let segments = split_on_special_tokens("hello world", &["<|im_start|>"]);
        assert_eq!(segments, vec!["hello world"]);
    }

    #[test]
    fn split_on_special_tokens_adjacent() {
        let segments =
            split_on_special_tokens("<|im_start|><|im_end|>", &["<|im_start|>", "<|im_end|>"]);
        assert_eq!(segments, vec!["<|im_start|>", "<|im_end|>"]);
    }

    #[test]
    fn split_on_special_tokens_empty_list() {
        let segments = split_on_special_tokens("hello", &[]);
        assert_eq!(segments, vec!["hello"]);
    }

    #[test]
    fn encode_with_special_tokens_direct_lookup() {
        let tokenizer = tokenizer_with_special_tokens();
        let ids = tokenizer
            .encode_with_special_tokens(
                "<|im_start|>abc<|im_end|>",
                &["<|im_start|>", "<|im_end|>"],
            )
            .unwrap();
        // <|im_start|>=6, then "abc" via BPE, <|im_end|>=7
        assert_eq!(ids[0], 6); // <|im_start|>
        assert_eq!(*ids.last().unwrap(), 7); // <|im_end|>
    }

    // --- Streaming decoder tests ---

    #[test]
    fn streaming_decoder_emits_text() {
        let tokenizer = tokenizer_for_test(&[]);
        let mut decoder = StreamingDecoder::new(&tokenizer);
        // "a"=0, "b"=1, "c"=2
        let t1 = decoder.next_token(0).unwrap();
        assert_eq!(t1, Some("a".to_string()));
        let t2 = decoder.next_token(1).unwrap();
        assert_eq!(t2, Some("b".to_string()));
    }

    #[test]
    fn streaming_decoder_unknown_token_errors() {
        let tokenizer = tokenizer_for_test(&[]);
        let mut decoder = StreamingDecoder::new(&tokenizer);
        assert!(decoder.next_token(999).is_err());
    }

    #[test]
    fn streaming_decoder_token_count() {
        let tokenizer = tokenizer_for_test(&[]);
        let mut decoder = StreamingDecoder::new(&tokenizer);
        assert_eq!(decoder.token_count(), 0);
        let _ = decoder.next_token(0);
        assert_eq!(decoder.token_count(), 1);
        let _ = decoder.next_token(1);
        assert_eq!(decoder.token_count(), 2);
    }
}
