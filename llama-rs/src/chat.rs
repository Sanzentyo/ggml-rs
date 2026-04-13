//! Chat message types and template formatting for multi-turn conversations.
//!
//! Provides [`ChatMessage`], [`Role`], and [`ChatFormat`] types plus
//! prompt formatting for ChatML-style models (Qwen3.5, etc.).

use crate::model::GgufModel;
use ggml_rs::GgufValue;
use std::fmt;
use thiserror::Error;

const CHAT_TEMPLATE_KEY: &str = "tokenizer.chat_template";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Role of a participant in a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => f.write_str("system"),
            Self::User => f.write_str("user"),
            Self::Assistant => f.write_str("assistant"),
        }
    }
}

/// A single message in a multi-turn conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
}

/// Chat template format that the model expects.
///
/// Currently only ChatML is supported (used by Qwen3.5 and many other models).
/// Extensible for future formats (Llama3, Mistral, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    /// ChatML format: `<|im_start|>role\ncontent\n<|im_end|>\n`
    ChatMl,
}

impl fmt::Display for ChatFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChatMl => f.write_str("ChatML"),
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ChatError {
    #[error("no chat template found in model metadata (key `{CHAT_TEMPLATE_KEY}`)")]
    NoChatTemplate,
    #[error("unrecognized chat template format (could not auto-detect from template string)")]
    UnrecognizedFormat,
    #[error("chat messages must not be empty")]
    EmptyMessages,
    #[error("message content contains reserved sentinel `{sentinel}` (role={role})")]
    ReservedSentinelInContent { sentinel: String, role: String },
}

// ---------------------------------------------------------------------------
// Template detection & formatting
// ---------------------------------------------------------------------------

/// Read the raw chat template string from a GGUF model's metadata.
pub fn read_chat_template(model: &GgufModel) -> Option<String> {
    match model.kv_value(CHAT_TEMPLATE_KEY) {
        Some(GgufValue::String(value)) => Some(value.clone()),
        _ => None,
    }
}

/// Auto-detect the chat format from a raw Jinja2 template string.
///
/// Currently recognizes ChatML by the presence of `<|im_start|>`.
pub fn detect_chat_format(template: &str) -> Option<ChatFormat> {
    if template.contains("<|im_start|>") {
        Some(ChatFormat::ChatMl)
    } else {
        None
    }
}

/// Format a sequence of chat messages into a prompt string.
///
/// Appends a generation prompt (`<|im_start|>assistant\n`) so the model
/// begins generating the assistant's response.
///
/// Returns an error if any message content contains reserved sentinel tokens
/// (`<|im_start|>`, `<|im_end|>`) which could corrupt the conversation structure.
pub fn format_chat_prompt(
    messages: &[ChatMessage],
    format: ChatFormat,
) -> Result<String, ChatError> {
    if messages.is_empty() {
        return Err(ChatError::EmptyMessages);
    }
    match format {
        ChatFormat::ChatMl => {
            validate_no_sentinels(messages, CHATML_SENTINELS)?;
            Ok(format_chatml(messages))
        }
    }
}

/// ChatML sentinel tokens that must not appear in user content.
const CHATML_SENTINELS: &[&str] = &["<|im_start|>", "<|im_end|>"];

/// The special tokens used by ChatML that the tokenizer must look up directly.
pub const CHATML_SPECIAL_TOKENS: &[&str] = &["<|im_start|>", "<|im_end|>"];

fn validate_no_sentinels(messages: &[ChatMessage], sentinels: &[&str]) -> Result<(), ChatError> {
    for msg in messages {
        for &sentinel in sentinels {
            if msg.content.contains(sentinel) {
                return Err(ChatError::ReservedSentinelInContent {
                    sentinel: sentinel.to_string(),
                    role: msg.role.to_string(),
                });
            }
        }
    }
    Ok(())
}

fn format_chatml(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&msg.role.to_string());
        prompt.push('\n');
        prompt.push_str(&msg.content);
        prompt.push_str("\n<|im_end|>\n");
    }
    // Generation prompt: tell the model to start generating as assistant
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_single_user_message() {
        let messages = vec![ChatMessage::user("Hello!")];
        let prompt = format_chat_prompt(&messages, ChatFormat::ChatMl).unwrap();
        assert_eq!(
            prompt,
            "<|im_start|>user\nHello!\n<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn chatml_system_and_user_message() {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("What is Rust?"),
        ];
        let prompt = format_chat_prompt(&messages, ChatFormat::ChatMl).unwrap();
        let expected = "\
<|im_start|>system\n\
You are a helpful assistant.\n\
<|im_end|>\n\
<|im_start|>user\n\
What is Rust?\n\
<|im_end|>\n\
<|im_start|>assistant\n";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn chatml_multi_turn_conversation() {
        let messages = vec![
            ChatMessage::system("Be concise."),
            ChatMessage::user("Hi"),
            ChatMessage::assistant("Hello!"),
            ChatMessage::user("How are you?"),
        ];
        let prompt = format_chat_prompt(&messages, ChatFormat::ChatMl).unwrap();
        assert!(prompt.contains("<|im_start|>assistant\nHello!\n<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn empty_messages_returns_error() {
        let result = format_chat_prompt(&[], ChatFormat::ChatMl);
        assert!(result.is_err());
    }

    #[test]
    fn sentinel_in_content_rejected() {
        let messages = vec![ChatMessage::user("Hello <|im_start|> world")];
        let result = format_chat_prompt(&messages, ChatFormat::ChatMl);
        assert!(matches!(
            result,
            Err(ChatError::ReservedSentinelInContent { .. })
        ));
    }

    #[test]
    fn sentinel_im_end_in_content_rejected() {
        let messages = vec![ChatMessage::user("Hello <|im_end|>")];
        let result = format_chat_prompt(&messages, ChatFormat::ChatMl);
        assert!(matches!(
            result,
            Err(ChatError::ReservedSentinelInContent { .. })
        ));
    }

    #[test]
    fn detect_chatml_from_template() {
        let template = "{%- for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}\n<|im_end|>\n{%- endfor %}";
        assert_eq!(detect_chat_format(template), Some(ChatFormat::ChatMl));
    }

    #[test]
    fn detect_unknown_format() {
        let template = "some random template without known markers";
        assert_eq!(detect_chat_format(template), None);
    }

    #[test]
    fn role_display() {
        assert_eq!(Role::System.to_string(), "system");
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
    }

    #[test]
    fn chat_format_display() {
        assert_eq!(ChatFormat::ChatMl.to_string(), "ChatML");
    }
}
