# Detokenization + Chat Infrastructure (simple-chat)

## Date
2026-04-16

## What changed

### Detokenization (`tokenizer.rs`)
- Added `decode()` and `decode_token()` methods (reverse GPT-2 byte-level BPE)
- Built reverse vocabulary (`HashMap<i32, String>`) from forward vocab
- Added `byte_decode()` (inverse of `byte_encode`) with `unicode_to_byte_table()`
- Added `encode_with_special_tokens()` for ChatML prompts — splits text at
  special tokens (`<|im_start|>`, `<|im_end|>`) and looks them up directly
  in vocab instead of passing through BPE regex
- Added `special_token_id()` for direct vocab lookup
- Added `StreamingDecoder` — buffers token IDs and only yields text when
  accumulated tokens decode to valid UTF-8 (handles partial byte sequences)
- Added `vocab_size()` and `bos_token_id()` accessors

### Chat module (`chat.rs` — new)
- `Role` enum: `System`, `User`, `Assistant`
- `ChatMessage` struct with convenience constructors (`system()`, `user()`, `assistant()`)
- `ChatFormat` enum: `ChatMl` (extensible for Llama3, Mistral, etc.)
- `format_chat_prompt()` — formats message history with ChatML markers and
  generation prompt suffix
- Content sanitization: rejects messages containing `<|im_start|>` or
  `<|im_end|>` sentinels to prevent prompt injection
- `read_chat_template()` — reads `tokenizer.chat_template` from GGUF metadata
- `detect_chat_format()` — auto-detects ChatML from template string

### simple_chat example (`examples/simple_chat.rs` — new)
- Interactive multi-turn chat loop
- CLI: `--model`, `--backend`, `--max-tokens`, `--system-prompt`
- Streaming token output via `StreamingDecoder`
- Stops on EOS or `<|im_end|>` turn terminator
- Full conversation history maintained across turns
- Per-turn statistics (token count, timing)

## Design decisions

1. **Reprocess full conversation each turn** — simple and correct for v1.
   KV cache reuse across turns would require `GenerationSession` extension.
2. **ChatFormat enum over Jinja2** — parsing general Jinja2 templates is
   complex and error-prone. Enum approach is extensible and reliable.
3. **Content sanitization** — reject rather than escape sentinel tokens in
   user content. Prevents prompt structure corruption.
4. **StreamingDecoder** — addresses rubber-duck critique that per-token
   `decode_token()` can produce invalid UTF-8 for multi-byte sequences.
5. **encode_with_special_tokens** — addresses rubber-duck critique that
   `encode()` would mangle ChatML markers through GPT-2 regex splitting.

## Test coverage

- 5 decode/roundtrip tests (tokenizer)
- 8 special token / split / encode tests (tokenizer)
- 3 streaming decoder tests (tokenizer)
- 9 chat format tests (chat module: single/multi-turn, error, sentinel)
- Total: 25 new tests, 180 total pass (179 run + 1 ignored)

## Rubber-duck critique findings (addressed)

| Finding | Severity | Resolution |
|---------|----------|------------|
| `encode()` mangles ChatML special tokens | Blocking | `encode_with_special_tokens()` |
| Per-token decode not UTF-8 safe | Blocking | `StreamingDecoder` buffering |
| Content can contain sentinels | Important | Sentinel validation in `format_chat_prompt` |
| Need `<\|im_end\|>` stop detection | Important | Resolved in example loop |
| Template detection too weak | Important | Documented as "known supported family" |
