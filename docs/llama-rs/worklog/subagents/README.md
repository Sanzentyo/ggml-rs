# subagent worklog policy

Each parallel subagent writes progress to its own file to avoid merge conflicts.

- `foundation.md`
- `gpt2.md`
- `gptj_magika.md`
- `vision_mnist.md`

Lock protocol for heavy commands:

- cargo/build/test: `cargo +nightly -Zscript scripts/agent_lock.rs -- cargo <cmd ...>`
- C++/cmake compile: `cargo +nightly -Zscript scripts/agent_lock.rs -- cpp <cmd ...>`
- benchmark/parity runtime: `cargo +nightly -Zscript scripts/agent_lock.rs -- bench <cmd ...>`

If Python is needed, use `uv` (`uv run ...`) and keep scripts self-contained.
