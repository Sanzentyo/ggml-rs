# Subagent Mandatory Rules (read first)

This memo is the required preflight for every subagent run in this repository.

## 1) Skill policy (mandatory)

### Rust work

- Follow Rust best practices strictly:
  - idiomatic APIs and types,
  - avoid unnecessary procedural style when iterator-based style is clearer,
  - avoid broad `#[allow(...)]`,
  - avoid unstable features unless explicitly approved,
  - avoid unnecessary `unsafe`, `Box::leak`, `forget`.
- Prefer newtypes/typed config and explicit error propagation.

### Python work

- Python is `uv`-only.
- Use `uv run ...` for execution.
- Use `uv add ...` for dependencies.
- For standalone scripts, include inline uv script metadata.

## 2) Lock protocol (mandatory)

- All heavy commands must use lock wrappers:
  - cargo/build/test: `cargo +nightly -Zscript scripts/agent_lock.rs -- cargo ...`
  - c++/cmake build: `cargo +nightly -Zscript scripts/agent_lock.rs -- cpp ...`
  - runtime bench/parity: `cargo +nightly -Zscript scripts/agent_lock.rs -- bench ...`
- Never run heavy compile/bench commands without lock.

## 3) CLI policy for examples

- Use typed `clap` derive structs for argument parsing.
- Avoid ad-hoc manual argument state machines.
- Keep argument validation explicit with typed errors (`thiserror`).

## 4) Tensor constructor policy

- Do not use legacy shape/len wrapper constructors.
- Use generic constructor style (`new_tensor::<N>(...)` and generic typed helpers) for normal usage.

## 5) Docs update policy

- Each subagent must append progress incrementally (not only final):
  - `docs/llama-rs/worklog/subagents/<batch>.md`
- Record:
  - inspected files,
  - implementation decisions,
  - validation commands,
  - artifact paths,
  - blockers and missing assets.

## 6) Parity run policy

- Run both C++ and Rust counterparts when possible.
- Always report:
  - output/checksum deltas,
  - performance ratio (Rust/C++),
  - exact blocker if real-assets are missing.
