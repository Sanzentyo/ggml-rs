#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec cargo +nightly -Zscript "$SCRIPT_DIR/agent_lock.rs" -- "$@"
