#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Running Rust-Lua hlua #144 Reproduction ==="
set +e
cargo run
EXIT_CODE=$?
set -e
echo "Run exited with: $EXIT_CODE"
