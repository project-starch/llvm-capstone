#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Compiling Rust-Lua reproduction natively ==="
cargo build || true

echo "=== Build Complete ==="
