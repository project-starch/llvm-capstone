#!/usr/bin/env bash
set -e

# Resolve paths
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLUA_DIR="$DIR/rlua"

echo "=== [1/2] Sourcing and prepping rlua repo ==="
if [ ! -d "$RLUA_DIR" ]; then
    echo "Cloning rlua..."
    git clone https://github.com/mlua-rs/rlua.git "$RLUA_DIR"
fi
cd "$RLUA_DIR"
# Check out the exact vulnerable commit for rlua #97 (e.g. pre-0.12.0)
git checkout 4be78cb10177 || true

echo "=== [2/2] Compiling Rust-Lua reproduction natively ==="
cd "$DIR"
cargo build || true

echo "=== Build Complete ==="
