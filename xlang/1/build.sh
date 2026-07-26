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
# Check out the exact vulnerable commit for rlua #19 (vulnerable parent of 36134e6)
git checkout 396a4b09169be429381d5db8e9bb1cd6bd5d5139

echo "=== [2/2] Compiling Rust-Lua reproduction natively ==="
cd "$DIR"
cargo build

echo "=== Build Complete ==="
