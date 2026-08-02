#!/usr/bin/env bash
set -e

# Resolve paths
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MRUBY_DIR="$DIR/mruby"

echo "=== [1/2] Sourcing and prepping mruby repo ==="
if [ ! -d "$MRUBY_DIR" ]; then
    echo "Cloning mruby upstream..."
    git clone https://github.com/mruby/mruby.git "$MRUBY_DIR"
fi

cd "$MRUBY_DIR"
# Check out the exact vulnerable commit for CVE-2022-1934
git fetch origin aa7f98dedb68d735a1665d3a289036c88b0c47ce
git checkout af5acf3566d5732871b1dcb007aee4f474369d93

echo "=== [2/2] Building mruby with unified config (host-asan + riscv64) ==="
rake clean || true
MRUBY_CONFIG="$DIR/build_config.rb" rake

echo "=== Build Complete ==="
