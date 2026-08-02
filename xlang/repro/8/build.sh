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
# Check out the exact vulnerable commit for mruby #4926 (Row 8)
git fetch origin 70e574689664c10ed2c47581999cc2ce3e3c5afb
git checkout fc8fb41451b07b3fda0726ba80e88e509ad02452

echo "=== [2/2] Building mruby with unified config (host-asan + riscv64) ==="
rake clean || true
MRUBY_CONFIG="$DIR/build_config.rb" rake

echo "=== Build Complete ==="
