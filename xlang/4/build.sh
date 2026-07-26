#!/usr/bin/env bash
# Automated Build and Setup Script for Row 4 (mruby CVE-2022-1071 OP_GETCONST UAF)
#
# This script handles the retrieval, checkout, and compilation of the exact vulnerable
# version of the mruby core interpreter.
set -e

# Resolve absolute paths for the build workspace
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MRUBY_DIR="$DIR/mruby"

echo "=== [1/2] Sourcing and prepping mruby repo ==="
# Step 1: Obtain the vulnerable core mruby interpreter.
# We clone the official mruby core and checkout commit b4168c9b (from early 2022),
# which contains the vulnerable evaluation-order implementation of the OP_GETCONST instruction.
if [ ! -d "$MRUBY_DIR" ]; then
    echo "Cloning mruby upstream..."
    git clone https://github.com/mruby/mruby.git "$MRUBY_DIR"
fi

cd "$MRUBY_DIR"
git fetch origin aaa28a508903041dd7399d4159a8ace9766b022f
git checkout b4168c9b68daf759ce890c7da9e4ad5547058330

echo "=== [2/2] Building mruby with unified config (host + riscv64) ==="
# Step 2: Clean old compilation artifacts and run the compiler.
# We direct Rake (mruby's build engine) to use our target profile in build_config.rb,
# which compiles a native GCC + ASan target (for verification) and a RISC-V target.
rake clean || true
MRUBY_CONFIG="$DIR/build_config.rb" rake

echo "=== Build Complete ==="
