#!/usr/bin/env bash
# Build Row 1 (rlua #19 -- __gc resurrection UAF across the Lua/Rust boundary).
#
# Produces one binary: a native x86_64 build with AddressSanitizer.
# See README.md "RISC-V QEMU" for why there is no riscv64 target here.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLUA_DIR="$DIR/rlua"
RLUA_COMMIT=396a4b09169be429381d5db8e9bb1cd6bd5d5139

echo "=== [1/3] Sourcing and pinning rlua ==="
# The vulnerable rlua predates the move to the mlua-rs org but its history was
# carried over, so this SHA resolves in the current upstream repository.
if [ ! -d "$RLUA_DIR" ]; then
    echo "Cloning rlua..."
    git clone https://github.com/mlua-rs/rlua.git "$RLUA_DIR"
fi
git -C "$RLUA_DIR" fetch --quiet origin "$RLUA_COMMIT" 2>/dev/null || true
# Hard reset rather than a plain checkout so re-running over an already-patched
# tree is idempotent instead of failing on local modifications.
git -C "$RLUA_DIR" checkout --quiet --force "$RLUA_COMMIT"
git -C "$RLUA_DIR" reset --hard --quiet "$RLUA_COMMIT"

echo "=== [2/3] Applying the modern-rustc compatibility patch ==="
# rlua at this commit aborts on startup under rustc >=1.48 because of
# mem::uninitialized(). The patch swaps both sites for std::ptr::read, which
# preserves the double-drop under test. Read the patch header before changing it.
git -C "$RLUA_DIR" apply "$DIR/rlua-modern-rustc.patch"

echo "=== [3/3] Building native + ASan ==="
# ASan for Rust is nightly-only (-Zsanitizer=address). An explicit --target is
# required: without it cargo would also instrument build scripts and proc-macros,
# which run inside the host compiler and would fail to load.
#
# -Zbuild-std is deliberately NOT used. std stays uninstrumented, which is fine
# here: the object under test is a heap allocation, and ASan's malloc/free
# interceptors catch it regardless of whether std itself was rebuilt.
cd "$DIR"
RUSTFLAGS="-Zsanitizer=address" \
    cargo +nightly build --target x86_64-unknown-linux-gnu

echo "=== Build Complete ==="
