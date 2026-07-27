#!/usr/bin/env bash
# Build Row 2 (rlua #97 -- callback arguments escape their lifetime).
#
# NOTE ON THE PIN: an earlier version of this script checked out
# `4be78cb10177`, which is NOT an rlua commit at all -- it is a commit from the
# llvm-capstone repository, pasted in by mistake. It was masked by a `|| true`,
# so the build silently used whatever rlua HEAD happened to be. The correct pin
# is derived below from the upstream fix. See target.md.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLUA_DIR="$DIR/rlua"
# Parent of 0f5a9a3 ("Fix terrible soundness issue... barely"), which is the fix
# for #97: it removes the unconstrained 'callback lifetime from create_function.
VULN_COMMIT=78021185d55c562e48a7bd76582f42c8b3326cbc

echo "=== [1/2] Sourcing and pinning rlua ==="
if [ ! -d "$RLUA_DIR" ]; then
    echo "Cloning rlua..."
    git clone https://github.com/mlua-rs/rlua.git "$RLUA_DIR"
fi
git -C "$RLUA_DIR" fetch --quiet origin "$VULN_COMMIT" 2>/dev/null || true
# Force + reset so a re-run is idempotent regardless of prior state.
git -C "$RLUA_DIR" checkout --quiet --force "$VULN_COMMIT"
git -C "$RLUA_DIR" reset --hard --quiet "$VULN_COMMIT"
echo "rlua pinned at $(git -C "$RLUA_DIR" rev-parse --short HEAD) (v0.15.4-alpha.0)"

echo "=== [2/3] Applying the modern-rustc compatibility patch ==="
# rlua_panic! expands with a trailing semicolon and is used in expression
# position, which is a hard error on current rustc (6 errors, one macro body).
# Unlike Row 1's patch this one is nowhere near the defect. See the patch header.
git -C "$RLUA_DIR" apply "$DIR/rlua-modern-rustc.patch"

echo "=== [3/3] Building native + ASan (release) ==="
# ASan for Rust is nightly-only. --target is required so build scripts and
# proc-macros are not instrumented; -Zbuild-std is unnecessary because ASan's
# interceptors catch the violation without rebuilding std.
#
# --release is deliberate. In a debug build rlua's own reference-leak assertion
# (src/lua.rs:46) panics first and muddies the output; upstream noted the same
# thing in the PoC. -g keeps symbols so the trace still names Table::len.
cd "$DIR"
RUSTFLAGS="-Zsanitizer=address -g" \
    cargo +nightly build --release --target x86_64-unknown-linux-gnu

echo "=== Build Complete ==="
