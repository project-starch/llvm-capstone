#!/usr/bin/env bash
# Build Row 3 (GHSA-f56g-chqp-22m9 -- libpulse-binding proplist::Iterator UAF).
#
# NOTE ON THE TARGET: an earlier version of this row targeted `hlua` (a Lua-in-Rust
# binding) with a placeholder trigger. That was the wrong crate -- the task spec
# and the benchmark table both define Row 3 as the Rust->C iterator lifetime bug in
# libpulse-binding. This build targets the correct crate. See target.md.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBR_DIR="$DIR/pulse-binding-rust"
# Parent of 9e31c82 ("proplist: fix `Iterator` use-after-free"), i.e. the last
# state before the 2.5.0 security fix. Crate version 2.4.0.
VULN_COMMIT=b6b1010847c1eb2d3a533820c8ff5cdbf9993d9e

echo "=== [1/2] Sourcing and pinning libpulse-binding ==="
if [ ! -d "$PBR_DIR" ]; then
    echo "Cloning pulse-binding-rust..."
    git clone https://github.com/jnqnfe/pulse-binding-rust.git "$PBR_DIR"
fi
git -C "$PBR_DIR" fetch --quiet origin "$VULN_COMMIT" 2>/dev/null || true
# Force + reset so a re-run is idempotent.
git -C "$PBR_DIR" checkout --quiet --force "$VULN_COMMIT"
git -C "$PBR_DIR" reset --hard --quiet "$VULN_COMMIT"
echo "pinned at $(git -C "$PBR_DIR" rev-parse --short HEAD) (libpulse-binding 2.4.0)"

# Requires the system libpulse (headers + shared library) to link against; the
# defect is in the Rust binding's lifetime handling, not in libpulse itself.
if ! pkg-config --exists libpulse; then
    echo "error: libpulse not found. Install libpulse-dev / pulseaudio-libs-devel." >&2
    exit 1
fi
echo "system libpulse: $(pkg-config --modversion libpulse)"

echo "=== [2/2] Building native + ASan ==="
# ASan for Rust is nightly-only; --target keeps build scripts uninstrumented.
# -Zbuild-std is unnecessary: the freed object is a C heap allocation made by
# libpulse, and ASan's malloc/free interceptors see it regardless.
cd "$DIR"
RUSTFLAGS="-Zsanitizer=address -g" \
    cargo +nightly build --target x86_64-unknown-linux-gnu

echo "=== Build Complete ==="
