#!/usr/bin/env bash
# Build Row 7 (mruby-bigint mrb_bint_reduce GC hazard).
#
# NOTE ON THE PIN: the task spec groups this row into the "single mruby 3.1.0
# build" Tier-1 cluster. That is not possible -- mrb_bint_reduce() does not exist
# in 3.1.0, 3.2.0 or 3.3.0. It first appears in the 3.4.0 line. This row is
# therefore pinned to the same 3.4.0-era commit as Row 6. See target.md.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MRUBY_DIR="$DIR/mruby"
VULN_COMMIT=cda2567c36ca33cd404908ce2fa7bd55ea2a8ed9

echo "=== [1/2] Sourcing and pinning mruby ==="
if [ ! -d "$MRUBY_DIR" ]; then
    echo "Cloning mruby upstream..."
    git clone https://github.com/mruby/mruby.git "$MRUBY_DIR"
fi
git -C "$MRUBY_DIR" fetch --quiet origin "$VULN_COMMIT" 2>/dev/null || true
# Force + reset so a re-run over a dirty tree is idempotent.
git -C "$MRUBY_DIR" checkout --quiet --force "$VULN_COMMIT"
git -C "$MRUBY_DIR" reset --hard --quiet "$VULN_COMMIT"

echo "=== [2/2] Building mruby (host+ASan and riscv64), with bigint + rational ==="
cd "$MRUBY_DIR"
rake clean || true
MRUBY_CONFIG="$DIR/build_config.rb" rake

echo "=== Build Complete ==="
