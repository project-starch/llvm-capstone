#!/usr/bin/env bash
# Clone mruby at a pinned commit and apply the capability patches.
#
#     bash fetch-mruby.sh            # clone/checkout the pin, patch, print the path
#     MRUBY_COMMIT=<sha> bash ...    # move the pin deliberately
#
# The pin matters more here than for a normal dependency. Every specimen in
# ref/blindspot-cases/mruby.md names an mruby version, and a bug fixed upstream is
# not a specimen -- it is a control. Bumping this without re-reading that file
# silently turns half the corpus into controls.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

MRUBY_REPO=${MRUBY_REPO:-https://github.com/mruby/mruby.git}
MRUBY_COMMIT=${MRUBY_COMMIT:-fdb2cca}
SRC=${MRUBY_SRC_DIR:-$CAPSTONE_TMP_ROOT/mruby-src}

if [[ ! -d "$SRC/.git" ]]; then
  echo "== cloning mruby into $SRC"
  git clone -q "$MRUBY_REPO" "$SRC"
fi

git -C "$SRC" fetch -q origin "$MRUBY_COMMIT" 2>/dev/null || git -C "$SRC" fetch -q origin
git -C "$SRC" checkout -q "$MRUBY_COMMIT"
git -C "$SRC" checkout -q -- .

for pf in "$SCRIPT_DIR"/patches/*.patch; do
  [[ -e "$pf" ]] || continue
  name=$(basename "$pf")
  if git -C "$SRC" apply --check "$pf" 2>/dev/null; then
    git -C "$SRC" apply "$pf"; echo "== applied $name"
  elif git -C "$SRC" apply --reverse --check "$pf" 2>/dev/null; then
    echo "== already present: $name"
  else
    # "does not apply" alone would hide a patch gone stale against a newer pin,
    # and a fetch that silently drops a capability fix produces a build that looks
    # fine and is not -- which on this target means a cleared tag rather than a
    # compile error.
    echo "ERROR: $name neither applies nor is already present -- stale against $MRUBY_COMMIT?" >&2
    exit 1
  fi
done

echo "mruby at $SRC ($(git -C "$SRC" rev-parse --short HEAD))"
