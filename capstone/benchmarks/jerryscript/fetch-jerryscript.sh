#!/usr/bin/env bash
# Fetch JerryScript at a pinned commit and apply the Capstone portability patches.
# Same shape as the MicroPython port's fetch script, deliberately: two ports that
# differ in how they are set up are two ports nobody can compare.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

JS_COMMIT=${JS_COMMIT:-b706935}
JS_SRC_DIR=${JS_SRC_DIR:-$CAPSTONE_TMP_ROOT/jerryscript}

if [[ ! -d $JS_SRC_DIR/.git ]]; then
  git clone --quiet https://github.com/jerryscript-project/jerryscript.git "$JS_SRC_DIR"
fi
git -C "$JS_SRC_DIR" fetch --depth 200 origin
git -C "$JS_SRC_DIR" checkout -q "$JS_COMMIT"
git -C "$JS_SRC_DIR" checkout -q -- .

for p in "$SCRIPT_DIR"/patches/*.patch; do
  [[ -e $p ]] || continue
  name=$(basename "$p")
  if git -C "$JS_SRC_DIR" apply --check "$p" 2>/dev/null; then
    git -C "$JS_SRC_DIR" apply "$p"
    echo "== applied $name"
  elif git -C "$JS_SRC_DIR" apply --reverse --check "$p" 2>/dev/null; then
    # Reverse-applies cleanly, so it is genuinely already in the tree. That is the
    # ONLY reason a patch may be skipped: "does not apply" on its own would hide a
    # patch gone stale against a newer pin, and a fetch that silently drops one
    # produces a build that looks fine and is not.
    echo "== already present, skipped $name"
  else
    echo "!! $name does NOT apply and is NOT present -- refusing to continue" >&2
    exit 1
  fi
done
echo "== jerryscript at $(git -C "$JS_SRC_DIR" rev-parse --short HEAD)"
