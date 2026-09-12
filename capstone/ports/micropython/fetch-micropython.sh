#!/usr/bin/env bash
# Fetch MicroPython at a pinned commit and apply the Capstone portability patches.
# Mirrors benchmarks/sqlite/fetch-sqlite.sh: the tree lives outside the repo, the
# patches live inside it.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAPSTONE_TMP_ROOT=${CAPSTONE_TMP_ROOT:-/tmp/capstone}

MPY_COMMIT=${MPY_COMMIT:-2e3304a}
MPY_URL=${MPY_URL:-https://github.com/micropython/micropython.git}
MPY_SRC_DIR=${MPY_SRC_DIR:-$CAPSTONE_TMP_ROOT/micropython}

mkdir -p "$CAPSTONE_TMP_ROOT"

if [[ ! -d "$MPY_SRC_DIR/.git" ]]; then
  git clone "$MPY_URL" "$MPY_SRC_DIR"
fi

git -C "$MPY_SRC_DIR" fetch --depth 200 origin
git -C "$MPY_SRC_DIR" checkout -q "$MPY_COMMIT"
git -C "$MPY_SRC_DIR" checkout -q -- .

for p in "$SCRIPT_DIR"/patches/*.patch; do
  name=$(basename "$p")
  if git -C "$MPY_SRC_DIR" apply --check "$p" 2>/dev/null; then
    git -C "$MPY_SRC_DIR" apply "$p"
    echo "== applied $name"
  elif git -C "$MPY_SRC_DIR" apply --reverse --check "$p" 2>/dev/null; then
    # Reverse-applies cleanly, so it is genuinely already in the tree. This is
    # the ONLY reason a patch may be skipped: "does not apply" on its own would
    # hide a patch that has gone stale against a newer pin, and a fetch that
    # silently drops a portability fix produces a build that looks fine and is
    # not.
    echo "== already present: $name"
  else
    echo "ERROR: $name neither applies nor is already present -- stale against $MPY_COMMIT?" >&2
    exit 1
  fi
done

echo "MicroPython at $MPY_SRC_DIR ($(git -C "$MPY_SRC_DIR" rev-parse --short HEAD))"
