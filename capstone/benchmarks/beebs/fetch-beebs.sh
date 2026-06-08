#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

BEEBS_REPO_URL=${BEEBS_REPO_URL:-https://github.com/mageec/beebs.git}
BEEBS_REF=${BEEBS_REF:-049ded9f3aeb5591f553879d3a0376b8614e9422}
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}

mkdir -p "$(dirname -- "$BEEBS_SRC_DIR")"

if [[ -d "$BEEBS_SRC_DIR/.git" ]]; then
  if [[ -n "$(git -C "$BEEBS_SRC_DIR" status --porcelain)" ]]; then
    echo "refusing to reuse dirty BEEBS checkout: $BEEBS_SRC_DIR" >&2
    echo "clean it or set BEEBS_SRC_DIR to a different path" >&2
    exit 1
  fi
else
  rm -rf "$BEEBS_SRC_DIR"
  git clone "$BEEBS_REPO_URL" "$BEEBS_SRC_DIR"
fi

git -C "$BEEBS_SRC_DIR" fetch --tags origin
git -C "$BEEBS_SRC_DIR" checkout --detach "$BEEBS_REF"

ACTUAL_COMMIT=$(git -C "$BEEBS_SRC_DIR" rev-parse HEAD)

echo "BEEBS repository : $BEEBS_REPO_URL"
echo "BEEBS source dir : $BEEBS_SRC_DIR"
echo "BEEBS ref        : $BEEBS_REF"
echo "BEEBS commit     : $ACTUAL_COMMIT"
