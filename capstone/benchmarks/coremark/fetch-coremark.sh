#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

COREMARK_REPO_URL=${COREMARK_REPO_URL:-https://github.com/eembc/coremark.git}
COREMARK_SITE_URL=${COREMARK_SITE_URL:-https://www.eembc.org/coremark/}
COREMARK_REF=${COREMARK_REF:-v1.01}
COREMARK_PINNED_COMMIT=${COREMARK_PINNED_COMMIT:-cfa9ab377835911f23d9b0831c7be302ed1f58de}
COREMARK_SRC_DIR=${COREMARK_SRC_DIR:-$CAPSTONE_TMP_ROOT/coremark-src}

mkdir -p "$(dirname -- "$COREMARK_SRC_DIR")"

if [[ -d "$COREMARK_SRC_DIR/.git" ]]; then
  if [[ -n "$(git -C "$COREMARK_SRC_DIR" status --porcelain)" ]]; then
    echo "refusing to reuse dirty CoreMark checkout: $COREMARK_SRC_DIR" >&2
    echo "clean it or set COREMARK_SRC_DIR to a different path" >&2
    exit 1
  fi
else
  rm -rf "$COREMARK_SRC_DIR"
  git clone "$COREMARK_REPO_URL" "$COREMARK_SRC_DIR"
fi

git -C "$COREMARK_SRC_DIR" fetch --tags origin

if git -C "$COREMARK_SRC_DIR" rev-parse -q --verify "refs/tags/$COREMARK_REF" >/dev/null; then
  git -C "$COREMARK_SRC_DIR" checkout --detach "$COREMARK_REF"
else
  git -C "$COREMARK_SRC_DIR" checkout --detach "$COREMARK_PINNED_COMMIT"
fi

ACTUAL_COMMIT=$(git -C "$COREMARK_SRC_DIR" rev-parse HEAD)

echo "CoreMark repository : $COREMARK_REPO_URL"
echo "CoreMark site       : $COREMARK_SITE_URL"
echo "CoreMark source dir : $COREMARK_SRC_DIR"
echo "CoreMark ref        : $COREMARK_REF"
echo "CoreMark commit     : $ACTUAL_COMMIT"

