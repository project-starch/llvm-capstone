#!/usr/bin/env bash
# Fetch WAMR at a PINNED commit. Pinned by SHA rather than by tag or branch,
# because a moving upstream silently changes what a census number describes.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

WAMR_COMMIT=${WAMR_COMMIT:-f73410e2ce39e50ea542cac494044a5f1a1d8733}
WAMR_SRC_DIR=${WAMR_SRC_DIR:-$CAPSTONE_TMP_ROOT/wamr-src/wasm-micro-runtime}
WAMR_URL=${WAMR_URL:-https://github.com/wasm-micro-runtime/wasm-micro-runtime.git}

if [[ ! -d "$WAMR_SRC_DIR/.git" ]]; then
  mkdir -p "$(dirname "$WAMR_SRC_DIR")"
  git clone -q "$WAMR_URL" "$WAMR_SRC_DIR"
fi
git -C "$WAMR_SRC_DIR" fetch -q origin "$WAMR_COMMIT" 2>/dev/null || git -C "$WAMR_SRC_DIR" fetch -q origin
git -C "$WAMR_SRC_DIR" checkout -q "$WAMR_COMMIT"
git -C "$WAMR_SRC_DIR" checkout -q -- .
echo "WAMR at $WAMR_SRC_DIR ($(git -C "$WAMR_SRC_DIR" rev-parse --short HEAD))"
