#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

RV8_REPO_URL=${RV8_REPO_URL:-https://github.com/michaeljclark/rv8-bench.git}
RV8_SITE_URL=${RV8_SITE_URL:-https://michaeljclark.github.io/bench}
RV8_PINNED_COMMIT=${RV8_PINNED_COMMIT:-d61b32f9e04bf04ce6dea88607ab93497fad95af}
RV8_SRC_DIR=${RV8_SRC_DIR:-$CAPSTONE_TMP_ROOT/rv8-src}

mkdir -p "$(dirname -- "$RV8_SRC_DIR")"

if [[ -d "$RV8_SRC_DIR/.git" ]]; then
  if [[ -n "$(git -C "$RV8_SRC_DIR" status --porcelain)" ]]; then
    echo "refusing to reuse dirty rv8-bench checkout: $RV8_SRC_DIR" >&2
    echo "clean it or set RV8_SRC_DIR to a different path" >&2
    exit 1
  fi
else
  rm -rf "$RV8_SRC_DIR"
  git clone "$RV8_REPO_URL" "$RV8_SRC_DIR"
fi

git -C "$RV8_SRC_DIR" fetch origin
if [[ -n "$RV8_PINNED_COMMIT" ]]; then
  git -C "$RV8_SRC_DIR" checkout --detach "$RV8_PINNED_COMMIT"
fi

ACTUAL_COMMIT=$(git -C "$RV8_SRC_DIR" rev-parse HEAD)
echo "rv8-bench repository : $RV8_REPO_URL"
echo "rv8-bench site       : $RV8_SITE_URL"
echo "rv8-bench source dir : $RV8_SRC_DIR"
echo "rv8-bench commit     : $ACTUAL_COMMIT"
