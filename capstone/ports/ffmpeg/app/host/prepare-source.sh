#!/usr/bin/env bash
# Fetch the pinned FFmpeg tarball, verify it against upstream.json, extract it and apply
# this port's patches at fuzz 0. Prints the prepared source directory on its last line.
#
#   prepare-source.sh             patched tree (what the domain and ffapp_native build)
#   prepare-source.sh --pristine  the verified tarball, NO patches: the stock reference
#                                 ffmpeg is built from this, so a wrong patch cannot hide by
#                                 appearing on both sides of the comparison
#   prepare-source.sh --pool      the patched tree PLUS FFmpeg's pools under the buffer-pool
#                                 port's lifetime hooks, for the pool arms of the DOMAIN build
#                                 only: this port's patches, then buffer-pool's 0001 and 0002
#                                 unchanged, then patches-pool/ (the app's rework), plus
#                                 buffer-pool's trace.h and this port's no-op recorder and
#                                 observer (src/pool/) copied into libavutil/. Keyed by all of
#                                 them. The native builds never read this tree.
#
# The tree is keyed by the patch set's hash, so editing a patch produces a fresh tree
# instead of silently reusing one prepared from the old patches.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APP_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$APP_DIR/../../../tests/capstone-test-env.sh"

WORK=${FFAPP_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-app}
mkdir -p "$WORK"

read -r URL SHA VERSION < <(python3 -c '
import json,sys; u=json.load(open(sys.argv[1])); print(u["url"], u["sha256"], u["version"])' "$APP_DIR/upstream.json")
TARBALL="$WORK/ffmpeg-$VERSION.tar.xz"

if [ ! -f "$TARBALL" ]; then
  # --retry: a transient reset from the mirror (seen 2026-09-23, curl rc 56) must not fail a build.
  curl -sSfL --retry 5 --retry-all-errors --retry-delay 3 -o "$TARBALL.part" "$URL"
  mv "$TARBALL.part" "$TARBALL"
fi
echo "$SHA  $TARBALL" | sha256sum -c --quiet - \
  || { echo "prepare-source: $TARBALL does not match upstream.json; refusing to build from it" >&2; exit 1; }

if [ "${1:-}" = --pristine ]; then
  SRC="$WORK/src-$VERSION-pristine"
  if [ ! -f "$SRC/.ffapp-prepared" ]; then
    rm -rf "$SRC" "$SRC.tmp"; mkdir -p "$SRC.tmp"
    tar xf "$TARBALL" -C "$SRC.tmp" --strip-components=1
    touch "$SRC.tmp/.ffapp-prepared"; mv "$SRC.tmp" "$SRC"
  fi
  echo "$SRC"; exit 0
fi

if [ "${1:-}" = --pool ]; then
  BP="$APP_DIR/../buffer-pool"
  BP_PATCHES=("$BP"/patches/ffmpeg-9.0.1-0001-*.patch "$BP"/patches/ffmpeg-9.0.1-0002-*.patch)
  POOL_FILES=("$APP_DIR"/patches/*.patch "${BP_PATCHES[@]}" "$APP_DIR"/patches-pool/*.patch
              "$BP/src/shared/trace.h" "$APP_DIR/src/pool/ff2_record.c" "$APP_DIR/src/pool/ff2_observe.c")
  POOL_KEY=$(cat "${POOL_FILES[@]}" | sha256sum | cut -c1-12)
  SRC="$WORK/src-$VERSION-pool-$POOL_KEY"
  if [ ! -f "$SRC/.ffapp-prepared" ]; then
    rm -rf "$SRC" "$SRC.tmp"; mkdir -p "$SRC.tmp"
    tar xf "$TARBALL" -C "$SRC.tmp" --strip-components=1
    for p in "$APP_DIR"/patches/*.patch "${BP_PATCHES[@]}" "$APP_DIR"/patches-pool/*.patch; do
      patch -d "$SRC.tmp" -p1 --batch --forward --fuzz=0 < "$p" >/dev/null \
        || { echo "prepare-source: $(basename "$p") does not apply at fuzz 0" >&2; exit 1; }
    done
    cp "$BP/src/shared/trace.h" "$SRC.tmp/libavutil/trace.h"
    cp "$APP_DIR/src/pool/ff2_record.c" "$SRC.tmp/libavutil/ff2_record.c"
    cp "$APP_DIR/src/pool/ff2_observe.c" "$SRC.tmp/libavutil/ff2_observe.c"
    touch "$SRC.tmp/.ffapp-prepared"; mv "$SRC.tmp" "$SRC"
  fi
  echo "$SRC"; exit 0
fi

PATCH_KEY=$(cat "$APP_DIR"/patches/*.patch | sha256sum | cut -c1-12)
SRC="$WORK/src-$VERSION-$PATCH_KEY"
if [ ! -f "$SRC/.ffapp-prepared" ]; then
  rm -rf "$SRC" "$SRC.tmp"
  mkdir -p "$SRC.tmp"
  tar xf "$TARBALL" -C "$SRC.tmp" --strip-components=1
  for p in "$APP_DIR"/patches/*.patch; do
    patch -d "$SRC.tmp" -p1 --batch --forward --fuzz=0 < "$p" >/dev/null \
      || { echo "prepare-source: $(basename "$p") does not apply at fuzz 0" >&2; exit 1; }
  done
  touch "$SRC.tmp/.ffapp-prepared"
  mv "$SRC.tmp" "$SRC"
fi
echo "$SRC"
