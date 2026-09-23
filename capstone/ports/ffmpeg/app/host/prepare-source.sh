#!/usr/bin/env bash
# Fetch the pinned FFmpeg tarball, verify it against upstream.json, extract it and apply
# this port's patches at fuzz 0. Prints the prepared source directory on its last line.
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
