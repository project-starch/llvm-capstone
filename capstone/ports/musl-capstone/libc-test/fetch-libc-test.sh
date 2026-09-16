#!/usr/bin/env bash
# Fetch musl's libc-test at a pinned commit, into the shared tmp root.
#
# Pinned rather than "latest" for the same reason musl itself is: a suite that
# moves under a measurement makes two runs incomparable. The canonical host,
# git.musl-libc.org, answered 502 on 2026-09-16; repo.or.cz carries the same
# history and is tried first, with the canonical host as fallback.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh"

LIBC_TEST_COMMIT=${LIBC_TEST_COMMIT:-7b95dfa}
DEST=${LIBC_TEST_DIR:-$CAPSTONE_TMP_ROOT/libc-test}

if [ ! -d "$DEST/.git" ]; then
  for u in https://repo.or.cz/libc-test.git https://git.musl-libc.org/git/libc-test; do
    git clone --quiet "$u" "$DEST" 2>/dev/null && break
  done
fi
[ -d "$DEST/.git" ] || { echo "libc-test: no mirror answered" >&2; exit 2; }
git -C "$DEST" checkout --quiet "$LIBC_TEST_COMMIT" 2>/dev/null || {
  git -C "$DEST" fetch --quiet --unshallow 2>/dev/null || true
  git -C "$DEST" checkout --quiet "$LIBC_TEST_COMMIT"
}
# The tree is reset to the pinned commit and then patched, every time, so the
# result depends on the commit and the patch set and on nothing that happened
# in the tree before. Each patch removes one flat-memory assumption from a
# test; see patches/README.md for the rule.
git -C "$DEST" checkout --quiet -- . 
for p in "$SCRIPT_DIR"/patches/*.patch; do
  [ -f "$p" ] || continue
  git -C "$DEST" apply --whitespace=nowarn "$p" || { echo "libc-test: patch failed: $(basename "$p")" >&2; exit 2; }
done
printf '%s\n' "$DEST"
