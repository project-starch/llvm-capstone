#!/usr/bin/env bash
# Fetch the SQLLogicTest corpus subset, pinned and verified. Prints the test directory.
#
# NOT VENDORED, same policy as fetch-sqlite.sh and fetch-musl.sh: the corpus is 1.1 GB of
# third-party data and belongs in the cache, not in this repo.
#
# PINNED BY COMMIT **AND** BY PER-FILE SHA-256, deliberately. A GitHub branch tarball is
# not content-addressed -- the archive bytes for the same commit can change when GitHub
# changes its compression, so an archive checksum would fail spuriously while a bare
# branch name would silently drift the corpus under a published pass rate. The per-file
# hashes below are what the measured baseline was taken against; if one mismatches, the
# number in the docs no longer describes the file being run and the script stops.
#
# sqlite.org's own tarball is behind a login (302 to /login), which is why this uses the
# GitHub mirror that carries the same canonical tree.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

SLT_COMMIT=${SLT_COMMIT:-c67f97bf3ca7e590d12e073408bcacaf2ff0f3a0}   # 2026-04-15
SLT_CACHE_ROOT=${SLT_CACHE_ROOT:-$CAPSTONE_TMP_ROOT/slt-src}
SLT_ARCHIVE="$SLT_CACHE_ROOT/sqllogictest-$SLT_COMMIT.tar.gz"
SLT_DIR="$SLT_CACHE_ROOT/sqllogictest-$SLT_COMMIT"
SLT_URL=${SLT_URL:-https://github.com/gregrahn/sqllogictest/archive/$SLT_COMMIT.tar.gz}

mkdir -p "$SLT_CACHE_ROOT"

if [[ ! -d "$SLT_DIR/test" ]]; then
  [[ -f "$SLT_ARCHIVE" ]] || curl -fsSL "$SLT_URL" -o "$SLT_ARCHIVE"
  rm -rf "$SLT_DIR.tmp"
  mkdir -p "$SLT_DIR.tmp"
  tar xzf "$SLT_ARCHIVE" -C "$SLT_DIR.tmp" --strip-components=1
  mv "$SLT_DIR.tmp" "$SLT_DIR"
fi

# THE SUBSET AND ITS HASHES. This is the exact set the recorded pass rate refers to.
# select1-5 are the canonical select suite; the aggfunc evidence file is here because it
# is the ONLY file in all 622 that uses the R (real) type, and without it the runner's
# %.3f rendering and its (empty)/NULL text rules are never exercised at all -- measured
# by mutating each rule and watching select1-5 stay green.
cd "$SLT_DIR/test"
sha256sum -c --quiet <<'SUMS'
e93b83d64d06f78aee0e690455b6c604e86ad9a339f77d927a782cefb6b0e1d5  select1.test
a8ecc3d206c4d4b2cd6a154c18999e558ec97168cd7e327a4369e23aaf31be64  select2.test
d5c321683bfe903c9be95ebe80a8d23da4d8f4a39193b2087dbc34cb4e137623  select3.test
155ff6bb9bbf7c2dcf1e5659bb1688dec5dab58126f8dc66d23dcae6df43f59e  select4.test
049a5d0bf90999c56db2d5880ef28febdc88906526f87069cc96af9f84c99869  select5.test
2a9d602e33c76766cba21d2f9a0e6ffda286f76445950ab187b544276c37abc9  evidence/slt_lang_aggfunc.test
SUMS

printf '%s\n' "$SLT_DIR/test"
