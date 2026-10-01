#!/usr/bin/env bash
# Build the batch-1 images of the repro322 silicon campaign: each case of the SQLite 3.22.0
# temporal-bug corpus (ports/sqlite/repro322, PR #171/#174) rebuilt in the SILICON config by
# build-sqlite-silicon.sh -- gp-captable, one translation unit, interp glue -- at its own entry VA.
#
# The corpus's own build (build-sqlite-row322.sh) links with start.S + link.ld, which on the
# emulator relies on a fabricated gp; these images are what the board can run. Same cases, same
# 3.22.0 adapted amalgamation, the corpus group's feature defines passed through DOMAIN_EXTRA_DEFS
# (appended after the silicon build's own SQLite defines and trims, so its -U/-D win).
#
#   CAPSTONE_LLVM_BIN=<toolchain bin> bash build-images.sh <out-dir>
#
# Writes <out-dir>/<tag>/r322-<tag>.dom; compare against SHA256SUMS here.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
P=$REPO/capstone/ports/sqlite
OUT=${1:?usage: build-images.sh <out-dir>}
: "${CAPSTONE_LLVM_BIN:?set CAPSTONE_LLVM_BIN to the bin directory of the toolchain}"
export CAPSTONE_TMP_ROOT=${CAPSTONE_TMP_ROOT:-$OUT/tmp}
source "$REPO/capstone/tests/capstone-test-env.sh"
echo "== compiler: $("$CAPSTONE_CLANG" --version | head -1)"

export SQLITE_VERSION=3220000 SQLITE_YEAR=2018
export SQLITE_ARCHIVE_SHA3=69bc5ee8f08d747494dd3a4bfe075e5b078fe200dfc671d76dd9e1ccb5b2decb
SRC=$(bash "$P/fetch-sqlite.sh")
ADAPTED=$CAPSTONE_TMP_ROOT/sqlite-322-adapted
[[ -f "$ADAPTED/sqlite3.c" ]] || bash "$P/adapt-sqlite-322.sh" "$SRC" "$ADAPTED"

# The corpus groups' defines, as corpus322.sh gives them.
CORE="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_PREUPDATE_HOOK -DSQLITE_COUNTOFVIEW_OPTIMIZATION"
CORET="-USQLITE_OMIT_INCRBLOB -USQLITE_OMIT_TEMPDB"
JSON="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_JSON1"

one() {  # va tag source group-defines...
  local va=$1 tag=$2 src=$3; shift 3
  mkdir -p "$OUT/$tag"
  echo "== $tag @ $va <- ${src#$REPO/}"
  DOMAIN_BASE_VA=$va PATCHED_SQLITE="$ADAPTED/sqlite3.c" OUT_DIR="$OUT/$tag" \
  DOMAIN_SRC="$src" DOMAIN_EXTRA_DEFS="-I$P/repro322 $*" \
    bash "$P/build-sqlite-silicon.sh" > "$OUT/$tag/build.log" 2>&1
  cp -f "$OUT/$tag/sqlite_silicon.dom" "$OUT/$tag/r322-$tag.dom"
  echo "   $(sha256sum "$OUT/$tag/r322-$tag.dom" | cut -c1-16)"
}
# 4 MiB apart, none at k800's 0x10000 (R-3 / preflight C15).
one 0x410000  base322      "$P/sqlite_capstone_domain.c"
one 0x810000  wschema      "$P/repro322/case_writable_schema.c"  $CORE
one 0xc10000  blobwrite    "$P/repro322/case_blobwrite.c"        $CORE
one 0x1010000 mem5design   "$P/repro322/case_mem5design.c"       $CORE
one 0x1410000 blobclose    "$P/sqlite_blobclose_domain.c"        $CORE
one 0x1810000 jsonstatic   "$P/repro322/case_json_each_static.c" $JSON
one 0x1c10000 jsonroot     "$P/repro322/case_json_each_root.c"   $JSON
one 0x2010000 backupattach "$P/repro322/case_backupattach.c"     $CORE
one 0x2410000 detachtrig   "$P/repro322/case_detach_trigger.c"   $CORET
