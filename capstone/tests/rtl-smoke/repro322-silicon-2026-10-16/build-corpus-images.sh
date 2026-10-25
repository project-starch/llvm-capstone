#!/usr/bin/env bash
# The engine-repros cells built from the CORPUS's own case.c (bug-corpora/sqlite/engine-repros/<case>/case.c),
# with ../repro322-silicon-2026-10-01/build-images.sh's recipe unchanged otherwise: the same silicon build, the
# same group defines, the same entry VAs, and the adapted input at $CAPSTONE_TMP_ROOT/sqlite-322-adapted.
#
# Why: build-images.sh compiles the cells from ports/sqlite. Seven of those files are byte-identical to their
# corpus case.c; blobclose is not (ports/sqlite/sqlite_blobclose_domain.c is the legacy standalone domain, and the
# corpus's 03 case.c was ported onto REPRO322_MAIN on 2026-10-05). A cell folds into a corpus case only if it was
# built from that case's case.c.
#
# REVIEW02_SRC=<case.c>, optional: one more cell, `backupattachrev`, from the external collaborator's review of
# case 02 (branch sqlite-review). It is built like 02 and placed at its own VA.
# The corpus's two controls (controls/control_uaf_mem5.c, control_bounds_mem5.c), which the stock configuration
# requires before a silent case reads MISSED, are built like the cells, in the core group, at their own VAs.
# CELLS="<tag> ...", optional: build only those cells.
#
#   CAPSTONE_LLVM_BIN=<toolchain bin> CAPSTONE_TMP_ROOT=<root holding sqlite-322-adapted> bash build-corpus-images.sh <out-dir>
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
P=$REPO/capstone/ports/sqlite
C=$REPO/capstone/bug-corpora/sqlite/engine-repros
OUT=${1:?usage: build-corpus-images.sh <out-dir>}
: "${CAPSTONE_LLVM_BIN:?set CAPSTONE_LLVM_BIN to the bin directory of the toolchain}"
: "${CAPSTONE_TMP_ROOT:?set CAPSTONE_TMP_ROOT to the root that holds sqlite-322-adapted}"
ADAPTED=$CAPSTONE_TMP_ROOT/sqlite-322-adapted
[[ -f "$ADAPTED/sqlite3.c" ]] || { echo "no adapted input at $ADAPTED/sqlite3.c" >&2; exit 1; }
echo "== adapted input: $(sha256sum "$ADAPTED/sqlite3.c" | cut -c1-16)"
source "$REPO/capstone/tests/capstone-test-env.sh"
echo "== compiler: $("$CAPSTONE_CLANG" --version | head -1)"

CORE="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_PREUPDATE_HOOK -DSQLITE_COUNTOFVIEW_OPTIMIZATION"
CORET="-USQLITE_OMIT_INCRBLOB -USQLITE_OMIT_TEMPDB"
JSON="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_JSON1"

one() {  # va tag source group-defines...
  local va=$1 tag=$2 src=$3; shift 3
  [[ -z "${CELLS:-}" || " $CELLS " == *" $tag "* ]] || return 0
  [[ -f "$src" ]] || { echo "no source $src" >&2; exit 1; }
  mkdir -p "$OUT/$tag"
  echo "== $tag @ $va <- ${src#$REPO/} ($(sha256sum "$src" | cut -c1-16))"
  DOMAIN_BASE_VA=$va PATCHED_SQLITE="$ADAPTED/sqlite3.c" OUT_DIR="$OUT/$tag" \
  DOMAIN_SRC="$src" DOMAIN_EXTRA_DEFS="-I$P/repro322 $*" \
    bash "$P/build-sqlite-silicon.sh" > "$OUT/$tag/build.log" 2>&1
  cp -f "$OUT/$tag/sqlite_silicon.dom" "$OUT/$tag/r322-$tag.dom"
  echo "   $(sha256sum "$OUT/$tag/r322-$tag.dom" | cut -c1-16)"
}
case_c() { local d; d=$(ls -d "$C/$1"_*/ | head -1); echo "${d%/}/case.c"; }
one 0x810000  wschema      "$(case_c 24)" $CORE
one 0xc10000  blobwrite    "$(case_c 04)" $CORE
one 0x1010000 mem5design   "$(case_c 18)" $CORE
one 0x1410000 blobclose03  "$(case_c 03)" $CORE
one 0x1810000 jsonstatic   "$(case_c 17)" $JSON
one 0x1c10000 jsonroot     "$(case_c 16)" $JSON
one 0x2010000 backupattach "$(case_c 02)" $CORE
one 0x2410000 detachtrig   "$(case_c 05)" $CORET
one 0x2810000 ctluafmem5    "$C/controls/control_uaf_mem5.c"    $CORE
one 0x3010000 ctlboundsmem5 "$C/controls/control_bounds_mem5.c" $CORE
[[ -n "${REVIEW02_SRC:-}" ]] && one 0x2c10000 backupattachrev "$REVIEW02_SRC" $CORE
exit 0
