#!/usr/bin/env bash
# Produce the Capstone-adapted SQLite 3.22.0 amalgamation: sqlite3.c and sqlite3.h.
#
# Counterpart of the sed pass in build-sqlite-capstone.sh, which targets 3.53.3.
# Of that script's six rewrites, one applies verbatim to 3.22.0 (saveBuf alignment), three
# address code 3.22.0 does not have (the Atoi64 typo, the c_atomic guard, YYDYNSTACK), and
# the two SZ_VDBECURSOR alignment rewrites are restated here against 3.22.0's ROUND8 form of
# allocateCursor. The SQLITE_TRANSIENT sentinel and the memsys5 methods table are retired in
# both scripts; build-sqlite-capstone.sh says why.
#
# saveBuf stays even though removing it leaves this 3.22.0 image byte-identical: the array is
# `align 1` in the IR, and it is 16-aligned only where this frame layout happens to put it.
#
# The old RowSet allocation hardcodes 64 bytes, but its capability-pointer
# layout is 112 bytes here. Allocate the actual rounded struct size; otherwise
# phase 210 faults while SQLite initializes the RowSet.
#
# One addition the 3.53.3 pass does not need: sqlite3_filename. SQLite 3.41.0 introduced
# it as an alias for const char * and changed sqlite3_vfs.xOpen to take it; the shared VFS
# (tests/runtime-qemu/sqlite-vfs-skeleton/capstone_sqlite_vfs.c) is written against that
# signature. Backporting the typedef keeps the shared VFS untouched.
#
# THE HEADER GETS IT TOO. The silicon build compiles the VFS in the amalgamation's translation
# unit, so the typedef in sqlite3.c is all it sees. build-sqlite-capstone.sh (the speedtest1
# path) compiles the VFS as its own translation unit against sqlite3.h, where the typedef must
# also exist. Adapting both keeps the .c and the .h declaring the same API; a .c/.h mismatch is
# what gap 9 was.
#
# The results are written to temporary files and moved into place only after every assertion
# passes. build-sqlite-silicon.sh reuses whatever file sits at $PATCHED, so a failed run must
# not leave an empty or half-rewritten one behind.
set -euo pipefail
SRC_DIR=${1:?usage: adapt-sqlite-322.sh <amalgamation-dir> <out-dir>}
OUT_DIR=${2:?usage: adapt-sqlite-322.sh <amalgamation-dir> <out-dir>}
for f in sqlite3.c sqlite3.h; do
  [ -f "$SRC_DIR/$f" ] || { echo "adapt-sqlite-322: no such file: $SRC_DIR/$f" >&2; exit 1; }
done
# The input must be the official amalgamation. An adapted one already carries the typedef, and
# adapting it again fails the count assertion below with no message -- which is what a stray
# SQLITE_SRC_DIR pointing fetch-sqlite.sh at the adapted directory looked like. Say so instead.
if grep -q '^typedef const char \*sqlite3_filename;$' "$SRC_DIR/sqlite3.c"; then
  echo "adapt-sqlite-322: $SRC_DIR is already adapted; pass the official amalgamation" \
       "(is SQLITE_SRC_DIR still exported?)" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"
TMP_C="$OUT_DIR/sqlite3.c.tmp"
TMP_H="$OUT_DIR/sqlite3.h.tmp"
trap 'rm -f "$TMP_C" "$TMP_H"' EXIT

FILENAME_TYPEDEF='/^typedef struct sqlite3_file sqlite3_file;$/a\
typedef const char *sqlite3_filename;'

sed \
  -e 's/^  char saveBuf\[PARSE_TAIL_SZ\];/  char saveBuf[PARSE_TAIL_SZ] __attribute__((aligned(16)));/' \
  -e 's/ROUND8(sizeof(VdbeCursor)) + 2\*sizeof(u32)\*nField + /((ROUND8(sizeof(VdbeCursor)) + 2*sizeof(u32)*nField + 15)\&~15) + /' \
  -e 's/&pMem->z\[ROUND8(sizeof(VdbeCursor))+2\*sizeof(u32)\*nField\]/\&pMem->z[(ROUND8(sizeof(VdbeCursor))+2*sizeof(u32)*nField+15)\&~15]/' \
  -e 's/pMem->zMalloc = sqlite3DbMallocRawNN(db, 64);/pMem->zMalloc = sqlite3DbMallocRawNN(db, ROUND8(sizeof(RowSet)));/' \
  -e "$FILENAME_TYPEDEF" \
  "$SRC_DIR/sqlite3.c" > "$TMP_C"
sed -e "$FILENAME_TYPEDEF" "$SRC_DIR/sqlite3.h" > "$TMP_H"

# Every rewrite is asserted: a silently missed substitution is the failure mode
# the 3.53.3 script guards against, and 3.22.0 deserves the same treatment.
grep -q 'char saveBuf\[PARSE_TAIL_SZ\] __attribute__((aligned(16)));' "$TMP_C"
grep -q '((ROUND8(sizeof(VdbeCursor)) + 2\*sizeof(u32)\*nField + 15)&~15) + ' "$TMP_C"
grep -q '&pMem->z\[(ROUND8(sizeof(VdbeCursor))+2\*sizeof(u32)\*nField+15)&~15\]' "$TMP_C"
grep -q 'pMem->zMalloc = sqlite3DbMallocRawNN(db, ROUND8(sizeof(RowSet)));' "$TMP_C"
for t in "$TMP_C" "$TMP_H"; do
  [ "$(grep -c '^typedef const char \*sqlite3_filename;$' "$t")" = 1 ]
done
mv -f "$TMP_C" "$OUT_DIR/sqlite3.c"
mv -f "$TMP_H" "$OUT_DIR/sqlite3.h"
echo "adapted 3.22.0 -> $OUT_DIR ($(wc -l < "$OUT_DIR/sqlite3.c") + $(wc -l < "$OUT_DIR/sqlite3.h") lines)"
