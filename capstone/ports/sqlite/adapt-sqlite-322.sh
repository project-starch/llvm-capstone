#!/usr/bin/env bash
# Produce the Capstone-adapted SQLite 3.22.0 amalgamation.
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
# One addition the 3.53.3 pass does not need: sqlite3_filename. SQLite 3.41.0 introduced
# it as an alias for const char * and changed sqlite3_vfs.xOpen to take it; the shared VFS
# (tests/runtime-qemu/sqlite-vfs-skeleton/capstone_sqlite_vfs.c) is written against that
# signature. Backporting the typedef keeps the shared VFS untouched.
#
# The result is written to a temporary file and moved into place only after every assertion
# passes. build-sqlite-silicon.sh reuses whatever file sits at $PATCHED, so a failed run must
# not leave an empty or half-rewritten one behind.
set -euo pipefail
SRC=${1:?usage: adapt-sqlite-322.sh <sqlite3.c> <out.c>}
OUT=${2:?usage: adapt-sqlite-322.sh <sqlite3.c> <out.c>}
[ -f "$SRC" ] || { echo "adapt-sqlite-322: no such file: $SRC" >&2; exit 1; }
TMP="$OUT.tmp"
trap 'rm -f "$TMP"' EXIT

sed \
  -e 's/^  char saveBuf\[PARSE_TAIL_SZ\];/  char saveBuf[PARSE_TAIL_SZ] __attribute__((aligned(16)));/' \
  -e 's/ROUND8(sizeof(VdbeCursor)) + 2\*sizeof(u32)\*nField + /((ROUND8(sizeof(VdbeCursor)) + 2*sizeof(u32)*nField + 15)\&~15) + /' \
  -e 's/&pMem->z\[ROUND8(sizeof(VdbeCursor))+2\*sizeof(u32)\*nField\]/\&pMem->z[(ROUND8(sizeof(VdbeCursor))+2*sizeof(u32)*nField+15)\&~15]/' \
  -e '/^typedef struct sqlite3_file sqlite3_file;$/a\
typedef const char *sqlite3_filename;' \
  "$SRC" > "$TMP"

# Every rewrite is asserted: a silently missed substitution is the failure mode
# the 3.53.3 script guards against, and 3.22.0 deserves the same treatment.
grep -q 'char saveBuf\[PARSE_TAIL_SZ\] __attribute__((aligned(16)));' "$TMP"
grep -q '((ROUND8(sizeof(VdbeCursor)) + 2\*sizeof(u32)\*nField + 15)&~15) + ' "$TMP"
grep -q '&pMem->z\[(ROUND8(sizeof(VdbeCursor))+2\*sizeof(u32)\*nField+15)&~15\]' "$TMP"
[ "$(grep -c '^typedef const char \*sqlite3_filename;$' "$TMP")" = 1 ]
mv -f "$TMP" "$OUT"
echo "adapted 3.22.0 -> $OUT ($(wc -l < "$OUT") lines)"
