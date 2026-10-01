#!/usr/bin/env bash
# Prepare SQLite 3.22.0 for the ordinary-program build (build-domain.sh) and its native oracle
# (build-native.sh), under <root>:
#   sqlite3.c        the amalgamation with the Capstone adaptations (../adapt-sqlite-322.sh)
#   sqlite3-stock.c  the amalgamation as released, for the native oracle
#   sqlite3.h, sqlite3ext.h, shell.c   as released
#   speedtest1.c     test/speedtest1.c from the full source release, with the result oracle
#                    (../study/speedtest1-oracle.patch; built with -DSQLITE_STUDY_ORACLE)
# Both archives are pinned by SHA3-256 and kept in a cache root of this version only, so a
# root holding another version's header can never be picked up (see run-sqlite-322-silicon.sh).
set -euo pipefail
ROOT=${1:?usage: prepare-sources.sh <root>}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORT=$(cd -- "$SCRIPT_DIR/.." && pwd)
CACHE=${SQLITE_322_CACHE_ROOT:-$ROOT/cache}
mkdir -p "$ROOT" "$CACHE"

AMALGAMATION=$(SQLITE_VERSION=3220000 SQLITE_YEAR=2018 \
  SQLITE_ARCHIVE_SHA3=69bc5ee8f08d747494dd3a4bfe075e5b078fe200dfc671d76dd9e1ccb5b2decb \
  SQLITE_CACHE_ROOT="$CACHE" bash "$PORT/fetch-sqlite.sh")

SOURCE_ZIP=$CACHE/sqlite-src-3220000.zip
SOURCE_SHA3=f8695d69b4b4e4c6e7ff72defd8b2c3a5db12cf07fade273e6bbc1fd61841a98
[[ -f "$SOURCE_ZIP" ]] || curl -fsSL https://www.sqlite.org/2018/sqlite-src-3220000.zip -o "$SOURCE_ZIP"
python3 - "$SOURCE_ZIP" "$SOURCE_SHA3" "$ROOT/speedtest1.c" <<'PY'
import hashlib, sys, zipfile
archive, expected, out = sys.argv[1:]
actual = hashlib.sha3_256(open(archive, 'rb').read()).hexdigest()
if actual != expected:
    raise SystemExit(f"sqlite-src-3220000.zip SHA3-256 mismatch: expected {expected}, got {actual}")
with zipfile.ZipFile(archive) as z:
    open(out, 'wb').write(z.read('sqlite-src-3220000/test/speedtest1.c'))
PY
patch -s -p1 -d "$ROOT" < "$PORT/study/speedtest1-oracle.patch"
grep -q 'STUDY-ORACLE phase=%d rows=%llu hash=%016llx' "$ROOT/speedtest1.c"

bash "$PORT/adapt-sqlite-322.sh" "$AMALGAMATION/sqlite3.c" "$ROOT/sqlite3.c" > /dev/null
# One more adaptation, for a path only this build takes. With temporary data in files (SQLite's
# default TEMP_STORE), the sorter packs its records into one buffer at ROUND8 offsets, and every
# record holds a pointer 16 bytes in (SorterRecord.u.pNext): a record 8 bytes off its boundary
# faults at its first link (a misaligned store, cause 6, in vdbeSorterSort; the first CREATE
# INDEX reaches it). Records go at 16-byte offsets. The freestanding builds define
# SQLITE_TEMP_STORE=3, so sqlite3TempInMemory() keeps them off this path; the rewrite stays out
# of adapt-sqlite-322.sh, whose output their recorded images are built from.
python3 - "$ROOT/sqlite3.c" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
old = "    pSorter->iMemory += ROUND8(nReq);\n"
new = "    pSorter->iMemory += (nReq+15)&~15;  /* Capstone: records hold pointers */\n"
if text.count(old) != 1:
    raise SystemExit("prepare-sources: the sorter's ROUND8(nReq) line is not there exactly once")
open(path, "w").write(text.replace(old, new))
PY
cp "$AMALGAMATION/sqlite3.c" "$ROOT/sqlite3-stock.c"
cp "$AMALGAMATION/sqlite3.h" "$AMALGAMATION/sqlite3ext.h" "$AMALGAMATION/shell.c" "$ROOT/"
printf '%s\n' "$ROOT"
