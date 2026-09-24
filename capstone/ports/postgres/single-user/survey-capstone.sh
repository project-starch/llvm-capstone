#!/usr/bin/env bash
# How much of PostgreSQL's backend compiles for a capstone64 musl domain, and
# what stops the rest: configure for riscv64-unknown-linux-musl with a
# capstone-cc wrapper as CC, then make -k over src/backend with one line per
# object, tallied at the end. results/<date>/compile-survey.txt and objects.tsv
# are its output.
#
#   CAPSTONE_CC=<capstone-cc> CAPSTONE_CC_ENV=<capstone-env.sh> bash survey-capstone.sh [<tarball>]
#
# CAPSTONE_CC is the CPython port's toolchain/capstone-cc (or anything with the
# same contract: compile against musl's headers, link a real domain, and with
# CPY_SURVEY_LOG set append "<object> <rc> <source> <first error>" per compile);
# CAPSTONE_CC_ENV is the capstone-env.sh its prepare script wrote, which names
# the clang, the musl tree, the runtime objects and the libc archive. Both are
# recorded in the result. -g is left out on purpose: Assignment Tracking (C-50)
# runs only with debug info, and PostgreSQL's release CFLAGS have no -g either.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh" >/dev/null
: "${CAPSTONE_CC:?path to a capstone-cc wrapper}" "${CAPSTONE_CC_ENV:?its capstone-env.sh}"
ROOT=${PG_SU_ROOT:-$CAPSTONE_TMP_ROOT/pg-single-user}
TARBALL=${1:-$CAPSTONE_TMP_ROOT/pg-mmgr-host/pg.tar.bz2}
[[ -f "$TARBALL" ]] || { echo "no tarball $TARBALL" >&2; exit 2; }
source "$CAPSTONE_CC_ENV"
mkdir -p "$ROOT/cross"
cd "$ROOT/cross"
[[ -d postgresql-17.5 ]] || tar xjf "$TARBALL"
cd postgresql-17.5
make distclean >/dev/null 2>&1 || true
# --with-system-tzdata: a cross build would otherwise need a zic for the host.
CC="$CAPSTONE_CC" CFLAGS="-O2" ./configure --host=riscv64-unknown-linux-musl \
  --without-readline --without-zlib --without-icu --with-system-tzdata=/usr/share/zoneinfo \
  > "$ROOT/cross-configure.log" 2>&1 || { tail -5 "$ROOT/cross-configure.log" >&2; echo "configure failed" >&2; exit 1; }
# The headers perl generates are a separate top-level target; without this
# step 181 objects fail on a missing header and the count says nothing.
make -C src/backend generated-headers > "$ROOT/cross-genheaders.log" 2>&1
export CPY_SURVEY_LOG=$ROOT/objects.tsv
: > "$CPY_SURVEY_LOG"
make -k -j"${JOBS:-16}" -C src/backend > "$ROOT/cross-make.log" 2>&1 || true
{
  echo "# PostgreSQL 17.5 backend compile survey, capstone64/musl, $(date +%F)"
  echo "# compiler: $("$CAPSTONE_CLANG" --version | head -1)"
  echo "# capstone-cc: $CAPSTONE_CC; env: $CAPSTONE_CC_ENV"
  echo "# CFLAGS=-O2; configure --host=riscv64-unknown-linux-musl --without-readline --without-zlib --without-icu --with-system-tzdata"
  python3 - "$CPY_SURVEY_LOG" <<'PY'
import sys, collections
rows = {}
for l in open(sys.argv[1]):
    r = l.rstrip('\n').split('\t')
    if len(r) >= 2: rows[r[0]] = r
ok = [r for r in rows.values() if r[1] == '0']; bad = [r for r in rows.values() if r[1] != '0']
print(f"objects compiled: {len(ok)} of {len(rows)} ({len(bad)} failed)")
c = collections.Counter()
for r in bad:
    msg = r[3] if len(r) > 3 else '?'
    for k in ('Assertion', 'error:', 'PLEASE submit', 'UNREACHABLE'):
        if k in msg: msg = msg[msg.find(k):]; break
    c[msg[:120]] += 1
for k, n in c.most_common(30): print(f"{n:4d}  {k}")
print("failed objects: " + ' '.join(sorted(r[0] for r in bad)))
PY
  errs=$(find src -name '*.o.err')
  echo "# warnings over what compiled: $(cat $errs | grep -o -E '\[-W[a-z-]+\]' | sort | uniq -c | sort -rn | awk '{printf "%s %s; ", $2, $1}')"
  echo "# -Wcapstone-pointer-roundtrip: $(cat $errs | grep 'Wcapstone-pointer-roundtrip' | grep -o -E '^[^:]+:[0-9]+' | sort -u | wc -l) distinct explicit sites in $(cat $errs | grep 'Wcapstone-pointer-roundtrip' | grep -o -E '^[^:]+' | sort -u | wc -l) files (DatumGetPointer, an inline function, fires once per translation unit)"
  echo "# configure answers:"
  grep -E 'define (SIZEOF_VOID_P|SIZEOF_SIZE_T|SIZEOF_LONG|MAXIMUM_ALIGNOF|ALIGNOF_DOUBLE|USE_SYSV_SHARED_MEMORY|USE_UNNAMED_POSIX_SEMAPHORES|HAVE_SYS_EPOLL_H|HAVE_SYS_SIGNALFD_H|HAVE_COMPUTED_GOTO|HAVE_SYNC_FILE_RANGE|HAVE_POSIX_FADVISE|PG_INT128_TYPE)' src/include/pg_config.h
} > "$ROOT/compile-survey.txt"
cat "$ROOT/compile-survey.txt"
