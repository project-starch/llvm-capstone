#!/usr/bin/env bash
# Build the same PostgreSQL 17.5 backend for CheriBSD purecap, with either the
# ordinary memory contexts or the existing PoisonCap context backend.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh" >/dev/null
: "${CHERI_SDK:?set CHERI_SDK}"
: "${CHERI_SYSROOT:?set CHERI_SYSROOT}"
MODE=${PG_CHERI_MODE:-spatial}
case "$MODE" in spatial|poisoncap) ;; *) echo "PG_CHERI_MODE=$MODE?" >&2; exit 2;; esac
ROOT=${PG_CHERI_ROOT:-$CAPSTONE_TMP_ROOT/pg-cheribsd-$MODE}
TARBALL=${1:-$CAPSTONE_TMP_ROOT/pg-mmgr-host/pg.tar.bz2}
JOBS=${JOBS:-12}
MANAGER=$HERE/../memory-contexts
[[ -f "$TARBALL" ]] || { echo "missing $TARBALL" >&2; exit 2; }
[[ $(sha256sum "$TARBALL" | cut -d' ' -f1) == fcb7ab38e23b264d1902cb25e6adafb4525a6ebcbd015434aeef9eda80f528d8 ]] \
  || { echo "PostgreSQL archive SHA-256 mismatch" >&2; exit 2; }
mkdir -p "$ROOT"
SRC=$ROOT/postgresql-17.5
if [[ -d $SRC && ! -f $ROOT/mode.txt ]]; then
  echo "incomplete preparation in $ROOT; select a fresh PG_CHERI_ROOT" >&2
  exit 2
fi
if [[ ! -d $SRC ]]; then
  tar -xjf "$TARBALL" -C "$ROOT"
  for n in 0001 0002 0004 0005 0007 0008 0009 0010 0011 0012 0013 0016; do
    patch -d "$SRC" -p1 --batch --forward --fuzz=0 -s < "$HERE"/patches/"$n"-*.patch
  done
  python3 "$HERE/cheribsd-compat.py" "$SRC/src/include/c.h"
  if [[ $MODE == poisoncap ]]; then
    for n in 0003 0004 0005 0006 0007; do
      patch -d "$SRC" -p1 --batch --forward --fuzz=0 -s \
        < "$MANAGER"/patches/postgresql-17.0-"$n"-*.patch
    done
  fi
  printf '%s\n' "$MODE" > "$ROOT/mode.txt"
fi
[[ $(cat "$ROOT/mode.txt") == "$MODE" ]] || { echo "build-root mode mismatch" >&2; exit 2; }
CC="$CHERI_SDK/bin/clang --target=riscv64-unknown-freebsd13 --sysroot=$CHERI_SYSROOT -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax -fuse-ld=lld -B$CHERI_SDK/bin"
FLAGS=-O1
if [[ $MODE == poisoncap ]]; then
  FLAGS="$FLAGS -DPG_POISONCAP -DPG_POISONCAP_BATCHED -I$MANAGER/src/allocators/sublet -I$MANAGER/src/cheribsd"
fi
cd "$SRC"
if [[ ! -f config.status ]]; then
  CC="$CC" CFLAGS="$FLAGS" pgac_cv_computed_goto=no \
    ./configure --host=riscv64-unknown-freebsd13 --build=x86_64-pc-linux-gnu \
      --prefix=/usr/local/pgsql-study --enable-depend --without-readline \
      --without-zlib --without-icu --with-system-tzdata=/usr/share/zoneinfo \
      > "$ROOT/configure.log" 2>&1
  grep -q '^#define MAXIMUM_ALIGNOF 8$' src/include/pg_config.h
  sed -i 's/^#define MAXIMUM_ALIGNOF 8$/#define MAXIMUM_ALIGNOF 16/' src/include/pg_config.h
  if [[ $MODE == poisoncap ]]; then
    cp "$MANAGER/src/cheribsd/poisoncap.c" src/backend/poisoncap.c
    cp "$HERE/poisoncap-app.c" src/backend/poisoncap-app.c
    python3 - <<'PY'
from pathlib import Path
p = Path('src/backend/Makefile')
s = p.read_text()
old = 'OBJS = \\\n'
assert s.count(old) == 1
s = s.replace(old, 'LOCALOBJS += poisoncap.o poisoncap-app.o\n\n' + old)
s += '\n# Select PoisonCap mode before PostgreSQL creates its first context.\n'
s += 'override LDFLAGS += -Wl,--wrap=main\n'
p.write_text(s)
PY
  fi
fi
make -C src/backend generated-headers > "$ROOT/headers.log" 2>&1
# The qsort object must see the capability-safe swap in patch 0013. PostgreSQL
# does not track this include as a dependency in all src/port configurations.
touch src/port/qsort.c
make -j"$JOBS" -C src/port libpgport_srv.a > "$ROOT/port.log" 2>&1
make -j"$JOBS" -C src/common libpgcommon_srv.a > "$ROOT/common.log" 2>&1
make -k -j"$JOBS" -C src/backend > "$ROOT/backend.log" 2>&1 || true
make -k -j"$JOBS" -C src/backend >> "$ROOT/backend.log" 2>&1 || true
make -j"$JOBS" -C src/backend postgres > "$ROOT/backend-final.log" 2>&1 \
  || { tail -n 60 "$ROOT/backend-final.log" >&2; exit 2; }
[[ -x src/backend/postgres ]] || { echo "backend link is missing" >&2; exit 2; }
if [[ $MODE == poisoncap ]]; then
  "$CHERI_SDK/bin/llvm-nm" src/backend/postgres | grep ' pg_poisoncap_init$' > /dev/null
fi
"$CHERI_SDK/bin/llvm-strip" --strip-all -o "$ROOT/postgres" src/backend/postgres
echo "$ROOT/postgres"
