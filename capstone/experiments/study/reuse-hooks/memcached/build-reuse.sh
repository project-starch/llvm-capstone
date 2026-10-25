#!/usr/bin/env bash
# The two instrumented memcached 1.6.45 servers of the reuse-distance row (pre-registration
# docs/plans/2026-10-11-nested-reuse-distance.md), from one prepared source tree:
#   upstream + ports/memcached/app/patches/*.patch (0001-0006, as host/build-domain.sh and
#   host/build-sublet.sh apply them) + memcached-1.6.45-reuse-gap-hooks.patch (this directory).
# Patch 0006 is inert without -DMC_CAPSTONE_SUBLET, so the hook patch applies to the same tree in
# both arms, and the two configure lines differ by that one define:
#   $OUT/stock/memcached.dom      -DEXP_REUSE_GAP
#   $OUT/protected/memcached.dom  -DEXP_REUSE_GAP -DMC_CAPSTONE_SUBLET
# The port's scripts are not edited: libevent comes from deps/build-libevent.sh, the SDK from
# deps/env.sh; configure and make are those of host/build-sublet.sh's build().
# Gates, each fatal: INSN (protected carries CDERIVE and CREVOKE, stock neither), HOOK (both carry
# the observer's report), LINK (as build-sublet.sh).
set -euo pipefail
HOOKS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APP=$HOOKS/../../../../ports/memcached/app; APP=$(cd -- "$APP" && pwd)
STUDY=$(cd -- "$HOOKS/../.." && pwd)
[[ $# == 1 ]] || { echo 'usage: build-reuse.sh NEW_OUTPUT' >&2; exit 2; }
export MC_WORK; MC_WORK=$(realpath -m "$1")
[[ ! -e "$MC_WORK" ]] || { echo "use a fresh output directory: $MC_WORK" >&2; exit 2; }
mkdir -p "$MC_WORK"
export CAPSTONE_APPLICATION_PROFILE=virtual
bash "$APP/deps/build-libevent.sh"
source "$APP/deps/env.sh"
RT_INC=$CAPSTONE_REPO_ROOT/capstone/runtime/include
MSRC=$(CAPSTONE_TMP_ROOT=$MC_WORK bash "$APP/../fetch-memcached.sh" | tail -1)
COMMON="-I$RT_INC -I$HOOKS -I$STUDY -DEXP_REUSE_GAP"

build() {   # arm, extra CPPFLAGS
  local arm=$1 defs=$2 X=$MC_WORK/$1/src L=$MC_WORK/$1/logs
  mkdir -p "$L"; cp -a "$MSRC" "$X"
  for p in "$APP"/patches/*.patch "$HOOKS/memcached-1.6.45-reuse-gap-hooks.patch"; do
    ( cd "$X" && patch --batch --forward --fuzz=0 -p1 < "$p" > "$L/patch-$(basename "$p").log" ) || { echo "patch $p FAILED" >&2; exit 1; }
  done
  cp "$APP/src/mcapp-safety.c" "$X/"
  ( cd "$X" && ./configure --host=riscv64-unknown-linux-musl --with-libevent="$MC_DEPS_PREFIX" \
      --disable-extstore --disable-proxy --disable-tls --disable-sasl --disable-docs \
      CPPFLAGS="$COMMON $defs" > "$L/configure.log" 2>&1 )
  ( cd "$X" && make -j"${JOBS:-8}" memcached > "$L/build.log" 2>&1 ) || { echo "build $arm FAILED"; grep -E 'error' "$L/build.log" | head -5; exit 1; }
  cp "$X/memcached" "$MC_WORK/$arm/memcached.dom"
}
build stock ""
build protected "-DMC_CAPSTONE_SUBLET"
for f in slabs.c cache.c slabs_mover.c memcached.h; do
  cmp -s "$MC_WORK/stock/src/$f" "$MC_WORK/protected/src/$f" || { echo "SOURCE GATE: $f differs between the arms"; exit 1; }
done
echo "source: slabs.c cache.c slabs_mover.c memcached.h identical in both arms; slabs.c $(sha256sum < "$MC_WORK/stock/src/slabs.c" | cut -c1-16) cache.c $(sha256sum < "$MC_WORK/stock/src/cache.c" | cut -c1-16)"

insn() {   # image -> "cderive crevoke" (as host/build-sublet.sh)
  local t; t=$(mktemp)
  "$CAPSTONE_LLVM_BIN/llvm-objcopy" -O binary --only-section=.text "$1" "$t"
  python3 - "$t" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
words = struct.unpack('<%dI' % (len(data) // 4), data[:len(data) // 4 * 4])
count = lambda f7: sum(1 for w in words if w & 0xfe00707f == (f7 << 25) | (1 << 12) | 0x5b)
print(count(0x51), count(0x52))
PY
  rm -f "$t"
}
for arm in stock protected; do
  f=$MC_WORK/$arm/memcached.dom
  read -r d r < <(insn "$f")
  echo "$arm memcached.dom: sha256 $(sha256sum < "$f" | cut -c1-16) bytes $(stat -c %s "$f") cderive=$d crevoke=$r"
  if [ $arm = protected ]; then
    [ "$d" -gt 0 ] && [ "$r" -gt 0 ] || { echo "INSN GATE: $f carries no CDERIVE or no CREVOKE"; exit 1; }
  else
    [ "$d" = 0 ] && [ "$r" = 0 ] || { echo "INSN GATE: $f carries cderive=$d crevoke=$r"; exit 1; }
  fi
  python3 -c 'import sys; sys.exit(0 if b"REUSE-GAP %s arm=" in open(sys.argv[1],"rb").read() else 1)' "$f" \
    || { echo "HOOK GATE: $f lacks the observer report"; exit 1; }
  undef=$("$CAPSTONE_LLVM_BIN/llvm-nm" -u "$f" | grep -v ' w ' | grep -vE ' _DYNAMIC$' || true)
  weak=$("$CAPSTONE_LLVM_BIN/llvm-nm" "$f" | awk '$1 == "w" || ($2 == "w")' || true)
  [ -z "$undef" ] || { echo "LINK GATE: $f undefined symbols:"; echo "$undef" | head; exit 1; }
  [ -z "$weak" ] || { echo "LINK GATE: $f undefined weak symbols:"; echo "$weak" | head; exit 1; }
done
echo "reuse build: $MC_WORK/{stock,protected}/memcached.dom; protected carries the lifetime instructions, stock none; both report; links clean"
