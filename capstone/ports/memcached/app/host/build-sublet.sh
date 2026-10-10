#!/usr/bin/env bash
# The Sublet arm of the virtual server: memcached 1.6.45 with patch 0006 compiled in
# (-DMC_CAPSTONE_SUBLET), so every slab chunk and cache.c object is a child lifetime (CDERIVE) that
# slabs_free and cache_free revoke (CREVOKE). The heap under it is the virtual profile's mallocng, the
# same as the plain virtual images'; nothing else differs. Two images, from one configured tree:
#   $MC_WORK/sublet/memcached.dom          the oracle's
#   $MC_WORK/sublet/memcached-safety.dom   with patch 0005's fixture hook as well
# Gates, each fatal:
#   INSN  both images carry CDERIVE and CREVOKE encodings; the plain virtual images (domain/,
#         safety/, marker/), when present, carry none
#   HOOK  the safety image carries the fixture command; the oracle image does not
#   LINK  nothing undefined but _DYNAMIC, no undefined weak symbol (as build-domain.sh)
set -euo pipefail
APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ ${CAPSTONE_APPLICATION_PROFILE:-} == virtual ]] || { echo "build-sublet.sh: virtual profile only (CAPSTONE_APPLICATION_PROFILE=virtual)" >&2; exit 2; }
source "$APP/deps/env.sh"
[ -f "$MC_DEPS_PREFIX/lib/libevent_core.a" ] || { echo "no libevent in $MC_DEPS_PREFIX: run deps/build-libevent.sh" >&2; exit 2; }
RT_INC=$CAPSTONE_REPO_ROOT/capstone/runtime/include
MSRC=$(CAPSTONE_TMP_ROOT=$MC_WORK bash "$APP/../fetch-memcached.sh" | tail -1)
OUT=$MC_WORK/sublet; rm -rf "$OUT"; mkdir -p "$OUT/logs"

build() {   # name, extra CPPFLAGS
  local name=$1 defs=$2 X=$OUT/src-$1 L=$OUT/logs/$1
  mkdir -p "$L"; cp -a "$MSRC" "$X"
  for p in "$APP"/patches/*.patch; do
    ( cd "$X" && patch --batch --forward --fuzz=0 -p1 < "$p" > "$L/patch-$(basename "$p").log" ) || { echo "patch $p FAILED" >&2; exit 1; }
  done
  cp "$APP/src/mcapp-safety.c" "$X/"
  ( cd "$X" && ./configure --host=riscv64-unknown-linux-musl --with-libevent="$MC_DEPS_PREFIX" \
      --disable-extstore --disable-proxy --disable-tls --disable-sasl --disable-docs \
      CPPFLAGS="-DMC_CAPSTONE_SUBLET -I$RT_INC $defs" > "$L/configure.log" 2>&1 )
  ( cd "$X" && make -j"${JOBS:-8}" memcached > "$L/build.log" 2>&1 ) || { echo "build $name FAILED"; grep -E 'error' "$L/build.log" | head -5; exit 1; }
  cp "$X/memcached" "$OUT/$name.dom"
}
build memcached ""
build memcached-safety "-DMC_CAPSTONE_SAFETY_FIXTURES"

# CDERIVE and CREVOKE are R-type words with opcode 0x5b, funct3 1 and funct7 0x51 / 0x52
# (runtime/include/capstone/capability.h); counted in each image's .text.
insn() {   # image -> "cderive crevoke"
  local t; t=$(mktemp)
  "$CAPSTONE_LLVM_BIN/llvm-objcopy" -O binary --only-section=.text "$1" "$t"
  python3 - "$t" <<'EOF'
import struct, sys
data = open(sys.argv[1], 'rb').read()
words = struct.unpack('<%dI' % (len(data) // 4), data[:len(data) // 4 * 4])
count = lambda f7: sum(1 for w in words if w & 0xfe00707f == (f7 << 25) | (1 << 12) | 0x5b)
print(count(0x51), count(0x52))
EOF
  rm -f "$t"
}
for f in "$OUT/memcached.dom" "$OUT/memcached-safety.dom"; do
  read -r d r < <(insn "$f")
  echo "sublet $(basename "$f"): $(sha256sum < "$f" | cut -c1-16) cderive=$d crevoke=$r"
  [ "$d" -gt 0 ] && [ "$r" -gt 0 ] || { echo "INSN GATE: $f carries no CDERIVE or no CREVOKE"; exit 1; }
done
for f in "$MC_WORK/domain/memcached.dom" "$MC_WORK/safety/memcached-safety-virtual.dom" "$MC_WORK/marker/memcached.dom"; do
  [ -f "$f" ] || continue
  read -r d r < <(insn "$f")
  [ "$d" = 0 ] && [ "$r" = 0 ] || { echo "INSN GATE: the plain image $f carries cderive=$d crevoke=$r"; exit 1; }
  echo "plain $(basename "$f"): cderive=0 crevoke=0"
done
has_hook() { python3 -c 'import sys; sys.exit(0 if b"mc_capstone_fixture " in open(sys.argv[1],"rb").read() else 1)' "$1"; }
has_hook "$OUT/memcached-safety.dom" || { echo "HOOK GATE: the safety image lacks the fixture command"; exit 1; }
! has_hook "$OUT/memcached.dom" || { echo "HOOK GATE: the oracle image carries the fixture command"; exit 1; }
for f in "$OUT/memcached.dom" "$OUT/memcached-safety.dom"; do
  undef=$("$CAPSTONE_LLVM_BIN/llvm-nm" -u "$f" | grep -v ' w ' | grep -vE ' _DYNAMIC$' || true)
  weak=$("$CAPSTONE_LLVM_BIN/llvm-nm" "$f" | awk '$1 == "w" || ($2 == "w")' || true)
  [ -z "$undef" ] || { echo "LINK GATE: $f undefined symbols:"; echo "$undef" | head; exit 1; }
  [ -z "$weak" ] || { echo "LINK GATE: $f undefined weak symbols:"; echo "$weak" | head; exit 1; }
done
echo "sublet: both images carry the lifetime instructions, the plain images none; hook in the safety image only; links clean"
