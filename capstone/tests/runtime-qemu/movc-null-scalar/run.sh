#!/usr/bin/env bash
# capstone-qemu's CAPSTONE_MOVC_NULL_SCALAR: MOVC nulls a non-capability source, as the
# RTL does (Q-04). Boots the same -O2 domain twice, with the switch off and on.
#
#   bash run.sh     exit 0 if the probe in movc_test.c reads b=5 c=5 with the switch
#                   off (QEMU's default: an integer source survives) and b=5 c=0 with
#                   it on (the RTL: the first movc zeroed it), and the switch's
#                   first-use notice is in the log of the second boot;
#                   exit 1 if either reading is wrong;
#                   exit 2 if it could not check -- a boot that did not finish, or a
#                   QEMU without the switch (the probe reads b=5 c=5 in both boots).
#   The c32 line (the SQLite port's setupLookaside, reduced) is reported, not judged:
#   with the switch on it reads 0x1 while the compiler still copies that pointer with
#   a movc whose source is read again (C-32), and 0x5000 once it no longer does.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR, CAPSTONE_BUILDROOT_DIR and CAPSTONE_QEMU_BINARY (a
# capstone-qemu with the switch), and a python with pexpect (PYTHON=..., default python3).
# Takes the QEMU lock for each boot unless CAPSTONE_QEMU_LOCK_HELD=1.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/movc-null-scalar}
PYTHON=${PYTHON:-python3}
mkdir -p "$OUT/obj" "$OUT/share"

# musl and its archive, private to this test (prepare rewrites arch/ in place).
export MUSL_CACHE_ROOT=$OUT/musl-src
mkdir -p "$MUSL_CACHE_ROOT"
[[ -f "$MUSL_CACHE_ROOT/musl-1.2.5.tar.gz" || ! -f "$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5.tar.gz" ]] \
  || cp "$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5.tar.gz" "$MUSL_CACHE_ROOT/"
MUSL=$(bash "$REPO/capstone/ports/musl-capstone/prepare-musl-capstone.sh" | tail -1)
OUT_DIR=$OUT/musl-build bash "$REPO/capstone/ports/musl-capstone/build-musl-capstone.sh" >/dev/null
ARCHIVE=$OUT/musl-build/libc-capstone.a

# The runtime, as musl-capstone/libc-test/build-libc-test.sh builds it.
MRT=$REPO/capstone/ports/musl-capstone/runtime
INC=(-nostdinc -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic"
     -isystem "$MUSL/obj/include" -isystem "$MUSL/include")
CF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
    -Xclang -target-feature -Xclang +a -ffreestanding -fno-builtin -fno-jump-tables
    -ffunction-sections -fdata-sections -O1 -w -Wno-int-conversion "${INC[@]}")
RF=("${CF[@]}" -std=c99 -D_XOPEN_SOURCE=700
    -I"$MUSL/src/include" -I"$MUSL/src/internal" -I"$MUSL/obj/src/internal")
O=$OUT/obj
for s in start-musl set_thread_area setjmp; do
  "$CAPSTONE_CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
    -ffreestanding -O0 -c "$MRT/$s.S" -o "$O/$s.o"
done
for f in hostcall tls level0 atomic_libcalls; do
  "$CAPSTONE_CLANG" "${RF[@]}" -c "$MRT/$f.c" -o "$O/$f.o"
done
# The libc overrides, from the one list every musl domain links (runtime/libc_overrides.sh).
source "$MRT/libc_overrides.sh"
build_musl_overrides "$CAPSTONE_CLANG" "$O" "$MUSL" "${RF[@]}"
CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$REPO/compiler-rt/lib/builtins
COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
              -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
source "$REPO/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -c "$HERE/entry.c" -o "$O/entry.o"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O2 -c "$HERE/movc_test.c" -o "$O/movc_test.o"
rm -f "$OUT/share"/*.dom
"$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
  -o "$OUT/share/movc.dom" "$O/start-musl.o" "$O/hostcall.o" "$O/tls.o" \
  "$O/set_thread_area.o" "$O/setjmp.o" "${MUSL_OVERRIDE_OBJS[@]}" \
  "$O/atomic_libcalls.o" "$O/level0.o" \
  "${softfloat_objs[@]}" "$O/entry.o" "$O/movc_test.o" "$ARCHIVE"

# The host: libc-test's, which runs one .dom to DONE and prints its status.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

GUEST='echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; echo RUN-BEGIN; /tmp/lt.user /mnt/host/movc.dom 60; echo RUN-END rc=$?; echo __ALL_DONE__'
boot() { # boot <label> <switch value>; prints the log path
  local log=$OUT/run-$1-$(date +%Y%m%d-%H%M%S).log lock=()
  [[ ${CAPSTONE_QEMU_LOCK_HELD:-0} == 1 ]] || lock=(flock -w 21600 "$CAPSTONE_QEMU_LOCK")
  set +e
  CAPSTONE_MOVC_NULL_SCALAR=$2 "${lock[@]}" "$PYTHON" \
    "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
    --share-dir "$OUT/share" --log-file "$log" --timeout-multiplier 8 \
    --guest-command "$GUEST" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ \
    >/dev/null 2>&1
  set -e
  echo "$log"
}

verdict=0
for arm in off on; do
  if [[ $arm == off ]]; then log=$(boot off 0); want_c=5; else log=$(boot on 1); want_c=0; fi
  if ! grep -aq 'RUN-END' "$log" || ! grep -aq 'MOVC-PROBE' "$log"; then
    echo "  switch $arm: NO RESULT (the domain did not finish), log $log"
    exit 2
  fi
  probe=$(grep -aoE 'MOVC-PROBE b=[0-9]+ c=[0-9]+' "$log" | tail -1)
  b=$(sed -E 's/.*b=([0-9]+) c=.*/\1/' <<<"$probe") c=$(sed -E 's/.* c=([0-9]+)/\1/' <<<"$probe")
  got=$(grep -aoE 'MOVC-C32 got=0x[0-9a-f]+' "$log" | tail -1 | sed 's/.*got=//')
  notice=$(grep -ac 'MOVC-NULL-SCALAR first non-zero source nulled' "$log" || true)
  echo "  switch $arm: probe b=$b c=$c (want b=5 c=$want_c); c32 got=$got; notice lines $notice"
  if [[ $arm == on && $b == 5 && $c == 5 ]]; then
    echo "  switch on changed nothing: this QEMU ($CAPSTONE_QEMU_BINARY) has no CAPSTONE_MOVC_NULL_SCALAR"
    exit 2
  fi
  [[ $b == 5 && $c == "$want_c" ]] || verdict=1
  [[ $arm == on && $notice != 1 ]] && { echo "  switch on: expected exactly one first-use notice"; verdict=1; }
  [[ $arm == off && $notice != 0 ]] && { echo "  switch off: a first-use notice was printed"; verdict=1; }
  [[ $arm == off && $got != 0x5000 ]] && { echo "  switch off: the c32 shape lost its address with QEMU's default -- not this test's question"; verdict=1; }
  if [[ $arm == on ]]; then
    case $got in
      0x1)    echo "  c32: NULLED -- the compiler still copies the bridged pointer with a movc whose source is read again (C-32 open)" ;;
      0x5000) echo "  c32: survived -- no such movc in this image" ;;
      *)      echo "  c32: unexpected value $got"; verdict=1 ;;
    esac
  fi
  echo "    log $log"
done
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), qemu $CAPSTONE_QEMU_BINARY"
exit $verdict
