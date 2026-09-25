#!/usr/bin/env bash
# __intcap on QEMU (intcap plan, Phases B and C): intcap_test.c built twice, with D =
# __uintcap_t ("intcap") and D = unsigned long ("uptr", the control), both run in each of two
# boots: CAPSTONE_MOVC_NULL_SCALAR=0 (QEMU's default movc) and =1 (the RTL's movc).
#
#   exit 0 if, in both boots, "intcap" prints ok for all six cases and END bad=0, and "uptr"
#          prints ok for the four integer cases and then faults at its first dereference
#          (no "ptr ok", no END) -- the control showing the pointer cases can fail;
#   exit 1 if any of that is wrong;
#   exit 2 if it could not check (a boot that did not reach the end of the guest command).
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (a compiler with __intcap), CAPSTONE_BUILDROOT_DIR,
# CAPSTONE_QEMU_BINARY (a capstone-qemu with CAPSTONE_MOVC_NULL_SCALAR), and a python with
# pexpect (PYTHON=...). Takes the QEMU lock for each boot unless CAPSTONE_QEMU_LOCK_HELD=1.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/intcap}
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
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O2 -DVARIANT='"intcap"' -DUSE_INTCAP -c "$HERE/intcap_test.c" -o "$O/intcap_test-intcap.o"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O2 -DVARIANT='"uptr"' -c "$HERE/intcap_test.c" -o "$O/intcap_test-uptr.o"
rm -f "$OUT/share"/*.dom
for v in intcap uptr; do
  "$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
    -o "$OUT/share/intcap-$v.dom" "$O/start-musl.o" "$O/hostcall.o" "$O/tls.o" \
    "$O/set_thread_area.o" "$O/setjmp.o" "${MUSL_OVERRIDE_OBJS[@]}" \
    "$O/atomic_libcalls.o" "$O/level0.o" \
    "${softfloat_objs[@]}" "$O/entry.o" "$O/intcap_test-$v.o" "$ARCHIVE"
done
# The host: libc-test's, which runs one .dom to DONE and prints its status.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

GUEST='echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; echo RUN-BEGIN; /tmp/lt.user /mnt/host/intcap-intcap.dom 60; /tmp/lt.user /mnt/host/intcap-uptr.dom 60; echo RUN-END rc=$?; echo __ALL_DONE__'
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
  if [[ $arm == off ]]; then log=$(boot off 0); else log=$(boot on 1); fi
  if ! grep -aq 'RUN-END' "$log"; then
    echo "  switch $arm: NO RESULT (the guest command did not finish), log $log"
    exit 2
  fi
  ok_i=$(grep -aoE 'INTCAP intcap [a-z]+ ok' "$log" | awk '{print $3}' | sort -u | tr '\n' ' ' || true)
  bad_i=$(grep -aE 'INTCAP intcap [a-z]+ BAD' "$log" | head -3 | tr '\n' ';' || true)
  end_i=$(grep -aoE 'INTCAP intcap END bad=[0-9]+' "$log" | tail -1 || true)
  ok_u=$(grep -aoE 'INTCAP uptr [a-z]+ ok' "$log" | awk '{print $3}' | sort -u | tr '\n' ' ' || true)
  end_u=$(grep -aoE 'INTCAP uptr END bad=[0-9]+' "$log" | tail -1 || true)
  echo "  switch $arm: intcap ok [$ok_i] ${end_i:-no END} ${bad_i}; uptr ok [$ok_u] ${end_u:-no END (faulted)}"
  [[ $ok_i == "arith cmp int ptr ptrarith switch " && $end_i == "INTCAP intcap END bad=0" ]] \
    || { echo "  intcap: expected all six cases ok and END bad=0"; verdict=1; }
  [[ $ok_u == "arith cmp int switch " && -z $end_u ]] \
    || { echo "  uptr (control): expected the four integer cases ok and a fault at the first dereference"; verdict=1; }
  echo "    log $log"
done
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), qemu $CAPSTONE_QEMU_BINARY"
exit $verdict
