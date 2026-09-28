#!/usr/bin/env bash
# Pointers computed through uintptr_t in a musl domain: CapstoneRecoverProvenance.
#
#   bash run.sh     exit 0 only if rt-O0 and rt-O2 pass all seven checks in
#                   rt_test.c (including musl's own, unmodified atexit()) AND the
#                   control halts:
#     rt-off        rt_test.c at -O2 with -mllvm -capstone-recover-provenance=false,
#                   the one difference: its first round-tripped pointer is
#                   untagged, and writing through it must halt the domain. It
#                   runs last, since a halted domain can take the guest with it.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (a clang with the pass), CAPSTONE_BUILDROOT_DIR
# and CAPSTONE_QEMU_BINARY, and a python with pexpect (PYTHON=..., default python3).
# Does not take the QEMU lock itself: the caller holds it (the nightly does).
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/round-trip}
PYTHON=${PYTHON:-python3}
AR=$CAPSTONE_LLVM_BIN/llvm-ar
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
# Except atexit's: this test is about musl's OWN atexit(), which round-trips its handler
# through uintptr_t and works only because CapstoneRecoverProvenance recovers it.
keep=(); for o in "${MUSL_OVERRIDE_OBJS[@]}"; do [[ $o == */atexit_capability_safe.o ]] || keep+=("$o"); done
MUSL_OVERRIDE_OBJS=("${keep[@]}")
CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$REPO/compiler-rt/lib/builtins
COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
              -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
source "$REPO/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -c "$HERE/entry.c" -o "$O/entry.o"
RUNTIME=("$O/start-musl.o" "$O/hostcall.o" "$O/tls.o" "$O/set_thread_area.o" "$O/setjmp.o"
         "${MUSL_OVERRIDE_OBJS[@]}" "$O/atomic_libcalls.o"
         "$O/level0.o" "${softfloat_objs[@]}" "$O/entry.o")

rm -f "$OUT/share"/*.dom
LD=("$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld")
for arm in O0 O2 off; do
  opt=(-$arm); [[ $arm == off ]] && opt=(-O2 -mllvm -capstone-recover-provenance=false)
  "$CAPSTONE_CLANG" "${CF[@]}" -std=c11 "${opt[@]}" -Wno-capstone-pointer-roundtrip \
    -c "$HERE/rt_test.c" -o "$O/test-$arm.o"
  "${LD[@]}" -o "$OUT/share/rt-$arm.dom" "${RUNTIME[@]}" "$O/test-$arm.o" "$ARCHIVE"
done

# The host: libc-test's, which runs one .dom to DONE and prints its status.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
GUEST='echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; for d in rt-O0 rt-O2 rt-off; do echo RUN-BEGIN $d; /tmp/lt.user /mnt/host/$d.dom 60; echo RUN-END $d rc=$?; done; echo __ALL_DONE__'
set +e
"$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$OUT/share" --log-file "$LOG" --timeout-multiplier 8 \
  --guest-command "$GUEST" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ >/dev/null 2>&1
set -e

verdict=0
for arm in O0 O2; do
  block=$(sed -n "/RUN-BEGIN rt-$arm\$/,/RUN-END rt-$arm /p" "$LOG")
  passes=$(grep -ac 'RT-TEST PASS' <<<"$block" || true)
  if grep -aq "LT-RESULT rt-$arm.dom status=0 " <<<"$block" && [[ $passes == 6 ]] &&
     grep -aq "RT-TEST atexit handler ran" <<<"$block" &&
     ! grep -aq "halted by capability fault" <<<"$block"; then
    echo "  rt-$arm: PASS (6 checks, and musl's atexit handler ran)"
  else
    echo "  rt-$arm: FAIL ($passes of 6 checks passed)"
    grep -aE "RT-TEST|LT-RESULT|halted" <<<"$block" | sed 's/^/    /'
    verdict=1
  fi
done
block=$(sed -n "/RUN-BEGIN rt-off\$/,\$p" "$LOG")
if grep -aq "halted by capability fault" <<<"$block" && ! grep -aq "RT-TEST PASS align-up" <<<"$block"; then
  echo "  rt-off: halts at the first round-tripped pointer, as it must without the pass"
else
  echo "  rt-off: did NOT halt there -- the test cannot tell the pass from its absence"
  grep -aE "RT-TEST|LT-RESULT|halted" <<<"$block" | sed 's/^/    /'
  verdict=1
fi
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), log $LOG"
exit $verdict
