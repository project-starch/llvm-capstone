#!/usr/bin/env bash
# Atomics whose value is a capability, run in a musl domain under QEMU (C-54).
#
#   bash run.sh        build with $CAPSTONE_CLANG at -O0 and -O2, run both in one
#                      boot, exit 0 only if BOTH print
#                      "CAP-ATOMICS-DONE failures=0" and a PASS status line
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (clang, ld.lld, llvm-ar), CAPSTONE_BUILDROOT_DIR
# and CAPSTONE_QEMU_BINARY (the tree that has them), and a python with pexpect
# (PYTHON=..., default python3). Takes the shared QEMU lock for the run.
# A run without the DONE marker is a FAIL, never "no result": a domain that
# faults part-way prints the lines before the fault and nothing after.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/capability-atomics}
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
# The libc overrides come from the one list every musl domain links (runtime/libc_overrides.sh),
# not a hand copy: a hand list here missed atexit_capability_safe when that override was added.
# atomic_libcalls stays above -- it defines the __atomic_* libcalls, not a replacement for a musl object.
source "$MRT/libc_overrides.sh"
build_musl_overrides "$CAPSTONE_CLANG" "$O" "$MUSL" "${RF[@]}"
CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$REPO/compiler-rt/lib/builtins
COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
              -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
source "$REPO/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -c "$HERE/entry.c" -o "$O/entry.o"
RUNTIME=("$O/start-musl.o" "$O/hostcall.o" "$O/tls.o" "$O/set_thread_area.o" "$O/setjmp.o"
         "${MUSL_OVERRIDE_OBJS[@]}"
         "$O/level0.o" "$O/atomic_libcalls.o" "${softfloat_objs[@]}" "$O/entry.o")

rm -f "$OUT/share"/*.dom
for lvl in O0 O2; do
  "$CAPSTONE_CLANG" "${CF[@]}" -std=c11 "-$lvl" -c "$HERE/cap_atomics.c" -o "$O/test-$lvl.o"
  "$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
    -o "$OUT/share/cap-$lvl.dom" "${RUNTIME[@]}" "$O/test-$lvl.o" "$ARCHIVE"
done

# The host: libc-test's, which runs one .dom to DONE and prints its status.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
GUEST='echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; for d in cap-O0 cap-O2; do echo RUN-BEGIN $d; /tmp/lt.user /mnt/host/$d.dom 60; echo RUN-END $d rc=$?; done; echo __ALL_DONE__'
set +e
flock -w 3600 "$CAPSTONE_QEMU_LOCK" "$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$OUT/share" --log-file "$LOG" --timeout-multiplier 8 \
  --guest-command "$GUEST" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ >/dev/null 2>&1
set -e

verdict=0
for lvl in O0 O2; do
  if grep -aq "CAP-ATOMICS-DONE failures=0" <(sed -n "/RUN-BEGIN cap-$lvl/,/RUN-END cap-$lvl/p" "$LOG") &&
     grep -aq "LT-RESULT cap-$lvl.dom status=0 .*PASS" "$LOG"; then
    echo "  cap-$lvl: PASS"
  else
    echo "  cap-$lvl: FAIL"
    sed -n "/RUN-BEGIN cap-$lvl/,/RUN-END cap-$lvl/p" "$LOG" | grep -aE "^FAIL|DONE|halted|LT-RESULT" | sed 's/^/    /'
    verdict=1
  fi
done
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), log $LOG"
exit $verdict
