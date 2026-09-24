#!/usr/bin/env bash
# __thread in a musl domain (C-47): the compiler's local-exec lowering, the TLS
# segment in my_first_domain/link.ld, and the block runtime/tls.c builds from it.
#
#   bash run.sh     exit 0 only if tls-O0 and tls-O2 pass every check in tls_test.c
#                   AND both controls fail:
#     tls-overrun   writes one byte past a 100-byte thread-local: must halt on the
#                   capability's bounds (the variable's capability is narrowed);
#     tls-oldrt     the -O2 test linked against the tls.c this branch starts from
#                   (56c39b13369d, pinned so it stays a control after this lands),
#                   which never copies the template: must not pass.
#   Each control halts its guest, so each runs last in a boot of its own.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (a clang with C-47), CAPSTONE_BUILDROOT_DIR and
# CAPSTONE_QEMU_BINARY, and a python with pexpect (PYTHON=..., default python3).
# Does not take the QEMU lock itself: the caller holds it (the nightly does).
set -euo pipefail


HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/thread-local}
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
CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$REPO/compiler-rt/lib/builtins
COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
              -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
source "$REPO/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -c "$HERE/entry.c" -o "$O/entry.o"
RUNTIME=("$O/start-musl.o" "$O/hostcall.o" "$O/tls.o" "$O/set_thread_area.o" "$O/setjmp.o"
         "${MUSL_OVERRIDE_OBJS[@]}" "$O/atomic_libcalls.o"
         "$O/level0.o" "${softfloat_objs[@]}" "$O/entry.o")

# The control's one difference: the tls.c this branch starts from.
git -C "$REPO" show 56c39b13369d:capstone/ports/musl-capstone/runtime/tls.c > "$O/tls-oldrt.c"
grep -q '__capstone_tls_image' "$O/tls-oldrt.c" \
  && { echo "56c39b13369d's tls.c already copies a TLS template; it controls nothing" >&2; exit 2; }
"$CAPSTONE_CLANG" "${RF[@]}" -I"$MRT" -c "$O/tls-oldrt.c" -o "$O/tls-oldrt.o"

rm -f "$OUT/share"/*.dom
LD=("$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld")
for opt in O0 O2; do
  "$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -$opt -c "$HERE/tls_test.c" -o "$O/test-$opt.o"
  "$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -$opt -c "$HERE/tls_other.c" -o "$O/other-$opt.o"
  "${LD[@]}" -o "$OUT/share/tls-$opt.dom" "${RUNTIME[@]}" "$O/test-$opt.o" "$O/other-$opt.o" "$ARCHIVE"
done
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O2 -DOVERRUN -c "$HERE/tls_test.c" -o "$O/test-overrun.o"
"${LD[@]}" -o "$OUT/share/tls-overrun.dom" "${RUNTIME[@]}" "$O/test-overrun.o" "$O/other-O2.o" "$ARCHIVE"
OLDRT=("${RUNTIME[@]/#$O\/tls.o/$O/tls-oldrt.o}")
"${LD[@]}" -o "$OUT/share/tls-oldrt.dom" "${OLDRT[@]}" "$O/test-O2.o" "$O/other-O2.o" "$ARCHIVE"

# The image carries a TLS segment aligned to the page-aligned thread-local, so the
# page-offset placement in tls.c is what the align4096 check exercises.
tls_ph=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -l "$OUT/share/tls-O2.dom" | awk '$1 == "TLS" { print $NF }')
[[ $tls_ph == 0x1000 ]] || { echo "tls-O2.dom: PT_TLS alignment '$tls_ph', want 0x1000" >&2; exit 2; }

# The host: libc-test's, which runs one .dom to DONE and prints its status.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

# boot <log> <image...>: one QEMU boot running the images in order.
boot() {
  local log=$1; shift
  local guest="echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; for d in $*; do echo RUN-BEGIN \$d; /tmp/lt.user /mnt/host/\$d.dom 60; echo RUN-END \$d rc=\$?; done; echo __ALL_DONE__"
  set +e
  "$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
    --share-dir "$OUT/share" --log-file "$log" --timeout-multiplier 8 \
    --guest-command "$guest" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ >/dev/null 2>&1
  set -e
}
LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
LOG2=${LOG%.log}-oldrt.log
boot "$LOG" tls-O0 tls-O2 tls-overrun
boot "$LOG2" tls-oldrt

verdict=0
for arm in O0 O2; do
  block=$(sed -n "/RUN-BEGIN tls-$arm\$/,/RUN-END tls-$arm /p" "$LOG")
  passes=$(grep -ac 'TLS-TEST PASS' <<<"$block" || true)
  if grep -aq "LT-RESULT tls-$arm.dom status=0 " <<<"$block" && [[ $passes == 9 ]] &&
     ! grep -aq "halted by capability fault" <<<"$block"; then
    echo "  tls-$arm: PASS (9 checks)"
  else
    echo "  tls-$arm: FAIL ($passes of 9 checks passed)"
    grep -aE "TLS-TEST|LT-RESULT|halted" <<<"$block" | sed 's/^/    /'
    verdict=1
  fi
done
block=$(sed -n "/RUN-BEGIN tls-overrun\$/,\$p" "$LOG")
if grep -aq "writing zeroed" <<<"$block" && grep -aq "halted by capability fault" <<<"$block" &&
   ! grep -aq "NOT stopped" <<<"$block"; then
  echo "  tls-overrun: halts at the write past the thread-local, as it must"
else
  echo "  tls-overrun: did NOT halt at the overrun -- the bounds check cannot be seen"
  grep -aE "TLS-TEST|LT-RESULT|halted" <<<"$block" | sed 's/^/    /'
  verdict=1
fi
block=$(sed -n "/RUN-BEGIN tls-oldrt\$/,\$p" "$LOG2")
if grep -aq "RUN-BEGIN tls-oldrt" <<<"$block" && ! grep -aq "LT-RESULT tls-oldrt.dom status=0 " <<<"$block"; then
  echo "  tls-oldrt: does not pass, as it must without the template copy ($(grep -aoE 'halted by capability fault|LT-RESULT[^\n]*' <<<"$block" | head -1))"
else
  echo "  tls-oldrt: PASSED or never ran -- the test cannot tell the runtime change from its absence"
  grep -aE "TLS-TEST|LT-RESULT|halted" <<<"$block" | sed 's/^/    /'
  verdict=1
fi
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), log $LOG"
exit $verdict
