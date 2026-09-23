#!/usr/bin/env bash
# The standard descriptors of a musl domain (fstat, isatty, lseek, write, close), under QEMU.
#
#   bash run.sh        exit 0 only if the domain built with this tree's runtime
#                      prints "STDIO-FD-DONE failures=0" with a PASS status
#                      line, AND the control fails fstat on stdout
#
# The control is the same test linked against the hostcall.c of 40eefa09420c (dev before this;
# pinned, so it stays a control after this lands), which
# answers EBADF for every descriptor it did not open itself. The two differ in
# that one file only.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (clang, ld.lld, llvm-ar), CAPSTONE_BUILDROOT_DIR
# and CAPSTONE_QEMU_BINARY, and a python with pexpect (PYTHON=..., default
# python3). Takes the shared QEMU lock for the run.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/stdio-descriptors}
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
for f in hostcall tls level0 string_bounds_safe fputwc_null_safe; do
  "$CAPSTONE_CLANG" "${RF[@]}" -c "$MRT/$f.c" -o "$O/$f.o"
done
# The control's one difference: 40eefa09420c's hostcall.c, whose fstat knows no stdout.
# Its one relative include is made absolute, since the copy lives in $O.
git -C "$REPO" show 40eefa09420c:capstone/ports/musl-capstone/runtime/hostcall.c \
  | sed "s#\"../../../tests/runtime-qemu/#\"$REPO/capstone/tests/runtime-qemu/#" > "$O/hostcall-control.c"
grep -q "\"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe/" "$O/hostcall-control.c" \
  || { echo "control copy: the include rewrite did not apply" >&2; exit 2; }
if grep -q "S_IFCHR" "$O/hostcall-control.c"; then
  echo "40eefa09420c's hostcall.c describes stdout; this control no longer controls anything" >&2
  exit 2
fi
"$CAPSTONE_CLANG" "${RF[@]}" -c "$O/hostcall-control.c" -o "$O/hostcall-control.o"
"$CAPSTONE_CLANG" "${RF[@]}" -I"$MUSL/src/multibyte" -c "$MRT/mbsrtowcs_bounds_safe.c" \
  -o "$O/mbsrtowcs_bounds_safe.o"
CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$REPO/compiler-rt/lib/builtins
COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
              -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
source "$REPO/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -c "$HERE/entry.c" -o "$O/entry.o"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O2 -c "$HERE/stdio_descriptors.c" -o "$O/stdio_descriptors.o"
COMMON=("$O/start-musl.o" "$O/tls.o" "$O/set_thread_area.o" "$O/setjmp.o"
        "$O/string_bounds_safe.o" "$O/mbsrtowcs_bounds_safe.o" "$O/fputwc_null_safe.o"
        "$O/level0.o" "${softfloat_objs[@]}" "$O/entry.o" "$O/stdio_descriptors.o")
rm -f "$OUT/share"/*.dom
for arm in test control; do
  hc=$O/hostcall.o; [[ $arm == control ]] && hc=$O/hostcall-control.o
  "$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
    -o "$OUT/share/fs-$arm.dom" "${COMMON[@]}" "$hc" "$ARCHIVE"
done

# The host: libc-test's, built from this tree, the same in both arms.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
GUEST='echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; for d in fs-test fs-control; do echo RUN-BEGIN $d; /tmp/lt.user /mnt/host/$d.dom 60; echo RUN-END $d rc=$?; done; echo __ALL_DONE__'
set +e
CAPSTONE_GP_NONLIN=${CAPSTONE_GP_NONLIN:-1} \
capstone_with_qemu_lock "$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$OUT/share" --log-file "$LOG" --timeout-multiplier 8 \
  --guest-command "$GUEST" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ >/dev/null 2>&1
set -e

arm_log() { sed -n "/RUN-BEGIN fs-$1/,/RUN-END fs-$1/p" "$LOG"; }
verdict=0
if grep -aq "STDIO-FD-DONE failures=0" <(arm_log test) &&
   grep -aq "LT-RESULT fs-test.dom status=0 .*PASS" "$LOG"; then
  echo "  test:    PASS"
else
  echo "  test:    FAIL"; verdict=1
fi
arm_log test | grep -aE "^STDIO-FD|halted|LT-RESULT" | sed 's/^/    /'
if grep -aq "STDIO-FD FAIL fstat-stdout" <(arm_log control) &&
   grep -aq "STDIO-FD-DONE" <(arm_log control); then
  echo "  control: FAILS fstat on stdout, as it must without this change"
else
  echo "  control: did NOT fail fstat on stdout -- the test cannot tell the change from its absence"; verdict=1
fi
arm_log control | grep -aE "^STDIO-FD|halted|LT-RESULT" | sed 's/^/    /'
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), log $LOG"
exit $verdict
