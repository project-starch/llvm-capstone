#!/usr/bin/env bash
# Large file reads and writes on the 9p share from a musl domain, under QEMU.
#
#   bash run.sh        exit 0 only if the domain served by this tree's host
#                      prints "LARGE-READ-DONE failures=0" with a PASS status
#                      line, AND the control fails its whole read
#
# The control is the SAME domain served by a host built with origin/dev's
# host_service.h, which reads straight into the payload region: 9p's zero-copy
# read cannot pin that mapping and fails with EFAULT. The two hosts differ in
# that one header only.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (clang, ld.lld, llvm-ar), CAPSTONE_BUILDROOT_DIR
# and CAPSTONE_QEMU_BINARY, and a python with pexpect (PYTHON=..., default
# python3). Takes the shared QEMU lock for the run.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/large-read}
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
"$CAPSTONE_CLANG" "${RF[@]}" -I"$MUSL/src/multibyte" -c "$MRT/mbsrtowcs_bounds_safe.c" \
  -o "$O/mbsrtowcs_bounds_safe.o"
CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$REPO/compiler-rt/lib/builtins
COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
              -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
source "$REPO/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -c "$HERE/entry.c" -o "$O/entry.o"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O2 -c "$HERE/large_read.c" -o "$O/large_read.o"
COMMON=("$O/start-musl.o" "$O/tls.o" "$O/set_thread_area.o" "$O/setjmp.o"
        "$O/string_bounds_safe.o" "$O/mbsrtowcs_bounds_safe.o" "$O/fputwc_null_safe.o"
        "$O/level0.o" "${softfloat_objs[@]}" "$O/entry.o" "$O/large_read.o" "$O/hostcall.o")
rm -f "$OUT/share"/*.dom
"$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
  -o "$OUT/share/large-read.dom" "${COMMON[@]}" "$ARCHIVE"

# The file to read, in the share the guest mounts at /mnt/host: byte i is i*7+3.
python3 -c "import sys; sys.stdout.buffer.write(bytes((i*7+3) & 255 for i in range(65536)))" > "$OUT/share/large.bin"
rm -f "$OUT/share/written.bin"

# Two hosts, libc-test's, differing only in host_service.h: this tree's, and
# origin/dev's for the control (put first on the include path).
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
mkdir -p "$O/control-inc"
git -C "$REPO" show origin/dev:capstone/ports/musl-capstone/runtime/host_service.h > "$O/control-inc/host_service.h"
if grep -q hc_host_bounce "$O/control-inc/host_service.h"; then
  echo "origin/dev's host_service.h bounces already; this control no longer controls anything" >&2
  exit 2
fi
for arm in test control; do
  inc=(); [[ $arm == control ]] && inc=(-I"$O/control-inc")
  out=$OUT/share/lt.user; [[ $arm == control ]] && out=$OUT/share/lt-control.user
  "$GUEST_CC" -O2 "${inc[@]}" -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
    -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
    -o "$out" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" "$LIBCAPSTONE_DIR/libcapstone.c"
done

LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
GUEST='echo __BOOT_OK__; cp /mnt/host/lt.user /mnt/host/lt-control.user /tmp/ && chmod 0755 /tmp/lt.user /tmp/lt-control.user; for h in lt lt-control; do echo RUN-BEGIN $h; /tmp/$h.user /mnt/host/large-read.dom 60; echo RUN-END $h rc=$?; done; echo __ALL_DONE__'
set +e
CAPSTONE_GP_NONLIN=${CAPSTONE_GP_NONLIN:-1} \
flock -w 3600 "$CAPSTONE_QEMU_LOCK" "$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$OUT/share" --log-file "$LOG" --timeout-multiplier 8 \
  --guest-command "$GUEST" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ >/dev/null 2>&1
set -e

arm_log() { sed -n "/RUN-BEGIN $1\$/,/RUN-END $1 /p" "$LOG"; }
verdict=0
if grep -aq "LARGE-READ-DONE failures=0" <(arm_log lt) &&
   grep -aq "LT-RESULT large-read.dom status=0 .*PASS" <(arm_log lt); then
  echo "  test:    PASS"
else
  echo "  test:    FAIL"; verdict=1
fi
arm_log lt | grep -aE "^LARGE-READ|halted|LT-RESULT" | sed 's/^/    /'
if grep -aq "LARGE-READ FAIL read-whole" <(arm_log lt-control) &&
   grep -aq "LARGE-READ-DONE" <(arm_log lt-control); then
  echo "  control: FAILS the whole read, as it must without the bounce buffer"
else
  echo "  control: did NOT fail the whole read -- the test cannot tell the bounce buffer from its absence"; verdict=1
fi
arm_log lt-control | grep -aE "^LARGE-READ|halted|LT-RESULT" | sed 's/^/    /'
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), log $LOG"
exit $verdict
