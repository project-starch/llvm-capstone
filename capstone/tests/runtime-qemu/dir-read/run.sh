#!/usr/bin/env bash
# Directory listing in a musl domain through the hostcall's DIR_READ, under QEMU.
#
#   bash run.sh        exit 0 only if the domain built with this tree's runtime
#                      prints "DIR-READ-DONE failures=0" with a PASS status line,
#                      AND the control fails its listing
#
# The control is the same test linked against origin/dev's hostcall.c, which
# has no getdents64: readdir must come back empty there. It differs from the
# test in that one file only, so a listing that passes both ways is not being
# served by DIR_READ at all, and the run says so.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (clang, ld.lld, llvm-ar), CAPSTONE_BUILDROOT_DIR
# and CAPSTONE_QEMU_BINARY, and a python with pexpect (PYTHON=..., default
# python3). Takes the shared QEMU lock for the run.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/dir-read}
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
# The control's one difference: origin/dev's hostcall.c, without getdents64.
# Its one relative include is made absolute, since the copy lives in $O.
git -C "$REPO" show origin/dev:capstone/ports/musl-capstone/runtime/hostcall.c \
  | sed "s#\"../../../tests/runtime-qemu/#\"$REPO/capstone/tests/runtime-qemu/#" > "$O/hostcall-control.c"
grep -q "\"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe/" "$O/hostcall-control.c" \
  || { echo "control copy: the include rewrite did not apply" >&2; exit 2; }
if grep -q SYS_getdents64 "$O/hostcall-control.c"; then
  echo "origin/dev's hostcall.c serves getdents64 already; this control no longer controls anything" >&2
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
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O2 -c "$HERE/dir_read.c" -o "$O/dir_read.o"
COMMON=("$O/start-musl.o" "$O/tls.o" "$O/set_thread_area.o" "$O/setjmp.o"
        "$O/string_bounds_safe.o" "$O/mbsrtowcs_bounds_safe.o" "$O/fputwc_null_safe.o"
        "$O/level0.o" "${softfloat_objs[@]}" "$O/entry.o" "$O/dir_read.o")
rm -f "$OUT/share"/*.dom
for arm in test control; do
  hc=$O/hostcall.o; [[ $arm == control ]] && hc=$O/hostcall-control.o
  "$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
    -o "$OUT/share/dir-$arm.dom" "${COMMON[@]}" "$hc" "$ARCHIVE"
done

# The directories to list, in the share the guest mounts at /mnt/host.
rm -rf "$OUT/share/dirtest" "$OUT/share/manyfiles"
mkdir -p "$OUT/share/dirtest/sub" "$OUT/share/manyfiles"
for f in alpha beta gamma; do echo "$f" > "$OUT/share/dirtest/$f"; done
for i in $(seq 1 300); do : > "$OUT/share/manyfiles/file-with-a-longish-name-$i"; done

# The host: libc-test's, built from this tree, so it serves DIR_READ in both arms.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
GUEST='echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; for d in dir-test dir-control; do echo RUN-BEGIN $d; /tmp/lt.user /mnt/host/$d.dom 60; echo RUN-END $d rc=$?; done; echo __ALL_DONE__'
set +e
CAPSTONE_GP_NONLIN=${CAPSTONE_GP_NONLIN:-1} \
flock -w 3600 "$CAPSTONE_QEMU_LOCK" "$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$OUT/share" --log-file "$LOG" --timeout-multiplier 8 \
  --guest-command "$GUEST" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ >/dev/null 2>&1
set -e

arm_log() { sed -n "/RUN-BEGIN dir-$1/,/RUN-END dir-$1/p" "$LOG"; }
verdict=0
if grep -aq "DIR-READ-DONE failures=0" <(arm_log test) &&
   grep -aq "LT-RESULT dir-test.dom status=0 .*PASS" "$LOG"; then
  echo "  test:    PASS"
else
  echo "  test:    FAIL"; verdict=1
fi
arm_log test | grep -aE "^DIR-READ|halted|LT-RESULT" | sed 's/^/    /'
if grep -aq "DIR-READ FAIL list" <(arm_log control) &&
   grep -aq "DIR-READ-DONE" <(arm_log control); then
  echo "  control: FAILS the listing, as it must without getdents64"
else
  echo "  control: did NOT fail the listing -- the test cannot tell DIR_READ from its absence"; verdict=1
fi
arm_log control | grep -aE "^DIR-READ|halted|LT-RESULT" | sed 's/^/    /'
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), log $LOG"
exit $verdict
