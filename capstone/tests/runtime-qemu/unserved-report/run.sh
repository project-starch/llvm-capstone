#!/usr/bin/env bash
# The runtime's unserved-syscall report from a program that closed fd 1 (ISSUES I-11).
#
#   bash run.sh        build closefd1.c twice, run both in one boot; exit 0 only if
#                      unserved-report prints "capstone-domain: UNSERVED syscalls: 160x2"
#                      (uname, twice) and returns status 0, AND the control does not print it
#
# unserved-report-control is unserved-report linked against the hostcall.c of 7b10f5f86db5
# (dev before this fix; pinned, so it stays a control after the fix lands). That runtime wrote
# its report through the program's fd 1, which the program had closed, so the report was
# refused and the line is missing. Neither image faults, so both run in one boot.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (clang, ld.lld, llvm-ar), CAPSTONE_BUILDROOT_DIR and
# CAPSTONE_QEMU_BINARY (the tree that has them), and a python with pexpect (PYTHON=...,
# default python3). Takes the shared QEMU lock for the run.
# A missing status line is a FAIL, never "no result".
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/unserved-report}
PYTHON=${PYTHON:-python3}
CONTROL_REV=7b10f5f86db5
mkdir -p "$OUT/obj" "$OUT/share"

# musl and its archive, private to this test (prepare rewrites arch/ in place).
export MUSL_CACHE_ROOT=$OUT/musl-src
mkdir -p "$MUSL_CACHE_ROOT"
[[ -f "$MUSL_CACHE_ROOT/musl-1.2.5.tar.gz" || ! -f "$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5.tar.gz" ]] \
  || cp "$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5.tar.gz" "$MUSL_CACHE_ROOT/"
MUSL=$(bash "$REPO/capstone/ports/musl-capstone/prepare-musl-capstone.sh" | tail -1)
OUT_DIR=$OUT/musl-build bash "$REPO/capstone/ports/musl-capstone/build-musl-capstone.sh" >/dev/null
ARCHIVE=$OUT/musl-build/libc-capstone.a

# The runtime, as the exit-hook test builds it (musl-capstone/libc-test/build-libc-test.sh).
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
for f in hostcall tls level0; do
  "$CAPSTONE_CLANG" "${RF[@]}" -c "$MRT/$f.c" -o "$O/$f.o"
done
source "$MRT/libc_overrides.sh"
build_musl_overrides "$CAPSTONE_CLANG" "$O" "$MUSL" "${RF[@]}"
CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$REPO/compiler-rt/lib/builtins
COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
              -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
source "$REPO/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -c "$HERE/entry.c" -o "$O/entry.o"
RUNTIME=("$O/start-musl.o" "$O/hostcall.o" "$O/tls.o" "$O/set_thread_area.o" "$O/setjmp.o"
         "${MUSL_OVERRIDE_OBJS[@]}"
         "$O/level0.o" "${softfloat_objs[@]}" "$O/entry.o")

# The control's one difference: the hostcall.c of $CONTROL_REV, whose one relative include is
# made absolute, since the copy lives in $O.
git -C "$REPO" show "$CONTROL_REV:capstone/ports/musl-capstone/runtime/hostcall.c" \
  | sed "s#\"../../../tests/runtime-qemu/#\"$REPO/capstone/tests/runtime-qemu/#" > "$O/hostcall-control.c"
grep -q 'write(1, buf, p);' "$O/hostcall-control.c" \
  || { echo "$CONTROL_REV's hostcall.c does not write the report through fd 1; it controls nothing" >&2; exit 2; }
"$CAPSTONE_CLANG" "${RF[@]}" -c "$O/hostcall-control.c" -o "$O/hostcall-control.o"

rm -f "$OUT/share"/*.dom
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O1 -c "$HERE/closefd1.c" -o "$O/closefd1.o"
LDS=$REPO/capstone/my_first_domain/link.ld
"$CAPSTONE_LD_LLD" --gc-sections -T "$LDS" -o "$OUT/share/unserved-report.dom" "${RUNTIME[@]}" "$O/closefd1.o" "$ARCHIVE"
CONTROL_RT=("${RUNTIME[@]/#$O\/hostcall.o/$O/hostcall-control.o}")
"$CAPSTONE_LD_LLD" --gc-sections -T "$LDS" -o "$OUT/share/unserved-report-control.dom" \
  "${CONTROL_RT[@]}" "$O/closefd1.o" "$ARCHIVE"

# The host: libc-test's, which runs one .dom to DONE and prints its status.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
guest="echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; for d in unserved-report unserved-report-control; do echo RUN-BEGIN \$d; /tmp/lt.user /mnt/host/\$d.dom 60; echo RUN-END \$d rc=\$?; done; echo __ALL_DONE__"
set +e
capstone_with_qemu_lock "$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$OUT/share" --log-file "$LOG" --timeout-multiplier 8 \
  --guest-command "$guest" --success-marker __BOOT_OK__ >/dev/null 2>&1
set -e

verdict=0
for arm in unserved-report:1 unserved-report-control:0; do
  name=${arm%%:*}; want=${arm##*:}
  block=$(tr -d '\r' < "$LOG" | sed -n "/^RUN-BEGIN $name\$/,/^RUN-END $name /p")
  n=$(grep -ac '^capstone-domain: UNSERVED syscalls: 160x2$' <<<"$block" || true)
  if grep -aq "UNSERVED-TEST two uname calls" <<<"$block" && grep -aq "LT-RESULT $name.dom status=0 " <<<"$block" &&
     [[ $n == "$want" ]]; then
    [[ $want == 1 ]] && echo "  $name: PASS (the report survives the closed fd 1)" \
                     || echo "  $name: the report is missing, as the runtime before I-11 must lose it"
  else
    echo "  $name: FAIL (want $want report line(s), got $n; or no status line)"
    grep -aE "UNSERVED|LT-RESULT|halted" <<<"$block" | sed 's/^/    /'
    verdict=1
  fi
done
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), log $LOG"
exit $verdict
