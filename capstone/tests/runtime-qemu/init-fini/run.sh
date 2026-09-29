#!/usr/bin/env bash
# Constructors and destructors in a musl domain (ISSUES C-64).
#
#   bash run.sh        build ctors.c twice, run both in one boot; exit 0 only if init-fini
#                      prints exactly what the same file built natively prints (constructors
#                      with and without a priority, one reading the environment the entry
#                      gives, main, destructors, in the native order),
#                      returns status 0 and does not fault, AND the control halts
#
# init-fini-control is init-fini linked against the hostcall.c of 7b10f5f86db5 (dev before
# this fix; pinned, so it stays a control after the fix lands). That runtime ran no
# constructor, and musl's exit() loaded the first .fini_array slot through an integer
# address: cause 24. A halted domain can take the guest with it, so the control runs last.
# The build also checks that link.ld leaves no constructor section outside the array
# markers (a priority section left as an orphan would silently never run).
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (clang, ld.lld, llvm-ar), CAPSTONE_BUILDROOT_DIR and
# CAPSTONE_QEMU_BINARY (the tree that has them), a host cc for the reference, and a python
# with pexpect (PYTHON=..., default python3). Takes the shared QEMU lock for the run.
# A missing status line is a FAIL, never "no result".
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/init-fini}
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
! grep -q 'hc_run_init_array\|__libc_exit_fini' "$O/hostcall-control.c" \
  || { echo "$CONTROL_REV's hostcall.c already runs the arrays; it controls nothing" >&2; exit 2; }
"$CAPSTONE_CLANG" "${RF[@]}" -c "$O/hostcall-control.c" -o "$O/hostcall-control.o"

rm -f "$OUT/share"/*.dom
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O1 -c "$HERE/ctors.c" -o "$O/ctors.o"
LDS=$REPO/capstone/my_first_domain/link.ld
"$CAPSTONE_LD_LLD" --gc-sections -T "$LDS" -o "$OUT/share/init-fini.dom" "${RUNTIME[@]}" "$O/ctors.o" "$ARCHIVE"
CONTROL_RT=("${RUNTIME[@]/#$O\/hostcall.o/$O/hostcall-control.o}")
"$CAPSTONE_LD_LLD" --gc-sections -T "$LDS" -o "$OUT/share/init-fini-control.dom" \
  "${CONTROL_RT[@]}" "$O/ctors.o" "$ARCHIVE"
orphans=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT/share/init-fini.dom" \
  | grep -oE '\.(init_array|fini_array)\.[^ ]+|\.(ctors|dtors)[^ ]*' | sort -u | tr '\n' ' ' || true)
[[ -z $orphans ]] || { echo "  FAIL: constructor sections outside link.ld's markers: $orphans"; exit 1; }
arrays=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT/share/init-fini.dom" \
  | awk '$0 ~ /\] \.(init|fini)_array / { for (i = 1; i <= NF; i++) if ($i ~ /_array$/) { print $i, $(i+4) } }' | tr '\n' ' ')
echo "  arrays in init-fini.dom (name, size): $arrays"

# The reference: the same file, natively.
cc -O1 -o "$OUT/ctors-native" "$HERE/ctors.c"
env -i INIT_FINI_ENV=before-main "$OUT/ctors-native" > "$OUT/native.txt"

# The host: libc-test's, which runs one .dom to DONE and prints its status.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
guest="echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; for d in init-fini init-fini-control; do echo RUN-BEGIN \$d; /tmp/lt.user /mnt/host/\$d.dom 60; echo RUN-END \$d rc=\$?; done; echo __ALL_DONE__"
set +e
capstone_with_qemu_lock "$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$OUT/share" --log-file "$LOG" --timeout-multiplier 8 \
  --guest-command "$guest" --success-marker __BOOT_OK__ >/dev/null 2>&1
set -e

verdict=0
block=$(tr -d '\r' < "$LOG" | sed -n '/^RUN-BEGIN init-fini$/,/^RUN-END init-fini /p')
got=$(grep -aE '^(CTOR|MAIN|DTOR) ' <<<"$block" || true)
if [[ $got == "$(cat "$OUT/native.txt")" ]] && grep -aq "LT-RESULT init-fini.dom status=0 " <<<"$block" &&
   ! grep -aq "halted by capability fault" <<<"$block"; then
  echo "  init-fini: PASS ($(tr '\n' ' ' < "$OUT/native.txt"), status 0)"
else
  echo "  init-fini: FAIL (want: $(tr '\n' ' ' < "$OUT/native.txt")status 0)"
  grep -aE "CTOR|MAIN|DTOR|LT-RESULT|halted" <<<"$block" | sed 's/^/    /'
  verdict=1
fi
block=$(tr -d '\r' < "$LOG" | sed -n '/^RUN-BEGIN init-fini-control$/,$p')
# That runtime ran no constructor, so main's line is the first, and exit() faults on the first
# .fini_array slot.
if grep -aq "halted by capability fault: cause = 24" <<<"$block" &&
   [[ $(grep -aE '^(CTOR|MAIN|DTOR) ' <<<"$block" | head -1) == "MAIN 1" ]]; then
  echo "  init-fini-control: ran no constructor (MAIN 1) and halted in exit() with cause 24, as the runtime before C-64 must"
else
  echo "  init-fini-control: did NOT show the defect -- the test cannot tell the fix from its absence"
  grep -aE "CTOR|MAIN|DTOR|LT-RESULT|halted" <<<"$block" | sed 's/^/    /'
  verdict=1
fi
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), log $LOG"
exit $verdict
