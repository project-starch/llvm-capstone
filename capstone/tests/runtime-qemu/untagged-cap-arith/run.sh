#!/usr/bin/env bash
# CINCOFFSET and SCC on an untagged value (arith_test.c), in two boots, because a domain that traps
# ends the guest session: case 0 (the control, both on a real capability) then case 1 (cincoffset on
# an integer) in the first, case 2 (scc on an integer) alone in the second -- where a QEMU without
# the scc fix aborts the whole machine instead of raising.
#
#   bash run.sh            expects today's ISA: cases 1 and 2 raise Unexpected operand type (24)
#   EXPECT=cheri bash run.sh   expects CHERI's rule: they print 0x5008 and 0x6000
#
#   exit 0 if case 0 stores through both results and cases 1 and 2 do what EXPECT says; 1 if not;
#   2 if it could not check (no result from case 0, or the boot ended early).
#
# Needs: CAPSTONE_LLVM_BUILD_DIR, CAPSTONE_BUILDROOT_DIR, CAPSTONE_QEMU_BINARY, and a python with
# pexpect (PYTHON=..., default python3). Takes the QEMU lock unless CAPSTONE_QEMU_LOCK_HELD=1.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/untagged-cap-arith}
EXPECT=${EXPECT:-trap}
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
rm -f "$OUT/share"/*.dom
for c in 0 1 2; do
  "$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O2 -DCASE=$c -c "$HERE/arith_test.c" -o "$O/arith-$c.o"
  "$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
    -o "$OUT/share/arith-$c.dom" "$O/start-musl.o" "$O/hostcall.o" "$O/tls.o" \
    "$O/set_thread_area.o" "$O/setjmp.o" "${MUSL_OVERRIDE_OBJS[@]}" \
    "$O/atomic_libcalls.o" "$O/level0.o" \
    "${softfloat_objs[@]}" "$O/entry.o" "$O/arith-$c.o" "$ARCHIVE"
done

# The host: libc-test's, which runs one .dom to DONE and prints its status.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

lock=()
[[ ${CAPSTONE_QEMU_LOCK_HELD:-0} == 1 ]] || lock=(flock -w 21600 "$CAPSTONE_QEMU_LOCK")
boot() { # boot <cases...>; one guest session running those domains in order; prints its log path
  local log=$OUT/run-$(printf '%s' "$@")-$(date +%Y%m%d-%H%M%S).log
  local guest="echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; for c in $*; do echo RUN-BEGIN \$c; /tmp/lt.user /mnt/host/arith-\$c.dom 60; echo RUN-END \$c rc=\$?; done; echo __ALL_DONE__"
  set +e
  "${lock[@]}" "$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
    --share-dir "$OUT/share" --log-file "$log" --timeout-multiplier 8 \
    --guest-command "$guest" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ >/dev/null 2>&1
  set -e
  echo "$log"
}
LOG1=$(boot 0 1)
LOG2=$(boot 2)
# A case's block runs from its RUN-BEGIN to its RUN-END or, when its trap ends the session, to the
# end of the log.
block() { local log=$LOG1; [[ $1 == 2 ]] && log=$LOG2; sed -n "/RUN-BEGIN $1\r\{0,1\}\$/,/RUN-END $1 /p" "$log"; }
verdict=0
b0=$(block 0)
if ! grep -aq 'ARITH-CASE 0 ' <<<"$b0"; then
  echo "  case 0: NO RESULT -- nothing below would mean anything; log $LOG1"; exit 2
fi
if grep -aq 'ARITH-CASE 0 cinc=1 scc=1' <<<"$b0"; then
  echo "  case 0: both instructions work on a capability (control)"
else
  echo "  case 0: FAIL: $(grep -ao 'ARITH-CASE 0 .*' <<<"$b0")"; verdict=1
fi
for c in 1 2; do
  b=$(block $c)
  want=$([[ $EXPECT == cheri ]] && { [[ $c == 1 ]] && echo 'result=0x5008' || echo 'result=0x6000'; } || echo 'cause = 24')
  if [[ -z $b ]] || { ! grep -aq "RUN-END $c " <<<"$b" && ! grep -aq 'halted by capability fault' <<<"$b"; }; then
    echo "  case $c: NO RESULT -- neither a result nor a trap: the machine stopped (a QEMU whose scc"
    echo "          still asserts aborts at case 2)"
    exit 2
  elif grep -aq "$want" <<<"$b"; then
    echo "  case $c: $want, as $EXPECT expects"
  else
    echo "  case $c: FAIL, wanted '$want': $(grep -aoE 'ARITH-CASE [0-9] .*|halted by capability fault: cause = [0-9]+' <<<"$b" | head -1)"
    verdict=1
  fi
done
echo "  qemu $CAPSTONE_QEMU_BINARY, logs $LOG1 $LOG2"
exit $verdict
