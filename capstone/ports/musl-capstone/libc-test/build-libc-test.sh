#!/usr/bin/env bash
# Build every libc-test functional test as its own pure-capability domain.
#
# Three outcomes per test, and all three are written down, because a suite
# result that only lists what ran is a result that hides its denominator:
#
#   built      a .dom in the share dir, ready to run
#   excluded   not attempted, with the reason: the service it needs has no
#              opcode (threads, processes, sockets, SysV IPC, dynamic loading,
#              a time source), or it would wait forever for one
#   failed     attempted and did not compile or link, with the first error
#
# The manifest is the input to run-libc-test.sh's summary, so the final table
# accounts for all 77 sources and not only the ones that produced a domain.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$PORT_DIR/../../tests/capstone-test-env.sh"

REPO_ROOT=$CAPSTONE_REPO_ROOT
CLANG=${CAPSTONE_CLANG:?}
LD_LLD=${CAPSTONE_LD_LLD:?}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-libc-test}
SHARE_DIR=${SHARE_DIR:-$OUT_DIR/share}
ARCHIVE=${ARCHIVE:-$CAPSTONE_TMP_ROOT/musl-capstone-build/libc-capstone.a}
LINKER_SCRIPT="$REPO_ROOT/capstone/my_first_domain/link.ld"
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR="${CAPSTONE_BUILDROOT_DIR:?set CAPSTONE_BUILDROOT_DIR}/package/modcapstone/userspace/lib"
HOSTCALL_H_DIR="$REPO_ROOT/capstone/tests/runtime-qemu/hostcall-stdout-probe"

[ -f "$ARCHIVE" ] || { echo "no $ARCHIVE; run build-musl-capstone.sh first" >&2; exit 2; }
MUSL=$(bash "$PORT_DIR/prepare-musl-capstone.sh" | tail -1)
LT=$(bash "$SCRIPT_DIR/fetch-libc-test.sh" | tail -1)
mkdir -p "$OUT_DIR/obj" "$SHARE_DIR"
rm -f "$SHARE_DIR"/lt-*.dom "$SHARE_DIR"/lt.user
MANIFEST="$OUT_DIR/manifest.txt"; : > "$MANIFEST"

INC=(-nostdinc -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic"
     -isystem "$MUSL/obj/include" -isystem "$MUSL/include"
     -I"$MUSL/src/include" -I"$MUSL/src/internal" -I"$MUSL/obj/src/internal")
CF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
    -Xclang -target-feature -Xclang +a -ffreestanding -fno-builtin -fno-jump-tables
    -ffunction-sections -fdata-sections -std=c99 -O1 -w -Wno-int-conversion
    -D_XOPEN_SOURCE=700 "${INC[@]}")
# The tests set their own feature macros but several rely on the GNU default
# for fileno, fdopen, setenv and PATH_MAX; -std=c99 alone hides them and 19
# tests fail to compile for no reason of the port's.
TF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
    -Xclang -target-feature -Xclang +a -ffreestanding -fno-builtin -fno-jump-tables
    -ffunction-sections -fdata-sections -std=c99 -O1 -w -Wno-int-conversion
    -D_GNU_SOURCE -nostdinc -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic"
    -isystem "$MUSL/obj/include" -isystem "$MUSL/include" -I"$LT/src/common")

COMPILER_RT="$REPO_ROOT/compiler-rt/lib/builtins"
OBJ_DIR="$OUT_DIR/obj"
CLANG="$CLANG" COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
  -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
source "$REPO_ROOT/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"

# The port's runtime, once.
"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -ffreestanding -O0 -c "$PORT_DIR/runtime/start-musl.S" -o "$OBJ_DIR/start-musl.o"
"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -ffreestanding -O0 -c "$PORT_DIR/runtime/set_thread_area.S" -o "$OBJ_DIR/set_thread_area.o"
"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -ffreestanding -O0 -c "$PORT_DIR/runtime/setjmp.S" -o "$OBJ_DIR/setjmp.o"
for f in hostcall tls level0 string_bounds_safe; do
  "$CLANG" "${CF[@]}" -c "$PORT_DIR/runtime/$f.c" -o "$OBJ_DIR/$f.o"
done
"$CLANG" "${TF[@]}" -c "$SCRIPT_DIR/libc_test_domain.c" -o "$OBJ_DIR/libc_test_domain.o"
RUNTIME=("$OBJ_DIR/start-musl.o" "$OBJ_DIR/hostcall.o" "$OBJ_DIR/tls.o" "$OBJ_DIR/set_thread_area.o"
         "$OBJ_DIR/setjmp.o"
         "$OBJ_DIR/string_bounds_safe.o" "$OBJ_DIR/level0.o" "${softfloat_objs[@]}" "$OBJ_DIR/libc_test_domain.o")

# libc-test's own harness, minus runtest.c (a fork-based driver, replaced by
# the guest loop) -- everything else is what the tests link against.
COMMON=()
for f in "$LT"/src/common/*.c; do
  b=$(basename "$f" .c); [ "$b" = runtest ] && continue
  "$CLANG" "${TF[@]}" -c "$f" -o "$OBJ_DIR/common-$b.o" && COMMON+=("$OBJ_DIR/common-$b.o")
done

# Excluded with reasons. Each is a service with no opcode in HostCall v0, or a
# test that would block on one. Recorded, not skipped silently.
declare -A EXCLUDE=(
  [dlopen]="dynamic loading: no capability relocation type exists"
  [dlopen_dso]="dynamic loading: no capability relocation type exists"
  [tls_align_dlopen]="dynamic loading: no capability relocation type exists"
  [tls_init_dlopen]="dynamic loading: no capability relocation type exists"
  [pthread_cancel-points]="threads: no clone, and the join would wait forever"
  [pthread_cancel]="threads: no clone, and the join would wait forever"
  [pthread_cond]="threads: no clone, and the wait would wait forever"
  [pthread_mutex]="threads: no clone, and the wait would wait forever"
  [pthread_mutex_pi]="threads: no clone, and the wait would wait forever"
  [pthread_robust]="threads: no clone, and the wait would wait forever"
  [pthread_tsd]="threads: no clone"
  [sem_init]="threads: no clone, and the wait would wait forever"
  [sem_open]="threads and shared memory: no clone, no shm"
  [vfork]="processes: no fork or exec"
  [spawn]="processes: no fork or exec"
  [popen]="processes: no fork or exec"
  [wordexp]="processes: musl expands words by forking a shell; it hung and cost a whole boot"
  [fcntl]="processes: the test forks a child to take the other end of every lock"
  [socket]="network: no socket opcode"
  [ipc_msg]="SysV IPC: no opcode"
  [ipc_sem]="SysV IPC: no opcode"
  [ipc_shm]="SysV IPC: no opcode"
  [clock_gettime]="time: HostCall v0 has no time opcode, and the test loops until time advances"
)

built=0; excluded=0; failed=0
for f in "$LT"/src/functional/*.c; do
  b=$(basename "$f" .c)
  if [ -n "${EXCLUDE[$b]:-}" ]; then
    printf 'excluded %s %s\n' "$b" "${EXCLUDE[$b]}" >> "$MANIFEST"; excluded=$((excluded+1)); continue
  fi
  if ! "$CLANG" "${TF[@]}" -Dmain=libc_test_main -c "$f" -o "$OBJ_DIR/t-$b.o" 2>"$OBJ_DIR/t-$b.err"; then
    printf 'failed %s compile: %s\n' "$b" "$(grep -m1 -oE 'error: .{0,90}' "$OBJ_DIR/t-$b.err" | tr -d '\n')" >> "$MANIFEST"
    failed=$((failed+1)); continue
  fi
  if ! "$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$SHARE_DIR/lt-$b.dom" \
        "${RUNTIME[@]}" "$OBJ_DIR/t-$b.o" "${COMMON[@]}" "$ARCHIVE" 2>"$OBJ_DIR/t-$b.ld"; then
    printf 'failed %s link: undefined %s\n' "$b" "$(grep -oE 'undefined symbol: [A-Za-z_0-9]+' "$OBJ_DIR/t-$b.ld" | sed 's/undefined symbol: //' | sort -u | tr '\n' ' ')" >> "$MANIFEST"
    failed=$((failed+1)); rm -f "$SHARE_DIR/lt-$b.dom"; continue
  fi
  printf 'built %s\n' "$b" >> "$MANIFEST"; built=$((built+1))
done

"$GUEST_CC" -O2 -I"$SCRIPT_DIR" -I"$PORT_DIR/runtime" -I"$LIBCAPSTONE_DIR" -I"$HOSTCALL_H_DIR" \
  -I"$REPO_ROOT/capstone/tests/runtime-qemu" \
  -o "$SHARE_DIR/lt.user" "$SCRIPT_DIR/libc_test_host.c" "$LIBCAPSTONE_DIR/libcapstone.c"

printf 'libc-test at %s\n' "$(git -C "$LT" rev-parse --short HEAD)"
printf 'built %d   excluded %d   failed %d   of %d\n' "$built" "$excluded" "$failed" "$((built+excluded+failed))"
printf 'manifest %s\n' "$MANIFEST"
grep '^failed' "$MANIFEST" || true
