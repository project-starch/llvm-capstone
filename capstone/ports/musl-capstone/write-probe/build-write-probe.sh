#!/usr/bin/env bash
# Link musl's own write(2) into a pure-capability domain, through the hostcall.
#
# WHAT THIS PROVES AND WHAT IT DOES NOT. It proves the call chain closes: the
# C program calls write(), musl's write() reaches __syscall_cp, arch-capstone64
# turns that into __capstone_hostcall, and the hostcall implementation resolves
# it. It does NOT prove a byte reached a host -- that needs the domain to run,
# which needs a guest helper to share the two regions and service the protocol.
# Keep the two claims apart; a clean link is not a working syscall.
#
# THE NEGATIVE CONTROL IS PART OF THE BUILD, not an optional extra. A link that
# succeeds says nothing on its own: hostcall.o is on the command line either
# way, so its symbols would be present whether or not musl's write reaches them.
# So the script links a second image with a stub domain_main and NO hostcall
# implementation, and requires __capstone_hostcall to come back undefined. If it
# does not, the chain is not what this script claims and the build fails.
#
# The first attempt at this found a real defect: runtime/hostcall.c used
# syscall_arg_t without including the header that defines it. The file had never
# been compiled, so nothing had ever asked.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$PORT_DIR/../../tests/capstone-test-env.sh"

REPO_ROOT=$CAPSTONE_REPO_ROOT
CLANG=${CAPSTONE_CLANG:?}
LD_LLD=${CAPSTONE_LD_LLD:?}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-write-probe}
ARCHIVE=${ARCHIVE:-$CAPSTONE_TMP_ROOT/musl-capstone-build/libc-capstone.a}
LINKER_SCRIPT="$REPO_ROOT/capstone/my_first_domain/link.ld"
OUT_DOM=${OUT_DOM:-$OUT_DIR/write_probe.dom}
OUT_HOST=${OUT_HOST:-$OUT_DIR/write_probe.user}
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR="${CAPSTONE_BUILDROOT_DIR:?set CAPSTONE_BUILDROOT_DIR}/package/modcapstone/userspace/lib"
LIBCAPSTONE_C="$LIBCAPSTONE_DIR/libcapstone.c"
HOSTCALL_H_DIR="$REPO_ROOT/capstone/tests/runtime-qemu/hostcall-stdout-probe"
[ -f "$LIBCAPSTONE_C" ] || { echo "no $LIBCAPSTONE_C; in a worktree the buildroot submodule is empty, so point CAPSTONE_BUILDROOT_DIR at the main clone" >&2; exit 2; }

[ -f "$ARCHIVE" ] || { echo "no $ARCHIVE; run build-musl-capstone.sh first" >&2; exit 2; }
MUSL=$(bash "$PORT_DIR/prepare-musl-capstone.sh" | tail -1)
mkdir -p "$OUT_DIR"

INC=(-nostdinc -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic"
     -isystem "$MUSL/obj/include" -isystem "$MUSL/include"
     -I"$MUSL/src/include" -I"$MUSL/src/internal" -I"$MUSL/obj/src/internal")
CF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
    -Xclang -target-feature -Xclang +a -ffreestanding -fno-builtin -fno-jump-tables
    -ffunction-sections -fdata-sections -std=c99 -O1 -w -Wno-int-conversion
    # Same feature-test level the survey compiles musl with. Without it musl's
    # locale.h does not expose locale_t, and pthread_impl.h has a member of that
    # type, so any file here that reaches musl's internals fails to compile.
    -D_XOPEN_SOURCE=700
    ${MUSL_WRITE_PROBE_BADFD:+-DMUSL_WRITE_PROBE_WANT_BADFD} "${INC[@]}")

# Soft-float builtins from the shared list. A domain has no FP hardware ABI, so
# every float and double operation lowers to a compiler-rt libcall, and musl's
# vfprintf drags in the 128-bit long-double family whether or not the program
# formats one. Sourcing the BEEBS list rather than keeping a second one is the
# point: its own comment asks to be extended when a new undefined __*tf symbol
# turns up, and that is where the TF family was added.
COMPILER_RT="$REPO_ROOT/compiler-rt/lib/builtins"
OBJ_DIR="$OUT_DIR"
CLANG="$CLANG" COMMON_FLAGS=(-target capstone64-unknown-elf
  -Xclang -target-feature -Xclang +m -ffreestanding -fno-builtin
  -ffunction-sections -fdata-sections -O1 -w)
source "$REPO_ROOT/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$PORT_DIR/runtime/start-musl.S" -o "$OUT_DIR/start-musl.o"
"$CLANG" "${CF[@]}" -c "$PORT_DIR/runtime/hostcall.c"             -o "$OUT_DIR/hostcall.o"
"$CLANG" "${CF[@]}" -c "$PORT_DIR/runtime/level0.c" -o "$OUT_DIR/level0.o"
# The libc overrides, from the one list every musl domain links
# (runtime/libc_overrides.sh). A probe that never calls them pays nothing:
# --gc-sections drops what is unreachable.
source "$PORT_DIR/runtime/libc_overrides.sh"
build_musl_overrides "$CLANG" "$OUT_DIR" "$MUSL" "${CF[@]}"
"$CLANG" "${CF[@]}" -c "$PORT_DIR/runtime/tls.c"                  -o "$OUT_DIR/tls.o"
"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$PORT_DIR/runtime/set_thread_area.S" -o "$OUT_DIR/set_thread_area.o"
# hostcall.c longjmps out of exit(), so setjmp.S is part of the runtime now and
# not only of the libc-test harness. The control below counts undefined symbols,
# so a missing one shows up there first.
"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$PORT_DIR/runtime/setjmp.S" -o "$OUT_DIR/setjmp.o"
"$CLANG" "${CF[@]}" -c "$SCRIPT_DIR/write_probe_domain.c"         -o "$OUT_DIR/write_probe.o"

"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OUT_DIR/start-musl.o" "$OUT_DIR/hostcall.o" "$OUT_DIR/tls.o" \
  "$OUT_DIR/set_thread_area.o" "$OUT_DIR/setjmp.o" "${MUSL_OVERRIDE_OBJS[@]}" "$OUT_DIR/level0.o" "${softfloat_objs[@]}" "$OUT_DIR/write_probe.o" "$ARCHIVE"

# THE CONTROL. domain_main present so --gc-sections keeps the chain alive, no
# hostcall implementation, so the reference musl's write makes has nothing to
# resolve to. Anything other than __capstone_hostcall undefined means the chain
# under test is not the chain being linked above.
cat > "$OUT_DIR/stub_main.c" <<'STUB'
int capstone_main(void);
void domain_main(unsigned *res, unsigned func) { (void)func; if (res) *res = (unsigned)capstone_main(); }
/* The unserved-syscall accessors live in hostcall.c too, so the stub provides
   them. Without this the control reports three missing symbols instead of one
   and stops isolating the symbol it exists to isolate. */
unsigned long __capstone_unserved_count(void) { return 0; }
long __capstone_unserved_at(unsigned long i) { (void)i; return -1; }
STUB
"$CLANG" "${CF[@]}" -c "$OUT_DIR/stub_main.c" -o "$OUT_DIR/stub_main.o"
set +e
control=$("$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DIR/nohostcall.dom" \
  "$OUT_DIR/start-musl.o" "$OUT_DIR/stub_main.o" "$OUT_DIR/tls.o" \
  "$OUT_DIR/set_thread_area.o" "$OUT_DIR/setjmp.o" "${MUSL_OVERRIDE_OBJS[@]}" "$OUT_DIR/level0.o" "${softfloat_objs[@]}" "$OUT_DIR/write_probe.o" "$ARCHIVE" 2>&1)
set -e
undef=$(printf '%s\n' "$control" | grep -oE 'undefined symbol: [A-Za-z_][A-Za-z0-9_]*' | sed 's/undefined symbol: //' | sort -u)
if [ "$undef" != "__capstone_hostcall" ]; then
  echo "CONTROL FAILED: expected exactly '__capstone_hostcall' undefined, got: ${undef:-<none>}" >&2
  exit 1
fi

# The guest-side host: shares the two regions and services WRITE_STDOUT. Built
# with the buildroot toolchain because it runs as an ordinary Linux process
# inside the QEMU guest, not in a domain.
"$GUEST_CC" -O2 -I"$SCRIPT_DIR" -I"$LIBCAPSTONE_DIR" -I"$HOSTCALL_H_DIR" \
  -o "$OUT_HOST" "$SCRIPT_DIR/write_probe_host.c" "$LIBCAPSTONE_C"

printf 'built   %s\n' "$OUT_DOM"
printf 'built   %s\n' "$OUT_HOST"
printf 'control fired: musl write reaches __capstone_hostcall and nothing else is missing\n'
"$CAPSTONE_LLVM_BIN/llvm-nm" --defined-only "$OUT_DOM" \
  | awk '$2=="T"||$2=="t"{print $3}' | grep -xE 'write|__syscall_cp|__syscall_ret|__errno_location|__capstone_hostcall|__capstone_yield' \
  | sed 's/^/  linked: /'
