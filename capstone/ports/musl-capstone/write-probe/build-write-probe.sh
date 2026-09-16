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

[ -f "$ARCHIVE" ] || { echo "no $ARCHIVE; run build-musl-capstone.sh first" >&2; exit 2; }
MUSL=$(bash "$PORT_DIR/prepare-musl-capstone.sh" | tail -1)
mkdir -p "$OUT_DIR"

INC=(-nostdinc -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic"
     -isystem "$MUSL/obj/include" -isystem "$MUSL/include"
     -I"$MUSL/src/include" -I"$MUSL/src/internal" -I"$MUSL/obj/src/internal")
CF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
    -Xclang -target-feature -Xclang +a -ffreestanding -fno-builtin -fno-jump-tables
    -ffunction-sections -fdata-sections -std=c99 -O1 -w "${INC[@]}")

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$PORT_DIR/runtime/start-musl.S" -o "$OUT_DIR/start-musl.o"
"$CLANG" "${CF[@]}" -c "$PORT_DIR/runtime/hostcall.c"             -o "$OUT_DIR/hostcall.o"
"$CLANG" "${CF[@]}" -c "$SCRIPT_DIR/write_probe_domain.c"         -o "$OUT_DIR/write_probe.o"

"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DIR/write_probe.dom" \
  "$OUT_DIR/start-musl.o" "$OUT_DIR/hostcall.o" "$OUT_DIR/write_probe.o" "$ARCHIVE"

# THE CONTROL. domain_main present so --gc-sections keeps the chain alive, no
# hostcall implementation, so the reference musl's write makes has nothing to
# resolve to. Anything other than __capstone_hostcall undefined means the chain
# under test is not the chain being linked above.
cat > "$OUT_DIR/stub_main.c" <<'STUB'
int capstone_main(void);
void domain_main(unsigned *res, unsigned func) { (void)func; if (res) *res = (unsigned)capstone_main(); }
STUB
"$CLANG" "${CF[@]}" -c "$OUT_DIR/stub_main.c" -o "$OUT_DIR/stub_main.o"
set +e
control=$("$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DIR/nohostcall.dom" \
  "$OUT_DIR/start-musl.o" "$OUT_DIR/stub_main.o" "$OUT_DIR/write_probe.o" "$ARCHIVE" 2>&1)
set -e
undef=$(printf '%s\n' "$control" | grep -oE 'undefined symbol: [A-Za-z_][A-Za-z0-9_]*' | sed 's/undefined symbol: //' | sort -u)
if [ "$undef" != "__capstone_hostcall" ]; then
  echo "CONTROL FAILED: expected exactly '__capstone_hostcall' undefined, got: ${undef:-<none>}" >&2
  exit 1
fi

printf 'built   %s\n' "$OUT_DIR/write_probe.dom"
printf 'control fired: musl write reaches __capstone_hostcall and nothing else is missing\n'
"$CAPSTONE_LLVM_BIN/llvm-nm" --defined-only "$OUT_DIR/write_probe.dom" \
  | awk '$2=="T"||$2=="t"{print $3}' | grep -xE 'write|__syscall_cp|__syscall_ret|__errno_location|__capstone_hostcall|__capstone_yield' \
  | sed 's/^/  linked: /'
