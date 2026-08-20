#!/usr/bin/env bash
# Build the resumable-yield probe: a pure-capability .dom plus its Linux host.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

REPO_ROOT=$CAPSTONE_REPO_ROOT
CLANG=${CAPSTONE_CLANG:?}
LD_LLD=${CAPSTONE_LD_LLD:?}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-yield-probe}
OBJ_DIR="$OUT_DIR/obj"
OUT_DOM=${OUT_DOM:-$OUT_DIR/yield_probe.dom}
OUT_HOST=${OUT_HOST:-$OUT_DIR/yield_probe.user}

START_SRC="$SCRIPT_DIR/../runtime/start-musl.S"
LINKER_SCRIPT="$REPO_ROOT/capstone/my_first_domain/link.ld"
# YIELD_PROBE_GPCT=1 builds the SAME probe against the gp-captable glue instead of
# start-musl.S. The point of the switch is that everything else -- probe source, host
# program, run script, success markers -- stays identical, so a difference in the result
# is a difference between the two glues and nothing else. See
# agent-handoff/plans/20-08-2026_mruby-gp-captable.md.
GPCT=${YIELD_PROBE_GPCT:-0}
LADDER="$REPO_ROOT/capstone/tests/runtime-qemu/silicon-ladder"
GPFREE="$REPO_ROOT/capstone/tests/runtime-qemu/gp-free-domain"
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_C="$REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"

mkdir -p "$OBJ_DIR"

if [[ "$GPCT" != "0" ]]; then
  # gp-captable: globals are reached through a cap table built at entry, so the domain
  # object must be compiled with -capstone-gp-captable and the image linked twice -- pass 1
  # only measures .text so pass 2 can place the globals region above it.
  "$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
    -ffreestanding -DCAPSTONE_GLUE_YIELD=1 \
    -c "$LADDER/start-gp-captable-interp.S" -o "$OBJ_DIR/start-musl.o"
  "$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
    -ffreestanding -c "$LADDER/../gct-section-end.S" -o "$OBJ_DIR/gct.o"
  "$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
    -ffreestanding -fno-builtin -fno-jump-tables -fno-optimize-sibling-calls \
    -mllvm -capstone-gp-captable \
    -O1 -c "$SCRIPT_DIR/yield_probe_domain.c" -o "$OBJ_DIR/yield_probe_domain.o"

  gpct_link() {  # $1 = globals offset literal, $2 = output
    sed "s/0x10000 + 0x1000/0x10000 + $1/" "$GPFREE/link-gpfree.ld" > "$OBJ_DIR/link.ld"
    "$LD_LLD" -T "$OBJ_DIR/link.ld" -o "$2" \
      "$OBJ_DIR/start-musl.o" "$OBJ_DIR/yield_probe_domain.o" "$OBJ_DIR/gct.o"
  }
  gpct_link 0x800000 "$OBJ_DIR/pass1.dom"
  TEXT=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OBJ_DIR/pass1.dom" 2>/dev/null | python3 -c '
import sys,re
for l in sys.stdin:
    m=re.match(r"\s*\[\s*\d+\]\s+(\.text)\s+\S+\s+[0-9a-f]+\s+[0-9a-f]+\s+([0-9a-f]+)", l)
    if m: print(int(m.group(2),16)); break
else: print(0)')
  [[ "${TEXT:-0}" -gt 0 ]] || { echo "could not measure .text from pass 1" >&2; exit 1; }
  GOFF=$(( ((TEXT + 0xFFFF) / 0x10000) * 0x10000 ))
  [[ $GOFF -lt 65536 ]] && GOFF=65536
  printf '   .text = %d bytes -> globals offset 0x%x\n' "$TEXT" "$GOFF"
  gpct_link "$(printf '0x%x' $GOFF)" "$OUT_DOM"

  # Same gate the interpreter builds use: a cjalr means a call went through gp as a
  # capability, i.e. the image is not gp-free and the glue's assumptions do not hold.
  NCJALR=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT_DOM" | grep -cE '\bcjalr\b' || true)
  echo "   cjalr=$NCJALR (must be 0)"
  [[ "$NCJALR" == "0" ]] || { echo "FAIL: cjalr present (not gp-free)" >&2; exit 1; }
else
"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start-musl.o"

# start-musl.S asks for __capstone_dom_data and __capstone_init_tp, which
# libc-capstone.a normally provides. This probe links no archive at all, so it
# links the stubs instead. Leaving them out fails the link naming both symbols,
# which is the intended failure -- see the note in start-musl.S about why a weak
# definition inside the glue would be silently wrong instead.
"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$SCRIPT_DIR/../runtime/no-libc-stubs.c" \
  -o "$OBJ_DIR/no-libc-stubs.o"

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -fno-builtin -fno-jump-tables -fno-optimize-sibling-calls \
  -ffunction-sections -fdata-sections \
  -O1 -c "$SCRIPT_DIR/yield_probe_domain.c" -o "$OBJ_DIR/yield_probe_domain.o"

"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start-musl.o" "$OBJ_DIR/no-libc-stubs.o" "$OBJ_DIR/yield_probe_domain.o"

fi

"$GUEST_CC" -O2 -o "$OUT_HOST" "$SCRIPT_DIR/yield_probe_host.c" "$LIBCAPSTONE_C"

printf 'built %s\nbuilt %s\n' "$OUT_DOM" "$OUT_HOST"
