#!/usr/bin/env bash
# WAS DECKT .capstone_cap_init AB?
#
# A tag cannot live in an ELF image, so every pointer in static data holds its
# link-time address, untagged, on disk. The compiler synthesizes one
# __capstone_cap_init per module to turn those words into capabilities at startup,
# and the entry glue runs it before main. This is the Capstone analogue of CHERI's
# __cap_relocs plus crt_init_globals, with the difference that theirs is a data
# table a loader interprets and ours is compiled code.
#
# This probe pins BOTH halves of the current behaviour:
#   covered      function pointers, pointer arrays, string pointers
#   NOT covered  block addresses (&&label), i.e. computed-goto dispatch tables
#
# The second half is not a wish, it is what makes WAMR need
# WASM_ENABLE_LABELS_AS_VALUES=0: its 256-entry handle_table gets no initializer and
# the first dispatch jumps to a link-time address. If someone closes that gap in the
# backend, THIS SCRIPT FAILS, which is the point -- the knob can then be retired.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/capstone-test-env.sh"
CL=${CAPSTONE_CLANG:-$CAPSTONE_LLVM_BIN/clang}
NM=$CAPSTONE_LLVM_BIN/llvm-nm
OBJDUMP=$CAPSTONE_LLVM_BIN/llvm-objdump
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
       -ffreestanding -fno-builtin -nostdinc -isystem "$($CL -print-resource-dir)/include"
       -fno-optimize-sibling-calls -fno-jump-tables -ffunction-sections -fdata-sections
       -O1 -w -mllvm -capstone-gp-captable -DCAPSTONE_GP_CAPTABLE_ABI=1)

stores_in_cap_init() {  # $1 = object file; prints the store count, 0 if no initializer
  if ! "$NM" "$1" 2>/dev/null | grep -q '__capstone_cap_init'; then echo 0; return; fi
  "$OBJDUMP" -d --no-show-raw-insn "$1" 2>/dev/null \
    | sed -n '/<__capstone_cap_init>:/,/[[:space:]]ret/p' | grep -c 'stc' || true
}

# --- half 1: pointer-valued globals MUST be covered -------------------------
# External linkage on purpose: a static that is never written gets constant-folded
# away at -O1 and the probe then measures nothing, which looks like a clean result.
cat > "$TMP/ptrs.c" <<'EOF'
typedef void (*fp_t)(void);
void real_target(void);
void real_target(void) { }
fp_t g_void  = (fp_t)real_target;
fp_t g_int32 = (fp_t)real_target;
fp_t g_table[3] = { (fp_t)real_target, (fp_t)real_target, (fp_t)real_target };
const char *g_str = "hallo";
int g_plain = 7;                      /* no pointer: must get NO store */
fp_t take_void(void) { return g_void; }
EOF
"$CL" "${FLAGS[@]}" -c "$TMP/ptrs.c" -o "$TMP/ptrs.o"
got=$(stores_in_cap_init "$TMP/ptrs.o")
want=6                                # g_void, g_int32, g_table[0..2], g_str
echo "  pointer-valued globals: $got stores, expected $want"
[ "$got" = "$want" ] || { echo "FAIL: the initializer does not cover what it used to" >&2; exit 1; }

# --- half 2: block addresses are NOT covered (the known gap) ----------------
cat > "$TMP/labels.c" <<'EOF'
int dispatch(int n)
{
    static void *tbl[] = { &&one, &&two };
    goto *tbl[n & 1];
one:  return 1;
two:  return 2;
}
EOF
"$CL" "${FLAGS[@]}" -c "$TMP/labels.c" -o "$TMP/labels.o"
got=$(stores_in_cap_init "$TMP/labels.o")
echo "  block addresses:        $got stores, expected 0 (the documented gap)"
if [ "$got" != "0" ]; then
  echo "GAP CLOSED: block addresses now get initializers." >&2
  echo "Retire WASM_ENABLE_LABELS_AS_VALUES=0 in benchmarks/wamr/build-wamr-silicon.sh," >&2
  echo "re-run benchmarks/wamr/run-wamr.sh, and update this probe." >&2
  exit 1
fi

echo "__CAPSTONE_CAP_INIT_COVERAGE_AS_DOCUMENTED__"
