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
#   covered      function pointers, pointer arrays, string pointers, aliases,
#                and block addresses (&&label), i.e. computed-goto dispatch tables
#   never        null slots and absolute addresses, which cannot carry a tag
#
# Block addresses were NOT covered until 2026-08-27, and that omission is what made
# WAMR's interpreter jump to a link-time address on its first dispatch: 224 live
# slots in one table, all silently untagged. Every arm here fails LOUDLY, because
# the failure it guards against is silent at build time and only shows up as a
# capability fault far from its cause.
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

# --- half 2: block addresses, the case that used to be missed ----------------
# A computed-goto dispatch table is an array of these. They were skipped because
# needsMaterialization() accepted only GlobalVariable and Function, so the table
# kept its link-time bytes and the first `goto *tbl[i]` left the image. Found the
# expensive way: WAMR's interpreter has 224 live slots in one table.
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
want=2
echo "  block addresses:        $got stores, expected $want"
[ "$got" = "$want" ] || {
  echo "FAIL: block addresses are no longer materialized." >&2
  echo "That regression is SILENT at build time and shows up as a jump to a" >&2
  echo "link-time address at run time. See llvm/test/CodeGen/Capstone/" >&2
  echo "static-cap-global-init-blockaddress.ll and CapstoneCapGlobalInit.cpp." >&2
  exit 1
}

# --- half 3: what must still NOT get a store --------------------------------
# A null slot needs no tag, and an absolute address cannot carry one. If either
# started producing stores, the pass would be writing over a deliberate value.
# MP_ROM_INT in MicroPython is the inttoptr case and is why this arm exists.
cat > "$TMP/none.c" <<'EOF'
typedef void (*fp_t)(void);
fp_t g_null = 0;
fp_t g_abs  = (fp_t)0x1234;
int  g_int  = 7;
EOF
"$CL" "${FLAGS[@]}" -c "$TMP/none.c" -o "$TMP/none.o"
got=$(stores_in_cap_init "$TMP/none.o")
echo "  null / absolute / int:  $got stores, expected 0"
[ "$got" = "0" ] || { echo "FAIL: the pass is materializing something it must not" >&2; exit 1; }

echo "__CAPSTONE_CAP_INIT_COVERAGE_AS_DOCUMENTED__"
