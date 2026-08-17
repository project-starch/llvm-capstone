#!/usr/bin/env bash
# Build the MicroPython interpreter as a Capstone domain (.dom).
#
# The whole py/ core, the port and the freestanding support are compiled as ONE translation
# unit. That is not a size optimisation: `getGpCaptableIndex` numbers globals PER MODULE, so a
# multi-TU domain emits several descriptor headers whose indices collide on the single gp cap
# table. The link-time gate below asserts exactly one header for that reason.
#
# Two-pass link, copied from build-sqlite-silicon.sh: the globals boundary in link-gpfree.ld is a
# fixed offset, so pass 1 links at a deliberately oversized offset only to MEASURE .text, and
# pass 2 links at the real one. MicroPython's .text is ~320 KiB, so the 0x1000 default cannot
# work -- the globals region would land inside the code.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

CAPSTONE_TMP_ROOT=${CAPSTONE_TMP_ROOT:-/tmp/capstone}
MPY_SRC_DIR=${MPY_SRC_DIR:-$CAPSTONE_TMP_ROOT/micropython}
LADDER="$REPO_ROOT/capstone/tests/runtime-qemu/silicon-ladder"
GPFREE="$REPO_ROOT/capstone/tests/runtime-qemu/gp-free-domain"
BEEBS_STRING="$REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
PORT="$SCRIPT_DIR/port"
SHIM="$SCRIPT_DIR/adapted/include"

# MPY_STAGE bisects startup: each value returns a marker from a different point instead of
# running on. Build one .dom per stage and run them ALL IN ONE BOOT, ascending, with a
# known-good control first -- a wedge takes the rest of the boot with it, so the first stage
# that fails to return IS the bisection point.
MPY_STAGE=${MPY_STAGE:-}
# MPY_TESTS=<n>|all builds the TEST RUNNER instead of a single baked-in program: the domain holds
# a table of MicroPython's own tests and runs the next one on each call, so the loader's existing
# `capstone-test.user <dom> <times>` runs n tests in ONE boot. One boot per test does not scale --
# basics/ alone is 576 files.
MPY_TESTS=${MPY_TESTS:-}
# Zero-based offset into the sorted candidate list. This permits a suite that is too large for
# one domain image to be covered by several independently runnable chunks.
MPY_TEST_OFFSET=${MPY_TEST_OFFSET:-0}
# Additional direct children of tests/, space-separated. The default remains basics-only.
MPY_TEST_BASE_DIR=${MPY_TEST_BASE_DIR:-basics}
MPY_TEST_DIRS=${MPY_TEST_DIRS:-}
DOM_NAME=${DOM_NAME:-micropython}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/micropython-silicon}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj}
mkdir -p "$OUT_DIR" "$OBJ_DIR"

CLANG=${CAPSTONE_CLANG}
LD_LLD=${CAPSTONE_LD_LLD}

[[ -d $MPY_SRC_DIR/py ]] || { echo "no MicroPython at $MPY_SRC_DIR -- run fetch-micropython.sh" >&2; exit 1; }
# The qstr / module / root-pointer headers are generated FROM mpconfigport.h, so they are
# port-specific and cannot be borrowed from ports/minimal: MICROPY_STACK_CHECK alone pulls in a
# qstr that port does not define, and a mismatched string pool is an error that compiles. So the
# port is mirrored into the MicroPython tree and MicroPython's own machinery generates them, with
# a STOCK host compiler -- none of this reaches the domain, only the headers do.
MPY_PORT_DIR="$MPY_SRC_DIR/ports/capstone"
echo "== generating this port's headers (stock toolchain, host only)"
rm -rf "$MPY_PORT_DIR"
cp -r "$PORT" "$MPY_PORT_DIR"
make -C "$MPY_PORT_DIR" -j"${HDR_JOBS:-8}" \
    CFLAGS_EXTRA="${DOMAIN_EXTRA_DEFS:-}" >"$OBJ_DIR/genhdr.log" 2>&1 || {
  echo "header generation failed; see $OBJ_DIR/genhdr.log" >&2; tail -5 "$OBJ_DIR/genhdr.log" >&2; exit 1; }
GEN_DIR="$MPY_PORT_DIR/build"
[[ -f $GEN_DIR/genhdr/qstrdefs.generated.h ]] || {
  echo "header generation produced no qstrdefs.generated.h" >&2; exit 1; }

# The GC patches are not optional: without them the interpreter builds and then loses a capability
# tag at its first collection (PTR_FROM_BLOCK) and at gc_init. Applied by fetch-micropython.sh;
# check rather than assume, because a re-fetch or a `git checkout` in the source tree drops them.
grep -q "gc_misalign" "$MPY_SRC_DIR/py/gc.c" || {
  echo "py/gc.c is missing the capability patches -- re-run fetch-micropython.sh" >&2; exit 1; }

SILICON=(-mllvm -capstone-gp-captable
         -mllvm -capstone-shrink-stack=false
         -mllvm -capstone-shrink-globals=false
         # String merging is load-bearing here, not an optimisation: with it the domain has 232
         # carves, without it 633. SQLite runs with 179, so the smaller number is the one inside
         # anything this project has demonstrated.
         -mllvm -capstone-merge-string-constants=true
         -DCAPSTONE_GP_CAPTABLE_ABI=1
         ${MPY_STAGE:+-DMPY_STAGE=$MPY_STAGE}
         ${MPY_TESTS:+-DMPY_TEST_RUNNER}
         # Extra -D flags for one build, word-split on purpose. Without this a
         # parameterised probe silently builds the DEFAULT value for every arm and the whole
         # sweep measures one thing N times -- caught here by hashing the images, not by the
         # exit code, which was 0 for all of them.
         ${DOMAIN_EXTRA_DEFS:-})
COMMON=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
        -ffreestanding
        # -nostdlibinc, or clang searches /usr/include even for a bare-metal triple and
        # <string.h> silently resolves to the HOST glibc header instead of adapted/include.
        -nostdlibinc
        -fno-builtin -fno-optimize-sibling-calls
        # -fno-jump-tables: a dense switch otherwise lowers to a table of code addresses in
        # .rodata plus an indirect jump, and under gp-captable .rodata is not reachable as plain
        # data. MicroPython's bytecode dispatch is exactly that shape.
        -fno-jump-tables
        -std=c99 -O0 -w
        -I"$SHIM" -I"$MPY_PORT_DIR" -I"$MPY_SRC_DIR" -I"$GEN_DIR" -I"$OBJ_DIR")

if [[ -n $MPY_TESTS ]]; then
  echo "== generating the test table"
  TEST_GEN_EXTRA=()
  for test_dir in $MPY_TEST_DIRS; do
    TEST_GEN_EXTRA+=(--add-tests-dir "$MPY_SRC_DIR/tests/$test_dir")
  done
  if [[ -n ${MPY_TEST_INCLUDE_UNSUPPORTED:-} ]]; then
    TEST_GEN_EXTRA+=(--include-unsupported)
  fi
  if [[ -n ${MPY_TEST_EXPECT_TIMEOUT:-} ]]; then
    TEST_GEN_EXTRA+=(--expect-timeout "$MPY_TEST_EXPECT_TIMEOUT")
  fi
  python3 "$SCRIPT_DIR/tools/gen-test-table.py" "$MPY_SRC_DIR/tests/$MPY_TEST_BASE_DIR" \
      "$OBJ_DIR/mpy_tests.h" ${MPY_TESTS:+--limit ${MPY_TESTS/all/0}} \
      --offset "$MPY_TEST_OFFSET" \
      "${TEST_GEN_EXTRA[@]}"
fi

echo "== amalgamating py/ + the port into one translation unit"
AMALGAM="$OBJ_DIR/mpy_all.c"
{
  # <stdarg.h> first: without it py/objexcept.c re-declares mp_obj_new_exception_msg_vlist with a
  # different va_list than the header saw, which is the single fix the amalgamation needs.
  echo '#include <stdarg.h>'
  for f in "$MPY_SRC_DIR"/py/*.c; do
    echo "#include \"$f\""
    if [[ $f == */malloc.c ]]; then
      # malloc.c redirects these names only for its own module. A normal multi-TU build drops the
      # macros at EOF; the amalgamation must reproduce that boundary explicitly.
      echo '#undef malloc'
      echo '#undef malloc_with_finaliser'
      echo '#undef free'
      echo '#undef realloc'
      echo '#undef realloc_ext'
    fi
  done
  echo "#include \"$MPY_PORT_DIR/mpy_domain.c\""
} > "$AMALGAM"

echo "== compiling (this is the whole interpreter in one go)"
"$CLANG" "${COMMON[@]}" "${SILICON[@]}" -c "$AMALGAM" -o "$OBJ_DIR/mpy.o"

# The freestanding string/memory functions the core needs. BEEBS_STRING_LINEAR_SAFE keeps the
# primitives from advancing a capability that may be LINEAR; see the comment on strlen in that file.
"$CLANG" "${COMMON[@]}" "${SILICON[@]}" -DBEEBS_STRING_LINEAR_SAFE=1 \
  -c "$BEEBS_STRING" -o "$OBJ_DIR/beebs_string.o"

# setjmp/longjmp. MICROPY_NLR_SETJMP=1 makes these MicroPython's exception mechanism, so this is
# not a stub: it is the capability-aware pair proven as the nlrjmp ladder rung. ra and sp are
# capabilities, so every slot is 16 bytes and saved with stc/ldc.
"$CLANG" "${COMMON[@]}" "${SILICON[@]}" -DCJ_DEFINE_SETJMP=1 \
  -c "$MPY_PORT_DIR/capstone_setjmp.c" -o "$OBJ_DIR/setjmp.o"

# strncmp/strchr: the two the core needs that beebs_freestanding_string.c does not define.
# Kept out of that file because it is shared with every BEEBS rung and the SQLite domain, whose
# published numbers were measured on those exact artifacts.
"$CLANG" "${COMMON[@]}" "${SILICON[@]}" \
  -c "$MPY_PORT_DIR/capstone_str_extra.c" -o "$OBJ_DIR/str_extra.o"

"$CLANG" -target capstone64-unknown-elf -ffreestanding \
  -c "$LADDER/../gct-section-end.S" -o "$OBJ_DIR/gct.o"

link() {  # $1 = globals offset literal, $2 = output
  local lds="$OBJ_DIR/link.ld"
  sed "s/0x10000 + 0x1000/0x10000 + $1/" "$GPFREE/link-gpfree.ld" > "$lds"
  "$CLANG" -target capstone64-unknown-elf -ffreestanding \
    ${INTERP_EXTRA_CFLAGS:-} \
    -c "$LADDER/start-gp-captable-interp.S" -o "$OBJ_DIR/start.o"
  "$LD_LLD" -T "$lds" -o "$2" \
    "$OBJ_DIR/start.o" "$OBJ_DIR/mpy.o" "$OBJ_DIR/beebs_string.o" \
    "$OBJ_DIR/setjmp.o" "$OBJ_DIR/str_extra.o" "$OBJ_DIR/gct.o"
}

echo "== pass 1: link at a provisional 8 MiB offset, only to measure .text"
link 0x800000 "$OUT_DIR/pass1.dom"
TEXT=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT_DIR/pass1.dom" 2>/dev/null | python3 -c '
import sys,re
for l in sys.stdin:
    m=re.match(r"\s*\[\s*\d+\]\s+(\.text)\s+\S+\s+[0-9a-f]+\s+[0-9a-f]+\s+([0-9a-f]+)", l)
    if m: print(int(m.group(2),16)); break
else: print(0)')
: "${TEXT:=0}"
[[ "$TEXT" -gt 0 ]] || { echo "could not measure .text from pass 1" >&2; exit 1; }
GOFF=$(( ((TEXT + 0xFFFF) / 0x10000) * 0x10000 ))
[[ $GOFF -lt 65536 ]] && GOFF=65536
printf "   .text = %d bytes -> globals offset 0x%x\n" "$TEXT" "$GOFF"

echo "== pass 2: link with the real globals offset"
link "$(printf '0x%x' $GOFF)" "$OUT_DIR/$DOM_NAME.dom"

echo "== gates"
DIS=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT_DIR/$DOM_NAME.dom")
NCJALR=$(grep -cE '\bcjalr\b' <<<"$DIS" || true)
NLDCGP=$(grep -cE 'ldc[[:space:]].*\(gp\)' <<<"$DIS" || true)
NCINCGP=$(grep -cE 'cincoffset[[:space:]]+[a-z0-9]+,[[:space:]]*gp,' <<<"$DIS" || true)
echo "   cjalr=$NCJALR  ldc-gp=$NLDCGP  cincoffset-gp=$NCINCGP"
[[ "$NCJALR" == "0" ]] || { echo "FAIL: cjalr present (not gp-free)" >&2; exit 1; }
[[ "$NLDCGP" -ge 1 || "$NCINCGP" -ge 1 ]] || { echo "FAIL: no gp[i] global access" >&2; exit 1; }

NHDR=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT_DIR/$DOM_NAME.dom" | grep -c "capstone_gp_table" || true)
echo "   .capstone_gp_table sections: $NHDR (must be 1)"
[[ "$NHDR" == "1" ]] || { echo "FAIL: expected exactly one gp-table header, got $NHDR" >&2; exit 1; }

python3 "$LADDER/domdata-budget.py" "$OUT_DIR/$DOM_NAME.dom" || true
echo "Built $OUT_DIR/$DOM_NAME.dom"
