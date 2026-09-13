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
BEEBS_SOFTFLOAT="$REPO_ROOT/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
COMPILER_RT="$REPO_ROOT/compiler-rt/lib/builtins"
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
# MPY_VFS=1 adds the filesystem stack: extmod/vfs*.c (listed in port/Makefile, so header
# generation and the amalgam cannot disagree) plus lib/oofatfs. Needed only by MPY-T14 and
# MPY-T15, whose reproductions define their block device in PYTHON, so this needs no host
# filesystem and no device -- the earlier note calling the block device a design question
# was reading the issues rather than their PoCs.
MPY_VFS=${MPY_VFS:-}
# lib/ sources outside libm had no mechanism before this. Same shape as LIBM_UNITS: a name
# list, an #include per unit into the amalgam.
OOFATFS_UNITS=''
if [[ -n $MPY_VFS ]]; then
  OOFATFS_UNITS='ff ffunicode'
fi
MPY_TEST_DIRS=${MPY_TEST_DIRS:-}
DOM_NAME=${DOM_NAME:-micropython}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/micropython-silicon}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj}
mkdir -p "$OUT_DIR" "$OBJ_DIR"

# MPY_FLOAT_CORE=1 gives the interpreter float OBJECTS: literals, arithmetic, repr, struct.
# MPY_FLOAT_MATH=1 adds the math and cmath modules and complex on top, which is a much bigger
# libm (see LIBM_UNITS below) and only worth linking when those modules are actually wanted.
FLOAT_DEFS=()
LIBM_UNITS=''
if [[ -n ${MPY_FLOAT_CORE:-} ]]; then
  FLOAT_DEFS=(-DMICROPY_FLOAT_IMPL=2
              -DMICROPY_FLOAT_FORMAT_IMPL=1
              -DMPY_CSTACK_MAX=393216)
  # No exp/log here: only math, cmath and objcomplex call them, and all three are off without
  # MPY_FLOAT_MATH. pow.c is self-contained (fdlibm), needing only sqrt, scalbn and fabs.
  LIBM_UNITS='fmod copysign rint nearbyint scalbn pow floor sqrt'
  if [[ -n ${MPY_FLOAT_MATH:-} ]]; then
    FLOAT_DEFS+=(-DMICROPY_PY_BUILTINS_COMPLEX=1
                 -DMICROPY_PY_MATH=1
                 -DMICROPY_PY_CMATH=1
                 -DMICROPY_PY_MATH_CONSTANTS=1)
    # Everything libm_dbl ships, minus the ARM-only sqrt and __fpclassify.c -- the latter needs
    # FP_NAN and friends, which the freestanding <math.h> shim does not define and nothing calls.
    LIBM_UNITS=$(cd "$MPY_SRC_DIR/lib/libm_dbl" && ls ./*.c |
                 sed 's#^\./##; s#\.c$##' | grep -vx 'thumb_vfp_sqrt\|__fpclassify' | tr '\n' ' ')
  else
    FLOAT_DEFS+=(-DMICROPY_PY_BUILTINS_COMPLEX=0
                 -DMICROPY_PY_MATH=0
                 -DMICROPY_PY_CMATH=0
                 -DMICROPY_PY_MATH_CONSTANTS=0)
  fi
fi

# Upstream compiles each libm unit separately; amalgamated into this port's single translation
# unit their fdlibm polynomial constants collide by name. Rename every colliding file static per
# unit, so the include order does not matter. The list is clang's own, from the redefinition
# errors it reports for the whole directory under -ferror-limit=0; a MicroPython bump that adds a
# collision therefore fails loudly at compile time rather than silently picking one copy.
declare -A LIBM_STATICS=(
  [__rem_pio2]='toint' [ceil]='toint' [floor]='toint' [rint]='toint' [round]='toint'
  [acos]='R pS0 pS1 pS2 pS3 pS4 pS5 pio2_hi pio2_lo qS1 qS2 qS3 qS4'
  [asin]='R pS0 pS1 pS2 pS3 pS4 pS5 pio2_hi pio2_lo qS1 qS2 qS3 qS4'
  [atan2]='pi' [tgamma]='pi'
  [exp]='P1 P2 P3 P4 P5 invln2' [expm1]='invln2 ln2_hi ln2_lo'
  [log]='Lg1 Lg2 Lg3 Lg4 Lg5 Lg6 Lg7 ln2_hi ln2_lo'
  [log1p]='Lg1 Lg2 Lg3 Lg4 Lg5 Lg6 Lg7 ln2_hi ln2_lo'
  [pow]='P1 P2 P3 P4 P5 tiny' [sqrt]='tiny'
  # __expo2's k against the sha256 round-constant table modhashlib.c pulls in.
  [__expo2]='k')

# Same problem, same fix, for the extmod units: modbinascii defines its own bytes_fromhex_obj
# next to py/objstr.c's.
# WORD is not a static, it is a TYPEDEF, and it collides the same way: sha256.h says
# `typedef unsigned int WORD` while lib/oofatfs/ff.h says `typedef uint16_t WORD`, and the
# amalgamation puts both in one translation unit. Renaming sha256's copy over the span of
# modhashlib.c is the same mechanism and keeps both modules, where dropping hashlib left a
# dangling mp_module_hashlib in the module table that genhdr had already emitted.
declare -A EXTMOD_STATICS=([modbinascii]='bytes_fromhex_obj' [modhashlib]='WORD')

# Single source of truth for the extmod modules this port carries; port/Makefile reads it from
# here (EXTMOD_SRC_C) so header generation and the amalgam can never disagree about which
# MP_REGISTER_MODULE entries exist.
EXTMOD_UNITS=$(sed -n 's#^[[:space:]]\{1,\}extmod/\([a-z0-9_]\{1,\}\)\.c.*#\1#p' \
               "$PORT/Makefile" | tr '\n' ' ')
[[ -n ${EXTMOD_UNITS// /} ]] || { echo "no EXTMOD_SRC_C found in $PORT/Makefile" >&2; exit 1; }
CLANG=${CAPSTONE_CLANG}
LD_LLD=${CAPSTONE_LD_LLD}

[[ -d $MPY_SRC_DIR/py ]] || { echo "no MicroPython at $MPY_SRC_DIR -- run fetch-micropython.sh" >&2; exit 1; }
# The qstr / module / root-pointer headers are generated FROM mpconfigport.h, so they are
# port-specific and cannot be borrowed from ports/minimal: MICROPY_STACK_CHECK alone pulls in a
# qstr that port does not define, and a mismatched string pool is an error that compiles. So the
# port is mirrored into the MicroPython tree and MicroPython's own machinery generates them, with
# a STOCK host compiler -- none of this reaches the domain, only the headers do.
MPY_PORT_DIR="$MPY_SRC_DIR/ports/capstone"
# MPY_FEATURE_LEVEL -- a name for a set of defines, because the interesting ones are not
# all the ROM level. mpconfigport.h starts from "nothing optional is enabled" so that a test
# failing for want of a builtin says nothing about capabilities, and that stays the default.
# The levels below are what the upstream test set answers to, measured in a domain, 421
# tests, every count from one round with no restart:
#
#   minimum   270 pass  151 fail  0 fault      the default, as mpconfigport.h ships
#   core      349 pass   72 fail  0 fault      MICROPY_CONFIG_ROM_LEVEL=CORE_FEATURES
#   extra     417 pass    4 fail  0 fault      + EXTRA_FEATURES and mpz long integers
#   full      420 pass    1 fail  0 fault      + the three remaining feature switches
#
# The four failures at `extra` are not one thing: three are single switches that upstream
# gates at EVERYTHING or FULL_FEATURES, and `full` turns exactly those three on. The fourth,
# fun_code_full.py, needs MICROPY_PY_BUILTINS_CODE at FULL, which is reachable only through
# MICROPY_PY_SYS_SETTRACE -- and that is NOT offered here, deliberately. Turning it on builds
# and then takes the suite to 312 pass with 89 FAULTS, 85 of them at one site,
# mp_emit_common_populate_module_context+0x1b0, where the compiler fills a module's constant
# table. That is a capability defect in a code path only settrace reaches, and it wants the
# same treatment S-14 got rather than a knob that hands someone 89 faults.
#
# ROM_LEVEL_EVERYTHING is also not offered: it does not compile at this pin, wanting
# QSTR_LAST_STATIC and a qstr_table member that persistentcode.c expects and this port's
# configuration does not produce.
case "${MPY_FEATURE_LEVEL:-minimum}" in
  minimum) MPY_LEVEL_DEFS="" ;;
  core)    MPY_LEVEL_DEFS="-DMICROPY_CONFIG_ROM_LEVEL=MICROPY_CONFIG_ROM_LEVEL_CORE_FEATURES" ;;
  extra)   MPY_LEVEL_DEFS="-DMICROPY_CONFIG_ROM_LEVEL=MICROPY_CONFIG_ROM_LEVEL_EXTRA_FEATURES -DMICROPY_LONGINT_IMPL=MICROPY_LONGINT_IMPL_MPZ" ;;
  full)    MPY_LEVEL_DEFS="-DMICROPY_CONFIG_ROM_LEVEL=MICROPY_CONFIG_ROM_LEVEL_EXTRA_FEATURES -DMICROPY_LONGINT_IMPL=MICROPY_LONGINT_IMPL_MPZ -DMICROPY_PY_BUILTINS_RANGE_BINOP=1 -DMICROPY_PY_ALL_INPLACE_SPECIAL_METHODS=1 -DMICROPY_PY_FUNCTION_ATTRS_CODE=1" ;;
  *) echo "MPY_FEATURE_LEVEL must be minimum, core, extra or full" >&2; exit 1 ;;
esac
DOMAIN_EXTRA_DEFS="${MPY_LEVEL_DEFS}${DOMAIN_EXTRA_DEFS:+ $DOMAIN_EXTRA_DEFS}"

echo "== generating this port's headers (stock toolchain, host only)"
rm -rf "$MPY_PORT_DIR"
cp -r "$PORT" "$MPY_PORT_DIR"
make -C "$MPY_PORT_DIR" -j"${HDR_JOBS:-8}" \
    CFLAGS_EXTRA="${DOMAIN_EXTRA_DEFS:-} ${FLOAT_DEFS[*]} ${MPY_VFS:+-DMICROPY_VFS=1 -DFFCONF_H='\"ffconf.h\"' -I../../lib/oofatfs}" >"$OBJ_DIR/genhdr.log" 2>&1 || {
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
         ${MPY_VFS:+-DMICROPY_VFS=1 -DFFCONF_H=\"ffconf.h\" -I$MPY_SRC_DIR/lib/oofatfs}
         ${DOMAIN_EXTRA_DEFS:-}
         "${FLOAT_DEFS[@]}")

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
  # The self-contained extmod modules: pure computation, no clock, no filesystem, no device. Each
  # is wrapped in its own `#if MICROPY_PY_<NAME>` and so compiles to nothing below EXTRA level, and
  # each #includes whatever it needs from lib/ (re1.5, uzlib, crypto-algorithms) at its own foot,
  # so listing the module file is the whole dependency. Deliberately absent: time (the domain has
  # no clock, and a stub one would make time.time() confidently wrong), and everything backed by a
  # filesystem, socket, or peripheral. Keep this list in step with EXTMOD_SRC_C in port/Makefile,
  # which is what puts their MP_REGISTER_MODULE into genhdr/moduledefs.h.
  for m in $EXTMOD_UNITS; do
    for n in ${EXTMOD_STATICS[$m]:-}; do echo "#define $n mpyext_${m}_$n"; done
    echo "#include \"$MPY_SRC_DIR/extmod/$m.c\""
    for n in ${EXTMOD_STATICS[$m]:-}; do echo "#undef $n"; done
  done
  if [[ -n ${MPY_FLOAT_CORE:-} ]]; then
    # MicroPython's OWN libm, not an approximation of it. Its APPROX float formatting and parsing
    # both scale by pow(5, n) (py/parsenum.c:275), so pow's accuracy IS repr's accuracy: with the
    # BEEBS exp(y*log(x)) pow, whose relative error on pow(5,16) is -1.3e-12, repr(1.0) printed
    # 0.9999999999986986 -- the same 1.3e-12 -- and 24 float tests failed on their last digits.
    for u in $LIBM_UNITS; do
      for n in ${LIBM_STATICS[$u]:-}; do echo "#define $n mpylibm_${u}_$n"; done
      echo "#include \"$MPY_SRC_DIR/lib/libm_dbl/$u.c\""
      for n in ${LIBM_STATICS[$u]:-}; do echo "#undef $n"; done
    done
    echo "#include \"$MPY_PORT_DIR/capstone_math_extra.c\""
  fi
  if [[ -n $MPY_VFS ]]; then
    # vfs_fat.c's stat() converts FAT timestamps through this. It is shared/, not lib/, and
    # it is the only file from there this port needs.
    echo "#include \"$MPY_SRC_DIR/shared/timeutils/timeutils.c\""
  fi
  for u in $OOFATFS_UNITS; do
    # FatFs is one translation unit per file with no statics that collide with py/, so
    # unlike libm it needs no renaming. FFCONF_H and the include path are passed in the
    # SILICON defines above, because ff.h resolves the config by macro.
    echo "#include \"$MPY_SRC_DIR/lib/oofatfs/$u.c\""
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

FLOAT_OBJS=()
if [[ -n ${MPY_FLOAT_CORE:-} ]]; then
  COMMON_FLAGS=("${COMMON[@]}" "${SILICON[@]}" -D__SOFTFP__)
  source "$BEEBS_SOFTFLOAT"
  FLOAT_OBJS+=("${softfloat_objs[@]}")
  for builtin in extendhfsf2 truncsfhf2; do
    "$CLANG" "${COMMON_FLAGS[@]}" -I"$COMPILER_RT" \
      -c "$COMPILER_RT/$builtin.c" -o "$OBJ_DIR/softfloat-$builtin.o"
    FLOAT_OBJS+=("$OBJ_DIR/softfloat-$builtin.o")
  done
fi

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
    "$OBJ_DIR/setjmp.o" "$OBJ_DIR/str_extra.o" "${FLOAT_OBJS[@]}" "$OBJ_DIR/gct.o" \
    ${DOMREQ_OBJ:+"$DOMREQ_OBJ"}
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
# Slack between .text and the globals region was suspected when a build landed 2,772 bytes under
# the boundary and hung; forcing a whole spare 64 KiB (TEXT + 0x1FFFF) was tried and the image hung
# just the same, so the rounding is left alone. Do not re-propose it without a new measurement.
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

# THE DOMAIN DECLARES WHAT IT NEEDS, instead of leaving the module to infer it from the
# CODE size. That fallback holds only while text >= cap table + carved storage + stack.
# SQLite satisfies it with 2.2 MB of text against a modest carve. This port does not:
# 390 KB of text against ~500 KB of storage, because the GC heap is a 384 KiB global and
# under gp-captable a global's storage comes out of dom_data. Measured over seven
# test-table sizes, two came up 68,864 and 125,296 bytes short and died in the entry glue
# with the globals blob overwritten.
#
# THE CARVE is computed from the linked image, which is the only place that knows it.
# THE STACK is the one number a build has to choose, and it is MEASURED, not guessed: the
# full upstream test set in a domain, 421 tests, has its deepest stack write 61,416 bytes
# below the top (capstone-qemu CAPSTONE_STOREWATCH_LOW). 128 KiB is that doubled. It is
# not a bound -- this pin of MicroPython has no reachable recursion guard, MP_STACK_CHECK()
# appears zero times in the built translation unit -- so a script that recurses deeper than
# this will still run past it. Bounding that needs the stack's own capability to stop at
# the blob, which is a separate change.
#
# EXPECT A BIGGER ALLOCATION, and that is the requirement becoming visible rather than a
# regression: at 60 tests the image runs today on 71,824 bytes of stack against 61,416
# measured use, ten kilobytes of margin by accident. Declaring an honest stack does not fit
# 1 MiB, so most sizes move to 2 MiB.
MPY_DOMAIN_STACK=${MPY_DOMAIN_STACK:-$((128 * 1024))}
CARVE=$(CAPSTONE_BUILDROOT_DIR="$CAPSTONE_BUILDROOT_DIR" \
        python3 "$LADDER/domdata-budget.py" "$OUT_DIR/$DOM_NAME.dom" --carve)
[[ "$CARVE" =~ ^[0-9]+$ ]] || { echo "could not compute the carve requirement: $CARVE" >&2; exit 1; }
DOMREQ_DATA=$((CARVE + MPY_DOMAIN_STACK))
echo "== pass 3: declare dom_data >= $DOMREQ_DATA (carve $CARVE + stack $MPY_DOMAIN_STACK)"
"$CLANG" -target capstone64-unknown-elf -ffreestanding \
  -DCAPSTONE_DOMREQ_DATA=$DOMREQ_DATA -DCAPSTONE_DOMREQ_STACK=$MPY_DOMAIN_STACK \
  -c "$LADDER/../domreq.S" -o "$OBJ_DIR/domreq.o"
# The declaration is non-alloc, so relinking with it must not move a loaded byte. That
# check is why it is safe to add after the two passes that fix the layout.
_loaded() { "$CAPSTONE_LLVM_BIN/llvm-readelf" -lW "$1" | awk '/LOAD/ {print $3, $5, $6}'; }
_before=$(_loaded "$OUT_DIR/$DOM_NAME.dom")
DOMREQ_OBJ="$OBJ_DIR/domreq.o" link "$(printf '0x%x' $GOFF)" "$OUT_DIR/$DOM_NAME.dom"
_after=$(_loaded "$OUT_DIR/$DOM_NAME.dom")
[[ "$_before" == "$_after" ]] || {
  echo "domreq.S moved a loaded byte; the declaration must be non-alloc" >&2
  echo "  before: $_before" >&2; echo "  after:  $_after" >&2; exit 1; }

BUDGET=$(python3 "$LADDER/domdata-budget.py" "$OUT_DIR/$DOM_NAME.dom" 2>&1 || true)
echo "$BUDGET"
# "VERDICT: fits" is a STATIC budget check and does not predict whether the image runs. An image
# that lands in a >2 MiB domain allocation has twice been one that links, reports "fits", and then
# faults on EVERY normal call -- 200 tests, 200 reboots, no results. Not fatal (a 4 MiB chunk has
# also run clean), so this warns rather than stops; a runner that would spend an hour on the image
# should refuse it instead.
# REFUSE A BUILD THE BUDGET SAYS DOES NOT FIT. This used to print the verdict and carry
# on, so an image whose carve overruns dom_data got built, shipped to a runner and died in
# its entry glue with the globals blob overwritten -- a fault that looks like a compiler or
# monitor defect and costs a day. Two of seven test-table sizes were in that state
# (160 and 200 tests, short by 68,864 and 125,296 bytes) and the budget line said "fits",
# because the predictor modelled an allocation rule the kernel module does not have.
# The predictor is fixed; this is the half that makes its answer count.
if grep -q 'DOES NOT FIT' <<<"$BUDGET"; then
  echo "FAIL: the carve does not fit dom_data for this image. Lower MPY_HEAP_SIZE or the" >&2
  echo "      test count, or raise the domain allocation. The budget above has the numbers." >&2
  exit 1
fi
if grep -qE 'order=1[0-9]' <<<"$BUDGET"; then
  echo "   WARNING: domain allocation is larger than 2 MiB -- images this size have faulted on" >&2
  echo "            every call despite VERDICT: fits. Prefer smaller test chunks." >&2
fi
echo "Built $OUT_DIR/$DOM_NAME.dom"
