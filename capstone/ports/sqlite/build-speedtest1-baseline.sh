#!/usr/bin/env bash
# Build the MATCHED BASELINE half of the speedtest1 overhead measurement: the same translation unit
# as the capability domain, compiled as ordinary RISC-V and linked into one freestanding Linux
# binary.
#
#   usage: build-speedtest1-baseline.sh
#   out:   $OUT_DIR/speedtest1_baseline
#
# The point of this build is that it differs from build-sqlite-silicon.sh in exactly ONE dimension.
# That script compiles the amalgam TU with $CAPSTONE_CLANG at $OPT for -target capstone64-unknown-elf
# with the gp-captable silicon flags; this compiles THE SAME amalgam.c with THE SAME clang at THE
# SAME $OPT for -target riscv64-unknown-elf with no capability flags. Using buildroot gcc for the
# measured code would make the ratio a measurement of two compilers rather than of the capability
# ABI, so it is deliberately NOT used there -- only for the harness, which sits outside the counter
# brackets. Both sides are rv64imac/lp64, so the objects link together. Same reasoning, same shape,
# as build-ladder-base-fpga.sh.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/speedtest1-baseline}
OBJ_DIR=$OUT_DIR/obj
mkdir -p "$OBJ_DIR"

CLANG=${CLANG:-$CAPSTONE_CLANG}
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
ADAPTED=$SCRIPT_DIR/adapted
VFS_DIR=$REPO_ROOT/capstone/tests/runtime-qemu/sqlite-vfs-skeleton
BUILTINS=$REPO_ROOT/compiler-rt/lib/builtins
BEEBS_STRING=$REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c

# MUST MATCH THE DOMAIN'S. build-sqlite-silicon.sh compiles the amalgam at -O0 (SQLITE_OPT_LEVEL)
# and the support objects at -O1 (SUPPORT_OPT). A baseline built at a different level measures the
# optimiser, not the ABI -- that mismatch produced five bogus "silicon failures" on the ladder
# before ladder-rungs.spec was introduced to hold the two halves together.
OPT=${SQLITE_OPT_LEVEL:--O0}
# SQLITE_SUPPORT_OPT_LEVEL, spelled exactly as build-sqlite-silicon.sh:2791 spells it. It was
# SQLITE_SUPPORT_OPT here, which is a DIFFERENT variable: the two defaults coincide at -O1, so the
# arms agreed by accident and an A/B that set the domain's level would have moved one arm silently --
# the "two silent -O mismatches" failure this script's own header cites.
SUPPORT_OPT=${SQLITE_SUPPORT_OPT_LEVEL:--O1}
# From the shared geometry, not a third literal. This defaulted independently and no committed flow
# ever reached it, so the two arms could run different arena sizes with nothing gating it.
source "$SCRIPT_DIR/speedtest1-geometry.sh"
HEAP=${SQLITE_HEAP_SIZE:-$SPEEDTEST1_GEOM_HEAP}

# The patched amalgamation, produced by the existing script's sed pass. Reuse it rather than
# duplicating 25 fragile substitutions -- and rather than patching it differently here, which would
# make the two halves different engines.
PATCHED=${PATCHED_SQLITE:-$CAPSTONE_TMP_ROOT/sqlite-build/sqlite3-capstone.c}
if [[ ! -f "$PATCHED" ]]; then
  echo "patched amalgamation missing; running the existing build to produce it"
  OUT_DIR="$(dirname "$PATCHED")" bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
fi
[[ -f "$PATCHED" ]] || { echo "still no $PATCHED" >&2; exit 1; }

# THE AMALGAMATION'S OWN sqlite3.h, EXPLICITLY.
#
# The domain build never puts this directory on its include path and does not need to: the amalgam
# includes sqlite3-capstone.c first, which defines SQLITE3_H at line 355, so every later
# `#include "sqlite3.h"` expands to nothing. It gets away with it because clang searches the host's
# /usr/include for the capstone64 target and finds one there -- which on this machine is SQLite
# 3.45.1, eight point releases below the 3.53.3 we build. Inert in the amalgam, latent everywhere
# else: any TU that includes capstone_sqlite_vfs.h WITHOUT the amalgamation ahead of it compiles
# against a different SQLite's declarations. A bare-metal riscv64 target searches no host
# directories, so the baseline has to name the right one, and naming it is the correct behaviour
# rather than a concession.
SQLITE_AMALG_DIR=$(bash "$SCRIPT_DIR/fetch-sqlite.sh")

SPEEDTEST1_SRC=${SQLITE_SPEEDTEST1_SRC:-$(bash "$SCRIPT_DIR/fetch-sqlite-src.sh")/test/speedtest1.c}
[[ -f "$SPEEDTEST1_SRC" ]] || { echo "no speedtest1.c at $SPEEDTEST1_SRC" >&2; exit 1; }

# Stage exactly what the amalgam includes, under the names it includes them by.
cp -f "$PATCHED"                          "$OBJ_DIR/sqlite3-capstone.c"
cp -f "$VFS_DIR/capstone_sqlite_vfs.c"    "$OBJ_DIR/capstone_sqlite_vfs.c"
cp -f "$ADAPTED/capstone_sqlite_os.c"     "$OBJ_DIR/capstone_sqlite_os.c"
cp -f "$SCRIPT_DIR/speedtest1_measure.c"  "$OBJ_DIR/sqlite_capstone_domain.c"
cp -f "$SCRIPT_DIR/sqlite_silicon_amalgam.c" "$OBJ_DIR/amalgam.c"
cp -f "$SPEEDTEST1_SRC"                   "$OBJ_DIR/speedtest1.c"

# The SAME counted rename as the silicon build. Both halves must carry it or they are not the same
# translation unit, which is the entire premise of a matched baseline.
python3 - "$OBJ_DIR/speedtest1.c" <<'PYRF'
import sys
p = sys.argv[1]
s = open(p).read()
n = s.count("randomFunc")
if n != 2:
    sys.exit("speedtest1.c: expected 2 randomFunc occurrences, found %d -- the source shape "
             "changed, re-check the collision set against the amalgamation" % n)
open(p, "w").write(s.replace("randomFunc", "speedtest1_randomFunc"))
print("   renamed randomFunc -> speedtest1_randomFunc (2 sites)")
PYRF

# The define list, harvested from the one file that owns it so the two halves cannot drift.
mapfile -t DEFS < <(sed -n '/^SQLITE_DEFINES=(/,/^)/p' "$SCRIPT_DIR/build-sqlite-capstone.sh" \
                    | grep -oE '\-[DU][A-Za-z0-9_]+(=[^ )]*)?')
EXPECT_DEFS=27   # one variable, because a count written twice can disagree with itself

# SQLITE_FLOAT / SQLITE_FULL, read exactly as the domain build and the oracle read them. THE
# BASELINE MUST CARRY THE SAME FEATURE DEFINES OR THE COMPARISON IS MEANINGLESS: with floating point
# on one side only, the two arms are not running the same SQL, let alone the same code. This script
# did not have these knobs until 2026-09-11, which would have compared a floating-point domain
# against an integer-only baseline -- a worse version of the string-primitive mismatch retracted the
# day before, because that one changed optimisation and this one changes semantics.
SQLITE_FULL=${SQLITE_FULL:-off}
SQLITE_FLOAT=${SQLITE_FLOAT:-off}
[[ "$SQLITE_FULL" == "on" ]] && SQLITE_FLOAT=on
if [[ "$SQLITE_FLOAT" == "on" ]]; then
  DEFS+=(-USQLITE_OMIT_FLOATING_POINT)
  EXPECT_DEFS=$(( EXPECT_DEFS + 1 ))
fi
if [[ "$SQLITE_FULL" == "on" ]]; then
  DEFS+=(-DSQLITE_ENABLE_RTREE=1 -USQLITE_OMIT_INCRBLOB)
  EXPECT_DEFS=$(( EXPECT_DEFS + 2 ))
  [[ "${SQLITE_JSON:-off}" == "on" ]] && { DEFS+=(-USQLITE_OMIT_JSON); EXPECT_DEFS=$(( EXPECT_DEFS + 1 )); }
fi
(( ${#DEFS[@]} == EXPECT_DEFS )) || {
  echo "ERROR: harvested ${#DEFS[@]} defines, expected $EXPECT_DEFS." >&2
  echo "  A FLOOR TEST USED TO LIVE HERE and it could not do this job: it required only more than" >&2
  echo "  ten, so sixteen defines could vanish and it still passed. It fired only on total sed" >&2
  echo "  failure, which is the one case that also breaks the build loudly downstream." >&2
  echo "  If SQLITE_DEFINES legitimately changed, re-bless this number DELIBERATELY -- both arms of" >&2
  echo "  every comparison read the same block, so a silent drop moves them together and no ratio" >&2
  echo "  would look wrong." >&2
  exit 1; }
if [[ "${SQLITE_FEATURE_SET:-deployed}" == restored ]]; then
  mapfile -t -O "${#DEFS[@]}" DEFS < <(sed -n '/^SQLITE_RESTORE=(/,/^)/p' "$SCRIPT_DIR/build-sqlite-capstone.sh" \
                    | grep -oE '\-[DU][A-Za-z0-9_]+(=[^ )]*)?')
fi
echo "== defines shared with the domain: ${#DEFS[@]}"

# EMPTY STUBS FOR THE HEADERS THE GUARD ALREADY NEUTRALISES.
#
# capstone_sqlite_libc.h defines _STDIO_H, _STDLIB_H, _STRING_H, _ASSERT_H, _CTYPE_H, _TIME_H and
# _MATH_H, so glibc's copies of those headers contribute NOTHING when the amalgamation includes
# them -- the whole file sits inside its own include guard. The capstone64 build gets away with the
# includes anyway because clang searches the host's /usr/include for that target; a bare-metal
# riscv64 target does not, and the build stops at `'stdio.h' file not found`.
#
# So the stubs are not a workaround for the baseline: they are the domain build's ACTUAL behaviour
# made explicit. An empty file is exactly what a guarded glibc header expands to here, and having
# them on disk means the baseline provably cannot pick up a host declaration that the domain half
# never saw.
mkdir -p "$OBJ_DIR/stubs/sys"
for h in stdio.h stdlib.h string.h assert.h ctype.h time.h math.h unistd.h sys/types.h; do
  : > "$OBJ_DIR/stubs/$h"
done

# SPEEDTEST1_BASELINE_MARCH exists to MEASURE the encoding asymmetry, not to hide it. The default
# follows the ladder's existing baseline convention (build-ladder-base-fpga.sh uses -march=rv64imac),
# so these numbers stay comparable with every overhead figure this project has published.
COMMON=(-target riscv64-unknown-elf -march=${SPEEDTEST1_BASELINE_MARCH:-rv64imac_zicsr} -mabi=lp64
        -ffreestanding -fno-stack-protector -fno-jump-tables -fno-builtin
        # The same libc guard the domain build preincludes. Without it the amalgamation reaches for
        # the host's <stdio.h> and the build stops -- which is the good outcome; the bad one would
        # be a baseline that quietly compiled against different headers than the domain.
        -include "$ADAPTED/capstone_sqlite_libc.h"
        -DCAPSTONE_SPEEDTEST1_BASELINE=1 -DCAPSTONE_SQLITE_MCYCLE_CLOCK=1
        -DSQLITE_HEAP_SIZE=$HEAP
        -I"$ADAPTED" -I"$SCRIPT_DIR" -I"$VFS_DIR" -I"$OBJ_DIR" -I"$(dirname "$PATCHED")"
        -I"$BUILTINS" -I"$SQLITE_AMALG_DIR" -isystem "$OBJ_DIR/stubs")

echo "== compiling the amalgam TU as ordinary RISC-V ($OPT)"
"$CLANG" "${COMMON[@]}" "${DEFS[@]}" "$OPT" -c "$OBJ_DIR/amalgam.c" -o "$OBJ_DIR/amalgam.o"

# THE SUPPORT DEFINES THE DOMAIN CARRIES, and why the baseline must be able to carry them too.
#
# build-sqlite-silicon.sh compiles these two objects with three defines this script originally did
# not pass: BEEBS_STRING_LINEAR_SAFE, BEEBS_MEMCPY_OPTNONE and BEEBS_STRING_WRITERS_OPTNONE. They
# make memcpy, memset, memmove and strcpy `optnone, noinline` and change strlen/strcmp/strcpy to an
# indexing form. These two objects are where SQLite spends its tight loops, by that script's own
# account, so a baseline without them compares an -O0 un-inlinable memcpy against an -O1 inlinable
# one and charges the difference to the capability ABI.
#
# They are SILICON-DEFECT WORKAROUNDS (S-04 and the untagged ldc/stc high-half loss), not part of the
# ABI, so neither answer is simply right:
#
#   SPEEDTEST1_BASELINE_WORKAROUNDS=1 (default)  the arms differ only in -target. The ratio prices
#                                                the capability ABI.
#   SPEEDTEST1_BASELINE_WORKAROUNDS=0            the baseline is ordinary RISC-V with ordinary string
#                                                primitives. The ratio prices what it costs to run
#                                                SQLite safely on THIS silicon today, workarounds
#                                                included.
#
# Report both and name which is which. The first was the number published on 2026-09-10 and it was
# labelled as the second.
SUPPORT_DEFS=()
if [[ "${SPEEDTEST1_BASELINE_WORKAROUNDS:-1}" == "1" ]]; then
  SUPPORT_DEFS=(-DBEEBS_STRING_LINEAR_SAFE=1
                -DBEEBS_MEMCPY_OPTNONE=${SQLITE_MEMCPY_OPTNONE:-1}
                -DBEEBS_STRING_WRITERS_OPTNONE=${SQLITE_WRITERS_OPTNONE:-1})
fi
echo "== compiling the support objects ($SUPPORT_OPT, workarounds=${SPEEDTEST1_BASELINE_WORKAROUNDS:-1})"
for pair in "libc:$ADAPTED/capstone_sqlite_libc.c" "beebs_string:$BEEBS_STRING"; do
  "$CLANG" "${COMMON[@]}" "${DEFS[@]}" "${SUPPORT_DEFS[@]}" "$SUPPORT_OPT" \
    -c "${pair#*:}" -o "$OBJ_DIR/${pair%%:*}.o"
done

BUILTIN_OBJS=()
"$CLANG" "${COMMON[@]}" -O0 -c "$SCRIPT_DIR/capstone_floatdidf_noglobals.c" -o "$OBJ_DIR/floatdidf_ng.o"
BUILTIN_OBJS+=("$OBJ_DIR/floatdidf_ng.o")
# The same conditional builtin sets as the domain build, and for the same reason: with floating
# point on, the amalgamation references single-precision and conversion routines that the deployed
# set never needed. Every name here was produced by a LINK ERROR rather than predicted.
_extra_builtins=""
[[ "$SQLITE_FLOAT" == "on" ]] && _extra_builtins="extendsfdf2"
[[ "$SQLITE_FULL"  == "on" ]] && _extra_builtins="$_extra_builtins floatundidf truncdfsf2 comparesf2 subsf3 addsf3 mulsf3 divsf3 fixsfsi floatsisf"
_seen=" "
for b in eqdf2 fixdfdi fixdfsi gedf2 gtdf2 ltdf2 muldf3 nedf2 adddf3 subdf3 comparedf2 \
         fixunsdfdi fixunsdfsi floatsidf floatunsidf fp_mode divdf3 $_extra_builtins; do
  case "$_seen" in *" $b "*) continue ;; esac
  _seen+="$b "
  [[ -f "$BUILTINS/$b.c" ]] || continue
  # No 2>/dev/null and no `&&`, matching the domain build: a builtin that fails to compile stops
  # this build with its error rather than reappearing later as an undefined symbol.
  "$CLANG" "${COMMON[@]}" -O0 -c "$BUILTINS/$b.c" -o "$OBJ_DIR/$b.o" || {
    echo "build-speedtest1-baseline.sh: builtin '$b' FAILED TO COMPILE" >&2; exit 1; }
  BUILTIN_OBJS+=("$OBJ_DIR/$b.o")
done

echo "== linking with the harness (buildroot gcc, outside the counter brackets)"
"$GUEST_CC" -Os -static -no-pie -fno-pie -nostdlib -ffreestanding -fno-stack-protector \
  -march=rv64imac_zicsr -mabi=lp64 \
  -o "$OUT_DIR/speedtest1_baseline" "$SCRIPT_DIR/speedtest1_baseline.c" \
  "$OBJ_DIR/amalgam.o" "$OBJ_DIR/libc.o" "$OBJ_DIR/beebs_string.o" "${BUILTIN_OBJS[@]}"

# STATIC GATE. The baseline must contain NO capability instruction. If one leaked in, the "plain
# RISC-V" denominator would be pricing capabilities too and the whole ratio would be meaningless --
# so fail the BUILD rather than the analysis. Copied from build-ladder-base-fpga.sh, which is where
# this gate was first needed.
_CAPRE='cjalr|ldc|stc|scc|cincoffset|mrev|csdrop|shrink|delin|revoke'
# THE POSITIVE CONTROL FIRST, because this gate is structurally unable to fail on its own. Every
# input to the baseline is compiled -target riscv64 with no override knob, so no path in this script
# can emit a Capstone mnemonic and NCAP is 0 whatever happens -- including if llvm-objdump prints
# nothing at all, or spells these instructions differently for this backend. Build one throwaway
# object for the capability target and require the SAME grep to find some.
"$CLANG" -target capstone64-unknown-elf -ffreestanding -O1 -c -x c - -o "$OBJ_DIR/capprobe.o" <<'CAPPROBE'
/* One capability load and one capability store, which is all the grep needs to see. */
extern void *__capstone_slot;
void *capprobe(void **p) { void *v = *p; __capstone_slot = v; return v; }
CAPPROBE
_PROBE=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OBJ_DIR/capprobe.o")
_NPROBE=$(grep -cEw "$_CAPRE" <<<"$_PROBE" || true)
echo "static: positive control finds $_NPROBE capability instructions in a capability-target object"
[[ "$_NPROBE" -gt 0 ]] || {
  echo "FAIL: the capability-instruction grep finds NOTHING in a capability object -- the gate below" >&2
  echo "      would read 0 for a baseline full of them. Check the mnemonic list against llvm-objdump." >&2
  exit 1; }

DIS=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT_DIR/speedtest1_baseline")
NCAP=$(grep -cEw "$_CAPRE" <<<"$DIS" || true)
echo "static: capability-instructions=$NCAP (must be 0)"
[[ "$NCAP" == "0" ]] || { echo "FAIL: capability instructions in the baseline" >&2; exit 1; }

echo "Built $OUT_DIR/speedtest1_baseline ($(stat -c%s "$OUT_DIR/speedtest1_baseline") bytes)"
