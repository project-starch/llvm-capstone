#!/usr/bin/env bash
# Build the SQLite domain in the SILICON config (stage S5 of the SQLite-on-silicon plan).
#
#   -capstone-gp-captable + gp-free call/ret + shrink OFF + -fno-jump-tables,
#   descriptor-driven entry glue, one translation unit, globals offset sized to .text.
#
# Differences from build-sqlite-capstone.sh, and why each one is needed:
#
#  1. SILICON FLAGS. -capstone-gp-captable moves every global behind `ldc gp[i]`;
#     shrink-stack/globals are off for the RTL shrink->store hazard. The existing
#     build passes no -mllvm flags at all, so this is the first time SQLite is
#     compiled for the silicon ABI.
#
#  2. ONE MODULE. getGpCaptableIndex numbers globals per module and positionally, so
#     two TUs that both own globals collide silently on the single gp cap-table.
#     sqlite_silicon_amalgam.c #includes the five files that own globals (measured;
#     the other 14 objects own none and stay separate).
#
#  3. GLOBALS OFFSET SIZED TO .text. link-gpfree.ld puts globals at a fixed offset and
#     .text must fit BELOW it. 0x1000 suits a BEEBS kernel; SQLite's .text is ~2.2 MB.
#     This does a two-pass link: measure .text, round up, relink. The monitor learns
#     the value at run time (packed into entry_offset by libcapstone) -- issue C-12.
#
#  4. INTERP GLUE. The generated prologue cannot express SQLite's globals: it rejects
#     private `.L` symbols (SQLite has ~910) and dies above 2040 B per initialized
#     global. The descriptor-driven glue reads sizes/offsets at entry instead.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/sqlite-silicon}
OBJ_DIR=$OUT_DIR/obj
LADDER=$REPO_ROOT/capstone/tests/runtime-qemu/silicon-ladder
GPFREE=$REPO_ROOT/capstone/tests/runtime-qemu/gp-free-domain
ADAPTED=$SCRIPT_DIR/adapted
VFS_DIR=$REPO_ROOT/capstone/tests/runtime-qemu/sqlite-vfs-skeleton
BUILTINS=$REPO_ROOT/compiler-rt/lib/builtins
BEEBS_STRING=$REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c
OPT=${SQLITE_OPT_LEVEL:--O0}
# The memsys5 arena is charged against dom_data (see sqlite_capstone_domain.c), so the
# silicon build shrinks it. 1 MiB does not fit; 256 KiB leaves a workable stack.
HEAP=${SQLITE_HEAP_SIZE:-$((256*1024))}

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# The patched amalgamation is produced by the existing script's sed pass. Reuse it
# rather than duplicating 25 fragile substitutions.
PATCHED=${PATCHED_SQLITE:-$CAPSTONE_TMP_ROOT/sqlite-build/sqlite3-capstone.c}
if [[ ! -f "$PATCHED" ]]; then
  echo "patched amalgamation missing; running the existing build to produce it"
  bash "$SCRIPT_DIR/build-sqlite-capstone.sh" >/dev/null
fi
[[ -f "$PATCHED" ]] || { echo "still no $PATCHED" >&2; exit 1; }

# Stage the amalgamation's includes side by side so plain #include works.
cp -f "$PATCHED"                          "$OBJ_DIR/sqlite3-capstone.c"

# BUILTIN_LIMIT=<n> -- DIAGNOSTIC ONLY. Clamp how many builtin functions
# sqlite3RegisterBuiltinFunctions processes, so the count can be bisected on the board.
#
# That function is the bisected wedge point (stages 7/8/9 return rc=0, stage 10 wedges) and
# every construct inside it has now been isolated and cleared individually: string data
# (unaligned-copy fix, board-confirmed 0xFF), 512 capability stores, 128 local structs
# holding string pointers, strlen on cap-table literals, strcmp/strcpy linear-safety, a
# stack->global struct assignment carrying a capability, a >2048-byte register-built stack
# frame, and gp offsets (max index 175 of 176, in range). The remaining untested variable is
# SCALE: it processes ~72 entries and touches ~176 distinct cap-table slots, where every
# probe touched one or two.
#
# limit=1 wedging  -> the construct itself is broken, and one entry is a minimal reproducer.
# limit=1 passing and limit=N wedging -> it is a count/scale effect, and the bisection gives
# the exact threshold, which is the number to hand the board owner.
# SQLITE_STATIC_BUILTINS=1 -- WORKAROUND for R-14, and arguably the correct shape now.
#
# R-14: straight-line materialisation of distinct string constants into a struct array wedges
# the board (variant A), while the same data assigned in a loop from a static table is fine
# (variant C) and a flat pointer array is fine (variant D). sqlite3RegisterBuiltinFunctions
# builds exactly the wedging shape, because build-sqlite-capstone.sh strips `static` from
#     static FuncDef aBuiltinFunc[] = { ... }
# turning a compile-time-initialised GLOBAL into a STACK array constructed straight-line at
# run time, then copied element-wise into a separate static.
#
# That de-static predates CapstoneCapGlobalInit. We now emit __capstone_cap_init, which
# stores each capability leaf of a global initialiser at domain entry -- the current SQLite
# domain already does 394 such stores and works, so the machinery handles this at scale.
# Putting the array back to `static` therefore removes the straight-line stack construction
# ENTIRELY rather than reshaping it, and routes the same data through a path that is already
# exercised and passing.
#
# *** 2026-08-04: R-14 IS FIXED IN SILICON, so this is no longer needed as a workaround. ***
#
# R-14 was a capability operand-forwarding bug (capstone-ariane 7aac52f93), fixed by the
# bitstream `caplifive_fixed_forward.bit`. Verified on the board across two valid boots with
# controls green: k1200 and r14lp -- both previously failing -- now return the correct value.
#
# It is left DEFAULT ON deliberately, on its own merits: putting aBuiltinFunc back to a
# compile-time-initialised static removes the straight-line stack construction entirely rather
# than reshaping it, which is the better shape regardless of the silicon bug. Set
# SQLITE_STATIC_BUILTINS=0 for the old de-static shape (which was the R-14 reproducer).
#
# HISTORICAL, and the reason to be careful about flipping this: =1 was ALSO the shape that
# entry-stalled (R-16) -- st10, sb10, swa, swa8, swa9 all stalled before executing any code, so
# the workaround never actually ran at SQLite scale until the reflash. R-16 turned out to be the
# SAME forwarding defect, i.e. the R-14 workaround was creating R-16. Both are fixed together,
# and both return together if the board is reflashed to a bitstream lacking the fix -- see
# capstone/tests/fpga-repros/R16-entry-stall/ and .../ARCHIVED/R14-frame-pad/.
if [[ "${SQLITE_STATIC_BUILTINS:-1}" == "1" ]]; then
  echo "== R-14 WORKAROUND: restoring aBuiltinFunc to a compile-time-initialised static"
  python3 - "$OBJ_DIR/sqlite3-capstone.c" <<'PY'
import sys, re, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
before = s
# 1. undo the de-static: the stack array becomes the real static again.
s = s.replace("  FuncDef capstoneBuiltinFunc[] = {",
              "  static FuncDef aBuiltinFunc[] = {", 1)
# 2. drop the separate zero-init static that the patch introduced.
s = s.replace("  static FuncDef aBuiltinFunc[ArraySize(capstoneBuiltinFunc)];\n", "", 1)
# 3. drop the element-wise copy loop -- there is nothing to copy from any more.
s = re.sub(r"  for\(int capstoneI=0; capstoneI<[^\n]*ArraySize\(capstoneBuiltinFunc\)[^\n]*\)\{\n"
           r"    aBuiltinFunc\[capstoneI\] = capstoneBuiltinFunc\[capstoneI\];\n  \}\n",
           "", s, count=1)
# 4. any surviving reference to the removed name now means the static.
s = s.replace("ArraySize(capstoneBuiltinFunc)", "ArraySize(aBuiltinFunc)")
s = s.replace("capstoneBuiltinFunc", "aBuiltinFunc")
if s == before:
    sys.exit("SQLITE_STATIC_BUILTINS: nothing was rewritten -- the patch shape changed")
if "capstoneBuiltinFunc" in s:
    sys.exit("SQLITE_STATIC_BUILTINS: stale references remain")
p.write_text(s)
print("   rewrote aBuiltinFunc to a static initialiser; copy loop removed")
PY
fi

if [[ -n "${BUILTIN_LIMIT:-}" ]]; then
  echo "== DIAGNOSTIC: clamping builtin-function registration to $BUILTIN_LIMIT entries"
  sed -i \
    -e "s/capstoneI<ArraySize(capstoneBuiltinFunc)/capstoneI<(int)($BUILTIN_LIMIT)/" \
    -e "s/capstoneI<ArraySize(aBuiltinFunc)/capstoneI<(int)($BUILTIN_LIMIT)/" \
    -e "s/sqlite3InsertBuiltinFuncs(aBuiltinFunc, ArraySize(aBuiltinFunc))/sqlite3InsertBuiltinFuncs(aBuiltinFunc, (int)($BUILTIN_LIMIT))/" \
    "$OBJ_DIR/sqlite3-capstone.c"
  grep -c "$BUILTIN_LIMIT" "$OBJ_DIR/sqlite3-capstone.c" >/dev/null || {
    echo "BUILTIN_LIMIT clamp did not apply -- the amalgamation patch shape changed" >&2; exit 1; }
fi
# SQLITE_REG_BISECT=1 -- split sqlite3RegisterBuiltinFunctions into its SIX sub-steps behind a
# RUNTIME limit, so one build bisects all of them instead of one rebuild per point.
#
# Stage 10 (MallocInit + RegisterBuiltinFunctions) is the SQLite blocker as of 2026-08-04: it
# wedges first-position on a fresh boot in BOTH build shapes, while every hand-written probe
# (stages 11-16, 18) returns. So the bisection has to happen inside the real function.
#
# capstone_reg_limit == 0 runs the whole function, so a build with this applied and the limit
# left at 0 behaves exactly as before; the domain sets it from the stage selector.
if [[ "${SQLITE_REG_BISECT:-0}" == "1" ]]; then
  echo "== BISECT: splitting sqlite3RegisterBuiltinFunctions into runtime-limited sub-steps"
  python3 - "$OBJ_DIR/sqlite3-capstone.c" <<'PYBI'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
FN = "SQLITE_PRIVATE void sqlite3RegisterBuiltinFunctions(void){"
# Each entry: (anchor, limit value, place-before?).  A limit of K returns just AFTER step K-1,
# so K=1 runs nothing and K=7 (or 0) runs everything.
steps = [
    ("  sqlite3AlterFunctions();",                                        2, True),
    ("  for(int capstoneI=0;",                                            3, True),
    ("  sqlite3WindowFunctions();",                                       4, True),
    ("  sqlite3RegisterDateTimeFunctions();",                             5, True),
    ("  sqlite3RegisterJsonFunctions();",                                 6, True),
    ("  sqlite3InsertBuiltinFuncs(aBuiltinFunc, ArraySize(aBuiltinFunc))",7, True),
]
if FN not in s:
    sys.exit("REG_BISECT: sqlite3RegisterBuiltinFunctions not found -- patch shape changed")
s = s.replace(FN, "int capstone_reg_limit = 0;\n" + FN + "\n  if(capstone_reg_limit==1) return;", 1)
applied = []
for anchor, k, before in steps:
    if anchor not in s:
        continue                      # step compiled out (e.g. SQLITE_OMIT_ALTERTABLE)
    s = s.replace(anchor, "  if(capstone_reg_limit==%d) return;\n%s" % (k, anchor), 1)
    applied.append(k)
# Also clamp sqlite3AlterFunctions' OWN entry count at runtime. Stage 202 (MallocInit +
# AlterFunctions only) wedges first-position on a fresh boot, and AlterFunctions is a single
# sqlite3InsertBuiltinFuncs(aAlterTableFuncs, 9). Clamping that 9 bisects to the exact entry.
# -1 (the default) means "unclamped", so a build with this applied and the knob untouched
# behaves exactly as before.
ALT = "  sqlite3InsertBuiltinFuncs(aAlterTableFuncs, ArraySize(aAlterTableFuncs));"
if ALT in s:
    s = s.replace(ALT,
        "  sqlite3InsertBuiltinFuncs(aAlterTableFuncs,\n"
        "      capstone_alter_limit >= 0 ? capstone_alter_limit : (int)ArraySize(aAlterTableFuncs));", 1)
    s = s.replace("SQLITE_PRIVATE void sqlite3AlterFunctions(void){",
                  "int capstone_alter_limit = -1;\nSQLITE_PRIVATE void sqlite3AlterFunctions(void){", 1)
    applied.append("alter")
else:
    sys.exit("REG_BISECT: aAlterTableFuncs insert call not found -- patch shape changed")
if len(applied) < 3:
    sys.exit("REG_BISECT: only %d anchors matched (%r) -- patch shape changed" % (len(applied), applied))
p.write_text(s)
print("   sub-step returns installed at limits: 1 (entry) + %s" % applied)
PYBI
fi

# PREFLIGHT GATE (added 2026-08-04 after this exact mistake cost several board sessions).
#
# The staged dispatch in sqlite_capstone_domain.c -- INCLUDING the read of the host's stage
# selector -- lives inside `#ifdef CAPSTONE_SQLITE_STAGE`, and DOMAIN_EXTRA_DEFS defaults to
# empty. So a build without it SILENTLY IGNORES the selector and runs the full workload: every
# `dom:NNN` invocation returns the same thing, and the run looks like a legitimate verdict
# about stage NNN when the stage never executed. Three board sessions were spent bisecting
# sub-steps of a function that was never entered under a clamp.
#
# SQLITE_REG_BISECT's stages (200-206, 210-219) are reachable ONLY through that dispatch, so
# requesting the bisect without the staged build is always a mistake. Fail loudly instead.
if [[ "${SQLITE_REG_BISECT:-0}" == "1" ]] && [[ "${DOMAIN_EXTRA_DEFS:-}" != *CAPSTONE_SQLITE_STAGE* ]]; then
  echo "FATAL: SQLITE_REG_BISECT=1 needs the staged dispatch compiled in, or the stage" >&2
  echo "       selector is ignored and every stage silently runs the FULL workload." >&2
  echo "       Add: DOMAIN_EXTRA_DEFS=-DCAPSTONE_SQLITE_STAGE=0" >&2
  exit 1
fi

cp -f "$VFS_DIR/capstone_sqlite_vfs.c"    "$OBJ_DIR/capstone_sqlite_vfs.c"
cp -f "$VFS_DIR/../sqlite-vfs-skeleton/capstone_sqlite_os.c" "$OBJ_DIR/capstone_sqlite_os.c" 2>/dev/null \
  || cp -f "$ADAPTED/capstone_sqlite_os.c" "$OBJ_DIR/capstone_sqlite_os.c"
cp -f "${DOMAIN_SRC:-$SCRIPT_DIR/sqlite_capstone_domain.c}" "$OBJ_DIR/sqlite_capstone_domain.c"
cp -f "$SCRIPT_DIR/sqlite_silicon_amalgam.c" "$OBJ_DIR/amalgam.c"

SQLITE_DEFINES=$(sed -n '/^SQLITE_DEFINES=(/,/^)/p' "$SCRIPT_DIR/build-sqlite-capstone.sh" \
                 | grep -oE '\-D[A-Za-z0-9_]+(=[^ ]*)?' | tr '\n' ' ')

# CARVE-COUNT TRIM -- silicon only, and load-bearing rather than cosmetic.
#
# The entry glue performs one `split` per global to carve its capability, and every split
# allocates a revocation node. The RTL allocator's head is 10 bits starting at 3
# (capstone_rev_node.anvil:160,168), so allocation ~#1022 wraps to id 0 and reuses LIVE
# ids. Reuse can splice a node into the `next` chain twice, and REVOKE_NODE (:13-32) has
# no visit bound and no cycle detection -- it then walks forever and never answers another
# query. Since every `stc` blocks on a revocation-node query with no timeout
# (capstone_dyn_unit.anvil:395-404), the next capability store hangs with no trap.
#
# Untrimmed SQLite needs 1059 carves, which is over that limit. These OMITs each remove
# whole global tables rather than a few bytes, and none of them touches what the five
# success markers exercise (CREATE/INSERT/SELECT, transactions, a secondary index,
# prepared bound statements, UPDATE/DELETE, aggregates and the sorter, JOIN, GROUP BY,
# string functions). Verify with gp-carve-count.py and with run-sqlite-silicon.sh under
# QEMU -- a trim that fits the pool but breaks SQLite is worthless.
#
# This is a silicon-only deviation and must be reported alongside any board number.
# AMALGAMATION-SAFE ONLY. SQLite supports SQLITE_OMIT_* officially only when building
# from canonical sources, because most of them require regenerating parse.c with lemon.
# Against the prebuilt amalgamation, OMIT_TRIGGER / OMIT_VIEW / OMIT_CTE / OMIT_WINDOWFUNC
# / OMIT_UPSERT / OMIT_VIRTUALTABLE / OMIT_ATTACH / OMIT_ANALYZE / OMIT_VACUUM /
# OMIT_REINDEX / OMIT_AUTOVACUUM all fail to compile -- the parser tables still reference
# sqlite3TriggerInsertStep, sqlite3WindowListDelete, sqlite3CteNew and friends. Measured,
# not assumed: 20 errors of exactly that shape. Do not re-add them without regenerating
# the amalgamation.
SILICON_TRIM=(
  -DSQLITE_OMIT_AUTHORIZATION=1
  -DSQLITE_OMIT_TRACE=1
  -DSQLITE_OMIT_PROGRESS_CALLBACK=1
  -DSQLITE_OMIT_INTROSPECTION_PRAGMAS=1
  -DSQLITE_OMIT_XFER_OPT=1
  -DSQLITE_OMIT_COMPLETE=1
  # Non-grammar omissions: these delete function-definition tables and their name
  # strings without changing the token stream, so the prebuilt parser stays valid.
  -DSQLITE_OMIT_DATETIME_FUNCS=1
  -DSQLITE_OMIT_LIKE_OPTIMIZATION=1
  -DSQLITE_OMIT_OR_OPTIMIZATION=1
  -DSQLITE_OMIT_BETWEEN_OPTIMIZATION=1
  -DSQLITE_OMIT_TRUNCATE_OPTIMIZATION=1
  -DSQLITE_OMIT_QUICKBALANCE=1
  -DSQLITE_OMIT_SCHEMA_VERSION_PRAGMAS=1
  -DSQLITE_OMIT_FLAG_PRAGMAS=1
  # OMIT_PRAGMA compiles but does not LINK: the prebuilt parser still calls
  # sqlite3Pragma / sqlite3PragmaVtabRegister. Same amalgamation limit as the grammar set.
)
# OFF BY DEFAULT -- THE TRIM BREAKS SQLITE. Measured 2026-07-31: with these OMITs the
# silicon build compiles and links clean but fails under QEMU with a capability fault at the
# domain's first entry; with SQLITE_NO_TRIM=1 the SAME tree passes end-to-end
# (__CAPSTONE_SQLITE_SILICON_PASSED__). SQLite supports SQLITE_OMIT_* only when building
# from canonical sources; against the prebuilt amalgamation some of them compile and then
# misbehave, which is what happened here.
#
# This cost a long detour: the resulting fault was investigated as a cap-init bug and
# bisected to a specific initializer leaf before the trim itself was tested properly. The
# earlier "baseline without the trim also fails" check was INVALID -- SQLITE_NO_TRIM was set
# only as a prefix on the build command, and run-sqlite-silicon.sh:19 rebuilds the domain
# unconditionally, so the run silently restored the trim.
#
# Opt in with SQLITE_TRIM=1 only to re-measure the carve count; never for a correctness run.
[[ "${SQLITE_TRIM:-0}" == "1" ]] || SILICON_TRIM=()

# STRING MERGING IS REQUIRED FOR THIS DOMAIN, not an optimisation. One capability carve per
# global costs one revocation node, and the board's allocator wraps after 1021: untrimmed
# SQLite needs 1059 carves and overflowed the pool on silicon (measured 2026-07-31, head=74
# with the overflow flag set). Merging the private read-only string literals takes it to 179
# carves, ~215 allocations total. Enabled HERE rather than by default so the silicon-ladder
# rungs keep their measured geometry -- tab:spatialcost's BEEBS numbers were deliberately
# taken with merging off and must not silently change.
SILICON=(-mllvm -capstone-merge-string-constants=true
         -mllvm -capstone-gp-captable
         -mllvm -capstone-shrink-stack=false
         -mllvm -capstone-shrink-globals=false
         -fno-jump-tables
         # Tells domain sources they are on the gp-captable ABI, where every global is
         # reached through a cap-table storage capability that is ALREADY NONLIN. Domain
         # code must not delin such a capability: the RTL's DELIN takes CAP_TYPE_LINEAR
         # only and faults otherwise (QEMU's is idempotent, so it hides this). See
         # output_text() in sqlite_capstone_domain.c and C-13.
         -DCAPSTONE_GP_CAPTABLE_ABI=1)
# EXTRA_MLLVM lets a bisect turn one backend pass off without editing this script, e.g.
#   EXTRA_MLLVM="-mllvm -capstone-fix-destructive-copies=false"
read -r -a _extra_mllvm <<< "${EXTRA_MLLVM:-} ${SQLITE_DIAG:-}"
SILICON+=("${_extra_mllvm[@]}")

COMMON=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
        -ffreestanding -fno-builtin -fno-optimize-sibling-calls
        -include "$ADAPTED/capstone_sqlite_libc.h"
        -I"$ADAPTED" -I"$SCRIPT_DIR" -I"$VFS_DIR" -I"$OBJ_DIR"
        -I"$(dirname "$PATCHED")" -I"$BUILTINS")

echo "== compiling the single silicon TU (this is the first time SQLite sees the silicon ABI)"
# DOMAIN_EXTRA_DEFS reaches sqlite_capstone_domain.c (it is #included by amalgam.c), so a
# diagnostic build can be produced without editing this script -- e.g.
#   DOMAIN_EXTRA_DEFS=-DCAPSTONE_SQLITE_STAGE=2
# for the staged-return bisection. Empty by default, so the normal build is unaffected.
read -r -a _domain_defs <<< "${DOMAIN_EXTRA_DEFS:-}"
"$CAPSTONE_CLANG" "${COMMON[@]}" "${SILICON[@]}" $SQLITE_DEFINES "${SILICON_TRIM[@]}" "$OPT" \
  -DSQLITE_HEAP_SIZE=$HEAP "${_domain_defs[@]}" \
  -c "$OBJ_DIR/amalgam.c" -o "$OBJ_DIR/amalgam.o"

echo "== compiling the no-globals support objects separately (they cannot collide)"
# SUPPORT_OPT is separate from $OPT because these two objects hold the string primitives
# (strlen/strcmp/memcpy) and are therefore where the domain spends its tight loops, while
# the amalgamation is where -O1 currently cannot go.
#
# At -O0 every pointer is spilled, so `strlen` round-trips its walking pointer through a
# stack CAPABILITY slot twice per iteration -- `ldc`, `lbu`, `ldc` again from the same
# slot, `cincoffsetimm`, `stc`. The board froze at exactly the `cincoffsetimm` of that
# sequence (image VA 0x14d884, ra -> sqlite3Strlen30), pc not advancing under stepi with
# mcause=0. At -O1 the whole pattern is gone: the pointer stays in a register and the loop
# contains no ldc/stc at all. Whether the round-trip is the CAUSE is not established --
# QEMU executes the -O0 form happily, so the board is the only oracle -- but it is the one
# construct at the frozen pc that -O1 removes.
#
# Why not raise $OPT itself: the amalgamation does not compile above -O0. `cond ? capA :
# capB` reaches ISel as an i128 CapstoneISD::SELECT_CC, and the only patterns for it
# (Select_GPRCAP_Using_CC_GPR) are emitted under !is64Bit(), so on capstone64 there is no
# i128 select pattern and the backend aborts with "Cannot select". Reproducer:
# `char *pick(int n, char *a, char *b) { return n == 10 ? a : b; }` at -O1. The n==0 form
# compiles because SelectCC_GPR_rrirr adds a separate explicit Pat for a zero rhs.
SUPPORT_OPT=${SQLITE_SUPPORT_OPT_LEVEL:-$OPT}
# BEEBS_STRING_LINEAR_SAFE: index instead of walking, so the string primitives never copy
# or advance a capability that may be LINEAR. SQLite is the first thing here to call strlen
# on hardware (no ladder rung references it), and both the -O0 and -O1 pointer-walking
# forms freeze on the board at the instruction that advances the pointer. See the header
# comment on strlen in beebs_freestanding_string.c for the RTL citations. Set only here --
# the ladder rungs keep the walking form so their published geometry is unchanged.
SUPPORT_DEFS=(-DBEEBS_STRING_LINEAR_SAFE=1 ${BEEBS_STRING_EXTRA_DEFS:-})
for pair in "libc:$ADAPTED/capstone_sqlite_libc.c" "beebs_string:$BEEBS_STRING"; do
  "$CAPSTONE_CLANG" "${COMMON[@]}" "${SILICON[@]}" $SQLITE_DEFINES "${SILICON_TRIM[@]}" \
    "${SUPPORT_DEFS[@]}" "$SUPPORT_OPT" \
    -c "${pair#*:}" -o "$OBJ_DIR/${pair%%:*}.o"
done
BUILTIN_OBJS=()
"$CAPSTONE_CLANG" "${COMMON[@]}" "${SILICON[@]}" -O0 \
  -c "$SCRIPT_DIR/capstone_floatdidf_noglobals.c" -o "$OBJ_DIR/floatdidf_ng.o"
BUILTIN_OBJS+=("$OBJ_DIR/floatdidf_ng.o")
# floatdidf is replaced by our globals-free version (see the amalgam header).
for b in eqdf2 fixdfdi fixdfsi gedf2 gtdf2 ltdf2 muldf3 nedf2 adddf3 subdf3 comparedf2 \
         fixunsdfdi fixunsdfsi floatsidf floatunsidf fp_mode; do
  [[ -f "$BUILTINS/$b.c" ]] || continue
  "$CAPSTONE_CLANG" "${COMMON[@]}" "${SILICON[@]}" -O0 \
    -c "$BUILTINS/$b.c" -o "$OBJ_DIR/$b.o" 2>/dev/null && BUILTIN_OBJS+=("$OBJ_DIR/$b.o")
done

# Two-pass link: pass 1 only to measure .text, pass 2 with the real globals offset.
link() {  # $1 = globals offset literal, $2 = output
  local lds="$OBJ_DIR/link.ld"
  sed "s/0x10000 + 0x1000/0x10000 + $1/" "$GPFREE/link-gpfree.ld" > "$lds"
  # INTERP_BUILD_LIMIT=<N> (diagnostic only) clamps how many carve iterations the glue
  # runs while leaving the cap-table geometry byte-identical. It is the discriminator for
  # R-12: SQLite's descriptor count is 1059, so the builder performs ~1060 `split`s against
  # a 1024-entry rev-node pool whose head is 10 bits -- allocation #1025 wraps to id 0 and
  # reuses live ids silently. A limit below 1024 never exhausts the pool.
  "$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding \
    ${INTERP_EXTRA_CFLAGS:-} \
    -c "$LADDER/start-gp-captable-interp.S" -o "$OBJ_DIR/start.o"
  "$CAPSTONE_LD_LLD" -T "$lds" -o "$2" \
    "$OBJ_DIR/start.o" "$OBJ_DIR/amalgam.o" "$OBJ_DIR/libc.o" \
    "$OBJ_DIR/beebs_string.o" "${BUILTIN_OBJS[@]}"
}

# Pass 1 uses a deliberately OVERSIZED provisional offset. Linking at the 0x1000
# default cannot work -- SQLite's .text is ~1.3 MB and would overlap the globals
# region, which is the whole reason the offset has to be computed. 8 MiB is chosen to
# be larger than any plausible .text so pass 1 always links and can be measured.
echo "== pass 1: link at a provisional 8 MiB offset, only to measure .text"
link 0x800000 "$OUT_DIR/pass1.dom"
# Parsed in python, not awk: strtonum() is a gawk extension and mawk silently returns
# 0, which produced ".text = 0 bytes" and a bogus offset on the first attempt.
TEXT=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT_DIR/pass1.dom" 2>/dev/null | python3 -c '
import sys,re
for l in sys.stdin:
    m=re.match(r"\s*\[\s*\d+\]\s+(\.text)\s+\S+\s+[0-9a-f]+\s+[0-9a-f]+\s+([0-9a-f]+)", l)
    if m: print(int(m.group(2),16)); break
else: print(0)')
: "${TEXT:=0}"
[[ "$TEXT" -gt 0 ]] || { echo "could not measure .text from pass 1" >&2; exit 1; }
# Round up to 64 KiB so the boundary is representable and there is headroom.
GOFF=$(( ((TEXT + 0xFFFF) / 0x10000) * 0x10000 ))
[[ $GOFF -lt 65536 ]] && GOFF=65536
printf "   .text = %d bytes -> globals offset 0x%x\n" "$TEXT" "$GOFF"

echo "== pass 2: link with the real globals offset"
link "$(printf '0x%x' $GOFF)" "$OUT_DIR/sqlite_silicon.dom"

echo "== gates"
DIS=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT_DIR/sqlite_silicon.dom")
echo "   cjalr=$(grep -cE '\bcjalr\b' <<<"$DIS" || true)  ldc-gp=$(grep -cE 'ldc[[:space:]].*\(gp\)' <<<"$DIS" || true)"
NHDR=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT_DIR/sqlite_silicon.dom" | grep -c "capstone_gp_table" || true)
echo "   .capstone_gp_table sections: $NHDR (must be 1 -- more means a multi-TU index collision)"
python3 "$LADDER/domdata-budget.py" "$OUT_DIR/sqlite_silicon.dom" || true
echo "Built $OUT_DIR/sqlite_silicon.dom"

# Artifact check: a staged build MUST contain the 0x5A6E marker and the staged-only literal.
# Checking the flag is not enough -- this verifies what actually got compiled.
if [[ "${DOMAIN_EXTRA_DEFS:-}" == *CAPSTONE_SQLITE_STAGE* ]]; then
  python3 - "$OUT_DIR/sqlite_silicon.dom" <<'PYCHK' || exit 1
import sys, pathlib
d = pathlib.Path(sys.argv[1]).read_bytes()
marker = d.count(bytes.fromhex("6e5a"))          # lui rd, 0x5a6e0, any rd
probe  = b"capstone_probe_string" in d           # a literal only in the staged block
if marker == 0 or not probe:
    sys.exit(f"FATAL: staged build requested but the artifact is NOT staged "
             f"(marker={marker}, probe={probe}). The selector would be ignored and every "
             f"stage would silently run the full workload.")
print(f"   staged-dispatch verified in artifact (marker x{marker}, probe literal present)")
PYCHK
fi
