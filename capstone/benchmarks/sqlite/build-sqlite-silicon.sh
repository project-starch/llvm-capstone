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
  # GATE. The old check here was `grep -c "$BUILTIN_LIMIT" <9.5 MB C file>`, which searches for
  # the NUMBER AS A SUBSTRING and therefore matched thousands of unrelated hits for N=0/1/8.
  # It could never fail, and a build whose clamp silently did not apply would be measured as
  # though it had -- which is how a whole bisection series became VOID once already. This
  # version asserts the exact post-sed loop bound exists, and negative-tests itself by also
  # asserting the UNCLAMPED form is gone.
  _want="capstoneI<(int)($BUILTIN_LIMIT)"
  _n_clamped=$(grep -c -F "$_want" "$OBJ_DIR/sqlite3-capstone.c" || true)
  _n_orig=$(grep -c -F "capstoneI<ArraySize(capstoneBuiltinFunc)" "$OBJ_DIR/sqlite3-capstone.c" || true)
  echo "== clamp gate: '$_want' x$_n_clamped ; unclamped form x$_n_orig"
  if [[ "$_n_clamped" -lt 1 || "$_n_orig" -ne 0 ]]; then
    echo "BUILTIN_LIMIT clamp did NOT apply (clamped=$_n_clamped unclamped-remaining=$_n_orig)." >&2
    echo "The amalgamation patch shape changed. Refusing to build a domain that would be" >&2
    echo "measured as clamped while running the full array." >&2
    exit 1
  fi
fi
# REGBUILTIN_STOP=<n> -- DIAGNOSTIC ONLY. Clamp sqlite3RegisterBuiltinFunctions() to
# return after the n-th sub-step, so a wedge inside it bisects by CALL INDEX.
#
# WHY THIS EXISTS. BUILTIN_LIMIT only empties the arg-fixup loop and the
# InsertBuiltinFuncs count. Four other registration calls still run, so a wedge at
# BUILTIN_LIMIT=0 (measured 2026-08-09) says nothing about which call is at fault.
#
#   0 = return at entry            (nothing runs -- must RETURN or the fault is upstream)
#   1 = after sqlite3AlterFunctions()
#   2 = after the arg-fixup loop
#   3 = after sqlite3WindowFunctions()
#   4 = after sqlite3RegisterDateTimeFunctions()
#   5 = after sqlite3RegisterJsonFunctions()
#   6 = after sqlite3InsertBuiltinFuncs()   (== unclamped)
if [[ -n "${REGBUILTIN_STOP:-}" ]]; then
  echo "== DIAGNOSTIC: clamping sqlite3RegisterBuiltinFunctions to stop after step $REGBUILTIN_STOP"
  python3 - "$OBJ_DIR/sqlite3-capstone.c" "$REGBUILTIN_STOP" <<'PYSTOP'
import sys,re
path,n = sys.argv[1], int(sys.argv[2])
s=open(path).read()
m=re.search(r'void sqlite3RegisterBuiltinFunctions\(void\)\s*\{', s)
if not m: print("REGBUILTIN_STOP: function not found",file=sys.stderr); sys.exit(1)
# marker so the artifact can be PROVEN to carry the clamp
mark='  volatile int capstone_regstop_mark = 0x5C0%X;\n' % n
body_start=m.end()
# anchors, in order; each gets a return AFTER it when n <= index
anchors=[(0,None),
         (1,'sqlite3AlterFunctions();'),
         (2,'  }\n  sqlite3WindowFunctions();'),
         (3,'sqlite3WindowFunctions();'),
         (4,'sqlite3RegisterDateTimeFunctions();'),
         (5,'sqlite3RegisterJsonFunctions();')]
ins=0
if n==0:
    s = s[:body_start] + '\n' + mark + '  if(capstone_regstop_mark) return;\n' + s[body_start:]
    ins=1
else:
    tgt=[a for i,a in anchors if i==n and a]
    if not tgt: print("REGBUILTIN_STOP: no anchor for n=%d"%n,file=sys.stderr); sys.exit(1)
    t=tgt[0]
    idx=s.find(t, body_start)
    if idx<0: print("REGBUILTIN_STOP: anchor %r not found"%t,file=sys.stderr); sys.exit(1)
    after=idx+len(t)
    s = s[:after] + '\n' + mark + '  if(capstone_regstop_mark) return;\n' + s[after:]
    ins=1
open(path,'w').write(s)
print("REGBUILTIN_STOP: inserted %d early-return(s) for step %d" % (ins,n))
PYSTOP
  # GATE, with BOTH directions checked against the file that is actually compiled.
  _mark=$(printf '0x5C0%X' "$REGBUILTIN_STOP")
  _nmark=$(grep -c -F "capstone_regstop_mark = $_mark" "$OBJ_DIR/sqlite3-capstone.c" || true)
  _nret=$(grep -c -F "if(capstone_regstop_mark) return;" "$OBJ_DIR/sqlite3-capstone.c" || true)
  echo "== regstop gate: marker $_mark x$_nmark ; early-return x$_nret"
  if [[ "$_nmark" -ne 1 || "$_nret" -ne 1 ]]; then
    echo "REGBUILTIN_STOP did NOT apply cleanly (marker=$_nmark return=$_nret). Refusing to build." >&2
    exit 1
  fi
fi

# CAPSTONE_TEXT_PAD=N -- insert N bytes of dead, never-called code at the TOP of the
# amalgamation and change NOTHING else. No early return, no control-flow change: this shifts
# every following function by N bytes and is the null-shift control the audit asked for.
#
# Why: uc vs g1 differ by an opaque early return inside an UNCALLED function, and g1's stage 11
# hangs while uc's returns -- verified in one boot with position controlled both ways. But that
# edit also moved 1931 functions by +52 bytes and relocated `strlen` itself from 0x14fc1c to
# 0x14fc50, a different 16-byte alignment. "The edit" and "the shift" are the same intervention
# there. Varying ONLY the shift separates them.
if [[ -n "${CAPSTONE_TEXT_PAD:-}" ]]; then
  echo "== TEXT PAD: inserting ${CAPSTONE_TEXT_PAD} bytes of dead code ahead of everything"
  python3 - "$OBJ_DIR/sqlite3-capstone.c" "$CAPSTONE_TEXT_PAD" <<'PYPAD'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); n = int(sys.argv[2])
if n % 4:
    sys.exit("FATAL: CAPSTONE_TEXT_PAD must be a multiple of 4 (instruction size)")
nops = "\n".join(['  __asm__ volatile("nop");'] * (n // 4))
pad = ('__attribute__((used,noinline)) static void capstone_text_pad(void)\n{\n'
       + nops + '\n}\n')
p.write_text(pad + p.read_text())
print(f"   inserted {n} bytes ({n//4} nops) of dead code")
PYPAD
fi

# SQLITE_REG_BISECT=1 -- split sqlite3RegisterBuiltinFunctions into its sub-steps behind a
# COMPILE-TIME limit, one image per point.
#
# Compile-time, not runtime, and that is the whole point. The runtime version added two
# globals (179 -> 183 carves) and the resulting image ENTRY-STALLED 3/3 -- domain entry is
# exquisitely sensitive to gp-captable layout, so instrumentation must not touch the global
# set at all. A macro compare folds at compile time: no new globals, no layout change, and
# exactly one early return survives per build.
#
# CAPSTONE_REG_LIMIT=K returns just BEFORE sub-step K (K=1 runs nothing, unset/0 runs all).
# CAPSTONE_ALTER_LIMIT=N clamps sqlite3AlterFunctions' own entry count (-1 = unclamped).
# Pass them via SQLITE_EXTRA_DEFS so they reach the amalgamation compile.
if [[ "${SQLITE_REG_BISECT:-0}" == "1" ]]; then
  echo "== BISECT: compile-time sub-step limit CAPSTONE_REG_LIMIT=${CAPSTONE_REG_LIMIT:-unset}"
  python3 - "$OBJ_DIR/sqlite3-capstone.c" <<'PYBI'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
FN = "SQLITE_PRIVATE void sqlite3RegisterBuiltinFunctions(void){"
if FN not in s:
    sys.exit("REG_BISECT: sqlite3RegisterBuiltinFunctions not found -- patch shape changed")
steps = [("  sqlite3AlterFunctions();", 2),
         ("  for(int capstoneI=0;", 3),
         ("  sqlite3WindowFunctions();", 4),
         ("  sqlite3RegisterDateTimeFunctions();", 5),
         ("  sqlite3RegisterJsonFunctions();", 6),
         ("  sqlite3InsertBuiltinFuncs(aBuiltinFunc, ArraySize(aBuiltinFunc))", 7)]
s = ("#ifndef CAPSTONE_REG_LIMIT\n#define CAPSTONE_REG_LIMIT 0\n#endif\n"
     "#ifndef CAPSTONE_ALTER_LIMIT\n#define CAPSTONE_ALTER_LIMIT -1\n#endif\n"
     "#ifndef CAPSTONE_INSERT_STOP\n#define CAPSTONE_INSERT_STOP 0\n#endif\n"
     "#define CAPSTONE_STOP() do { int cs_=1; __asm__ volatile(\"\" : \"+r\"(cs_)); "
     "if (cs_) return; } while (0)\n") + s
s = s.replace(FN, FN + "\n#if CAPSTONE_REG_LIMIT==1\n  CAPSTONE_STOP();\n#endif", 1)
applied = []
for anchor, k in steps:
    if anchor not in s:
        continue
    s = s.replace(anchor, "#if CAPSTONE_REG_LIMIT==%d\n  CAPSTONE_STOP();\n#endif\n%s" % (k, anchor), 1)
    applied.append(k)
# CAPSTONE_INSERT_STOP=N: split the ONE loop iteration of sqlite3InsertBuiltinFuncs that
# board-bisected as the wedge (0 entries returns, 1 entry wedges). Compile-time, so no new
# globals. N returns just after sub-operation N; 0 = unclamped.
BODY = ("    const char *zName = aDef[i].zName;\n"
        "    int nName = sqlite3Strlen30(zName);\n"
        "    int h = SQLITE_FUNC_HASH(zName[0], nName);\n")
if BODY in s:
    s = s.replace(BODY,
        "    const char *zName = aDef[i].zName;\n"
        "#if CAPSTONE_INSERT_STOP==1\n    if(zName) return;\n#endif\n"
        # 7 = read ONE byte of zName inline; 5 = full inline length loop. Both dereference
        # zName WITHOUT calling sqlite3Strlen30. n1 (load only) returns and n2 (after the
        # call) wedges, so the fault is the call boundary or the dereference; these split it.
        "#if CAPSTONE_INSERT_STOP==7\n    if(zName[0]) return;\n#endif\n"
        "#if CAPSTONE_INSERT_STOP==5\n    { int k_=0; while(zName[k_]) k_++; if(k_>=0) return; }\n#endif\n"
        # 8 = the SAME byte-walk but BOUNDED to 32. p5 (unbounded) does not return while p7
        # (one byte) does, so either the loads themselves are fatal or the string never
        # reaches a NUL and the loop spins forever. A bounded walk separates them: if 8
        # RETURNS, the loads are fine and p5 was an infinite loop, i.e. the string data or
        # its capability is wrong -- a wrong ANSWER, not a fault. Converting the hang into a
        # returning run is the only way to tell those apart.
        "#if CAPSTONE_INSERT_STOP==8\n    { int k_=0; while(k_<32 && zName[k_]) k_++; if(k_>=0) return; }\n#endif\n"
        # 9 = returns ONLY if a NUL is actually found within 32 bytes; otherwise it falls
        # through to the real code and wedges. 8 returns unconditionally after &lt;=32 loads, so
        # it proves the loads work but not that the string is terminated. This is the yes/no:
        # RETURN  -> zName IS a terminated string, and p5 spun for some other reason
        # NO RET  -> no NUL in 32 bytes, so zName does not point at the expected name and
        #            the "wedge" is an infinite strlen over wrong data.
        "#if CAPSTONE_INSERT_STOP==9\n    { int k_=0; while(k_<32 && zName[k_]) k_++; if(k_<32) return; }\n#endif\n"
        # Value channel: the clamp sits in a void function, so a byte can only be reported as
        # "returned or not". 10 returns iff zName[0] is the expected 's'; 11 returns iff the
        # first EIGHT bytes sum to 867 == sum("sqlite_r"). The literal is verified present and
        # correctly placed inside merged string container descriptor 177 in the IMAGE, so
        # these test what the glue copy produced at RUNTIME.
        "#if CAPSTONE_INSERT_STOP==10\n    if(zName[0]==0x73) return;\n#endif\n"
        "#if CAPSTONE_INSERT_STOP==11\n    { int k_=0,s_=0; for(k_=0;k_<8;k_++) s_+=(unsigned char)zName[k_]; if(s_==867) return; }\n#endif\n"
        "    int nName = sqlite3Strlen30(zName);\n"
        "#if CAPSTONE_INSERT_STOP==2\n    if(nName>=0) return;\n#endif\n"
        "    int h = SQLITE_FUNC_HASH(zName[0], nName);\n"
        "#if CAPSTONE_INSERT_STOP==3\n    if(h>=0) return;\n#endif\n", 1)
    applied.append("body")
    SEARCH = "    pOther = sqlite3FunctionSearch(h, zName);"
    if SEARCH in s:
        s = s.replace(SEARCH, SEARCH + "\n#if CAPSTONE_INSERT_STOP==4\n    return;\n#endif", 1)
        applied.append("search")
ALT = "  sqlite3InsertBuiltinFuncs(aAlterTableFuncs, ArraySize(aAlterTableFuncs));"
if ALT in s:
    s = s.replace(ALT,
        "  sqlite3InsertBuiltinFuncs(aAlterTableFuncs,\n"
        "      (CAPSTONE_ALTER_LIMIT >= 0 ? CAPSTONE_ALTER_LIMIT : (int)ArraySize(aAlterTableFuncs)));", 1)
    applied.append("alter")
if len(applied) < 4:
    sys.exit("REG_BISECT: only %d anchors matched (%r) -- patch shape changed" % (len(applied), applied))
# No new globals may be introduced -- that is what broke the runtime version.
if "capstone_reg_limit" in s or "capstone_alter_limit" in s:
    sys.exit("REG_BISECT: a runtime global leaked in; the clamp must be compile-time only")
p.write_text(s)
print("   compile-time returns installed at limits: 1 (entry) + %s" % applied)
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

# FAILSTOP=<n> -- DIAGNOSTIC ONLY. Clamp inside fail(), which WEDGES (S-02 Site A,
# measured 2026-08-09: nk takes the fail() path and wedges, n8 skips it and returns in 4 s
# with an otherwise identical image).
#
#   1 = return at fail() entry, before ANY output_text  -> isolates the call/prologue of fail
#   2 = after output_text("SQLITE ERROR stage=")        -> a STRING LITERAL capability
#   3 = after output_text(stage)                        -> a PASSED string capability
#
# Pair with INITSTOP=9 so the fail() path is actually taken.
if [[ -n "${FAILSTOP:-}" ]]; then
  echo "== DIAGNOSTIC: clamping fail() after sub-step $FAILSTOP"
  python3 - "$OBJ_DIR/sqlite_capstone_domain.c" "$FAILSTOP" <<'PYFAIL'
import sys,re
path,n=sys.argv[1],int(sys.argv[2])
s=open(path).read()
m=re.search(r'static int fail\(const char \*stage, int rc, sqlite3 \*db\) \{', s)
if not m: print("FAILSTOP: fail() not found",file=sys.stderr); sys.exit(1)
start=m.end()
if n==1:
    after=start
else:
    tgt={2:'output_text("SQLITE ERROR stage=");', 3:'output_text(stage);'}[n]
    i=s.find(tgt, start)
    if i<0: print("FAILSTOP: anchor %r not found"%tgt,file=sys.stderr); sys.exit(1)
    after=i+len(tgt)
ins='\n  { volatile int capstone_failstop = 0x7C0%X; if(capstone_failstop) return capstone_failstop; }\n' % n
s=s[:after]+ins+s[after:]
open(path,'w').write(s)
print("FAILSTOP: early return inserted at sub-step %d"%n)
PYFAIL
  _m=$(printf '0x7C0%X' "$FAILSTOP")
  _n=$(grep -c -F "capstone_failstop = $_m" "$OBJ_DIR/sqlite_capstone_domain.c" || true)
  _r=$(grep -A1 -F "capstone_failstop = $_m" "$OBJ_DIR/sqlite_capstone_domain.c" | grep -c -F "return capstone_failstop;" || true)
  echo "== failstop gate: marker $_m x$_n ; return x$_r"
  if [[ "$_n" -ne 1 || "$_r" -ne 1 ]]; then
    echo "FAILSTOP did NOT apply (marker=$_n return=$_r). Refusing to build." >&2; exit 1
  fi
fi

# INITSTOP=<n> -- DIAGNOSTIC ONLY. Early-return from sqlite3_initialize() after sub-step n,
# to bisect S-02 (the wedge is inside initialize(); measured 2026-08-09 by rn1 RETURN /
# rn2 WEDGE, both on caplifive_65536_r18_fix.bit with a green control).
#
#   1 = after sqlite3MutexInit()
#   2 = after sqlite3MallocInit()
#   3 = after sqlite3RegisterBuiltinFunctions()   (already EXONERATED by rs0 -- ordering control)
#   4 = after sqlite3PcacheInitialize()
#   5 = after sqlite3OsInit()
#   6 = after sqlite3MemdbInit()
#
# Lightweight on purpose: one early return, nothing else. Heavyweight staged builds do NOT
# ENTER on this bitstream (0/4) while early-return builds enter 4/4.
if [[ -n "${INITSTOP:-}" ]]; then
  echo "== DIAGNOSTIC: clamping sqlite3_initialize() after sub-step $INITSTOP"
  python3 - "$OBJ_DIR/sqlite3-capstone.c" "$INITSTOP" <<'PYINIT'
import sys,re
path,n=sys.argv[1],int(sys.argv[2])
s=open(path).read()
m=re.search(r'int sqlite3_initialize\(void\)\s*\{', s)
if not m: print("INITSTOP: sqlite3_initialize not found",file=sys.stderr); sys.exit(1)
start=m.end()
anchors={8:'__NAKED0__',
         9:'__NAKED__',
         0:'__ENTRY__',
         7:'__BEFORE_MUTEXINIT__',
         1:'rc = sqlite3MutexInit();',
         2:'rc = sqlite3MallocInit();',
         3:'sqlite3RegisterBuiltinFunctions();',
         4:'rc = sqlite3PcacheInitialize();',
         5:'rc = sqlite3OsInit();',
         6:'rc = sqlite3MemdbInit();'}
t=anchors.get(n)
if not t: print("INITSTOP: no anchor for %d"%n,file=sys.stderr); sys.exit(1)
if t in ('__NAKED__','__NAKED0__'):
    ZERO = (t=='__NAKED0__')   # NAKED0 returns 0 so run_sqlite does NOT take the fail() path
    # Replace the ENTIRE body with a bare return. At -O0 the frame covers every local
    # declared anywhere in the function, so an early return at the top (INITSTOP=0) still
    # emits the full prologue including the `stc s0` capability spill. Deleting the body
    # shrinks the frame to the minimum and removes that spill -- which is what separates
    # "the CALL wedges" from "the PROLOGUE wedges".
    depth=1; j=start
    while j<len(s) and depth>0:
        if s[j]=='{': depth+=1
        elif s[j]=='}': depth-=1
        j+=1
    body_end=j-1
    _ret = '0' if ZERO else 'capstone_initstop'
    s = s[:start] + ('\n  volatile int capstone_initstop = 0x7B0%X;\n  return %s;\n' % (n,_ret)) + s[body_end:]
    open(path,'w').write(s)
    print("INITSTOP: sqlite3_initialize body REPLACED with a bare return (naked)")
    raise SystemExit
if t=='__ENTRY__':
    after=start                      # immediately after the opening brace: nothing runs
elif t=='__BEFORE_MUTEXINIT__':
    i=s.find('rc = sqlite3MutexInit();', start)
    if i<0: print("INITSTOP: MutexInit call not found",file=sys.stderr); sys.exit(1)
    after=i                          # BEFORE the call, so MutexInit never executes
else:
    i=s.find(t, start)
    if i<0: print("INITSTOP: anchor %r not found"%t,file=sys.stderr); sys.exit(1)
    after=i+len(t)
# MUST return NON-ZERO. Returning 0 (= SQLITE_OK) makes run_sqlite carry on to
# sqlite3_open/exec, so the domain wedges DOWNSTREAM and the arm says nothing about
# initialize() at all. That flaw invalidated the F0/in1/in2/in4 results of 2026-08-09,
# exactly as it earlier invalidated rs1..rs5. A non-zero rc trips run_sqlite's own
# `if (rc != SQLITE_OK) return fail("initialize", rc, 0);` so the DOMAIN returns.
# INITSTOP_ZERO=1 makes the clamp return 0 instead of the marker. Pair it with RUNSTOP=2 so
# run_sqlite returns DIRECTLY after the call: initialize then runs only up to sub-step n, and
# NO arm enters fail(). Without this, every non-zero return routes through fail(), which is
# S-02 Site A and would be measured instead of Site B.
_iz = __import__('os').environ.get('INITSTOP_ZERO','')
_rv = '0' if _iz else 'capstone_initstop'
ins='\n  { volatile int capstone_initstop = 0x7B0%X; if(capstone_initstop) return %s; }\n' % (n,_rv)
s=s[:after]+ins+s[after:]
open(path,'w').write(s)
print("INITSTOP: early return inserted after sub-step %d (%s)" % (n,t))
PYINIT
  _m=$(printf '0x7B0%X' "$INITSTOP")
  _n=$(grep -c -F "capstone_initstop = $_m" "$OBJ_DIR/sqlite3-capstone.c" || true)
  # Three clamp shapes exist: the `if(...) return capstone_initstop;` inline form, the naked
  # form (`return capstone_initstop;`), and the naked-zero form (`return 0;`). Checking for a
  # bare "return 0;" globally would match thousands of unrelated lines, so instead require the
  # line IMMEDIATELY AFTER the marker declaration to be a return -- unique by construction.
  _r=$(grep -A1 -F "capstone_initstop = $_m" "$OBJ_DIR/sqlite3-capstone.c" | grep -c -E "return (capstone_initstop|0);" || true)
  echo "== initstop gate: marker $_m x$_n ; return x$_r"
  if [[ "$_n" -ne 1 || "$_r" -ne 1 ]]; then
    echo "INITSTOP did NOT apply (marker=$_n return=$_r). Refusing to build." >&2; exit 1
  fi
fi

# ALTERSTOP=<n> -- DIAGNOSTIC ONLY. S-03 is sqlite3AlterFunctions(); this clamps how many of
# its NINE aAlterTableFuncs entries are actually registered.
#   0 = return before calling sqlite3InsertBuiltinFuncs at all
#   1..9 = call it with that many entries
# Pair with RUNSTOP=2 so the domain returns instead of running on into sqlite3_open.
if [[ -n "${ALTERSTOP:-}" ]]; then
  echo "== DIAGNOSTIC: sqlite3AlterFunctions clamped to $ALTERSTOP entries"
  python3 - "$OBJ_DIR/sqlite3-capstone.c" "$ALTERSTOP" <<'PYALT'
import sys,re
path,n=sys.argv[1],int(sys.argv[2])
s=open(path).read()
t='sqlite3InsertBuiltinFuncs(aAlterTableFuncs, ArraySize(aAlterTableFuncs));'
i=s.find(t)
if i<0: print("ALTERSTOP: call site not found",file=sys.stderr); sys.exit(1)
if n==0:
    rep='{ volatile int capstone_alterstop = 0x7D00; if(capstone_alterstop) return; }\n  /* skipped */'
else:
    rep='{ volatile int capstone_alterstop = 0x7D0%X; (void)capstone_alterstop; }\n  sqlite3InsertBuiltinFuncs(aAlterTableFuncs, %d);' % (n,n)
s=s[:i]+rep+s[i+len(t):]
open(path,'w').write(s); print("ALTERSTOP: applied n=%d"%n)
PYALT
  _m=$(printf '0x7D0%X' "$ALTERSTOP")
  _n=$(grep -c -F "capstone_alterstop = $_m" "$OBJ_DIR/sqlite3-capstone.c" || true)
  echo "== alterstop gate: marker $_m x$_n"
  [[ "$_n" -eq 1 ]] || { echo "ALTERSTOP did NOT apply" >&2; exit 1; }
fi

# RUNSTOP=<n> -- DIAGNOSTIC ONLY. Clamp run_sqlite() to return after step n.
#
# WHY THIS AND NOT CAPSTONE_SQLITE_STAGE. The staged block drags in probe arrays and,
# measured 2026-08-09, EVERY staged build fails to ENTER on this bitstream (4/4: st0, st1,
# s1-p4096, fx1) while unstaged builds enter 2/2. Removing the 580-element holder default
# cut __capstone_cap_init from 1257 to 608 stc and did NOT fix entry, so the holder was a
# real defect but not the cause. This knob inserts ONLY an early return -- the same
# lightweight mechanism as REGBUILTIN_STOP, whose images DID enter (rs0/rs1 entered and
# wedged, i.e. they ran).
#
#   1 = after sqlite3_config(SQLITE_CONFIG_HEAP)
#   2 = after sqlite3_initialize()
#   3 = after sqlite3_open(":memory:")
#   4 = after CREATE TABLE
if [[ -n "${RUNSTOP:-}" ]]; then
  echo "== DIAGNOSTIC: clamping run_sqlite() to return after step $RUNSTOP"
  python3 - "$OBJ_DIR/sqlite_capstone_domain.c" "$RUNSTOP" <<'PYRUN'
import sys,re
path,n=sys.argv[1],int(sys.argv[2])
s=open(path).read()
m=re.search(r'static int run_sqlite\(void\)\s*\{', s)
if not m: print("RUNSTOP: run_sqlite not found",file=sys.stderr); sys.exit(1)
anchors={1:'return fail("config-heap", rc, 0);',
         2:'return fail("initialize", rc, 0);',
         3:'return fail("open", rc, db);',
         4:'return fail("create", rc, db);'}
t=anchors.get(n)
if not t: print("RUNSTOP: no anchor for %d"%n,file=sys.stderr); sys.exit(1)
i=s.find(t, m.end())
if i<0: print("RUNSTOP: anchor %r not found"%t,file=sys.stderr); sys.exit(1)
after=i+len(t)
ins='\n  { volatile int capstone_runstop = 0x7A0%X; if(capstone_runstop) return 0x7A0%X; }\n' % (n,n)
s=s[:after]+ins+s[after:]
open(path,'w').write(s)
print("RUNSTOP: early return inserted after step %d"%n)
PYRUN
  _m=$(printf '0x7A0%X' "$RUNSTOP")
  _n=$(grep -c -F "capstone_runstop = $_m" "$OBJ_DIR/sqlite_capstone_domain.c" || true)
  _r=$(grep -c -F "if(capstone_runstop) return $_m;" "$OBJ_DIR/sqlite_capstone_domain.c" || true)
  echo "== runstop gate: marker $_m x$_n ; return x$_r"
  if [[ "$_n" -ne 1 || "$_r" -ne 1 ]]; then
    echo "RUNSTOP did NOT apply (marker=$_n return=$_r). Refusing to build." >&2; exit 1
  fi
fi

# CAPSTONE_DOMAIN_VAR=1 -- append a dead integer GLOBAL (not a function) to the domain file.
#
# Separates two hypotheses that the function-pad conflates. A `used` static FUNCTION added five
# .gct entries (the static capability table __capstone_cap_init walks) and zero carves. A dead
# INTEGER global does the reverse: one more carve, and no .gct entry, because an integer
# initialiser needs no capability. So:
#   returns -> .gct / cap-init is the variable, and carve count is not
#   hangs   -> any structural change hangs, and .gct is incidental
if [[ "${CAPSTONE_DOMAIN_VAR:-0}" == "1" ]]; then
  echo "== DOMAIN VAR: appending one dead integer global (adds a carve, no .gct entry)"
  printf '\n__attribute__((used)) static volatile long capstone_pad_var = 7;\n' \
    >> "$OBJ_DIR/sqlite_capstone_domain.c"
fi

# CAPSTONE_DOMAIN_PAD=N -- append N bytes of dead, never-called code to the DOMAIN file,
# leaving the amalgamation completely untouched.
#
# Control for a confound: every perturbation so far (g1, pad0..pad52) was injected by the python
# rewrite of sqlite3-capstone.c, which also renumbers every __LINE__ in the file (178 constants
# changed in g1). The two builds that WORK (uc, f10) are both un-rewritten. So "modified" is
# confounded with "amalgamation rewritten". This knob perturbs code layout WITHOUT rewriting the
# amalgamation, separating the two.
if [[ -n "${CAPSTONE_DOMAIN_PAD:-}" ]]; then
  echo "== DOMAIN PAD: ${CAPSTONE_DOMAIN_PAD} bytes of dead code appended to the domain file"
  python3 - "$OBJ_DIR/sqlite_capstone_domain.c" "$CAPSTONE_DOMAIN_PAD" <<'PYDPAD'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); n = int(sys.argv[2])
if n % 4: sys.exit("FATAL: CAPSTONE_DOMAIN_PAD must be a multiple of 4")
nops = "\n".join(['  __asm__ volatile("nop");'] * (n // 4))
p.write_text(p.read_text() +
  "\n__attribute__((used,noinline)) static void capstone_domain_pad(void)\n{\n" + nops + "\n}\n")
print(f"   appended {n} bytes ({n//4} nops); amalgamation untouched")
PYDPAD
fi



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
# The string primitives default to -O1, NOT to $OPT. Board-measured 2026-08-05: at -O0
# `strlen` re-loads its string capability from a stack slot with `ldc` on EVERY iteration, and
# on silicon that sporadically yields 1 instead of the true length -- stage 13 returned 15, then
# 26, then hung across three boots of the same source, where QEMU returns 36 every time. At -O1
# the pointer stays in a register (zero `ldc` in the loop) and stage 13 returns 36 on silicon,
# twice. This is a real wrong-answer defect, not a workaround, so it is the default.
#
# It does NOT fix the stage-10 hang -- that was measured separately and is independent. And it
# cannot be raised to a whole-image -O1: that hits C-17 (`Cannot select: i128 =
# CapstoneISD::SELECT_CC`), which is why this knob is scoped to the support files.
SUPPORT_OPT=${SQLITE_SUPPORT_OPT_LEVEL:--O1}
# BEEBS_STRING_LINEAR_SAFE: index instead of walking, so the string primitives never copy
# or advance a capability that may be LINEAR. SQLite is the first thing here to call strlen
# on hardware (no ladder rung references it), and both the -O0 and -O1 pointer-walking
# forms freeze on the board at the instruction that advances the pointer. See the header
# comment on strlen in beebs_freestanding_string.c for the RTL citations. Set only here --
# the ladder rungs keep the walking form so their published geometry is unchanged.
# BEEBS_MEMCPY_OPTNONE: compile memcpy, and ONLY memcpy, at -O0 while everything else in that
# file stays at $SUPPORT_OPT. This is what lets the build avoid BOTH known string defects at
# once. Before it, the two were coupled through one file-wide flag: -O1 gives a correct strlen
# and a memcpy whose small aligned copies vanish on silicon (S-04), -O0 gives a working memcpy
# and a strlen that sporadically returns the wrong length. SQLITE_SUPPORT_OPT_LEVEL=-O0 was
# therefore a trade, not a fix, and every measurement taken under it was taken under a second
# known defect -- which is the leading suspect for the SQLITE_CORRUPT at CREATE (S-05).
#
# Verified on the artifact, not assumed: with this define at -O1, strlen/strcmp/strcpy/memset/
# memmove are BYTE-IDENTICAL to the plain -O1 build, and memcpy alone changes to the form that
# spills the destination capability at entry and reloads it with `ldc` before each `sb` -- the
# structure the working -O0 build has. Set SQLITE_MEMCPY_OPTNONE=0 to turn it off for an A/B.
_memcpy_optnone=${SQLITE_MEMCPY_OPTNONE:-1}
SUPPORT_DEFS=(-DBEEBS_STRING_LINEAR_SAFE=1 -DBEEBS_MEMCPY_OPTNONE=$_memcpy_optnone \
              ${BEEBS_STRING_EXTRA_DEFS:-})
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

# DESCRIPTOR-IDENTITY GATE. This image's behaviour depends on its GLOBAL SET: adding globals
# entry-stalls it (the runtime clamp, 179->183 carves, stalled 3/3 without executing an
# instruction), and dropping globals via DCE behind a clamp changes which code hangs (the
# compile-time clamp, 181->178, made stage 11 hang when it returns unclamped). Four separate
# conclusions died to this. So any instrumented build MUST leave .capstone_gp_initdesc
# byte-identical to its uninstrumented reference, and is refused otherwise.
#
#   CAPSTONE_DESC_REF=/path/to/unclamped.dom
if [[ -n "${CAPSTONE_DESC_REF:-}" ]]; then
  python3 - "$OUT_DIR/sqlite_silicon.dom" "$CAPSTONE_DESC_REF" <<'PYDESC' || exit 1
import sys, pathlib, subprocess, struct
def desc(p):
    out = subprocess.run(["llvm/cmake-build-debug/bin/llvm-readelf","-SW",p],
                         capture_output=True, text=True).stdout
    off = size = None
    for l in out.splitlines():
        if "initdesc" in l:
            f = l.split(); off = int(f[5],16); size = int(f[6],16)
    if off is None: sys.exit(f"FATAL: no .capstone_gp_initdesc in {p}")
    d = pathlib.Path(p).read_bytes()[off:off+size]
    return d, struct.unpack_from("<Q", d, 8)[0]
a, na = desc(sys.argv[1]); b, nb = desc(sys.argv[2])
if a != b:
    first = next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
    sys.exit(f"FATAL: descriptor table differs from the reference "
             f"(carves {na} vs {nb}, first difference at byte {first}). The instrumentation "
             f"changed the global set, so any verdict from this image would be about the "
             f"instrument, not the bug.")
print(f"   descriptor table identical to reference ({na} carves)")
PYDESC
fi

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
