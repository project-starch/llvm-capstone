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
# CAPSTONE_OOB_PROBE=1 -- convert the vdbeMemClearExternAndSetNull OUT_OF_BOUNDS wedge into a
# RETURN, and report what the offending capability's bounds actually were.
#
# The domain wedges at vdbeMemClearExternAndSetNull+0x3c on `ldc a1, 0x0(a0)` with mcause 29
# (OUT_OF_BOUNDS). a0 is the incoming `Mem *p`, spilled by the -O0 prologue and reloaded with NO
# intervening call, so the bad capability was handed in by the caller. The LDC rule
# (capstone_dyn_unit.anvil:317-348) is: fault iff cursor < start || cursor > end - 16.
#
# Two readings remain and static analysis cannot separate them:
#   A  the address is right and the BOUNDS are wrong -- p is a real Mem whose capability metadata
#      is bad. MEM_Agg is genuine; the workload does use COUNT/SUM/MAX/GROUP BY.
#   B  the address is WRONG -- p is wild, the flags read junk that happened to have bit 15 set,
#      and the ldc is simply the first bounds-checked use.
#
# The probe reads p's type with LCC selector 1 and, when p really is a capability, its
# cursor/start/end with selectors 2/3/4, records them, and then FALLS THROUGH so the original
# code faults exactly as it would have.
#
# IT CHANGES NO CONTROL FLOW, and that is the whole design. Version 1 detected the bad capability
# and RETURNED early, skipping the aggregate finalize; the run diverged from a real one, the
# domain faulted at a DIFFERENT Mem dereference one function further on, the numbers never came
# out, and the second fault was attributable to nothing. A diagnostic that perturbs the thing it
# measures answers a question nobody asked.
#
# Reporting only works because the entry glue can install an in-domain trap vector
# (INTERP_DOMAIN_MTVEC), which converts the fault into an ordinary domreturn; sqlite_host.c then
# writes the payload region to stdout as it does on any return. BOTH halves are required, so a
# build using this MUST be run in a boot where the trapctl rung passes -- otherwise "no output"
# means either "the trap handler does not work" or "SQLite wedged elsewhere", and those are not
# the same finding. See tests/runtime-qemu/silicon-ladder/trapctl_kernel.h.
#
# The type query goes FIRST and guards the other three. Only selector 1 was made total by the
# S-06 enabler; selectors 2/3/4 still RAISE on a non-capability, so querying bounds
# unconditionally would make the probe fault on precisely the input that matters most.
#
# DEFAULT OFF: it is a DIAGNOSTIC build and must never carry a timing or correctness number.
if [[ "${CAPSTONE_OOB_PROBE:-0}" == "1" ]]; then
  python3 - "$OBJ_DIR/sqlite3-capstone.c" <<'PYOOB'
import sys
p = sys.argv[1]
s = open(p).read()
FN = "static SQLITE_NOINLINE void vdbeMemClearExternAndSetNull(Mem *p){"
if FN not in s:
    sys.exit("OOB_PROBE: vdbeMemClearExternAndSetNull not found -- patch shape changed")
probe = FN + """
  /* CAPSTONE OOB PROBE (non-perturbing) -- see build-sqlite-silicon.sh. Diagnostic only.
     Records and falls through; the fault below is left to happen exactly as it would. */
  {
    unsigned long cs_ty_, cs_cur_ = 0, cs_st_ = 0, cs_en_ = 0;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(cs_ty_) : "r"(p));
    if (cs_ty_ != 7UL) {
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x2" : "=r"(cs_cur_) : "r"(p));
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x3" : "=r"(cs_st_)  : "r"(p));
      __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x4" : "=r"(cs_en_)  : "r"(p));
    }
    /* WHO handed us this capability, and WHAT REGION its cursor points into.
       The first run established that p is the HEAP capability (bounds exactly the 1 MiB heap)
       with a cursor 54336 bytes past its end -- so the bounds are right and the cursor is wrong,
       and the next question is who computed it and what lives at that address. Both are read
       here rather than inferred: `ra` still holds the caller's return address (nothing has been
       called yet at this point, and the -O0 prologue has not reused it), and sp's bounds ARE the
       stack region, so "is that address in the stack" stops being arithmetic on assumed layout.
       rs1 is written as the literal register number -- x1 is ra, x2 is sp -- because these are
       .insn forms, not named operands. */
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, x1, x2" : "=r"(capstone_oob_ra));
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, x2, x3" : "=r"(capstone_oob_spst));
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, x2, x4" : "=r"(capstone_oob_spen));
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, x2, x2" : "=r"(capstone_oob_spcur));
    capstone_oob_calls++;
    capstone_oob_last_ty = cs_ty_;   capstone_oob_last_cur = cs_cur_;
    capstone_oob_last_st = cs_st_;   capstone_oob_last_en  = cs_en_;
    /* Sticky: the FIRST p that fails the hardware's own LDC predicate. Without this, hundreds
       of healthy calls after the interesting one would scroll it away. */
    if (cs_ty_ != 7UL && (cs_cur_ < cs_st_ || cs_cur_ > cs_en_ - 16UL)) {
      capstone_oob_bad_seen++;
      if (capstone_oob_bad_seen == 1UL) {
        capstone_oob_bad_n   = capstone_oob_calls;
        capstone_oob_bad_ty  = cs_ty_;   capstone_oob_bad_cur = cs_cur_;
        capstone_oob_bad_st  = cs_st_;   capstone_oob_bad_en  = cs_en_;
      }
    }
    capstone_oob_report();
  }
"""
s = s.replace(FN, probe, 1)
# Declarations before first use. The definitions and capstone_oob_report() live in
# sqlite_capstone_domain.c, which the amalgam includes AFTER this file -- legal in one TU,
# and it keeps the reporter next to output_text() rather than injected as a second patch.
decl = ("extern unsigned long capstone_oob_calls, capstone_oob_bad_seen, capstone_oob_bad_n;\n"
        "extern unsigned long capstone_oob_last_ty, capstone_oob_last_cur;\n"
        "extern unsigned long capstone_oob_last_st, capstone_oob_last_en;\n"
        "extern unsigned long capstone_oob_bad_ty, capstone_oob_bad_cur;\n"
        "extern unsigned long capstone_oob_bad_st, capstone_oob_bad_en;\n"
        "extern unsigned long capstone_oob_ra, capstone_oob_spcur;\n"
        "extern unsigned long capstone_oob_spst, capstone_oob_spen;\n"
        "void capstone_oob_report(void);\n")
i = s.find(FN)
j = s.rfind("\n", 0, i)
s = s[:j+1] + decl + s[j+1:]
open(p, "w").write(s)
print("   OOB probe injected into vdbeMemClearExternAndSetNull (non-perturbing)")
PYOOB
  # Turns on the reporter half in sqlite_capstone_domain.c. Set here rather than left to the
  # caller so the two halves cannot be enabled independently: an injected call site with no
  # definition is a link error, and a definition with no call site is dead code that silently
  # reports nothing. Assigned BEFORE _domain_defs is read, a few dozen lines below.
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_OOB_PROBE=1"
fi

# CAPSTONE_ARG_PROBE=<fn> -- report the TYPE of a function's two pointer arguments, and its
# caller, on the first call. Non-perturbing: it records and falls through.
#
# WHY. With the codegen guard enabled SQLite stops faulting at vdbeMemClearExternAndSetNull and
# instead faults in sqlite3_strnicmp at `cincoffsetimm a0, a0, 1` on a pointer reloaded from a
# stack slot -- mcause 25, an UNTAGGED pointer. That function is BYTE-IDENTICAL in the guarded
# and unguarded builds, so the guard did not produce it; the pointer arrives that way or loses
# its tag in the spill. Those are different bugs and static reading cannot separate them: the
# earlier `lbu` dereferences in the same function do NOT fault, because plain integer accesses
# are unchecked in our domains, so `cincoffsetimm` is merely the first tag-REQUIRING use and
# says nothing about when the tag went.
#
# The probe queries the type of both incoming pointers BEFORE anything else runs. A type of 7
# on entry means the caller handed over plain data; anything else means the tag was alive at the
# boundary and the spill/reload lost it.
if [[ -n "${CAPSTONE_ARG_PROBE:-}" ]]; then
  python3 - "$OBJ_DIR/sqlite3-capstone.c" "$CAPSTONE_ARG_PROBE" <<'PYARG'
import sys, re
path, fn = sys.argv[1], sys.argv[2]
s = open(path).read()
m = re.search(r'^[^\n]*\b' + re.escape(fn) + r'\s*\(([^)]*)\)\s*\{', s, re.M)
if not m:
    sys.exit(f"ARG_PROBE: {fn} definition not found -- patch shape changed")
args = [a.strip().split()[-1].lstrip('*') for a in m.group(1).split(',') if a.strip()]
if len(args) < 2:
    sys.exit(f"ARG_PROBE: {fn} has fewer than two arguments: {args}")
probe = m.group(0) + """
  /* CAPSTONE ARG PROBE -- records and falls through; changes no control flow. */
  {
    unsigned long ap_t1_, ap_t2_, ap_ra_;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(ap_t1_) : "r"(""" + args[0] + """));
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(ap_t2_) : "r"(""" + args[1] + """));
    __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, x1, x2" : "=r"(ap_ra_));
    if (capstone_arg_calls == 0UL) {
      capstone_arg_ty1 = ap_t1_; capstone_arg_ty2 = ap_t2_; capstone_arg_ra = ap_ra_;
    }
    capstone_arg_calls++;
    capstone_arg_report();
  }
"""
s = s.replace(m.group(0), probe, 1)
decl = ("extern unsigned long capstone_arg_calls, capstone_arg_ty1, capstone_arg_ty2;\n"
        "extern unsigned long capstone_arg_ra;\n"
        "void capstone_arg_report(void);\n")
i = s.find(probe); j = s.rfind("\n", 0, i)
s = s[:j+1] + decl + s[j+1:]
open(path, "w").write(s)
print(f"   ARG probe injected into {fn} (args: {args[0]}, {args[1]})")
PYARG
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_ARG_PROBE=1"
fi

# CAPSTONE_PROBE_LOOKASIDE=1 -- report the lookaside small-free head WITHOUT dereferencing it,
# then fall back to the general allocator so the run CONTINUES rather than wedging.
#
# The board faults at sqlite3DbMallocRawNN+0xf8 on `pBuf->pNext`, where pBuf came out of
# db->lookaside.pSmallFree already untagged. The compiler is exonerated: every writer of that
# list is capability-clean, and `stc` traps on an untagged BASE, so an untagged pointer reaching
# the free path would have trapped one instruction earlier than it does. The value was tagged
# when stored and untagged when read back -- so the question is WHICH value came back, and that
# is answered by printing it, not by dereferencing it.
#
# Falling through to dbMallocRawFinish is semantically fine: lookaside is an optional fast path
# and the general allocator is already exercised heavily by this workload. So this build both
# reports the value AND has a chance of completing the extended workload -- unlike the two
# earlier lookaside-disable attempts, which never changed the executed path at all.
if [[ "${CAPSTONE_PROBE_LOOKASIDE:-0}" == "1" ]]; then
  python3 - "$OBJ_DIR/sqlite3-capstone.c" <<'PYLOOK'
import sys
path = sys.argv[1]
s = open(path).read()
NEEDLE = "    if( (pBuf = db->lookaside.pSmallFree)!=0 ){"
if NEEDLE not in s:
    sys.exit("PROBE_LOOKASIDE: pSmallFree pop not found -- patch shape changed")
probe = NEEDLE + """
      /* CAPSTONE LOOKASIDE PROBE -- report, never dereference, then use the general allocator.
         `db` is passed AS A CAPABILITY, not as an integer, because the reporter measures its
         LCC type/cursor/base/end: boot25/boot26 showed this `db` is not the connection the
         workload opened, and only the bounds say whether it is a wrong OBJECT or a shifted
         CURSOR on the right one. The address of the field being popped is computed HERE,
         where struct sqlite3 is complete, so the probe never has to assume an offset. */
      capstone_look_report2((unsigned long)(void *)pBuf,
                            (const void *)db,
                            (unsigned long)(void *)&db->lookaside.pSmallFree,
                            (unsigned long)n,
                            (unsigned long)db->lookaside.sz,
                            capstone_look_pops);
      capstone_look_pops++;
      return dbMallocRawFinish(db, n);
"""
s = s.replace(NEEDLE, probe, 1)
decl = ("extern unsigned long capstone_look_pops;\n"
        "void capstone_look_report2(unsigned long, const void *, unsigned long,\n"
        "                           unsigned long, unsigned long, unsigned long);\n")
i = s.find(probe); j = s.rfind("\n", 0, i)
s = s[:j+1] + decl + s[j+1:]
open(path, "w").write(s)
print("   LOOKASIDE probe injected into the pSmallFree pop")
PYLOOK
  # THE ALLOCATOR SENTINEL. memsys5 must never return a block overlapping the live sqlite3
  # connection; db is measured at 2048 bytes and correctly aligned, so db+0x200 is INTERIOR to it.
  # Refusing such a block returns OOM, which SQLite handles gracefully -- so this both localises
  # the bug and prevents the smash, instead of merely observing it.
  python3 - "$OBJ_DIR/sqlite3-capstone.c" <<'PYOVL'
import sys
path = sys.argv[1]
s = open(path).read()
NEEDLE = "  return (void*)&mem5.zPool[i*mem5.szAtom];"
if NEEDLE not in s:
    sys.exit("PROBE_LOOKASIDE: memsys5 return site not found -- patch shape changed")
guard = """  {
    unsigned long a_ = (unsigned long)(void *)&mem5.zPool[i*mem5.szAtom];
    unsigned long n_ = (unsigned long)(mem5.szAtom << iLogsize);
    if (capstone_db_lo && a_ < capstone_db_hi && a_ + n_ > capstone_db_lo) {
      capstone_ovl_report(a_, n_, 1UL);
      return 0;            /* OOM -- SQLite handles it; the smash never happens */
    }
  }
""" + NEEDLE
s = s.replace(NEEDLE, guard, 1)
decl = ("extern unsigned long capstone_db_lo, capstone_db_hi, capstone_ovl_hits;\n"
        "void capstone_ovl_report(unsigned long, unsigned long, unsigned long);\n"
        "void capstone_dump_window(const void *, unsigned long, unsigned long,\n"
        "                          unsigned long, unsigned long);\n")
i = s.find(guard); j = s.rfind("\n", 0, i)
s = s[:j+1] + decl + s[j+1:]
open(path, "w").write(s)
print("   ALLOCATOR OVERLAP sentinel injected into memsys5MallocUnsafe")
PYOVL
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_PROBE_LOOKASIDE=1"
fi

# CAPSTONE_DBWHO_PROBE=1 -- WHICH CALLER hands sqlite3DbMallocRawNN a `db` that is not the
# connection? Records and falls through: no control flow changes, so the run fails exactly where
# it failed without the probe.
#
# WHY THIS AND NOT ANOTHER CALLEE-SIDE PROBE. Boot27 measured `db` arriving with a cursor 0x3e00
# from the connection, inside a 64-byte block of SQL text, WITH A VALID TAG and with bounds
# spanning the whole heap -- so neither the tag nor the hardware's bounds check can see anything
# wrong, and every property of the callee is consistent. The wrong object was chosen by the
# caller, which is the one thing no probe inside the allocator has looked at.
#
# AND IT DOES NOT REPRODUCE UNDER QEMU. The same lookaside probe, same build path, run under QEMU
# with no stage limit: the report never fires and the extended workload passes. That is a zero, so
# it is only evidence because the identical instrument DID fire on the board -- and because the
# two agree on the mechanism: `pSmallFree` is legitimately null under QEMU, and reads nonzero on
# the board only because it is being read out of a string block. The instrument is the same; the
# subject differs. So the board is the only oracle here and the probe has to be self-contained.
#
# Two levels, because one is not enough: almost every call site reaches the allocator through a
# wrapper, so the immediate return address names the wrapper and stops. Each wrapper below pushes
# its own (id, ra) into a ring on entry, and the ring entry immediately preceding the bad call
# carries the caller one level up. Ids are the 1-based position in DBWHO_WRAPPERS.
if [[ "${CAPSTONE_DBWHO_PROBE:-0}" == "1" ]]; then
  DBWHO_WRAPPERS=${DBWHO_WRAPPERS:-auto}
  python3 - "$OBJ_DIR/sqlite3-capstone.c" "$DBWHO_WRAPPERS" <<'PYDBWHO'
import sys, re
path, wrappers = sys.argv[1], sys.argv[2].split(',')
s = open(path).read()

# ONE SCAN, AND IT CARRIES LINE NUMBERS.
#
# Every function is located once by brace matching, and each injection then addresses its
# definition by INDEX rather than by re-searching for its text. Two things forced this:
#
#  * sqlite3SelectDup is defined TWICE, the real one and a SQLITE_OMIT_SELECT stub under #else.
#    A name-based search cannot tell them apart, and treating "more than one match" as an error
#    -- which the first version did -- simply failed the build. Selecting by body content picks
#    the compiled one for free, because the body is what the auto list is derived from.
#  * `}else if( sqlite3StrAccumEnlarge(pAccum, szBufNeeded)<szBufNeeded ){` matched an earlier,
#    looser pattern: `[^;{]*\)` runs greedily past the inner close-paren to the one before the
#    brace, so a CALL was mistaken for a definition. Requiring column zero excludes call sites,
#    since definitions in the amalgamation always start there.
#
# DBWHO_WRAPPERS=auto selects every function that calls the allocator DIRECTLY. An earlier
# guessed list -- attributing each call to the nearest preceding function header -- produced 45
# names including sqlite3DbMallocZero, which does not call sqlite3DbMallocRawNN at all; the real
# count is 21. Wrong in that direction is the expensive kind: the breadcrumb ring then stays
# silent for the caller that mattered, and the silence reads as "not that path".
HOP2 = ["sqlite3DbMallocZero", "sqlite3DbStrDup", "sqlite3VdbeMemGrow", "sqlite3StrAccumEnlarge"]
lines = s.split("\n")
dpat = re.compile(r'^[A-Za-z_][^\n;=]*?\b(\w+)\s*\([^;{]*\)\s*\{\s*$')
defs = []
for i, l in enumerate(lines):
    m = dpat.match(l)
    if not m:
        continue
    depth, j = 0, i
    while j < len(lines):
        depth += lines[j].count("{") - lines[j].count("}")
        if depth <= 0 and j > i:
            break
        j += 1
    defs.append((i, m.group(1), "\n".join(lines[i+1:j])))
if not defs:
    sys.exit("DBWHO: no function definitions parsed -- the scan, not the amalgamation, is wrong")

CALLS = re.compile(r'\bsqlite3DbMallocRawNN\s*\(')
if wrappers == ["auto"]:
    targets = [(i, n) for i, n, b in defs if CALLS.search(b)]
    if not targets:
        sys.exit("DBWHO: auto found NO direct callers -- the scan is wrong, not the amalgamation")
    have = {n for _, n in targets}
    extra = [f for f in HOP2 if f not in have]
else:
    targets, extra = [], list(wrappers)
for f in extra:
    cand = [(i, n) for i, n, b in defs if n == f]
    if not cand:
        sys.exit("DBWHO: %s is not defined in the amalgamation" % f)
    targets.append(cand[0])          # first definition; the #else stub always follows the real one

targets.sort()
ids = {i: k for k, (i, _) in enumerate(targets, start=1)}
print("   DBWHO callers: " + " ".join("%d=%s" % (ids[i], n) for i, n in targets))
# Descending, so an injection never invalidates the index of one not yet done.
for i, n in sorted(targets, reverse=True):
    lines[i] += ("\n  { unsigned long dw_rt_, dw_ra_;          /* breadcrumb: who called us */\n"
                 "    CAPSTONE_DBWHO_QRA(dw_rt_, dw_ra_);\n"
                 "    capstone_dbwho_push(%dUL, dw_ra_); }" % ids[i])
s = "\n".join(lines)

DECL = ("extern unsigned long capstone_dbwho_calls, capstone_dbwho_ok, capstone_dbwho_bad;\n"
        "extern unsigned long capstone_dbwho_badn, capstone_dbwho_rat;\n"
        "extern unsigned long capstone_dbwho_ctl_ra, capstone_real_db;\n"
        "void capstone_dbwho_push(unsigned long, unsigned long);\n"
        "void capstone_dbwho_report(const void *, unsigned long, unsigned long);\n"
        "void *capstone_heap_ptr(unsigned long);\n"
        # THE TYPE GUARD LIVES INSIDE THE ASM, and the cursor query is never issued unguarded.
        #
        # Selector 1 is total on both sides (RTL by the change we landed; QEMU by the matching
        # imm==1 case in helper_cslcc). SELECTOR 2 IS NOT: on an untagged operand the RTL raises
        # UNEXPECTED_OPERAND and QEMU trips `assert(rs1_v->tag)`, which ABORTS THE EMULATOR --
        # not a guest fault, an abort, so a probe that gets this wrong takes the whole run with
        # it. The first version of this probe did exactly that and died at SQ: G/enter.
        #
        # Guarding in C is not enough, and that is the subtle half. At -O0 the pointer is spilled
        # between the test and the use, so the operand the cursor query sees is a RELOAD of the
        # one the type query approved -- and a value that does not survive the round trip passes
        # the guard and faults anyway. output_capinfo() carries the same construction and the
        # same comment for the same reason. One asm block, one materialisation, no reload.
        #
        # THE RETURN ADDRESS IS A SCALAR, NOT A CAPABILITY, and that was measured, not assumed.
        # A first cut queried ra's cursor with selector 2 and got 0 out of the guard on every
        # call; printing ra's TYPE alongside it gave 7 -- NOT_CAP. So `lcc` can never read a
        # return address here, and any ra a probe reports through selector 2 is the guard's zero.
        # x1 holding a plain integer means it can simply be MOVED, which is the first arm below;
        # the capability arm is kept because nothing guarantees that stays true.
        "#define CAPSTONE_DBWHO_QP(t_, c_, p_)                        \\\n"
        "  __asm__ volatile(\".insn r 0x5b, 0x1, 0x4, %0, %2, x1\\n\\t\"   \\\n"
        "                   \"li   %1, 0\\n\\t\"                          \\\n"
        "                   \"li   t0, 3\\n\\t\"                          \\\n"
        "                   \"bltu t0, %0, 1f\\n\\t\"                     \\\n"
        "                   \".insn r 0x5b, 0x1, 0x4, %1, %2, x2\\n\\t\"   \\\n"
        "                   \"1:\\n\\t\"                                  \\\n"
        "                   : \"=&r\"(t_), \"=&r\"(c_) : \"r\"(p_) : \"t0\")\n"
        "#define CAPSTONE_DBWHO_QRA(t_, c_)                           \\\n"
        "  __asm__ volatile(\".insn r 0x5b, 0x1, 0x4, %0, x1, x1\\n\\t\"   \\\n"
        "                   \"li   %1, 0\\n\\t\"                          \\\n"
        "                   \"li   t0, 7\\n\\t\"                          \\\n"
        "                   \"bne  t0, %0, 2f\\n\\t\"                     \\\n"
        "                   \"mv   %1, x1\\n\\t\"                         \\\n"
        "                   \"j    3f\\n\\t\"                             \\\n"
        "                   \"2:\\n\\t\"                                  \\\n"
        "                   \"li   t0, 3\\n\\t\"                          \\\n"
        "                   \"bltu t0, %0, 3f\\n\\t\"                     \\\n"
        "                   \".insn r 0x5b, 0x1, 0x4, %1, x1, x2\\n\\t\"   \\\n"
        "                   \"3:\\n\\t\"                                  \\\n"
        "                   : \"=&r\"(t_), \"=&r\"(c_) :: \"t0\")\n")

# --- the boundary probe, at the top of the allocator ------------------------------------------
# The type query (selector 1) goes first and GUARDS the cursor query (selector 2): selector 2
# RAISES on a NOT_CAP operand, so an unguarded read would make the probe fault on precisely the
# input that would matter most -- a `db` that arrived untagged.
OPEN = "SQLITE_PRIVATE void *sqlite3DbMallocRawNN(sqlite3 *db, u64 n){"
if OPEN not in s:
    sys.exit("DBWHO: sqlite3DbMallocRawNN signature not found -- patch shape changed")
probe = OPEN + """
  {  /* CAPSTONE DBWHO PROBE -- records and falls through */
    unsigned long dw_t_, dw_c_, dw_rt_, dw_ra_;
    CAPSTONE_DBWHO_QP(dw_t_, dw_c_, db);
    CAPSTONE_DBWHO_QRA(dw_rt_, dw_ra_);
    capstone_dbwho_calls++;
    capstone_dbwho_rat = (dw_t_ << 4) | dw_rt_;   /* db type and ra type, both on the last call */
    if (capstone_dbwho_calls == 1UL) capstone_dbwho_ctl_ra = dw_ra_;
    if (capstone_real_db != 0UL) {
      if (dw_c_ == capstone_real_db) {
        capstone_dbwho_ok++;
      } else {
        if (capstone_dbwho_bad == 0UL) {
          capstone_dbwho_bad = capstone_dbwho_calls;
          capstone_dbwho_report((const void *)db, (unsigned long)n, dw_ra_);
        }
        capstone_dbwho_badn++;
#ifdef CAPSTONE_DBFIX_PROBE
        /* THE MATCHED ARM. This workload holds exactly one connection, so a `db` that is not it
           cannot be a second one -- substituting is semantically a no-op for every call that was
           already correct, and the counters above say how many were not. If the extended workload
           then completes, the defect is confined to the delivery of THIS argument; if it fails
           somewhere else instead, the wrong `db` was a symptom and something upstream is wrong
           too. Either outcome is worth more than the observation alone. */
        { void *dw_fix_ = capstone_heap_ptr(capstone_real_db);
          if (dw_fix_) db = (sqlite3 *)dw_fix_; }
#endif
      }
    }
  }
"""
s = s.replace(OPEN, probe, 1)

# The declarations go at the TOP, not next to the allocator. The other probes in this script get
# away with declaring immediately above the function they patch because that function is their
# only injection site; this one instruments eight wrappers, two of which sit ABOVE
# sqlite3DbMallocRawNN in the amalgamation, so a local declaration left them calling an
# undeclared function -- a hard error under C99, which is the only reason it was not a silent
# implicit-int declaration instead. Anchored after SQLITE_CORE, which precedes all SQLite code.
ANCHOR = "#define SQLITE_CORE 1\n"
if ANCHOR not in s:
    sys.exit("DBWHO: amalgamation prologue anchor not found -- patch shape changed")
s = s.replace(ANCHOR, ANCHOR + DECL, 1)
open(path, "w").write(s)
print("   DBWHO probe injected into sqlite3DbMallocRawNN + %d callers" % len(targets))
PYDBWHO
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_DBWHO_PROBE=1"
  # CAPSTONE_DBFIX_PROBE=1 turns the observation into a matched arm: substitute the connection
  # for the object the caller got wrong, and see whether the workload then completes.
  #
  # This define was missing at first while the `#ifdef CAPSTONE_DBFIX_PROBE` it gates was already
  # in the injected code, so a "DBFIX" build compiled to the SAME BYTES as the plain DBWHO build
  # -- identical sha256 -- and its QEMU run looked like a clean pass of the fix arm. A matched
  # pair whose two halves are the same binary measures nothing at all. bake-sqlite-doms.sh now
  # refuses two byte-identical variants for this reason.
  if [[ "${CAPSTONE_DBFIX_PROBE:-0}" == "1" ]]; then
    DOMAIN_EXTRA_DEFS="$DOMAIN_EXTRA_DEFS -DCAPSTONE_DBFIX_PROBE=1"
  fi
fi

# CAPSTONE_PDEST_PROBE=1 -- read the capability OP_Column is about to use, and escape the way the
# clamp does.
#
# This is the measurement three earlier instruments failed to get. Reporting at the fault is
# impossible (the wedge takes the core and the payload with it), and every escape from inside
# sqlite3VdbeMemGrow has failed: SQLITE_NOMEM wedges in the error path, SQLITE_OK wedges on
# untagged arithmetic, and short-circuiting abort_due_to_error misses the path entirely. So report
# from inside OP_Column at the LAST point before any Mem is dereferenced, and leave through
# `goto vdbe_return` -- the exact escape CAPSTONE_VDBE_CLAMP uses, which is MEASURED to return on
# silicon.
#
# WHAT IT DECIDES. The RTL's LDC bounds predicate is `cursor + imm > end - 16` and it reserves the
# last 16 bytes of every capability, so a perfectly valid capability whose cursor sits in that tail
# raises OUT_OF_BOUNDS at offset 0 WITHOUT anything being corrupt. A SEALEDRET-typed operand is
# checked against a different window (start+48 .. start+1008) and can fault benignly too. Both are
# alive as explanations until pDest's type, cursor, base and end are on the table. A correctly
# derived &aMem[i] sits ~63 KB from its end -- measured -- so if the reported capability shows the
# same, the benign readings are dead and corruption is the only one left.
if [[ "${CAPSTONE_PDEST_PROBE:-0}" == "1" ]]; then
  python3 - "$OBJ_DIR/sqlite3-capstone.c" <<'PYPD'
import sys
path = sys.argv[1]
s = open(path).read()
# The main-path assignment, not the nullRow one: take the LAST occurrence inside OP_Column, which
# is the one on the ordinary column-extraction path.
# SCOPED TO OP_Column'S BODY. `pDest = &aMem[pOp->p3];` also appears in other opcodes, and a
# whole-file rfind picked one of those -- in a function where OP_Column's locals are not even in
# scope, so the build failed on an undeclared identifier. Loud, but only by luck.
NEEDLE = "  pDest = &aMem[pOp->p3];\n"
cs = s.find("case OP_Column: {")
if cs < 0:
    sys.exit("PDEST_PROBE: OP_Column case not found -- patch shape changed")
ce = s.find("\ncase OP_", cs + 10)
if ce < 0:
    sys.exit("PDEST_PROBE: end of OP_Column not found")
i = s.rfind(NEEDLE, cs, ce)
if i < 0:
    sys.exit("PDEST_PROBE: no pDest assignment inside OP_Column -- patch shape changed")
probe = NEEDLE + """  if( capstone_vdbe_armed && !capstone_pdest_seen ){
    capstone_pdest_seen = 1;
    capstone_pdest_report((const void*)pDest, (unsigned long)pOp->p3, (unsigned long)pOp->p2);
    rc = SQLITE_DONE;
    goto vdbe_return;      /* the clamp's escape -- measured to return on silicon */
  }
"""
s = s[:i] + probe + s[i+len(NEEDLE):]
ANCHOR = "#define SQLITE_CORE 1\n"
if ANCHOR not in s:
    sys.exit("PDEST_PROBE: amalgamation prologue anchor not found")
s = s.replace(ANCHOR, ANCHOR +
              "extern unsigned long capstone_vdbe_armed, capstone_pdest_seen;\n"
              "void capstone_pdest_report(const void *, unsigned long, unsigned long);\n", 1)
open(path, "w").write(s)
print("   PDEST probe injected into OP_Column")
PYPD
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_PDEST_PROBE=1"
fi

# CAPSTONE_AMEM_PROBE=1 -- report the VDBE register array's capability, from a point that RETURNS.
#
# The Mem dereferences that fault are `&aMem[i]` elements, and the bounds that matter are the ones
# `aMem` carries. Every attempt to measure them AT the fault has failed for the same structural
# reason: the fault wedges the core, the host never returns, and the payload dies with it. The two
# escapes tried are recorded so nobody repeats them -- returning SQLITE_NOMEM sends OP_Column into
# an error path that dereferences another Mem and wedges there, and returning SQLITE_OK leaves an
# ungrown Mem whose pointer arithmetic aborts on an untagged operand.
#
# So measure it EARLY instead. This reports at the top of sqlite3VdbeExec, before a single opcode
# has run, and is paired with CAPSTONE_VDBE_CLAMP=6 -- an arm already MEASURED to return on
# silicon. Report-only: no control flow changes, so the arm behaves exactly as the clamp arm that
# is known to work.
if [[ "${CAPSTONE_AMEM_PROBE:-0}" == "1" ]]; then
  python3 - "$OBJ_DIR/sqlite3-capstone.c" <<'PYAMEM'
import sys
path = sys.argv[1]
s = open(path).read()
NEEDLE = "  Mem *aMem = p->aMem;       /* Copy of p->aMem */\n"
if NEEDLE not in s:
    sys.exit("AMEM_PROBE: sqlite3VdbeExec aMem anchor not found -- patch shape changed")
probe = NEEDLE + """  if( capstone_vdbe_armed && !capstone_amem_seen ){
    capstone_amem_seen = 1;
    capstone_amem_report((const void*)aMem, (unsigned long)p->nMem, (unsigned long)p->nCursor);
  }
"""
s = s.replace(NEEDLE, probe, 1)
ANCHOR = "#define SQLITE_CORE 1\n"
if ANCHOR not in s:
    sys.exit("AMEM_PROBE: amalgamation prologue anchor not found")
s = s.replace(ANCHOR, ANCHOR +
              "extern unsigned long capstone_vdbe_armed, capstone_amem_seen;\n"
              "void capstone_amem_report(const void *, unsigned long, unsigned long);\n", 1)
open(path, "w").write(s)
print("   AMEM probe injected into sqlite3VdbeExec")
PYAMEM
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_AMEM_PROBE=1"
fi

# CAPSTONE_MEMGROW_PROBE=1 -- MEASURE the capability that faults, instead of inferring it.
#
# The latched trap state says what happened without saying why: mcause 29 = OUT_OF_BOUNDS,
# identical on all four wedged arms, at an `ldc` of offset 0x30 inside sqlite3VdbeMemGrow.
# Offset 0x30 in struct Mem is `sqlite3 *db`. An OUT_OF_BOUNDS there means the Mem capability
# being dereferenced does not span its own `db` field -- so the question is what its bounds
# actually are, and that is measurable rather than arguable.
#
# It REPORTS AND THEN RETURNS AN ERROR, which is the only shape that works here: the run wedges
# a few instructions later, a wedge takes the core, and the host never writes the payload out.
# Returning SQLITE_NOMEM propagates out of OP_Column through abort_due_to_error, sqlite3_step
# returns an error, the row loop exits and the ladder's stage 66 returns -- so the report reaches
# the host by the front door.
#
# ARMED by the same flag the VDBE clamp uses, so it fires on the first sqlite3VdbeMemGrow of the
# SELECT's step and not on the hundreds of calls during CREATE and INSERT that work fine.
if [[ "${CAPSTONE_MEMGROW_PROBE:-0}" == "1" ]]; then
  python3 - "$OBJ_DIR/sqlite3-capstone.c" <<'PYMG'
import sys
path = sys.argv[1]
s = open(path).read()
OPEN = "SQLITE_PRIVATE SQLITE_NOINLINE int sqlite3VdbeMemGrow(Mem *pMem, int n, int bPreserve){"
if OPEN not in s:
    sys.exit("MEMGROW_PROBE: sqlite3VdbeMemGrow signature not found -- patch shape changed")
probe = OPEN + """
  if( capstone_vdbe_armed && !capstone_memgrow_seen ){
    capstone_memgrow_seen = 1;
    capstone_memgrow_report((const void*)pMem, (unsigned long)n, (unsigned long)bPreserve);
    /* RETURNS SQLITE_NOMEM, which is the semantically honest answer and the only one that keeps
       SQLite's invariants. Returning OK instead -- to skip the faulting `ldc` and escape via the
       VDBE clamp -- was tried and is NOT SAFE: OP_Column carries on with an ungrown Mem and does
       pointer arithmetic on a pointer that was never allocated, which aborted QEMU outright at
       `helper_cscincoffset: Assertion rs1_v->tag failed`. The NOMEM path's own wedge is a real
       result (see the history note); it is not worth trading for a corrupted run. */
    return 7;
  }
"""
s = s.replace(OPEN, probe, 1)

# AND SHORT-CIRCUIT THE ERROR PATH, which is the half that was missing.
# Returning SQLITE_NOMEM is semantically honest but sends OP_Column to abort_due_to_error, and
# that path dereferences ANOTHER Mem: measured, the wedge simply moved to
# vdbeMemClearExternAndSetNull and took the report with it. Cutting the label short returns
# through vdbe_return like the clamp does, touching no further Mem, so the report survives to the
# host. Only fires for the probe's own escape (memgrow_seen), so ordinary error handling in every
# other build and every earlier statement is untouched.
LBL = "abort_due_to_error:\n"
if LBL not in s:
    sys.exit("MEMGROW_PROBE: abort_due_to_error label not found -- patch shape changed")
s = s.replace(LBL, LBL + """  if( capstone_memgrow_seen ){ rc = SQLITE_DONE; goto vdbe_return; }
""", 1)

ANCHOR = "#define SQLITE_CORE 1\n"
if ANCHOR not in s:
    sys.exit("MEMGROW_PROBE: amalgamation prologue anchor not found")
s = s.replace(ANCHOR, ANCHOR +
              "extern unsigned long capstone_vdbe_armed, capstone_memgrow_seen, capstone_vdbe_ops;\n"
              "void capstone_memgrow_report(const void *, unsigned long, unsigned long);\n", 1)
open(path, "w").write(s)
print("   MEMGROW probe injected into sqlite3VdbeMemGrow")
PYMG
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_MEMGROW_PROBE=1"
fi

# CAPSTONE_VDBE_CLAMP=<n> -- stop sqlite3VdbeExec after n opcodes and return cleanly, recording
# which opcode was about to run.
#
# WHY IT HAS TO GO INSIDE THE VDBE. The ladder localised the wedge to a single call: on silicon
# with the schema fixup on, CREATE TABLE completes, the INSERT succeeds (rc=0), prepare of the
# SELECT succeeds (rc=0), and the arm that does nothing but sqlite3_step() and return -- no
# column access, no comparison -- WEDGES.  There is no smaller C-level step left to take; the
# next boundary is inside the bytecode interpreter.
#
# A CLAMP RATHER THAN A COUNTER, for the reason that governs every instrument on this path: a
# wedge takes the core, so the host never returns and any value recorded in the shared region
# dies with it.  Only a build that RETURNS can report anything.  Clamping converts the hang into
# a clean SQLITE_DONE, and the arm that stops one opcode short of the fault RETURNS while naming
# the opcode it declined to run -- which is the answer, not a bracket around it.
#
# Ladder on n, ascending, all arms returning until one does not.
if [[ -n "${CAPSTONE_VDBE_CLAMP:-}" ]]; then
  python3 - "$OBJ_DIR/sqlite3-capstone.c" <<'PYVDBE'
import sys
path = sys.argv[1]
s = open(path).read()
NEEDLE = "    assert( pOp>=aOp && pOp<&aOp[p->nOp]);\n    nVmStep++;\n"
if NEEDLE not in s:
    sys.exit("VDBE_CLAMP: main-loop anchor not found -- patch shape changed")
# `pOp->opcode` is read BEFORE the clamp fires, so the reported opcode is the one that was ABOUT
# to execute -- the suspect -- not the last one that succeeded.
clamp = NEEDLE + """
    if( capstone_vdbe_armed && ++capstone_vdbe_ops >= (unsigned long)(CAPSTONE_VDBE_CLAMP) ){
      capstone_vdbe_lastop = (unsigned long)pOp->opcode;
      rc = SQLITE_DONE;
      goto vdbe_return;
    }
"""
s = s.replace(NEEDLE, clamp, 1)
ANCHOR = "#define SQLITE_CORE 1\n"
if ANCHOR not in s:
    sys.exit("VDBE_CLAMP: amalgamation prologue anchor not found")
s = s.replace(ANCHOR, ANCHOR + "extern unsigned long capstone_vdbe_ops, capstone_vdbe_lastop, capstone_vdbe_armed;\n", 1)
open(path, "w").write(s)
print("   VDBE clamp injected into sqlite3VdbeExec")
PYVDBE
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_VDBE_CLAMP=${CAPSTONE_VDBE_CLAMP}"
fi

# CAPSTONE_CREATE_LADDER=<n> -- split CREATE TABLE into prepare/step/finalize and RETURN a
# 0x5A6E_ssrr marker after stage n. Only meaningful with SQLITE_LDC_HIGH_HALF_FIXUP=1, which is
# the configuration that wedges: a wedge takes the core, so every in-domain probe is silent on
# the one path that matters. Stages ascend, so the whole ladder goes into ONE boot and the first
# arm that fails to return is the bisection point. n >= 4 runs the whole statement and falls
# through to the rest of the workload.
if [[ -n "${CAPSTONE_CREATE_LADDER:-}" ]]; then
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_CREATE_LADDER=${CAPSTONE_CREATE_LADDER}"
fi

if [[ "${CAPSTONE_DBWHO_PROBE:-0}" != "1" && "${CAPSTONE_DBFIX_PROBE:-0}" == "1" ]]; then
  echo "ERROR: CAPSTONE_DBFIX_PROBE=1 needs CAPSTONE_DBWHO_PROBE=1 -- the repair arm is part of"\
       "that probe, and on its own it would silently do nothing." >&2
  exit 1
fi

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
# THE S-06 AGGREGATE-COPY GUARD IS ON BY DEFAULT FOR THIS BUILD.
#
# Measured 2026-08-14, matched pair in one boot with the control returning: without it, the VDBE
# instruction operand pOp->p3 reads 1699 where it should read 1, so `&aMem[p3]` lands outside the
# heap and the hardware correctly raises OUT_OF_BOUNDS; with it, p3 reads 1 and the basic workload
# -- CREATE, INSERT, the SELECT with all three rows, finalize -- passes on silicon for the first
# time. The corruption is unguarded compiler-emitted 16-byte granule copies landing in the VdbeOp
# array, which is S-06 reaching the program through the compiler rather than through memcpy.
#
# ON HERE AND NOT GLOBALLY, deliberately. The pass emits `lcc` field 1, and that query is only
# TOTAL on enabler silicon; on an older bitstream it RAISES on plain data, so a target-wide default
# would break every build that does not run on this hardware. This script targets the current
# bitstream, so the knob belongs here.
#
# SQLITE_GRANULE_GUARD=0 turns it off for an A/B. It is a WORKAROUND, not a fix: it costs +33660
# bytes of .text and a branch per granule, and both are in the primitive every timing number is
# measured on. See ref/S06-WORKAROUNDS-TO-REVERT.md for what to remove when silicon is fixed.
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
# AMALGAM_EXTRA_MLLVM / SUPPORT_EXTRA_MLLVM apply -mllvm flags to ONE object instead of all of
# them. EXTRA_MLLVM goes into SILICON and therefore reaches the amalgamation AND both support
# objects at once, which makes a codegen flag that misbehaves impossible to localise: the whole
# domain either works or does not. These let a bisect answer "which object" in one QEMU run each.
read -r -a _amalgam_mllvm <<< "${AMALGAM_EXTRA_MLLVM:-}"
read -r -a _support_mllvm <<< "${SUPPORT_EXTRA_MLLVM:-}"
# CAPSTONE_MCP_TAGCHECK=1 -- instrument the REAL memcpy (the one the domain links) to ask whether
# it is HANDED an untagged destination or loses the tag mid-loop, and to convert the resulting
# wedge into a wrong answer so the recording reaches the host.
#
# The define has to reach BOTH objects: beebs_freestanding_string.c, which holds memcpy and the
# counters, and the amalgamation TU, which holds the reporter and the EXT_STOP call site. Setting
# it on only one produced a link error the first time a probe was split this way, which is the
# good version of that mistake -- the silent version is a reporter compiled out while its caller
# still thinks it ran.
#
# IT MUST SIT ABOVE `read -r -a _domain_defs`. Placed after it, the append to DOMAIN_EXTRA_DEFS
# reached BEEBS_STRING_EXTRA_DEFS (read later) but not the amalgamation TU (read earlier), so
# memcpy was instrumented, the counters were populated, and the reporter that prints them was
# compiled out -- a build that runs the experiment and cannot say what happened. Caught only
# because the MCP line was missing from an otherwise clean QEMU run.
if [[ "${CAPSTONE_MCP_TAGCHECK:-0}" == "1" ]]; then
  BEEBS_STRING_EXTRA_DEFS="${BEEBS_STRING_EXTRA_DEFS:-} -DBEEBS_MEMCPY_TAGCHECK=1"
  if [[ "${CAPSTONE_MCP_SELFTEST:-0}" == "1" ]]; then
    BEEBS_STRING_EXTRA_DEFS="$BEEBS_STRING_EXTRA_DEFS -DBEEBS_MCP_SELFTEST=1"
  fi
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DBEEBS_MEMCPY_TAGCHECK=1"
fi

# CAPSTONE_EVICT_PROBE=1 -- does a capability survive a spill/reload across a CACHE EVICTION?
# The last axis the ladder rungs cannot reach: they touch sixteen adjacent slots in a tight loop,
# which never leave the cache. Set ABOVE the _domain_defs read, like every other domain knob --
# below it the define reaches nothing and the probe is silently compiled out.
if [[ "${CAPSTONE_EVICT_PROBE:-0}" == "1" ]]; then
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_EVICT_PROBE=1"
fi

# S-07 operand discrimination. cincoffset raises mcause 25 for rs1 NOT_CAP *or* rs2 not-NOT_CAP
# (capstone_flu_unit.anvil:29-31), so the wedge alone cannot say which operand was wrong. With this
# on, output_text asks both per iteration with the TOTAL type query and retries a lost tag from its
# own stack slot, then returns a packed 0x5A...... count instead of wedging.
# ABOVE the _domain_defs read, like every other domain knob -- appended after it, the define
# reaches nothing and the probe silently is not in the binary.
if [[ "${SQLITE_OPERAND_PROBE:-0}" == "1" ]]; then
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_OPERAND_PROBE"
fi
read -r -a _domain_defs <<< "${DOMAIN_EXTRA_DEFS:-}"
"$CAPSTONE_CLANG" "${COMMON[@]}" "${SILICON[@]}" $SQLITE_DEFINES "${SILICON_TRIM[@]}" "$OPT" \
  -DSQLITE_HEAP_SIZE=$HEAP "${_domain_defs[@]}" "${_amalgam_mllvm[@]}" \
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
# BEEBS_STRING_WRITERS_OPTNONE and BEEBS_LDC_HIGH_HALF_FIXUP address two DIFFERENT measured
# defects in the same primitive, which is why both are on. Board stage 167, one boot, control
# green, decoding memmove's two arms:
#
#   writers at -O1     0x19   bit 0 SET (7-byte memmove lost)  bit 3 SET (32-byte lost)
#   writers optnone    0x78   bit 0 clear                      bit 3 STILL SET
#
# So optnone fixes the small-copy path (the S-04 family) and does nothing for the block path,
# because that one is the untagged ldc/stc high-half loss -- a separate defect, reproduced in
# RTL simulation, fixed by the compare-and-repair sequence. Target with both on: stage 167
# reads 0x70 and stage 169 reads 0x40 (all 32 bytes survive).
_writers_optnone=${SQLITE_WRITERS_OPTNONE:-1}
# BEEBS_LDC_HIGH_HALF_FIXUP is DEFAULT OFF, and that is a measured decision, not caution.
# It is correct at the PRIMITIVE level on silicon -- with it on, stage 169 reads 0x40 (all 32
# bytes of a block copy survive, dst32 byte-identical to src32) and stage 167 reads 0x70 (every
# writer clean) -- but it WEDGES the full SQLite workload. Isolated in one boot, control green
# (k800 = 4), three builds differing by one knob each:
#
#   qC  memcpy optnone only        RETURNS, stage=create rc=11
#   qB  + writers optnone          RETURNS, stage=create rc=11   (so writers optnone is neutral)
#   qA  + ldc high-half fixup      WEDGES
#
# RESOLVED 2026-08-11: the fixup is NOT what destabilises the workload. It repairs the schema
# text, so SQLite stops bailing out early on a corrupt schema and runs deeper into CREATE TABLE
# than it ever has on silicon, where it takes mcause 25 INVALID_CAPABILITY. Measured: a REDRAW
# control (baseline + one dead function) RETURNS, so it is not image sensitivity; stages 170/171
# show tags survive the fixup; and a RUNSTOP ladder returns after initialize and after open and
# wedges only after CREATE. Default stays OFF only because an error return is easier to work
# with than a wedge -- turn it ON to chase the CREATE-path fault, which needs correct data to be
# reachable at all. See ISSUES.md S-06.
SUPPORT_DEFS=(-DBEEBS_STRING_LINEAR_SAFE=1 -DBEEBS_MEMCPY_OPTNONE=$_memcpy_optnone \
              -DBEEBS_STRING_WRITERS_OPTNONE=$_writers_optnone \
              ${BEEBS_STRING_EXTRA_DEFS:-})
for pair in "libc:$ADAPTED/capstone_sqlite_libc.c" "beebs_string:$BEEBS_STRING"; do
  "$CAPSTONE_CLANG" "${COMMON[@]}" "${SILICON[@]}" $SQLITE_DEFINES "${SILICON_TRIM[@]}" \
    "${SUPPORT_DEFS[@]}" "$SUPPORT_OPT" "${_support_mllvm[@]}" \
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
