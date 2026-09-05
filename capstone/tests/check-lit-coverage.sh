#!/usr/bin/env bash
# check-lit-coverage.sh -- the "definition of complete" for Capstone compiler tests,
# as a deterministic gate.
#
# A script, not a subagent, on purpose: a completeness claim about test coverage
# has to be reproducible and cheap enough to run on every commit.  It reads the
# backend sources for the things that MUST be covered (instructions, intrinsics,
# option flags, fatal-error routes) and the test tree for what IS covered, and
# exits non-zero on any gap.  It carries its own positive control (--self-test):
# run against a stub tree with no tests it must report gaps and exit 1, so a
# silent zero can never be mistaken for completeness.
#
#   exit 0  complete
#   exit 1  gaps (each printed as one line: SECTION | item | what is missing)
#   exit 2  internal error (a source file it depends on is missing)
#
# Usage:
#   check-lit-coverage.sh [--only instr,intrinsics,flags,fatal,olevels,cnn,mc,mutations]
#   check-lit-coverage.sh --self-test
#   check-lit-coverage.sh --root <repo-root>        (default: two levels above this file)
#
# Data files next to this script:
#   lit-coverage-inherited-flags.txt  capstone-* cl::opts inherited from RISCV (no coverage demanded)
#   lit-coverage-unreachable.txt      fatal routes proven unreachable: file:line | reason | pinning test
#   lit-coverage-olevel-exempt.txt    tests exempt from the -O0/-O1 arm rule: file | reason
#   lit-coverage-cnn.txt              open defect IDs: ID | test path, or REASON: <why untestable>

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ONLY=""
SELFTEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --self-test) SELFTEST=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$SELFTEST" -eq 1 ]; then
  # Positive control: a tree with the real backend sources but an EMPTY test dir
  # must produce gaps.  We build it by symlinking the backend into a scratch root.
  T="$(mktemp -d)"
  mkdir -p "$T/llvm/lib/Target" "$T/llvm/include/llvm/IR" "$T/llvm/test/CodeGen/Capstone" \
           "$T/llvm/test/MC" "$T/clang/test/CodeGen" "$T/clang/test/Sema" "$T/capstone/tests"
  ln -s "$ROOT/llvm/lib/Target/Capstone" "$T/llvm/lib/Target/Capstone"
  ln -s "$ROOT/llvm/include/llvm/IR/IntrinsicsCapstone.td" "$T/llvm/include/llvm/IR/IntrinsicsCapstone.td"
  for f in lit-coverage-inherited-flags.txt lit-coverage-unreachable.txt \
           lit-coverage-olevel-exempt.txt lit-coverage-cnn.txt; do
    [ -f "$HERE/$f" ] && cp "$HERE/$f" "$T/capstone/tests/$f"
  done
  printf '; RUN: llc < %%s\n; CHECK: nothing\ndefine void @f() { ret void }\n' > "$T/llvm/test/CodeGen/Capstone/empty.ll"
  "$0" --root "$T" > "$T/out.txt" 2>&1
  rc=$?
  rm -rf "$T"
  if [ "$rc" -eq 1 ]; then echo "self-test: OK (stub tree reported gaps, exit 1)"; exit 0; fi
  echo "self-test: FAILED -- stub tree exited $rc, expected 1" >&2; exit 2
fi

exec python3 - "$ROOT" "$ONLY" "$HERE" <<'PY'
import os, re, sys, glob
ROOT, ONLY, HERE = sys.argv[1], sys.argv[2], sys.argv[3]
sections = ONLY.split(",") if ONLY else ["instr","intrinsics","flags","fatal","olevels","cnn","mc","mutations"]
gaps = []
def gap(sec, item, what): gaps.append(f"{sec:10s} | {item} | {what}")
def rd(p):
    try: return open(p, encoding="utf-8", errors="replace").read()
    except OSError: return None
def data(name):
    p = os.path.join(HERE, name)
    s = rd(p)
    if s is None: return []
    return [l.strip() for l in s.splitlines() if l.strip() and not l.lstrip().startswith("#")]

TD  = os.path.join(ROOT, "llvm/lib/Target/Capstone/CapstoneInstrInfo.td")
INT = os.path.join(ROOT, "llvm/include/llvm/IR/IntrinsicsCapstone.td")
CG  = os.path.join(ROOT, "llvm/test/CodeGen/Capstone")
MCD = os.path.join(ROOT, "llvm/test/MC/Capstone")
DIS = os.path.join(ROOT, "llvm/test/MC/Disassembler/Capstone")
CL  = [os.path.join(ROOT, "clang/test/CodeGen"), os.path.join(ROOT, "clang/test/Sema")]
for must in (TD, INT):
    if rd(must) is None:
        print(f"internal error: missing {must}", file=sys.stderr); sys.exit(2)

def files(d, pats):
    out = []
    for p in pats: out += glob.glob(os.path.join(d, p))
    return sorted(set(out))
cg_tests  = files(CG, ["*.ll", "*.mir"])
mc_tests  = files(MCD, ["*.s"])
dis_tests = files(DIS, ["*.txt"])
cl_tests  = [f for d in CL for f in files(d, ["*capstone*.c", "*capstone*.cpp", "*capstone*.ll", "builtins-capstone.c"])]
cg_text   = {f: rd(f) or "" for f in cg_tests}
mc_text   = {f: rd(f) or "" for f in mc_tests}
dis_text  = {f: rd(f) or "" for f in dis_tests}
cl_text   = {f: rd(f) or "" for f in cl_tests}
CHECK_POS = re.compile(r'^\s*[;#/]+\s*[A-Z0-9_-]*CHECK(?:-NEXT|-SAME|-DAG|-COUNT-\d+|-LABEL)?:\s*(.*)$', re.M)
CHECK_NEG = re.compile(r'^\s*[;#/]+\s*[A-Z0-9_-]*CHECK-NOT:\s*(.*)$', re.M)
IMPL_NEG  = re.compile(r'--implicit-check-not[= ]["\']?([^\s"\']+)')
RUN       = re.compile(r'^\s*[;#/]+\s*RUN:\s*(.*)$', re.M)
def has_word(text, w): return re.search(r'(?<![\w.])' + re.escape(w) + r'(?![\w])', text) is not None

# ---- instructions -----------------------------------------------------------
if "instr" in sections:
    td = rd(TD)
    mnems = []
    # Defs may be indented inside a `let ... in {` block (the domain ops are), so
    # split on any line that begins, after whitespace, with `def`.
    for block in re.split(r'\n(?=\s*def\s)', td):
        m = re.match(r'\s*def\s+(\w+)', block)
        if not m: continue
        name = m.group(1)
        if name.startswith("Pseudo"): continue
        if "OPC_CAP_OP" not in block and name != "CJALR": continue
        q = re.search(r'"([a-z][a-z0-9.]*)"', block)
        if q: mnems.append((name, q.group(1)))
    if len(mnems) < 20:
        print(f"internal error: only {len(mnems)} capability instructions parsed from the .td", file=sys.stderr); sys.exit(2)
    for name, mn in mnems:
        pos = any(any(has_word(c, mn) for c in CHECK_POS.findall(t)) for t in cg_text.values())
        neg = any(any(has_word(c, mn) for c in CHECK_NEG.findall(t)) or any(has_word(x, mn) for x in IMPL_NEG.findall(t)) for t in cg_text.values())
        mcp = any(has_word(t, mn) for t in mc_text.values())
        mci = any(has_word(t, mn) for f, t in mc_text.items() if "invalid" in os.path.basename(f))
        dis = any(has_word(t, mn) for t in dis_text.values())
        if not pos: gap("instr", f"{name} ({mn})", "no positive CHECK in llvm/test/CodeGen/Capstone")
        if not neg: gap("instr", f"{name} ({mn})", "no CHECK-NOT / implicit-check-not control")
        if not mcp: gap("instr", f"{name} ({mn})", "no assembler test in llvm/test/MC/Capstone")
        if not mci: gap("instr", f"{name} ({mn})", "no invalid-operand diagnostic test (MC/Capstone/*invalid*.s)")
        if not dis: gap("instr", f"{name} ({mn})", "no disassembler test in llvm/test/MC/Disassembler/Capstone")

# ---- intrinsics -------------------------------------------------------------
if "intrinsics" in sections:
    names = re.findall(r'def\s+int_capstone_cap_(\w+)', rd(INT))
    if not names:
        print("internal error: no intrinsics parsed", file=sys.stderr); sys.exit(2)
    for n in names:
        ir = "@llvm.capstone.cap." + n.replace("_", ".")
        bi = "__builtin_capstone_cap_" + n
        if not any(ir in t for t in cg_text.values()): gap("intrinsics", n, f"no CodeGen test calls {ir}")
        if not any(bi in t for t in cl_text.values()):  gap("intrinsics", n, f"no clang test uses {bi}")

# ---- flags ------------------------------------------------------------------
CAPSTONE_FLAGS = {
  "capstone-shrink-globals": "bool", "capstone-shrink-stack": "bool", "capstone-gp-free": "bool",
  "capstone-gp-captable": "bool", "capstone-merge-string-constants": "bool",
  "capstone-merge-string-max-bytes": "num", "capstone-s12-movc-ldc-workaround": "bool",
  "capstone-s12-window": "num", "capstone-retry-untagged-ldc": "bool", "capstone-double-ldc": "bool",
  "capstone-memcpy-high-half-fixup": "bool", "capstone-memcpy-high-half-fixup-max-bytes": "num",
  "capstone-memcpy-fixup-no-stc": "bool", "capstone-memcpy-fixup-no-plain-stores": "bool",
  "capstone-lower-memops-via-libcall": "bool", "capstone-cap-init-limit": "num", "capstone-cap-init-print": "bool",
  "capstone-gp-captable-jump-tables": "bool",
}
if "flags" in sections:
    runs = [r for t in cg_text.values() for r in RUN.findall(t)]
    for flag, kind in CAPSTONE_FLAGS.items():
        vals = set()
        for r in runs:
            for m in re.finditer(r'-' + re.escape(flag) + r'(?:=([^\s|]+))?', r):
                vals.add(m.group(1) if m.group(1) is not None else "true")
        if kind == "bool":
            if "true" not in vals: gap("flags", flag, "no RUN line with =true (or bare)")
            if "false" not in vals: gap("flags", flag, "no RUN line with =false")
        else:
            if len(vals) < 2: gap("flags", flag, f"needs RUN lines at two distinct values, found {sorted(vals) or 'none'}")
    inherited = set(data("lit-coverage-inherited-flags.txt"))
    found = set()
    for f in glob.glob(os.path.join(ROOT, "llvm/lib/Target/Capstone/*.cpp")):
        for m in re.finditer(r'cl::opt<[^;]{0,400}?"(capstone-[a-z0-9-]+)"', rd(f) or "", re.S):
            found.add(m.group(1))
    for n in sorted(found - set(CAPSTONE_FLAGS) - inherited):
        gap("flags", n, "capstone-* cl::opt in the backend that is neither covered nor listed as inherited")

# ---- fatal / diagnostic routes ----------------------------------------------
if "fatal" in sections:
    # Entries are keyed `file:line` or, preferably, `file:msg=<prefix of the message>`:
    # a line number goes stale with every edit above it (it did three times in one
    # day), a message prefix does not.
    unreachable = {}
    for l in data("lit-coverage-unreachable.txt"):
        parts = [p.strip() for p in l.split("|")]
        if len(parts) >= 3: unreachable[parts[0]] = (parts[1], parts[2])
    def unreachable_entry(key, fname, msg):
        if key in unreachable: return unreachable[key]
        for k, v in unreachable.items():
            if k.startswith(fname + ":msg=") and msg.startswith(k[len(fname) + 5:]): return v
        return None
    all_checks = " ".join(c for t in list(cg_text.values()) + list(cl_text.values()) for c in CHECK_POS.findall(t))
    seen_fatal = set()
    for f in glob.glob(os.path.join(ROOT, "llvm/lib/Target/Capstone/*.cpp")):
        src = rd(f) or ""
        # Bound each match to ONE call: from the call name to the first `);` that
        # closes it, then take the string literals inside that span (adjacent
        # literals are one message).  A span may not cross another statement.
        for m in re.finditer(r'\b(report_fatal_error|reportFatalUsageError|diagnose|DiagnosticInfoUnsupported)\s*\(', src):
            end = src.find(");", m.end())
            if end < 0 or end - m.end() > 1200: continue
            span = src[m.end():end]
            if ";" in span: continue
            lits = re.findall(r'"((?:[^"\\]|\\.)*)"', span)
            if not lits: continue
            msg = "".join(lits)
            if "Capstone" not in msg and "PureCap" not in msg: continue
            line = src.count("\n", 0, m.start()) + 1
            key = f"{os.path.basename(f)}:{line}"
            if key in seen_fatal: continue
            seen_fatal.add(key)
            frag = max(lits, key=len)[:40]
            if frag in all_checks: continue
            if (ent := unreachable_entry(key, os.path.basename(f), msg)) is not None:
                reason, pin = ent
                if pin.upper().startswith("TODO") or not os.path.exists(os.path.join(ROOT, pin)):
                    gap("fatal", key, f"recorded unreachable but its pinning test is missing: {pin}")
                continue
            gap("fatal", key, f'no test CHECKs "{frag}..." and not in lit-coverage-unreachable.txt')

# ---- -O arms ----------------------------------------------------------------
if "olevels" in sections:
    exempt = {l.split("|")[0].strip() for l in data("lit-coverage-olevel-exempt.txt")}
    for f, t in cg_text.items():
        b = os.path.basename(f)
        if not b.endswith(".ll") or b in exempt: continue
        runs = " ".join(RUN.findall(t))
        for lvl in ("-O0", "-O1"):
            if not re.search(r'(?<![\w-])' + re.escape(lvl) + r'(?![\w])', runs):
                gap("olevels", b, f"no RUN line at {lvl}")

# ---- open defect IDs --------------------------------------------------------
if "cnn" in sections:
    for l in data("lit-coverage-cnn.txt"):
        parts = [p.strip() for p in l.split("|")]
        if len(parts) < 2: gap("cnn", l, "malformed line"); continue
        cid, target = parts[0], parts[1]
        if target.upper().startswith("REASON:"): continue
        if target.upper().startswith("TODO"): gap("cnn", cid, f"no regression test yet ({target})"); continue
        p = os.path.join(ROOT, target)
        if not os.path.exists(p): gap("cnn", cid, f"mapped test does not exist: {target}"); continue
        if cid not in (rd(p) or ""): gap("cnn", cid, f"mapped test {target} does not mention {cid}")

# ---- MC dirs ----------------------------------------------------------------
if "mc" in sections:
    if not mc_tests:  gap("mc", "llvm/test/MC/Capstone", "no assembler tests")
    if not dis_tests: gap("mc", "llvm/test/MC/Disassembler/Capstone", "no disassembler tests")

# ---- mutation headers -------------------------------------------------------
if "mutations" in sections:
    for f, t in list(cg_text.items()) + list(cl_text.items()) + list(mc_text.items()):
        if (CHECK_NEG.search(t) or IMPL_NEG.search(t)) and "MUTATION:" not in t:
            gap("mutations", os.path.relpath(f, ROOT), "has a negative check but no `MUTATION:` header showing it can fire")

for g in gaps: print(g)
print(f"--- {len(gaps)} gap(s); sections: {','.join(sections)}; tests: CodeGen {len(cg_tests)}, MC {len(mc_tests)}, Disassembler {len(dis_tests)}, clang {len(cl_tests)}")
sys.exit(1 if gaps else 0)
PY
