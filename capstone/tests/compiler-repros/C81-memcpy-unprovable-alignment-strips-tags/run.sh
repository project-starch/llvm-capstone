#!/usr/bin/env bash
# Two-sided, with a positive control, because a "no byte expansion" reading is worthless
# unless this script can produce one. Prints a verdict per function per -O level.
#
# The compiler: $CAPSTONE_LLVM_BIN, else the convention's llvm/cmake-build-debug/bin.
# A worktree has no cmake-build-debug, so the override is the normal case here.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
BIN=${CAPSTONE_LLVM_BIN:-$REPO/llvm/cmake-build-debug/bin}
CC=$BIN/clang
[ -x "$CC" ] || { echo "no clang at $CC (set CAPSTONE_LLVM_BIN)" >&2; exit 2; }
echo "compiler: $("$CC" --version 2>&1 | head -1)"
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT
FAIL=0
FLAGS=(--no-default-config -target capstone64-unknown-elf -nostdinc
       -Xclang -target-feature -Xclang +m -Xclang -target-feature -Xclang +a
       -Xclang -target-feature -Xclang +f -Xclang -target-feature -Xclang +d
       -Xclang -target-abi -Xclang lp64 -fno-jump-tables
       -mllvm -capstone-gp-free -mllvm -capstone-image-gp)
printf '%-28s %-6s %-14s %-5s %-5s %s\n' FUNCTION OPT ARM MOVED CAPS VERDICT
for opt in -O0 -O1 -O2; do
  for arm in plain nobuiltin; do
    extra=(); [ "$arm" = nobuiltin ] && extra=(-fno-builtin)
    "$CC" "${FLAGS[@]}" "${extra[@]+"${extra[@]}"}" "$opt" -S "$HERE/repro.c" -o "$OUT/r.s" 2>"$OUT/err" || {
      printf '%-28s %-6s %-14s COMPILE FAILED: %s\n' - "$opt" "$arm" "$(head -1 "$OUT/err")"; FAIL=1; continue; }
    python3 - "$OUT/r.s" "$opt" "$arm" <<'PY'
import re, sys
L = open(sys.argv[1]).read().splitlines(); opt, arm = sys.argv[2], sys.argv[3]
# SCALAR covers every tag-stripping unit, not just bytes: an EffAlign-8 ld/sd expansion
# strips a tag exactly as lbu/sb does, so the classifier is unit-aware by design.
EXPECT = {("copy32_unknown","plain"):"SCALAR1", ("copy32_unknown","nobuiltin"):"CALL",
          ("copy32_align8","plain"):"SCALAR8",  ("copy32_align8","nobuiltin"):"CALL",
          ("copy32_known","plain"):"CAPS",      ("copy32_known","nobuiltin"):"CALL",
          ("copy32_unknown_builtin","plain"):"SCALAR1",
          ("copy32_unknown_builtin","nobuiltin"):"SCALAR1"}
STORES = {"sb":1, "sh":2, "sw":4, "sd":8}
rc = 0
for fn in ("copy32_unknown", "copy32_align8", "copy32_known", "copy32_unknown_builtin"):
    a = next((i for i,l in enumerate(L) if re.match(rf'^{fn}:', l.strip())), None)
    b = next((i for i,l in enumerate(L) if a is not None and i>a
              and re.match(rf'\.size\s+{fn}', l.strip())), None)
    if a is None or b is None:
        print(f"{fn:<28} {opt:<6} {arm:<14} NO BRACKET -- instrument failed"); rc = 1; continue
    ins = [re.sub(r'\s+',' ',l.split('#')[0].strip()) for l in L[a+1:b]
           if l.strip() and not l.strip().startswith(('.','#','//'))]
    moved = sum(STORES[m] for m in (i.split()[0] for i in ins) if m in STORES)
    unit = min((STORES[m] for m in (i.split()[0] for i in ins) if m in STORES), default=0)
    nc = sum(1 for i in ins if re.match(r'(stc|ldc)\b', i))
    call = any('memcpy' in i for i in ins)
    got = (f"SCALAR{unit}" if moved >= 16 and unit and unit < 16
           else "CALL" if call else "CAPS" if nc else "?")
    want = EXPECT[(fn, arm)]
    ok = (got == want)
    note = "tags LOST" if got.startswith("SCALAR") else "tags kept" if got in ("CAPS","CALL") else "?"
    print(f"{fn:<28} {opt:<6} {arm:<14} {moved:<5} {nc:<5} {got} ({note})"
          f"{'' if ok else '   <-- UNEXPECTED, wanted ' + want}")
    if not ok: rc = 1
raise SystemExit(rc)
PY
    [ $? -eq 0 ] || FAIL=1
  done
done
echo
if [ "$FAIL" -eq 0 ]; then
  echo "RESULT: defect PRESENT at BOTH units and every control behaved --"
  echo "        copy32_unknown unit 1, copy32_align8 unit 8, both tag-stripping;"
  echo "        copy32_known keeps tags; copy32_unknown_builtin expands under -fno-builtin too."
else
  echo "RESULT: at least one arm did not match the recorded expectation -- read the table."
fi
exit "$FAIL"
