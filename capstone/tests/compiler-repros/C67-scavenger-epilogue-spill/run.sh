#!/usr/bin/env bash
# C-67: does the register scavenger, in a >2 KiB frame whose epilogue has no free register, spill
# only the integer half of a capability register and reload it after sp has been released?
# Usage: LLC=<path to llc> ./run.sh [file.mir]   Prints PRESENT or ABSENT; exit 0 = PRESENT, 1 = ABSENT,
# 2 = no verdict (llc failed, or the function did not come out as expected).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LLC=${LLC:?set LLC to an llc with the Capstone target}
MIR=${1:-$HERE/src/all-live-epilogue.mir}
out=$("$LLC" -mtriple=capstone64 -mattr=+m,+movc-keeps-integer-source -O2 -start-before=prologepilog \
      -o - "$MIR" 2>&1) || {
  # An llc without the live-source copy rule does not know the feature; retry without it.
  out=$("$LLC" -mtriple=capstone64 -mattr=+m -O2 -start-before=prologepilog \
        -o - "$MIR" 2>&1) || { echo "llc failed:"; echo "$out" | head -5; exit 2; }
}
python3 - "$out" <<'PY'
import re, sys
body = [l.split("#")[0].strip() for l in sys.argv[1].splitlines()]
body = [l for l in body if l and not l.startswith(".")]
if "cjalr zero, 0(ra)" not in [re.sub(r"\s+", " ", l) for l in body]:
    print("no verdict: the function has no return in the output"); sys.exit(2)
ins = [re.sub(r"\s+", " ", l) for l in body]
# The epilogue's release of the frame: the last `cincoffset sp, sp, <reg>` before the return.
rel = max((i for i, l in enumerate(ins) if re.fullmatch(r"cincoffset sp, sp, a\d|cincoffset sp, sp, t\d|cincoffset sp, sp, s\d+", l)), default=None)
if rel is None:
    print("no verdict: no register-sized frame release found"); sys.exit(2)
hits = []
for i, l in enumerate(ins):
    m = re.fullmatch(r"sd (\w+), (-?\d+)\(sp\)", l)
    if m and i < rel:
        reg = m.group(1)
        for j in range(rel + 1, len(ins)):
            if re.fullmatch(rf"ld {reg}, -?\d+\(sp\)", ins[j]):
                hits.append(f"{l} ... {ins[rel]} ... {ins[j]}")
if hits:
    print("PRESENT: " + "; ".join(hits)); sys.exit(0)
print("ABSENT"); sys.exit(1)
PY
