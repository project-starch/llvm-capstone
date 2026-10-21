#!/usr/bin/env python3
"""Select the Juliet 1.3 C test cases for one set and write the manifest every later step reads.

    select.py --juliet <dir>/C --set temporal|spatial --out <manifest.json>

SETS.
  temporal  CWE-415 double free, CWE-416 use after free (the temporal classes PoisonCap evaluated).
  spatial   CWE-121 stack overflow, CWE-122 heap overflow, CWE-124 buffer underwrite,
            CWE-126 buffer over-read, CWE-127 buffer under-read.

THE UNIT is a Juliet test case (SARD's unit): one functional variant x one flow variant, all of its files
(`_NN.c`, or `_NNa.c` ... `_NNe.c` for the multi-file flows). C only: `.cpp` cases are never selected.

WHAT IS LEFT OUT, by name, and recorded as such in the manifest: cases whose data source is a socket, stdin
(fgets/fscanf), rand() or the environment, and Windows-only (w32) cases. Their input would have to be supplied;
every selected case is self-contained.

FLAGS carried per case, so a reader can split the denominator:
  wchar_t       the case's name says wchar_t (Juliet's own Linux Makefile, and PoisonCap's count, leave these out)
  intra_object  an overflow inside one struct (`type_overrun`); the virtual profile has no sub-object bounds
  rand_flow     the bad path is chosen by rand() (flow variant 12); run.py --seeded settles whether it ran
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SETS = {"temporal": (415, 416), "spatial": (121, 122, 124, 126, 127)}
NAME = re.compile(r"(CWE(\d+)_.*?_\d\d)([a-e]|_bad|_good\w*)?\.c$")
LEFT_OUT = re.compile(r"w32|socket|fgets|fscanf|rand|environment")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--juliet", type=Path, required=True, help="the unpacked C/ directory (fetch-juliet.sh)")
    p.add_argument("--set", choices=sorted(SETS), required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    tc = a.juliet / "testcases"
    if not (tc.is_dir() and (a.juliet / "testcasesupport/io.c").is_file()):
        sys.exit(f"{a.juliet} is not Juliet's C/ directory")
    files, left_out = defaultdict(list), defaultdict(set)
    for f in sorted(tc.rglob("CWE*.c")):
        m = NAME.match(f.name)
        if not m or int(m.group(2)) not in SETS[a.set]:
            continue
        stem = m.group(1)
        hit = LEFT_OUT.search(stem.lower())
        if hit:
            left_out[(int(m.group(2)), hit.group(0))].add(stem)
            continue
        files[stem].append(f)
    cases = []
    for stem, fs in sorted(files.items()):
        text = "".join(f.read_text(errors="replace") for f in fs)
        cases.append({"case": stem, "cwe": int(stem[3:6]),
                      "files": [str(f.relative_to(tc)) for f in sorted(fs)],
                      "sha256": {str(f.relative_to(tc)): sha256(f) for f in sorted(fs)},
                      "wchar_t": "wchar_t" in stem.lower(), "intra_object": "type_overrun" in stem.lower(),
                      "rand_flow": "globalReturnsTrueOrFalse" in text or "rand()" in text})
    support = {n: sha256(a.juliet / "testcasesupport" / n) for n in ("io.c", "std_testcase.h", "std_testcase_io.h")}
    manifest = {"suite": "NIST SARD Juliet C/C++ 1.3", "set": a.set, "cwes": list(SETS[a.set]), "language": "C",
                "support": support, "left_out": {f"CWE-{c} {why}": len(s) for (c, why), s in sorted(left_out.items())},
                "cases": cases}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(manifest, indent=1))
    by = Counter(c["cwe"] for c in cases)
    print(f"{a.set}: {len(cases)} cases {dict(sorted(by.items()))}; intra_object "
          f"{sum(c['intra_object'] for c in cases)}, rand_flow {sum(c['rand_flow'] for c in cases)}, "
          f"wchar_t {sum(c['wchar_t'] for c in cases)}; left out {sum(len(s) for s in left_out.values())}")


if __name__ == "__main__":
    main()
