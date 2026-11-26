#!/usr/bin/env python3
"""mc-grade.py -- grade a misscost session's result lines against this kit's expected vectors (rtl lane, 2026-11-26).

  mc-grade.py <expected-vectors.json> <result-lines.txt>

Reads the `NN-<image> SR n SV ...` lines and checks every E position exactly (rule 1/2's control and count) and
PREREG rule 3's structure on both srvcost images. Prints the four per-miss costs. Exit 0 only when every check holds;
1 on any failure; 2 when an image has no row (no data is an error).
"""
import json, sys

def main():
    ev = json.load(open(sys.argv[1]))
    rows = {}
    for l in open(sys.argv[2]):
        p = l.split()
        if len(p) > 3 and p[1] == "SR" and p[0][:2].isdigit():
            rows[p[0][3:]] = [int(x, 16) for x in p[4:]]
    fail = 0
    for name, e in ev.items():
        got = rows.get(name)
        if got is None:
            print(name, "NO ROW"); sys.exit(2)
        vec = [int(x, 16) if isinstance(x, str) else x for x in e["vector"]]
        cls = e["classes"]
        bad = [i + 1 for i, (c, a, b) in enumerate(zip(cls, vec, got)) if c == "E" and a != b]
        if len(got) != len(vec) or bad:
            fail = 1
        print(name, "SR", len(got), "expected", len(vec), "E", cls.count("E"), "E-mismatch", bad)
    w, c = rows["b5b2-srvcost"], rows["b5b2-srvcost-cold"]
    P = lambda v, k: v[4 + k - 1]
    ok2 = (P(w, 3) > P(w, 2) and P(w, 7) > P(w, 2) and w[12] == 8
           and all(abs(P(w, k) - P(w, 2)) <= 0.2 * P(w, 2) for k in (4, 5, 6, 8)))
    ok3 = P(c, 4) > P(c, 2) and P(c, 3) > P(c, 4) and P(c, 7) > P(c, 2) and c[12] == 8
    print("rule 3: b5b2-srvcost", "AS PREDICTED" if ok2 else "FAIL", "/ b5b2-srvcost-cold", "AS PREDICTED" if ok3 else "FAIL")
    print("warm %.1f %.1f  cold %.1f %.1f cycles per miss" % ((P(w, 3) - P(w, 2)) / 33, (P(w, 7) - P(w, 2)) / 33,
                                                             (P(c, 3) - P(c, 4)) / 33, (P(c, 7) - P(c, 2)) / 33))
    sys.exit(0 if (ok2 and ok3 and not fail) else 1)

main()
