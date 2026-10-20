#!/usr/bin/env python3
"""rep-size-sweep.py MODEL < cases: re-check every rep-size-sweep case against the RTL's compress/decompress
(MODEL = capstone-ariane verif/tests/custom/capstone/bounds-model.py of a bitstream tree). Exit 0 only when cases were
read, every kind fired, and the RTL agrees with the expectation on every case."""
import collections, importlib.util, sys
spec = importlib.util.spec_from_file_location("bm", sys.argv[1]); bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)
n = bad = 0; kinds = collections.Counter()
for line in sys.stdin:
    kind, b, l, expect, qemu = line.split()
    b, l = int(b, 16), int(l, 16)
    iE, B, T, E = bm.compress(b, b + l)[:4]
    rb, re_ = bm.decompress(iE, B, T, b)
    rtl = "EXACT" if (rb, re_) == (b, b + l) else "INEXACT"
    n += 1; kinds[kind] += 1
    if rtl != expect or qemu != expect:
        bad += 1
        if bad <= 20: print(f"DISAGREE {kind} [{b:#x}, +{l:#x}) expect {expect} qemu {qemu} rtl {rtl}")
print(f"rep-size-sweep: {n} cases {dict(kinds)}, disagreements {bad}")
sys.exit(0 if n and not bad and all(kinds[k] for k in ("pad", "small", "raw", "skew", "short")) else 1)
