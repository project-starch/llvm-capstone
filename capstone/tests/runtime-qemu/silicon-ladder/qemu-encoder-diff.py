#!/usr/bin/env python3
"""Diff check-repr.py's port of the RTL's bounds encoder against capstone-qemu's cap_compress.c, compiled standalone.

check-repr.py's compress/decompress were ported from capstone-ariane's ariane_pkg.sv; the C is capstone-qemu's
own, but it mirrors the RTL's identifiers line for line, so agreement guards against a TYPO in either, not against a
shared misreading, and says nothing about whether either is right (both had the R-54 wrap until aad781e15a). The
script also counts the triples that decode NOT covering their input and checks them against R-54's predicate; at a
ref with the renormalisation (origin/bounds-encoder-renormalise) the word comparison is EXPECTED to fail on exactly
the wrapping triples until check-repr.py ports the fix.

    qemu-encoder-diff.py [--ref origin/isa/sublet-lifetimes] [--qemu <capstone-qemu checkout>] [-n 60000]

--format full (the default) compares the full scheme for every triple, the encoding capstone-qemu always uses.
Exit status 1 on any mismatch, or if the build or the run fails (an empty run is an error, never a pass).
"""
import argparse
import importlib.util
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

DRIVER = r"""
#include <stdio.h>
#include "cap_compress.h"
/* stdin: "start end cursor" (hex) per line; stdout: "lo hi decoded_base decoded_end" (hex) */
int main(void) {
    unsigned long long s, e, c;
    while (scanf("%llx %llx %llx", &s, &e, &c) == 3) {
        capfat_t f = {0};
        f.bounds.base = s; f.bounds.end = e; f.bounds.cursor = c; f.type = CAP_TYPE_LIN;
        uint64_t lo, hi;
        cap_compress(&f, &lo, &hi);
        capfat_t o;
        cap_uncompress(lo, hi, &o);
        printf("%llx %llx %llx %llx\n", (unsigned long long)lo, (unsigned long long)hi,
               (unsigned long long)o.bounds.base, (unsigned long long)o.bounds.end);
    }
    return 0;
}
"""


def git_show(qemu, ref, path):
    return subprocess.run(["git", "-C", qemu, "show", "%s:%s" % (ref, path)], capture_output=True, text=True,
                          check=True).stdout


def build(qemu, ref, d):
    src = "target/riscv/"
    open(os.path.join(d, "cap_compress.c"), "w").write(git_show(qemu, ref, src + "cap_compress.c"))
    open(os.path.join(d, "cap_compress.h"), "w").write(git_show(qemu, ref, src + "cap_compress.h"))
    cap = git_show(qemu, ref, src + "cap.h")
    # only the type definitions: cap.h's later inline helpers pull in QEMU headers
    cut = cap.index("typedef struct CapFat capfat_t;") + len("typedef struct CapFat capfat_t;")
    open(os.path.join(d, "cap.h"), "w").write(cap[:cut] + "\n#endif\n")
    open(os.path.join(d, "capstone_defs.h"), "w").write(
        "#define STATIC_ASSERT(COND, MSG) typedef char static_assertion_##MSG[(COND)?1:-1]\n")
    open(os.path.join(d, "drv.c"), "w").write(DRIVER)
    exe = os.path.join(d, "drv")
    subprocess.run(["cc", "-O1", "-o", exe, os.path.join(d, "drv.c"), os.path.join(d, "cap_compress.c")],
                   check=True)
    return exe


def cases(n, seed):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        k = rnd.randrange(0, 40)
        mode = rnd.randrange(4)
        if mode == 0:
            ln = rnd.randrange(1, 1 << (k + 1))
        elif mode == 1:                                  # just under a power of two: where the wrap lives
            ln = max(1, (1 << k) - rnd.randrange(0, 1 << max(0, k - 8)))
        elif mode == 2:
            ln = (1 << k) + rnd.randrange(0, 1 << max(0, k - 8))
        else:
            ln = rnd.randrange(1, 1 << 14)
        al = rnd.choice([0, 4, 8, 12, 16, 21])
        base = (rnd.randrange(0, 1 << 40) >> al) << al
        if rnd.randrange(8) == 0:                        # the top of the address space, where a 64-bit end add wraps
            base = max(0, (1 << 64) - ln - rnd.randrange(0, 1 << 20))
        cur = base if rnd.randrange(3) == 0 else base + rnd.randrange(0, ln + 1)
        out.append((base, base + ln, cur))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qemu", default=os.path.normpath(os.path.join(HERE, "../../../capstone-qemu")))
    ap.add_argument("--ref", default="origin/isa/sublet-lifetimes")
    ap.add_argument("-n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--renorm", action="store_true",
                    help="model R-54's fix (check-repr.py --renorm); use with a QEMU ref that carries it "
                         "(origin/bounds-encoder-renormalise); every decode must then cover its input")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("cr", os.path.join(HERE, "check-repr.py"))
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    cr.FORMAT = "full"
    cr.RENORM = a.renorm
    tri = cases(a.n, a.seed)
    with tempfile.TemporaryDirectory() as d:
        exe = build(a.qemu, a.ref, d)
        res = subprocess.run([exe], input="".join("%x %x %x\n" % t for t in tri), capture_output=True, text=True,
                             check=True).stdout.split("\n")
    if len([r for r in res if r]) != len(tri):
        print("ERROR: the C driver answered %d of %d triples" % (len([r for r in res if r]), len(tri)))
        return 1
    word = dec = notcover = pred_bad = unrep = 0
    first = None
    for (s, e, c), line in zip(tri, res):
        _lo, hi, qb, qe = (int(x, 16) for x in line.split())
        _, ie, bf, tf, E = cr.rtl_compress(s, e, c)
        if ie == 0:
            w = bf | ((tf & 7) << 14) | ((tf >> 3) << 17)
        else:
            w = (E & 7) | (bf << 3) | (((E >> 3) & 7) << 14) | (tf << 17) | (1 << 26)
        pb, pe = cr.rtl_decompress(("full", ie, bf, tf, E), c)
        if w != hi & ((1 << 27) - 1):
            word += 1
            first = first or "word %#x vs %#x for [%#x,%#x) cursor %#x" % (w, hi & ((1 << 27) - 1), s, e, c)
        if (pb, pe) != (qb, qe):
            dec += 1
            first = first or "decode [%#x,%#x) vs [%#x,%#x) for [%#x,%#x) cursor %#x" % (pb, pe, qb, qe, s, e, c)
        if s <= c <= e:
            gq = 1 if ie == 0 else 1 << (E + 3)
            if -(-e // gq) * gq >= 1 << 64:              # the rounded top is 2^64 or more: no 64-bit end can hold it
                unrep += 1                               # (both encoder forms decode it short); counted, not judged
                continue
            bad = qb > s or qe < e
            notcover += bad
            g = cr.full_granule(e - s)
            span = -(-e // g) * g - s // g * g
            pred_bad += bad if a.renorm else bad != (e - s >= 4096 and span >= g << 10)
    print("qemu-encoder-diff: %s at %s%s, %d triples: %d word mismatches, %d decode mismatches%s; %d decode not "
          "covering their input (%d more with a rounded top at or above 2^64, unrepresentable, not judged), %s %d" % (
              a.qemu, a.ref, " (--renorm)" if a.renorm else "", len(tri), word, dec,
                                                                    " (first: %s)" % first if first else "",
                                                                    notcover, unrep,
                                                                    "which must be 0:" if a.renorm else
                                                                    "R-54's predicate wrong on", pred_bad))
    return 1 if word or dec or pred_bad else 0


if __name__ == "__main__":
    sys.exit(main())
