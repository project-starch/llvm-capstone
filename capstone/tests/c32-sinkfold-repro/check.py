#!/usr/bin/env python3
"""C-32 / design A: does the bridged integer still get copied with `movc`?

Runs llc over shapes.ll and compares the movc count per function against the
EXPECT-MOVC line written beside each shape.  Sub-second, and it needs no SQLite
domain image -- which is the point: the only previous test of a C-32 fix was a
~90 minute image build on a host whose glibc can compile the domain TU.

EXIT STATUS
  0  every shape matches, and the positive control fired
  1  a shape disagrees with its expectation -- a real change in codegen
  2  CANNOT CHECK.  llc missing, a shape absent from the output, or the
     positive control failing to fire.  This is an error on purpose: a run that
     checked nothing must not read like a pass.

THE POSITIVE CONTROL is not optional.  Every "0" here is a claim that a copy is
absent, and an absent copy is exactly what a broken instrument also reports.  So
the same shapes are rebuilt with -capstone-enable-sink-fold=false, which is the
single switch design A's protection actually rides on; shape 1 must go from 0 to
non-zero.  If it does not, this script cannot tell "fixed" from "not looking".

WHAT IS AND IS NOT COUNTED.  `movc rd, zero` materialises cnull and copies no
bridged value, so it is excluded; every other movc inside the function body is
counted.  Run from the repo root, or pass --llc.
"""
import argparse, os, re, subprocess, sys

def die(msg):
    """Cannot-check exits 2, never 1 and never 0.  sys.exit(str) would exit 1,
    which is the "a shape differs" status -- a run that checked nothing would
    then be indistinguishable from a run that found a real change."""
    sys.stderr.write("cannot check: %s\n" % msg)
    sys.exit(2)


HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = os.path.join(HERE, "shapes.ll")
DEFAULT_LLC = "llvm/cmake-build-debug/bin/llc"
# Asserted, not derived. Deleting a `define` from shapes.ll otherwise exits 0 having checked
# one shape fewer, with a summary line that still reads like a full run (found by audit,
# 2026-09-17 -- it was the fifth exit path, and the one this script did not have).
EXPECTED_SHAPES = 5
TRIPLE = ["-mtriple=capstone64", "-mattr=+m", "-O2", "-verify-machineinstrs"]


def expectations(path):
    """EXPECT-MOVC lives next to the shape it describes, so the two cannot drift."""
    exp, pending = {}, None
    for line in open(path):
        m = re.search(r"EXPECT-MOVC:\s*(\d+)\s+(defect|correct)", line)
        if m:
            if pending is not None:
                die("two EXPECT-MOVC lines with no `define` between them in %s -- the first "
                    "would be silently discarded, so an expectation added ahead of its shape "
                    "would report clean having labelled nothing" % path)
            pending = (int(m.group(1)), m.group(2))
            continue
        m = re.match(r"define\s+.*?@([A-Za-z0-9_]+)\s*\(", line)
        if m:
            if pending is None:
                die("%s has no EXPECT-MOVC above @%s" % (path, m.group(1)))
            exp[m.group(1)] = pending
            pending = None
    if not exp:
        die("no shapes found in %s" % path)
    return exp


def movc_counts(llc, extra):
    cmd = [llc] + TRIPLE + extra + [SHAPES, "-o", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        die("llc failed (%d): %s" % (r.returncode, " ".join(cmd)))
    counts, cur = {}, None
    for line in r.stdout.splitlines():
        m = re.match(r"^([A-Za-z0-9_]+):\s+#\s*@", line)
        if m:
            cur = m.group(1)
            counts[cur] = 0
            continue
        if cur and re.search(r"\bmovc\b", line) and not re.search(r"\bmovc\s+\w+,\s*zero\b", line):
            counts[cur] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llc", default=DEFAULT_LLC)
    args = ap.parse_args()
    if not os.path.isfile(args.llc):
        die("no llc at %s (run from the repo root, or pass --llc)" % args.llc)

    exp = expectations(SHAPES)
    if len(exp) != EXPECTED_SHAPES:
        die("shapes.ll defines %d shapes, expected %d -- a shape was added or removed. "
            "Update EXPECTED_SHAPES deliberately; this check exists so that removing one "
            "cannot pass quietly." % (len(exp), EXPECTED_SHAPES))
    got = movc_counts(args.llc, [])

    # The control first: a clean report from a blind instrument looks identical.
    ctl = movc_counts(args.llc, ["-capstone-enable-sink-fold=false"])
    probe = "fold_ok_all_uses_conform"
    if probe not in got or probe not in ctl:
        die("%s absent from llc output" % probe)
    if not (got[probe] == 0 and ctl[probe] > 0):
        die("positive control did not fire -- %s is %d with sink-and-fold on "
            "and %d with it off; expected 0 then non-zero. This script cannot "
            "currently tell a fix from a blind spot."
            % (probe, got[probe], ctl[probe]))
    print("positive control: %s %d -> %d with -capstone-enable-sink-fold=false  OK"
          % (probe, got[probe], ctl[probe]))

    # Second control, and it is the one that pins the STORY rather than the instrument.
    # The recorded mechanism says the fold is declined outright on the other shapes, so
    # turning it off must change nothing there. If one of them moves, the fold IS
    # contributing to a shape this script reports as fold-declined, and the numbers below
    # can no longer be read the way the summary line reads them.
    moved = [fn for fn in exp
             if fn != probe and fn in got and fn in ctl and got[fn] != ctl[fn]]
    if moved:
        die("sink-and-fold invariance broken for %s: these shapes are recorded as ones the "
            "fold declines outright, so disabling it must not change them. The mechanism in "
            "docs/history/17-09-2026_14-23-55_c32-design-a-sinkfold-mechanism.md needs "
            "revisiting before these counts mean what the summary says."
            % ", ".join(sorted(moved)))
    print("invariance control: %d fold-declined shapes unchanged with the fold off  OK"
          % (len(exp) - 1))

    bad = defects = 0
    for fn, (want, kind) in exp.items():
        if fn not in got:
            die("shape @%s absent from llc output" % fn)
        if got[fn] != want:
            bad += 1
        if kind == "defect":
            defects += got[fn]
        print("%s %-34s movc=%d expected=%d (%s)"
              % ("ok  " if got[fn] == want else "DIFF", fn, got[fn], want, kind))
    if bad:
        print("\n%d shape(s) differ from the recorded behaviour. A design change "
              "that removes a defect movc lands here -- update the EXPECT line "
              "with the reason." % bad)
        return 1
    hit = sorted(fn for fn, (w, k) in exp.items() if k == "defect" and got[fn] > 0)
    print("\nAll %d shapes as recorded: %d movc still copy a bridged integer, so C-32 "
          "continues to reach silicon on %s.\nThe %d in real_cap_copy_control are correct "
          "and must stay."
          % (len(exp), defects, ", ".join(hit) or "no shape",
             got.get("real_cap_copy_control", 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
