#!/usr/bin/env python3
"""Negative-test ALLOWED_GROUP_SIZE_EDIT: the one authorised size edit passes, near misses do not.

A gate that has never refused anything is unproven, and this allowance sits on alloc_group's
decision region -- the thing the gate exists to freeze. So each mutation below is a change a future
edit might plausibly make, and every one of them must still FAIL.

  python3 test-group-size-narrowing.py <patched musl tree> <musl-1.2.5.tar.gz>
"""
import argparse, shutil, subprocess, sys, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE / 'check-mallocng-policy.py'

# (label, old, new, must_pass)
MUTATIONS = [
    ("unmutated, as authorised", None, None, True),
    ("reverted to upstream sizing (policy-identical, so allowed -- a PASS here does NOT mean the fix is present)",
     None, "REVERT", True),
    ("reserves for `req` instead of the original request `want`",
     "cap_single_slot_bytes(want, pagesize)", "cap_single_slot_bytes(req, pagesize)", False),
    ("a second edit in the region: the 25% drop count",
     "size*cnt>4*pagesize) cnt = 3;", "size*cnt>4*pagesize) cnt = 4;", False),
    ("the mmap-worthiness condition relaxed",
     "req>=4*pagesize && 2*cnt>usage", "req>=2*pagesize && 2*cnt>usage", False),
    ("the non-Capstone build loses upstream's sizing",
     "#else\n\t\t\t\tneeded = req;\n#endif", "#endif", False),
    ("the failure path dropped, so a non-convergence maps 0 bytes",
     "if (!needed) { free_meta(m); return 0; }", "/* no check */", False),
    ("the branch condition itself made Capstone-specific",
     "if (req<size+UNIT || (req>=4*pagesize && 2*cnt>usage)) {",
     "if (req<size+UNIT || (req>=4*pagesize && 2*cnt>usage) || 1) {", False),
]

def run_gate(tree, archive):
    r = subprocess.run([sys.executable, str(GATE), str(tree), str(archive)],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stdout

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tree', type=Path)
    ap.add_argument('archive', type=Path)
    a = ap.parse_args()
    work = Path('/tmp/claude-1000/group-size-narrowing')
    src = a.tree / 'src/malloc/mallocng/malloc.c'
    base = src.read_text()
    with tarfile.open(a.archive) as t:
        up = t.extractfile('musl-1.2.5/src/malloc/mallocng/malloc.c').read().decode()
    START = "\t\tif (!nosmall && cnt<=7) {"
    END_OLD, END_NEW = "\t\t\t\tneeded = req;\n", "#endif\n"
    up_blk = up[up.index(START):up.index(END_OLD, up.index(START))+len(END_OLD)]
    cur_blk = base[base.index(START):base.index(END_NEW, base.index(START))+len(END_NEW)]

    failures = 0
    for label, old, new, must_pass in MUTATIONS:
        if work.exists(): shutil.rmtree(work)
        shutil.copytree(a.tree, work, symlinks=True,
                        ignore=shutil.ignore_patterns('*.o', '*.a', 'obj'))
        text = base
        if new == "REVERT":
            text = base.replace(cur_blk, up_blk, 1)
            assert text != base, "revert mutation did not apply"
        elif old is not None:
            assert base.count(old) == 1, f"{label}: anchor appears {base.count(old)} times"
            text = base.replace(old, new, 1)
        (work / 'src/malloc/mallocng/malloc.c').write_text(text)
        passed, _ = run_gate(work, a.archive)
        ok = (passed == must_pass)
        failures += not ok
        print("  %-5s %-7s %s" % ("OK" if ok else "BROKEN",
                                  "PASS" if passed else "REFUSED", label))
    if work.exists(): shutil.rmtree(work)
    print("\n  %d of %d behaved as required" % (len(MUTATIONS)-failures, len(MUTATIONS)))
    return 1 if failures else 0

if __name__ == '__main__':
    raise SystemExit(main())
