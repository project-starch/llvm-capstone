#!/usr/bin/env python3
"""compare-stages.py A B [--expect-images-differ]: compare two make-stage.py stage directories file by file.

Reports, per relative path under stage/: only-in-A, only-in-B, and differing files, and compares plan.json field by field.
Without --expect-images-differ (the regeneration control) every file but run.sh must be identical and every plan field
equal; run.sh may differ only if the make-stage.py version changed (reported, with its diff size). With it (a new tree),
bin/ and the plan's `executables` must differ and everything else must be equal. Exit 0 = as expected, 1 = not, 2 = no data.
"""
import hashlib
import json
import re
import sys
from pathlib import Path


def files(root):
    st = root / "stage"
    if not st.is_dir():
        return None
    return {canon(str(p.relative_to(st))): hashlib.sha256(p.read_bytes()).hexdigest() for p in st.rglob("*") if p.is_file()}


def canon(x):
    """A staged reference is named ref/<sha256(absolute source path)[:8]>-<name>, so its name carries the tree's path;
    its content is compared, its name prefix is not."""
    return re.sub(r"ref/[0-9a-f]{8}-", "ref/", x)


def canon_plan(p):
    return json.loads(canon(json.dumps(p)))


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    newtree = "--expect-images-differ" in sys.argv
    a, b = Path(args[0]), Path(args[1])
    fa, fb = files(a), files(b)
    if not fa or not fb:
        print(f"NO DATA: {a if not fa else b} has no stage/ files")
        return 2
    ok = True
    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
    diff = sorted(k for k in set(fa) & set(fb) if fa[k] != fb[k])
    if only_a or only_b:
        ok = False
        print(f"  only in A: {only_a[:8]}  only in B: {only_b[:8]}")
    for k in diff:
        expected = (k == "run.sh") or (newtree and k.startswith("bin/"))
        print(f"  differs: {k}{'' if expected else '  <-- UNEXPECTED'}")
        ok &= expected
    if newtree and not any(k.startswith("bin/") for k in diff):
        print("  bin/ identical although the tree is new  <-- UNEXPECTED")
        ok = False
    pa, pb = canon_plan(json.loads((a / "plan.json").read_text())), canon_plan(json.loads((b / "plan.json").read_text()))
    for k in sorted(set(pa) | set(pb)):
        va, vb = pa.get(k), pb.get(k)
        if k == "refs" and isinstance(va, dict) and isinstance(vb, dict):  # values are paths relative to the tree root
            va, vb = sorted(va), sorted(vb)
        if va != vb:
            expected = newtree and k == "executables"
            print(f"  plan.{k} differs{'' if expected else '  <-- UNEXPECTED'}")
            ok &= expected
    print(f"{'SAME-OPTIONS' if ok else 'MISMATCH'} {a.name} vs {b} ({len(fa)} files, {len(diff)} differ)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
