#!/usr/bin/env python3
"""The manifest of shared-LLVM/clang files this fork patches, and the drift check.

The Capstone target lives in llvm/lib/Target/Capstone and friends, but the port also
touches shared code: ValueTypes (c128), SelectionDAG, the legalizer, AsmPrinter,
ValueTracking, TargetParser, clang CodeGen...  Nothing guards those patches except the
tests of the targets that happen to exercise them, so an upstream merge can drop one
silently.  This script keeps a MANIFEST of every shared file that differs from the
drift base (the parent of the RISCV copy, a pure upstream commit) with the diff's
line counts and one distinctive line the patch added -- a marker -- and checks:

  * every manifest file still contains its marker (a dropped patch fails loudly);
  * the file's diff against the base has the recorded size (a rewritten patch is
    reported as drift, which after an upstream merge is the prompt to re-baseline);
  * no shared file outside the manifest differs from the base (a new patch without an
    entry fails loudly: add it with --write).

The comparison is base-vs-WORKING-TREE, not base-vs-HEAD, so an uncommitted edit to a
shared file already reads as drift, and --write run before the commit records the new
counts in the same commit as the change.

    capstone-shared-drift.py --repo <root> [--manifest FILE] [--base REV]
    capstone-shared-drift.py --repo <root> --write        # (re)generate the manifest

Exit 0 when clean, 1 on drift, 2 on a tool problem.  Run by
llvm/test/CodeGen/Capstone/shared-patches-present.test.
"""
import argparse
import os
import re
import subprocess
import sys

BASE = "b3a1c7778245"
SCOPES = [":/llvm/lib", ":/llvm/include", ":/llvm/utils", ":/clang/lib", ":/clang/include"]
OWN = re.compile(r"(Target/Capstone/|/Capstone[A-Za-z]*\.|BuiltinsCapstone|IntrinsicsCapstone|CapstoneGen|capstone-shared-)")
MARK = re.compile(r"Capstone|capabilit|c128|CheriCapability|EM_CAPSTONE|addrspace\(200\)|AS200|purecap", re.I)


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo] + list(args), capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout


def numstat(repo, base):
    out = {}
    for l in git(repo, "diff", "--numstat", base, "--", *SCOPES).splitlines():
        a, d, path = l.split("\t", 2)
        if OWN.search(path):
            continue
        out[path] = (int(a) if a != "-" else -1, int(d) if d != "-" else -1)
    return out


def marker_for(repo, base, path):
    diff = git(repo, "diff", "-U0", base, "--", path)
    cands = []
    for l in diff.splitlines():
        if l.startswith("+") and not l.startswith("+++"):
            body = l[1:].strip()
            if 12 <= len(body) <= 120 and MARK.search(body):
                cands.append(body)
    if not cands:
        for l in diff.splitlines():
            if l.startswith("+") and not l.startswith("+++"):
                body = l[1:].strip()
                if 12 <= len(body) <= 120 and not body.startswith(("//", "/*", "*", "#")):
                    cands.append(body)
    # Prefer a line that occurs exactly once in the file, so the grep is unambiguous.
    text = open(os.path.join(repo, path), errors="replace").read()
    for c in cands:
        if text.count(c) == 1:
            return c
    return cands[0] if cands else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    manifest = a.manifest or os.path.join(repo, "llvm", "utils", "capstone-shared-patches.txt")
    cur = numstat(repo, a.base)

    if a.write:
        with open(manifest, "w") as f:
            f.write(f"# Shared LLVM/clang files patched by the Capstone port, relative to drift base {a.base}.\n")
            f.write("# path | +added -deleted | marker (one line the patch added; must still be present)\n")
            f.write("# Regenerate with: python3 llvm/utils/capstone-shared-drift.py --repo . --write\n")
            for path in sorted(cur):
                add, dele = cur[path]
                f.write(f"{path} | +{add} -{dele} | {marker_for(repo, a.base, path)}\n")
        print(f"wrote {manifest}: {len(cur)} files")
        return 0

    if not os.path.exists(manifest):
        print(f"ERROR: no manifest at {manifest}; generate it with --write", file=sys.stderr)
        return 2
    rec = {}
    for l in open(manifest):
        l = l.rstrip("\n")
        if not l or l.startswith("#"):
            continue
        path, counts, marker = [p.strip() for p in l.split("|", 2)]
        m = re.match(r"\+(\d+) -(\d+)", counts)
        rec[path] = ((int(m.group(1)), int(m.group(2))), marker)

    drift = []
    for path, (counts, marker) in rec.items():
        full = os.path.join(repo, path)
        if not os.path.exists(full):
            drift.append(f"{path}: MISSING (deleted or moved)"); continue
        if marker and marker not in open(full, errors="replace").read():
            drift.append(f"{path}: marker not found -- the patch may have been dropped: {marker!r}")
        if path not in cur:
            drift.append(f"{path}: no longer differs from the base (patch dropped?)")
        elif cur[path] != counts:
            drift.append(f"{path}: diff is +{cur[path][0]} -{cur[path][1]}, manifest says +{counts[0]} -{counts[1]}")
    for path in sorted(set(cur) - set(rec)):
        drift.append(f"{path}: differs from the base but is not in the manifest (+{cur[path][0]} -{cur[path][1]}); add it with --write")
    if drift:
        print(f"capstone-shared-drift: {len(drift)} problem(s) against {manifest}:")
        for d in drift:
            print("  " + d)
        return 1
    print(f"capstone-shared-drift: {len(rec)} shared files match the manifest (base {a.base})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
