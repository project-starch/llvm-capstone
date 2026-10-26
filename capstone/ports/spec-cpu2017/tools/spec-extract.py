#!/usr/bin/env python3
"""Extract a pristine SPEC CPU2017 tree (tools plus the named benchmarks) and
verify every extracted file listed in SPEC's MANIFEST against its SHA-512.

usage: spec-extract.py SPEC17.tar.gz DEST BENCH...   (DEST must not exist)
       spec-extract.py --verify DEST BENCH...           (check an existing tree)
BENCH may be ALL: the whole tree, strict for every benchmark. runcpu needs
every benchmark a benchmark set names, so a tree to build in must be ALL.
Exits non-zero on any mismatch or on a MANIFEST entry under an extracted
benchmark that is missing; prints the counts it checked."""
import hashlib, os, subprocess, sys, tarfile

tarball, dest, benches = sys.argv[1], sys.argv[2], sys.argv[3:]
verify_only = tarball == "--verify"
if os.path.exists(dest) != verify_only:
    sys.exit(f"spec-extract: {dest} {'does not exist' if verify_only else 'exists'}")
if not benches:
    sys.exit("spec-extract: name at least one benchmark")
keep = tuple(f"SPEC17/benchspec/CPU/{b}/" for b in benches)

def wanted(name):
    """Everything outside benchspec/CPU, and the named benchmarks inside it."""
    prefix = "SPEC17/benchspec/CPU/"
    if not name.startswith(prefix):
        return True
    top = name[len(prefix):].split("/")[0]
    return benches == ["ALL"] or top == "" or top in benches or not top[:3].isdigit()

if not verify_only:
    os.makedirs(dest)
    with tarfile.open(tarball) as t:
        members = [m for m in t if wanted(m.name)]
        t.extractall(dest, members=members, filter="tar")
root = os.path.join(dest, "SPEC17")
# Strict for the named benchmarks: every MANIFEST entry under them must be
# present and match. Elsewhere an installed tree legitimately lacks the install
# media's archives and tool sources, so absence there is counted, not fatal;
# a mismatch anywhere is fatal.
bench_paths = tuple(k[len("SPEC17/"):] for k in keep)
counts = {"bench_ok": 0, "other_ok": 0, "bench_missing": 0, "other_missing": 0, "mismatch": 0}
for line in open(os.path.join(root, "MANIFEST")):
    digest, _, _, path = line.rstrip("\n").split(" ", 3)
    in_bench = path.startswith(bench_paths) or (
        benches == ["ALL"] and path.startswith("benchspec/CPU/") and path.split("/")[2][:3].isdigit())
    if path.startswith("benchspec/CPU/") and not in_bench and path.split("/")[2][:3].isdigit():
        continue
    full = os.path.join(root, path)
    if not os.path.exists(full):
        counts["bench_missing" if in_bench else "other_missing"] += 1
        if in_bench:
            print("MISSING", path)
        continue
    if hashlib.sha512(open(full, "rb").read()).hexdigest() != digest:
        counts["mismatch"] += 1
        print("MISMATCH", path)
    else:
        counts["bench_ok" if in_bench else "other_ok"] += 1
print("spec-extract:", " ".join(f"{k}={v}" for k, v in counts.items()), f"({', '.join(benches)})")
sys.exit(1 if counts["mismatch"] or counts["bench_missing"] or counts["bench_ok"] == 0 else 0)
