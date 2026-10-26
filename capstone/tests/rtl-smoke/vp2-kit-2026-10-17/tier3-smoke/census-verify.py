#!/usr/bin/env python3
"""census-verify.py <build.log> [manifest-T1.tsv]: for every program in a perf build log (-DBENCH_ID="suite/name"), the
verify_benchmark bodies in the .c files ITS compile lines name (never a directory listing: perf/build/native/src holds
only some of them, the rest come from beebs-src), each found by brace matching. When a file defines it more than once,
the definition whose enclosing #ifdef/#elif macro the compile line defines is taken (trio_test.c: TRIO_SNPRINTF's is
`return -1`, TRIO_SSCANF's is real). Prints NO-OP (every body is upstream's `return -1`), REAL, or NONE per program, and
any disagreement with the manifest's own_verify column. Controls: minver must read NO-OP and st REAL, else exit 2;
no program found is exit 2. Written 2026-10-26."""
import re
import sys
from collections import defaultdict


def defs(text):
    """(enclosing #ifdef/#elif macro or None, body) for each verify_benchmark definition."""
    out = []
    for m in re.finditer(r"\bverify_benchmark\s*\([^;{]*?\)\s*\{", text, re.S):
        i, d = m.end(), 1
        while d and i < len(text):
            d += {"{": 1, "}": -1}.get(text[i], 0); i += 1
        guard = None
        for g in re.finditer(r"^#\s*(?:ifdef|elif|if)\s+(?:defined\s*\(?\s*)?(\w+)|^#\s*endif", text[:m.start()], re.M):
            guard = g.group(1)
        out.append((guard, re.sub(r"/\*.*?\*/|//[^\n]*", "", text[m.end():i - 1], flags=re.S).strip()))
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    files, flags = defaultdict(list), defaultdict(set)
    for l in open(sys.argv[1], errors="replace"):
        m = re.search(r'-DBENCH_ID="([^"]+)"', l)
        if not m:
            continue
        b = m.group(1)
        flags[b] |= set(re.findall(r"-D(\w+)", l))
        for f in re.findall(r"(?<=\s)(\S+\.c)(?=\s|$)", l):
            if f not in files[b]:
                files[b].append(f)
    if not files:
        print(f"NO DATA: no -DBENCH_ID compile line in {sys.argv[1]}"); sys.exit(2)
    man = {}
    if len(sys.argv) > 2:
        for l in open(sys.argv[2]):
            f = l.rstrip("\n").split("\t")
            man[f"{f[2]}/{f[1]}"] = f[8]
    res, n = {}, defaultdict(int)
    for b in sorted(files):
        got = []
        for f in files[b]:
            ds = defs(open(f, errors="replace").read())
            if len(ds) > 1:
                ds = [d for d in ds if d[0] in flags[b]] or ds
            got += [(f.split("/")[-1], "NO-OP" if re.fullmatch(r"return\s+-1\s*;", body) else "REAL") for _, body in ds]
        v = "NONE" if not got else "NO-OP" if all(g[1] == "NO-OP" for g in got) else "REAL"
        res[b] = v; n[v] += 1
        mv = man.get(b)
        note = "" if mv in (None, "control", "driver") or (mv == "no") == (v == "NO-OP") else f"  MANIFEST own_verify={mv}"
        print(f"VERIFY {v:5s} {b:34s} {got}{note}")
    print("COUNT " + " ".join(f"{k} {n[k]}" for k in ("NO-OP", "REAL", "NONE")) + f" of {len(res)}")
    if res.get("beebs/minver") != "NO-OP" or res.get("beebs/st") != "REAL":
        print(f"CONTROL FAILED: minver {res.get('beebs/minver')} (want NO-OP), st {res.get('beebs/st')} (want REAL)"); sys.exit(2)


if __name__ == "__main__":
    main()
