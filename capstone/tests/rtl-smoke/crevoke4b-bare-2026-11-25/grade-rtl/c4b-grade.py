#!/usr/bin/env python3
"""grade a crevoke4b kit session from its raw uart captures against expected-vectors.json (independent of the board
lane's grader). c4b-grade.py <session dir> <expected-vectors.json> new|base. Exit 1 on no data."""
import json, os, re, sys

def read(path):
    t = open(path, errors="replace").read()
    m = re.search(r"^SR ([0-9A-Fa-f]{16})", t, re.M)
    if not m:
        return None
    n = int(m.group(1), 16)
    sv = re.findall(r"^SV ([0-9A-Fa-f]{16})", t[m.end():], re.M)[:n]
    return n, [hex(int(x, 16)) for x in sv]

def s64(x):
    v = int(x, 16)
    return v - (1 << 64) if v >> 63 else v

def main():
    d, js, tree = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
    suf = {"base": "904cd9c6b"}.get(tree, tree)
    vk, ck, sk = ("vector", "classes", "sr") if tree == "new" else ("vector_" + suf, "classes_" + suf, "sr_" + suf)
    dirs = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)) and re.match(r"\d\d-", x))
    if not dirs:
        print("NO DATA: no image directories in", d); sys.exit(1)
    ok = 0
    for x in dirs:
        name = x[3:]
        e = js.get(name)
        r = read(os.path.join(d, x, "uart.txt"))
        if e is None or r is None:
            print(f"NO READING {name}"); continue
        n, v = r
        if n != e[sk]:
            print(f"VOID IMAGE {name}: SR {n} want {e[sk]}"); continue
        bad = [(i + 1, v[i], e[vk][i]) for i, c in enumerate(e[ck]) if c == "E" and v[i] != e[vk][i]]
        extra = ""
        if name == "b5b2-crevint-lat":
            lat = [s64(z) for z in v[12:20]]; cyc, tk = int(v[20], 16), int(v[21], 16)
            pos = [l for l in lat[1:] if l > 0]
            extra = f" lat={lat} cal={cyc}/{tk}" + (f" cycles={[round(l * cyc / tk) for l in pos]}" if pos and tk else " band VOID (no positive latency after the first)")
        M = {i + 1: v[i] for i, c in enumerate(e[ck]) if c in ("M", "R")}
        print(f"{'EXACT' if not bad else 'MISS '} {name}: E {sum(c == 'E' for c in e[ck])} {'' if not bad else bad} M/R {M}{extra}")
        ok += not bad
    print(f"C4B-GRADE {tree}: {ok}/{len(dirs)} exact on E")

main()
