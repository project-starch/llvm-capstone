#!/usr/bin/env python3
"""judge-blb.py <reference board txt> <candidate board txt>: boot BL's bands, applied mechanically (2026-10-18).

PASS needs every one of:
- the ten rung retvals identical;
- the three mruby hashes identical;
- each of the five benchmark TIME lines within 0.5 % of the reference;
- no EXCT, EXCS, RGNF or ILLX, and no INJS, NODM or INJL. These are the S-trap injection's tags: P's monitor had no
  injection, so a trap that would have wedged P is a difference, not a pass.
Prints one line per check and a verdict. Exit 0 PASS, 1 DIFFER, 2 no data. The run-scoped transcript (board-*.txt),
never the raw console capture, which replays earlier output.
"""
import re, sys

TAGS = ("EXCT", "EXCS", "RGNF", "ILLX", "INJS", "NODM", "INJL")


def read(p):
    t = open(p, errors="replace").read()
    r = dict(re.findall(r"^RESULT (\S+) retval=(-?\d+)", t, re.M))
    m = dict((k, sha) for k, sha in re.findall(r"MRUBY (\S+) sha ([0-9a-f]+) bytes \d+ want [0-9a-f]+ rc \d+ secs [0-9.]+", t))
    tm = dict((k, int(v)) for k, v in re.findall(r"TIME (\S+) (\d+) ms", t))
    tags = {k: len(re.findall(rf"\b{k}:[0-9A-Fa-f]{{8}}", t)) for k in TAGS}
    return r, m, tm, tags


def judge(ref, cand):
    R, C = read(ref), read(cand)
    if not R[0] or not C[0]:
        return 2, ["NO DATA: a transcript has no RESULT line"]
    out, bad = [], 0
    ok = R[0] == C[0] and len(R[0]) == 10
    bad += not ok; out.append(f"retvals {'IDENTICAL' if ok else 'DIFFER'} ({len(C[0])} rungs): {C[0]}")
    for k in R[1]:
        ok = C[1].get(k) == R[1][k]
        bad += not ok; out.append(f"mruby {k} {'same' if ok else 'DIFFERENT'} {C[1].get(k)} (ref {R[1][k]})")
    if len(R[1]) != 3:
        bad += 1; out.append(f"reference has {len(R[1])} mruby hashes, need 3")
    for k, v in R[2].items():
        c = C[2].get(k)
        d = None if c is None else 100.0 * (c - v) / v
        ok = d is not None and abs(d) <= 0.5
        bad += not ok; out.append(f"TIME {k} {c} ms vs {v}: {'n/a' if d is None else f'{d:+.2f} %'} {'within' if ok else 'OUTSIDE'} 0.5 %")
    if len(R[2]) != 5:
        bad += 1; out.append(f"reference has {len(R[2])} TIME lines, need 5")
    hits = {k: n for k, n in C[3].items() if n}
    bad += bool(hits); out.append(f"monitor tags: {hits or 'none'}")
    return (1 if bad else 0), out


if __name__ == "__main__":
    rc, lines = judge(sys.argv[1], sys.argv[2])
    print("\n".join(lines))
    print({0: "VERDICT PASS", 1: "VERDICT DIFFER", 2: "VERDICT NO DATA"}[rc])
    sys.exit(rc)
