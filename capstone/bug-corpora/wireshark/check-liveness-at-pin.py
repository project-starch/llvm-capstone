#!/usr/bin/env python3
"""Content-based liveness at the wireshark 4.6.8 pin, for every candidate.

Ancestry is not usable: 4.6.x is a release branch that cherry-picks, so a fix can be
absent by ancestry and present by content. So for each candidate this takes the lines
the fix ADDS to a C file and asks whether the pinned file already contains them.

Verdicts:
  FIXED-AT-PIN   most added lines are already there -- the fix was backported
  LIVE           none of them are -- the pinned tree carries the unfixed code
  PARTIAL        some are; needs a human
  NO-FILE        the file does not exist at the pin (code added after it)
  NO-SIGNAL      the diff has no line distinctive enough to search for
"""
import re, subprocess, sys
from pathlib import Path

WS = Path("/tmp/capstone/ws-git")
PIN = "v4.6.8"


def git(*a):
    return subprocess.run(["git", *a], cwd=WS, capture_output=True, text=True,
                          errors="replace").stdout


def added_lines(sha, path):
    d = git("show", "--format=", "--unified=0", sha, "--", path)
    out = []
    for l in d.splitlines():
        if l.startswith("+") and not l.startswith("+++"):
            t = l[1:].strip()
            # Only lines distinctive enough that finding them means something: skip
            # braces, comments and short fragments that occur all over the file.
            if len(t) < 18 or t.startswith(("/*", "*", "//", "#endif", "#else")):
                continue
            if re.fullmatch(r"[}{();\s]*", t):
                continue
            out.append(t)
    return out


def verdict(sha):
    files = [f for f in git("show", "--format=", "--name-only", sha).splitlines()
             if f.endswith((".c", ".h"))
             and f.split("/")[0] in ("wiretap", "epan", "wsutil", "ui", "capture")]
    best = None
    for f in files:
        adds = added_lines(sha, f)
        if not adds:
            continue
        pin = git("show", f"{PIN}:{f}")
        if not pin:
            best = best or ("NO-FILE", f, 0, 0)
            continue
        hit = sum(1 for a in adds if a in pin)
        state = ("FIXED-AT-PIN" if hit >= max(1, len(adds) * 0.6)
                 else "LIVE" if hit == 0 else "PARTIAL")
        # A LIVE reading on any touched file is the interesting one; prefer it.
        rank = {"LIVE": 0, "PARTIAL": 1, "FIXED-AT-PIN": 2, "NO-FILE": 3}
        if best is None or rank[state] < rank[best[0]]:
            best = (state, f, hit, len(adds))
    return best or ("NO-SIGNAL", "", 0, 0)


def main():
    shas = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    tally, live = {}, []
    for n, sha in enumerate(shas, 1):
        st, f, hit, tot = verdict(sha)
        tally[st] = tally.get(st, 0) + 1
        subj = git("log", "-1", "--format=%s", sha).strip()
        if st == "LIVE":
            live.append((sha[:10], f, subj))
            print(f"  LIVE  {sha[:10]}  {f:34.34s} {subj[:62]}", flush=True)
        if n % 25 == 0:
            print(f"  [{n}/{len(shas)}] {tally}", flush=True)
    print(f"\n{tally}")
    print(f"\nLIVE candidates: {len(live)}")
    Path("/tmp/capstone/ws-live.txt").write_text(
        "\n".join(f"{s}\t{f}\t{j}" for s, f, j in live) + "\n")


if __name__ == "__main__":
    main()
