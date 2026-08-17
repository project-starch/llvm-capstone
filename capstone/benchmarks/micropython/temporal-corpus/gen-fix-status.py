#!/usr/bin/env python3
"""Compute, per corpus row, whether the defect is present in our pinned MicroPython.

Ancestry is decided by `git merge-base --is-ancestor <fix> <pin>` against a full
clone, not by comparing dates and not by trusting NVD's prose. The result is
written to fix-status.json so the corpus can be rebuilt without the clone.
"""
import json, subprocess, sys, os

REPO = "/tmp/capstone/micropython"
PIN = "2e3304a128b3166d48d6877f4e5c9fbd2e48122f"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fix-status.json")

# ref -> fix commit, where one is known. Sources: NVD patch id, or the commit /
# PR GitHub reports as having closed the issue, or a commit whose message names
# the issue. Everything else is deliberately absent, meaning "not established".
KNOWN_FIX = {
    "CVE-2023-7152": "8b24aa36ba978eafc6114b6798b47b7bfecdca26",
    "CVE-2024-8947": "4bed614e707c0644c06e117f848fa12605c711cd",
    "CVE-2026-1998": "570744d06c5ba9dba59b4c3f432ca4f0abd396b6",
    "#12887": "74fd7b3d32e1",
    "#13283": "4bed614e707c0644c06e117f848fa12605c711cd",
    "#12543": "6db91dfefb1a",
    "#12670": "365913953a4e",
    "#17848": "64f0394d80ca",
    "#19060": "64f0394d80ca",   # same defect as #17848, closed as duplicate
    "#11781": "d2a3cd7ac428",
    "#5226": "f34e16dbc664",
}

# Issues upstream still has open. Open means the defect is expected present at
# the pin, but "expected" is doing real work: upstream can fix without closing.
OPEN_AT_SOURCE = {"#18168", "#18171", "#17941", "#18619", "#19075", "#5487",
                  "#3627", "#17442", "#10402", "#5272", "#11698"}


def git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args),
                          capture_output=True, text=True)


def main():
    if not os.path.isdir(os.path.join(REPO, ".git")):
        print(f"clone missing at {REPO}", file=sys.stderr)
        return 1
    if os.path.exists(os.path.join(REPO, ".git", "shallow")):
        print("clone is shallow, ancestry would be wrong; run git fetch --unshallow",
              file=sys.stderr)
        return 1

    # positive control: the pin must be its own ancestor
    if git("merge-base", "--is-ancestor", PIN, PIN).returncode != 0:
        print("sanity check failed: pin is not its own ancestor", file=sys.stderr)
        return 1
    # negative control: current upstream head must NOT be an ancestor of the pin
    head = git("rev-parse", "origin/master").stdout.strip()
    if head and git("merge-base", "--is-ancestor", head, PIN).returncode == 0:
        print("sanity check failed: upstream head reads as contained in the pin",
              file=sys.stderr)
        return 1

    out = {"pin": PIN,
           "pin_date": git("log", "-1", "--format=%cs", PIN).stdout.strip(),
           "rows": {}}

    for ref, fix in sorted(KNOWN_FIX.items()):
        r = git("rev-parse", fix + "^{commit}")
        if r.returncode != 0:
            out["rows"][ref] = {"fix_commit": fix, "fix_date": "",
                                "present_at_pin": "unknown",
                                "repro_base": "unknown",
                                "why": "fix commit not found in clone"}
            continue
        full = r.stdout.strip()
        date = git("log", "-1", "--format=%cs", full).stdout.strip()
        contained = git("merge-base", "--is-ancestor", full, PIN).returncode == 0
        out["rows"][ref] = {
            "fix_commit": full[:12],
            "fix_date": date,
            "present_at_pin": "no" if contained else "yes",
            "repro_base": (full[:12] + "^") if contained else "pin",
            "why": ("fix is an ancestor of the pin, so build the fix's parent to reproduce"
                    if contained else "fix landed after the pin, defect still present"),
        }

    for ref in sorted(OPEN_AT_SOURCE):
        out["rows"][ref] = {
            "fix_commit": "", "fix_date": "",
            "present_at_pin": "yes",
            "repro_base": "pin",
            "why": "open upstream at time of survey; reproduce on the pinned tree directly",
        }

    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
    n_no = sum(1 for v in out["rows"].values() if v["present_at_pin"] == "no")
    n_yes = sum(1 for v in out["rows"].values() if v["present_at_pin"] == "yes")
    print(f"pin {PIN[:12]} ({out['pin_date']})")
    print(f"  bereits gefixt im Pin (repro braucht Parent-Build): {n_no}")
    print(f"  im Pin vorhanden (direkt reproduzierbar):           {n_yes}")
    print(f"  geschrieben: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
