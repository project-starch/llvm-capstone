#!/usr/bin/env python3
"""Gather the controls' boots into one results.json, and judge each prediction.

    collect.py --meta META.json --out results.json REP.json...

REP.json: run.py's per-boot records (any arms, any repetitions). META.json: what the boots do not
record themselves (revision, snapshot, builds, configuration); it is copied into "inputs". Paths in
the records are cut to the repository-relative part. A prediction holds when every repetition of
its arm shows it; a mode without a complete reading in some repetition makes the verdict "no
reading", never a pass.
"""
import argparse
import json
from pathlib import Path
import re

PREDICTIONS = {
    # (arm, mode): (expected outcome, what else must hold, in words)
    ("protected", "parent"): ("completed", "A freed (MRB_TT_FREE, issued[i] NULL, read through the page); "
                              "A's slot read and written through the page pointer and through page->life"),
    ("protected", "sibling"): ("completed", "B on A's page, intact after A was freed, written and read back"),
    ("protected", "successor"): ("completed", "C's address equals A's; C read and written"),
    ("protected", "stale"): ("fault", "cause 25 at the stale-read-A probe"),
    ("release", "parent"): ("completed", "A freed; A's slot read and written through the page pointer"),
    ("release", "sibling"): ("completed", "B on A's page, intact after A was freed, written and read back"),
    ("release", "successor"): ("completed", "C's address equals A's; C read and written"),
    ("release", "stale"): ("completed", "the read through A returns C's data"),
}


def cut(text):
    return re.sub(r"\S*?(capstone/bug-corpora/)", r"\1", text) if isinstance(text, str) else text


def address(lines, name):
    for l in lines:
        m = re.match(rf"ADDR {name}=(0x[0-9a-f]+) base=(0x[0-9a-f]+) page=(0x[0-9a-f]+) objects=(0x[0-9a-f]+) "
                     rf"index=(\d+)", l)
        if m:
            return dict(zip(("address", "base", "page", "objects", "index"),
                            (*m.groups()[:4], int(m.group(5)))))
    return None


def mode_row(arm, mode, d):
    lines = d.get("lines", [])
    row = {"outcome": d.get("outcome"), "exit": d.get("exit"),
           "transcript": d.get("transcript"),
           "probes": [{"name": p["name"], "at": cut(p["at"])} for p in d.get("probes", [])],
           "A": address(lines, "A"), "B": address(lines, "B"), "C": address(lines, "C"),
           "lines": [cut(l) for l in lines if not l.startswith("CTL ") or " start " in l or "completed" in l
                     or "PRECONDITION" in l or "CHECK-FAILED" in l]}
    life = next((l for l in lines if l.startswith("ADDR life=")), None)
    if life:
        row["life"] = life
    if d.get("outcome") == "fault":
        at = d.get("at", {})
        row["fault"] = {"cause": d.get("cause"), "pc": d.get("pc"), "function": at.get("function"),
                        "offset": at.get("offset"), "image_va": at.get("image_va"),
                        "file_line": cut(at.get("file_line", "")),
                        "launcher_line": cut(d.get("fault_line", "")),
                        "last_probe": {"name": d["last_probe"]["name"], "at": cut(d["last_probe"]["at"])}
                        if d.get("last_probe") else None}
    return row


def holds(arm, mode, row):
    """(True/False/None, why). None = no reading."""
    out = row["outcome"]
    if out not in ("completed", "fault"):
        return None, f"outcome {out}"
    if out == "completed" and not (row["transcript"] or {}).get("complete"):
        return None, "transcript incomplete"
    text = "\n".join(row["lines"])
    want = PREDICTIONS[(arm, mode)][0]
    if out != want:
        return False, f"outcome {out}, predicted {want}"
    if mode in ("parent", "sibling", "successor", "stale") and "FREED A after" not in text:
        return False, "A's release not shown"
    if mode == "parent":
        ok = "read back 0x5a5a1234c0ffee11" in text and (arm == "release" or "READ life: tt=4" in text)
        return ok, "slot read (MRB_TT_FREE) and written through the page" + (" and page->life" if arm == "protected" else "")
    if mode == "sibling":
        ok = (row["A"] and row["B"] and row["A"]["page"] == row["B"]["page"] and "intact=yes" in text
              and 'match=yes' in text)
        return bool(ok), "B on A's page, intact, written and read back"
    if mode == "successor":
        ok = "equal=yes" in text and text.count("match=yes") == 2 and row["A"]["address"] == row["C"]["address"]
        return ok, "C at A's address, read and written"
    if mode == "stale":
        if out == "fault" and "equal=yes" not in text:
            return None, "the REUSE line (C at A's address) did not arrive"
        if arm == "protected":
            f = row["fault"]
            ok = ("equal=yes" in text and f["cause"] == 25 and f["function"] == "ctl_probe_stale_read_A"
                  and (f["last_probe"] or {}).get("name") == "stale-read-A")
            return ok, f"cause {f['cause']} in {f['function']} at {f['file_line']}"
        ok = "equal=yes" in text and "equals-C's-data=yes" in text
        return ok, "read through A returned C's data"
    return None, "unknown mode"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("reps", type=Path, nargs="+")
    a = p.parse_args()
    boots, runs = [], {}
    for path in a.reps:
        r = json.loads(path.read_text())
        boots.append({"arm": r["arm"], "rep": r["rep"], "boot_completed": r["boot_completed"],
                      "boot_wall_seconds": r["boot_wall_seconds"], "staged_sha256": r["staged_sha256"],
                      "platform_sha256": r["platform_sha256"], "gate_sha256": r["gate_sha256"],
                      "runner_sha256": r["runner_sha256"], "run_staged_sha256": r["run_staged_sha256"]})
        for mode, d in r["modes"].items():
            runs.setdefault(f"{r['arm']}/{mode}", []).append(dict(mode_row(r["arm"], mode, d), rep=r["rep"]))
    verdicts = {}
    for (arm, mode), (want, words) in PREDICTIONS.items():
        rows = runs.get(f"{arm}/{mode}", [])
        judged = [holds(arm, mode, row) for row in rows]
        if not rows or any(j[0] is None for j in judged):
            verdict = "no reading"
        else:
            verdict = "holds" if all(j[0] for j in judged) else "does not hold"
        verdicts[f"{arm}/{mode}"] = {"prediction": f"{want}: {words}", "verdict": verdict,
                                     "repetitions": len(rows), "per_rep": [j[1] for j in judged]}
    record = {"inputs": json.loads(a.meta.read_text()), "boots": sorted(boots, key=lambda b: (b["arm"], b["rep"])),
              "runs": {k: sorted(v, key=lambda r: r["rep"]) for k, v in sorted(runs.items())},
              "predictions": verdicts}
    a.out.write_text(json.dumps(record, indent=1) + "\n")
    for k, v in verdicts.items():
        print(f"{k:22} {v['verdict']:14} n={v['repetitions']}  {v['per_rep'][0] if v['per_rep'] else ''}")
    return 0 if all(v["verdict"] == "holds" for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
