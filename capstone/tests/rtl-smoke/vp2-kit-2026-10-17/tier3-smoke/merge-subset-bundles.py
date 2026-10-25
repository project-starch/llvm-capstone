#!/usr/bin/env python3
"""merge-subset-bundles.py --out DIR --subset NAME BUNDLE RECORD IMAGES [--subset ...]: one verdicts-v1 bundle from the
subset boots of one corpus arm (the fold maps an arm to ONE bundle; a subset boot runs a few cases with --only).

Rules (the runtime lane's ruling, 2026-10-18):
- the 11 platform fields that a subset boot does not change must be identical across the subsets: compiler, judge,
  launcher, machine, module, monitor, monitor_defines, profile, qemu, rootfs, runner; and so must every top-level field
  but platform, tally and cases. Any difference refuses the merge (two arms, not one), exit 2.
- platform.firmware and platform.kernel differ by construction (each boot bakes its own stage into the initramfs), so
  both become the string "per-subset", and a top-level `subsets` map records, per subset, its firmware, kernel, record
  path and the sha256 of its images list.
- verdicts.jsonl is the subsets' lines, byte for byte, in case order; a case in two subsets refuses the merge.
- cases and tally are recomputed from the merged verdicts (tally key: the verdict, plus ":reason" for NO-READING), and
  each subset's own tally must equal the same recomputation over its own verdicts (the formula's control).
Refuses an existing --out. Exit 0 merged, 2 refused."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

SAME = ["compiler", "judge", "launcher", "machine", "module", "monitor", "monitor_defines", "profile", "qemu", "rootfs",
        "runner"]
NOTE = ("platform.firmware and platform.kernel are per subset: each subset boot bakes its own stage into the initramfs. "
        "For the pym0 pair a claim-auditor showed the kernel to be a relink of the same text with a different initramfs, "
        "and the firmware equal outside its embedded Image (2026-10-18).")


def tally(rows):
    return dict(collections.Counter(r["verdict"] + (":" + r["reason"] if r["verdict"] == "NO-READING" and r.get("reason")
                                                    else "") for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--subset", nargs=4, action="append", required=True, metavar=("NAME", "BUNDLE", "RECORD", "IMAGES"))
    a = p.parse_args()
    if a.out.exists():
        print(f"REFUSED: {a.out} exists")
        return 2
    subs, rows, seen = {}, [], {}
    first = None
    for name, bundle, record, images in a.subset:
        d = json.loads((Path(bundle) / "inputs.json").read_text())
        lines = [l for l in (Path(bundle) / "verdicts.jsonl").read_text().splitlines() if l.strip()]
        v = [json.loads(l) for l in lines]
        if tally(v) != d.get("tally") or len(v) != d.get("cases"):
            print(f"REFUSED: {name}'s own tally/cases do not recompute ({d.get('tally')} vs {tally(v)})")
            return 2
        if first is None:
            first = d
        for k in SAME:
            if d["platform"].get(k) != first["platform"].get(k):
                print(f"REFUSED: platform.{k} differs: {name} {d['platform'].get(k)} vs {first['platform'].get(k)}")
                return 2
        for k in set(d) | set(first):
            if k not in ("platform", "tally", "cases") and d.get(k) != first.get(k):
                print(f"REFUSED: {k} differs: {name} {str(d.get(k))[:80]} vs {str(first.get(k))[:80]}")
                return 2
        for r in v:
            c = r["observation"]["case"]
            if c in seen:
                print(f"REFUSED: case {c} in both {seen[c]} and {name}")
                return 2
            seen[c] = name
        rows += [(r["observation"]["case"], l, r) for r, l in zip(v, lines)]
        subs[name] = {"firmware": d["platform"].get("firmware"), "kernel": d["platform"].get("kernel"),
                      "record": record, "images_sha256": hashlib.sha256(Path(images).read_bytes()).hexdigest()}
    rows.sort(key=lambda t: t[0])
    out = dict(first)
    out["platform"] = dict(first["platform"], firmware="per-subset", kernel="per-subset")
    out["cases"], out["tally"] = len(rows), tally([t[2] for t in rows])
    out["subsets"], out["subsets_note"] = subs, NOTE
    a.out.mkdir(parents=True)
    (a.out / "inputs.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    (a.out / "verdicts.jsonl").write_text("".join(t[1] + "\n" for t in rows))  # each subset's line, byte for byte
    print(f"MERGED {len(subs)} subsets, {len(rows)} cells, tally {out['tally']} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
