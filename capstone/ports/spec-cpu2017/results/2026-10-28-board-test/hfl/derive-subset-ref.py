#!/usr/bin/env python3
"""derive-subset-ref.py FULL_RESULT_DIR SUBSET_STAGE OUT_DIR: a twin reference for a make-stage --commands stage, cut
from a twin run of the FULL stage (run-stage.py's result.json), without re-running anything.

run-stage.py's twin comparison (vs_twin) compares every output hash of the reference arm with the run's; a reference
of the full stage therefore lists the dropped commands' outputs, which the subset run never makes, and reads DIFFER.
This keeps, per arm, only the commands the subset stage selects (plan.json `selection`), and only the hashes and in-band
files of outputs the subset stage names (plan.json `outputs`) or that are not outputs of the full stage's dropped
commands. The selected commands ran first in the full run, in the same fresh run directory, so their outputs are the
same files. Writes OUT_DIR/result.json with `derived_from` (the full result's sha256) and `selection`. Exit 2 on no data.
"""
import hashlib
import json
import sys
from pathlib import Path


def main():
    full, stage, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    fr = full / "result.json"
    plan = json.loads((stage / "plan.json").read_text())
    sel = plan.get("selection")
    if not fr.is_file() or not sel:
        print(f"NO DATA: {fr} missing or {stage} has no selection")
        return 2
    res = json.loads(fr.read_text())
    keep_out = set(plan["outputs"])
    dropped = set()
    for arm, d in res["arms"].items():
        for name in list(d.get("hashes", {})):
            if name.replace(arm, "ARM") not in keep_out:
                dropped.add(name)
    new = dict(res, arms={}, derived_from=hashlib.sha256(fr.read_bytes()).hexdigest(), selection=sel,
               dropped_outputs=sorted(dropped))
    new.pop("vs_twin", None)
    new.pop("cross_arm", None)
    for arm, d in res["arms"].items():
        cs = [c for c in d["commands"] if c["tag"] != "run" or c["i"] in sel["commands"]]
        keep = lambda k: k.replace(arm, "ARM") in keep_out
        new["arms"][arm] = dict(d, commands=cs, hashes={k: v for k, v in d.get("hashes", {}).items() if keep(k)},
                                files={k: v for k, v in d.get("files", {}).items() if keep(k)},
                                validation=[v for v in d.get("validation", []) if any(o in v["cmd"] for o in keep_out)])
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps(new, indent=1) + "\n")
    for arm, d in new["arms"].items():
        print(f"{out.name} {arm}: commands {[c['i'] for c in d['commands']]}, hashes {sorted(d['hashes'])}, "
              f"validation {[v['verdict'] for v in d['validation']]}; dropped {sorted(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
