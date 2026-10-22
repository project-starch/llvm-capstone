#!/usr/bin/env python3
"""tier3-hold-board.py <board kit> <out dir>: the compiler lane's live-peak `hold 100000` (capstone/tests/live-peak-sizing,
dev 29c47723586f; image <mnt-hdd>/capstone/live-peak/hold.dom, sha256/16 433f9f1cdcd0db9f) in ONE board boot through
boardvm (the dev worktree /tmp/capstone/wt-newarms, bcf5f1724110), the steps of its QEMU runner r60-fmtfp/replay-stats2.py:
  1. stats0  ./capstone-vexec --stats                                  (the module's counters before)
  2. hold    cd /tmp && capstone-vexec images/hold.dom hold 100000    (HOLD-* lines and the exit)
  3. stats1  ./capstone-vexec --stats                                  (nodes_high_water, nodes_allocated_total, node_capacity)
No module parameters: the kit's module runs at its DEFAULT capacity. Prints one READ line per quantity and the raw step
texts; exit 0 when all three steps ran and step 3's JSON was read, 2 when anything is missing (no data is an error, not a
zero). Written 2026-10-23 for the bitstream-2 discriminator (18-bit index) and its batch-bitstream control."""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/capstone/wt-newarms/capstone/bug-corpora/tools")
import boardvm  # noqa: E402

IMG = Path("/mnt/hdd") / Path.home().name / "capstone/live-peak/hold.dom"
IMG_SHA = "433f9f1cdcd0db9f"


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    kit, out = Path(sys.argv[1]), Path(sys.argv[2])
    sha = hashlib.sha256(IMG.read_bytes()).hexdigest()[:16]
    if sha != IMG_SHA:
        print(f"HOLD STOP: {IMG} is {sha}, not {IMG_SHA}"); sys.exit(2)
    if boardvm.virtualvm.module_params(kit):
        print(f"HOLD STOP: the kit sets module parameters {boardvm.virtualvm.module_params(kit)}; the run needs the default capacity")
        sys.exit(2)
    out.mkdir(parents=True, exist_ok=False)
    batch = boardvm.BoardBatch(out / "stage")
    img = batch.put("images/hold.dom", IMG)
    root = str(Path(img).parent.parent)
    batch.step("stats0", "./capstone-vexec --stats")
    batch.step("hold", f"cd /tmp && {root}/capstone-vexec {img} hold 100000")
    batch.step("stats1", "./capstone-vexec --stats")
    batch.bounds = {"stats0": 120, "hold": 1800, "stats1": 120}
    runs, transcript, completed = batch.execute(kit, out / "guest", timeout=3600)
    print(f"HOLD boot {'completed' if completed else 'DID NOT COMPLETE'}; transcript {transcript}; steps {sorted(runs)}")
    for name in ("stats0", "hold", "stats1"):
        text, result = runs.get(name, ("", {"kind": "none"}))
        print(f"## step {name} result {json.dumps(result, sort_keys=True)}")
        print(text.rstrip())
    rc = 0
    h_text, h_res = runs.get("hold", ("", {"kind": "none"}))
    hold_lines = re.findall(r"^HOLD-[A-Z]+[^\n]*", h_text, re.M)
    print("READ hold lines:", " | ".join(hold_lines) if hold_lines else "NONE")
    print("READ hold result:", json.dumps(h_res, sort_keys=True))
    fault = re.findall(r"capstone-exec: domain fault[^\n]*", h_text)
    print("READ hold fault:", fault[-1][:200] if fault else "none")
    for name in ("stats0", "stats1"):
        text = runs.get(name, ("", {}))[0]
        js = [l for l in text.splitlines() if l.startswith("{") and '"node_capacity"' in l]
        if not js:
            print(f"READ {name}: NO DATA"); rc = 2; continue
        d = json.loads(js[-1])
        print(f"READ {name}: nodes_high_water {d.get('nodes_high_water')} nodes_allocated_total {d.get('nodes_allocated_total')} "
              f"node_capacity {d.get('node_capacity')} nodes_live {d.get('nodes_live')}")
    if not hold_lines or not completed:
        rc = 2
    sys.exit(rc)


if __name__ == "__main__":
    main()
