#!/usr/bin/env python3
"""tier3-misal.py (--kit <QEMU kit> | --board-kit <board kit>) <out dir>: Q-34's store side in ONE boot. misal-store.dom
(misal-store.c, built with the Juliet SDK f5145cbb) runs each step through capstone-vexec with CAPSTONE_EXEC_DIAGNOSTICS=1:
  aligned-sw, aligned-sd       controls, before any misaligned access: each must RETURN its value
  misal-lw                     Q-34's load, the positive control (silicon cause 4)
  misal-sw, misal-sd, misal-sd4  the stores under test
  aligned-sw-post              the control again, after the faults: the context path still works
Prints one READ line per step: the exit, the RETURNED value or the domain fault's cause and pc offset in the image. Exit 0
when the boot completed and every step was read, 2 otherwise (no data is an error, not a zero). Written 2026-10-26."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/capstone/wt-newarms/capstone/bug-corpora/tools")
import boardvm  # noqa: E402

DOM, DOM_SHA = Path("/mnt/hdd") / Path.home().name / "capstone/misal-store/misal-store.dom", "e3107737d6cb6031"
STEPS = (("aligned-sw", "aligned-sw"), ("aligned-sd", "aligned-sd"), ("misal-lw", "misal-lw"), ("misal-sw", "misal-sw"),
         ("misal-sd", "misal-sd"), ("misal-sd4", "misal-sd4"), ("aligned-sw-post", "aligned-sw"))


def main():
    p = argparse.ArgumentParser()
    k = p.add_mutually_exclusive_group(required=True)
    k.add_argument("--kit", type=Path)
    k.add_argument("--board-kit", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    got = hashlib.sha256(DOM.read_bytes()).hexdigest()[:16]
    if got != DOM_SHA:
        print(f"MISAL STOP: {DOM} is {got}, not {DOM_SHA}"); sys.exit(2)
    Batch, kit = boardvm.select(a.kit, a.board_kit)
    a.out.mkdir(parents=True, exist_ok=False)
    batch = Batch(a.out / "stage")
    dom = batch.put("images/misal-store.dom", DOM)
    root = str(Path(dom).parent.parent)
    env = "export CAPSTONE_EXEC_DIAGNOSTICS=1;"
    for name, arg in STEPS:
        batch.step(name, f"{env} {root}/capstone-vexec {dom} {arg}")
    batch.bounds = {name: 120 for name, _ in STEPS}
    runs, transcript, completed = batch.execute(kit, a.out / "guest", timeout=3600)
    print(f"MISAL kit {kit} boot {'completed' if completed else 'DID NOT COMPLETE'}; transcript {transcript}; steps {len(runs)}")
    rc = 0 if completed else 2
    for name, arg in STEPS:
        text, result = runs.get(name, ("", {"kind": "none"}))
        print(f"## step {name} result {json.dumps(result, sort_keys=True)}")
        print(text.rstrip())
        ret = re.search(rf"^MISAL {re.escape(arg)} RETURNED (0x[0-9a-f]+)$", text, re.M)
        flt = re.search(r"domain fault cause=(\d+) pc=0x([0-9a-f]+).*?code=0x([0-9a-f]+)-", text)
        mark = re.search(rf"^MISAL {re.escape(arg)} mark$", text, re.M)
        if not mark or result.get("kind") == "none":
            print(f"READ {name}: NO DATA (mark {'seen' if mark else 'absent'}, result {result.get('kind')})"); rc = 2
        elif ret and not flt:
            print(f"READ {name}: RETURNED {ret.group(1)} exit {result.get('kind')} {result.get('value')}")
        elif flt and not ret:
            print(f"READ {name}: FAULT cause {flt.group(1)} at code+{int(flt.group(2), 16) - int(flt.group(3), 16):#x} "
                  f"exit {result.get('kind')} {result.get('value')}")
        else:
            print(f"READ {name}: UNREADABLE (returned {bool(ret)}, fault {bool(flt)})"); rc = 2
    sys.exit(rc)


if __name__ == "__main__":
    main()
