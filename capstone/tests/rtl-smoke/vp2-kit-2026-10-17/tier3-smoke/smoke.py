#!/usr/bin/env python3
"""smoke.py --qemu-kit K | --board-kit K --image controls.dom --raw DIR: the Tier 3 board backend's smoke batch.

The corpus's virtual positive controls (postgres c-repros shared/controls.c, built with the kit's SDK) as three steps
in one boot, through virtualvm.Batch (QEMU) or boardvm.BoardBatch (the vp2 board), the same steps either way:
- control-bounds-malloc: the configuration virtual-mallocng expects `fault`;
- control-uaf-malloc: expects `fault`;
- control-clean: an unknown control name, which prints `CONTROL  RETURNED` and exits 0, so `complete`.
Prints one line per step: name, observe_control()'s reading, the result dict, the fault line.
Exit 0 when every step was read, 2 when any step is missing (no data is never a pass).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[4] / "bug-corpora" / "tools"
sys.path.insert(0, str(TOOLS))
import boardvm  # noqa: E402
import virtualvm  # noqa: E402

spec = importlib.util.spec_from_file_location("rvc", TOOLS / "run-virtual-cases.py")
rvc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rvc)

STEPS = (("control-bounds-malloc", "bounds-malloc"), ("control-uaf-malloc", "uaf-malloc"), ("control-clean", "clean"))


def main():
    p = argparse.ArgumentParser()
    k = p.add_mutually_exclusive_group(required=True)
    k.add_argument("--qemu-kit", type=Path)
    k.add_argument("--board-kit", type=Path)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    a = p.parse_args()
    Batch, kit = (boardvm.BoardBatch, a.board_kit) if a.board_kit else (virtualvm.Batch, a.qemu_kit)
    b = Batch(a.raw / "stage")
    b.put(f"images/{a.image.name}", a.image)
    for name, which in STEPS:
        b.step(name, f"./capstone-vexec ./images/{a.image.name} {which}")
    runs, transcript, completed = b.execute(kit, a.raw / "guest", timeout=1800)
    print(f"SMOKE {'board' if a.board_kit else 'qemu'} completed={completed} transcript={transcript}")
    missing = 0
    for name, which in STEPS:
        if name not in runs:
            print(f"STEP {name}: NO DATA"); missing += 1; continue
        text, result = runs[name]
        seen = rvc.observe_control(which, text, result)
        print(f"STEP {name}: {seen} {json.dumps(result)}")
    return 2 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
