#!/usr/bin/env python3
"""smoke-diag.py --board-kit K --controls controls.dom --contract contract.dom --raw DIR: the Tier 3 smoke's diagnostic
boot, after two smoke boots hung in control-bounds-malloc (QEMU faults it with cause 28 in seconds).

One board boot through boardvm.BoardBatch, the same backend as smoke.py, with every step made to RETURN:
- contract-ok: Tier 1's contract.dom `ok argument`, which passes on silicon in every Tier 1 boot. It is the matched
  arm: the same gate script, launcher, module and batch environment as the controls, with a different image.
- the three controls, each bounded: the launcher runs in the background, a sleeper sends SIGTERM after BOUND_S
  seconds, and the step prints the module's ring-buffer lines (dmesg, KMSG: prefix) after it ends. A step that hits
  the bound exits 143, as run.py's preempt step does on silicon.
Prints one line per step: name, exit, whether the control's mark printed, its fault line, the KMSG lines.
Exit 0 when every step was read, 2 when any is missing (no data is never a pass).
"""
import argparse
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[4] / "bug-corpora" / "tools"
sys.path.insert(0, str(TOOLS))
import boardvm  # noqa: E402

BOUND_S = 150


def bounded(cmd):
    """A step command that always returns unless the launcher is unkillable. At BOUND_S the watcher prints every
    thread's state and wchan (R spinning, S sleeping, D uninterruptible), sends SIGTERM, and SIGKILL 10 s later. The
    step then prints the module's lines from the ring buffer."""
    watch = (f"( sleep {BOUND_S}; for t in /proc/$p/task/*; do echo \"BOUND-STATE $(cut -d' ' -f1-3 $t/stat) "
             f"wchan=$(cat $t/wchan 2>/dev/null)\"; done; kill -TERM $p; sleep 10; kill -KILL $p ) 2>/dev/null & w=$!")
    return (f"export CAPSTONE_VM_STATS=1; {cmd} & p=$!; {watch}; "
            f"wait $p; rc=$?; kill $w 2>/dev/null; echo \"BOUNDED-EXIT $rc\"; "
            f"dmesg | grep -a 'capstone_vm:' | tail -12 | sed 's/^/KMSG: /'; exit $rc")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--board-kit", type=Path, required=True)
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--print-script", action="store_true", help="print the guest script and run nothing")
    a = p.parse_args()
    b = boardvm.BoardBatch(a.raw / "stage")
    b.put("images/contract.dom", a.contract)
    b.put("images/controls.dom", a.controls)
    steps = [("contract-ok", bounded("./capstone-vexec ./images/contract.dom ok argument"), None)]
    steps += [(f"control-{w}", bounded(f"./capstone-vexec ./images/controls.dom {w}"), w)
              for w in ("bounds-malloc", "uaf-malloc", "clean")]
    for name, cmd, _ in steps:
        b.step(name, cmd)
    if a.print_script:
        print(b.script())
        return 0
    runs, transcript, completed = b.execute(a.board_kit, a.raw / "guest", timeout=1200)
    print(f"SMOKE-DIAG board completed={completed} transcript={transcript}")
    missing = 0
    for name, _, which in steps:
        if name not in runs:
            print(f"STEP {name}: NO DATA")
            missing += 1
            continue
        text, result = runs[name]
        mark = (f"CONTROL {which} mark" in text) if which else ("VIRTUAL_APPLICATION_OK" in text)
        bexit = [ln for ln in text.splitlines() if ln.startswith("BOUNDED-EXIT ")]
        kmsg = [ln[6:] for ln in text.splitlines() if ln.startswith("KMSG: ")]
        print(f"STEP {name}: {bexit[-1] if bexit else 'no BOUNDED-EXIT'} mark={mark} fault={result.get('fault')!r}")
        for ln in text.splitlines():
            if ln.startswith("BOUND-STATE "):
                print(f"  {ln[:200]}")
        for ln in kmsg:
            print(f"  KMSG {ln[:200]}")
    # The apertures sit after the rung segment in the driver's transcript: the wedge block, or the resting block when
    # nothing wedged (193 the commit count, 221 the commit queue's valid bits, 225 bit 3 the switcher busy).
    full = transcript.read_text(errors="replace").replace("\r", "") if transcript.exists() else ""
    for title in ("final apertures (no wedge)", "vbatch wedge apertures"):
        if f"===== {title} =====" in full:
            block = full.split(f"===== {title} =====", 1)[1].split("=====", 1)[0]
            print(f"APERTURES {title}:")
            for ln in block.splitlines():
                if ln.startswith(("sw=193", "sw=194", "sw=221", "sw=224", "sw=225", "sw=227", "sw=228", "trap-log")):
                    print(f"  {ln[:120]}")
    return 2 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
