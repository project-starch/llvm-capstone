#!/usr/bin/env python3
"""boardvm's offline checks: the guest script parses as sh, the controller routes the batch rung, and execute() in
the stub mode reads ONLY the batch rung's own segment: a replayed previous boot's markers, an echoed command line and
another rung's output never reach split(), and a rung without its END marker reads as not completed."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boardvm  # noqa: E402


def kit(root):
    k = Path(root) / "kit"
    (k / "adapter/module").mkdir(parents=True)
    (k / "board.env").write_text('BOARD_BITSTREAM=caplifive_vp2_bf3db3175\nBOARD_MONITOR_COMMIT=007d81d\n'
                                 'BOARD_FW_DEFINES="-DCAPSTONE_VP2_FETCH_RETRY"\n')
    (k / "board-run.sh").write_text("#!/bin/bash\nexit 99\n")
    (k / "adapter/capstone-vexec").write_bytes(b"\x7fELF-launcher")
    (k / "adapter/module/capstone_vm.ko").write_bytes(b"\x7fELF-module")
    return k


def seg(n, m, name, body, rc=0):
    return (f"# echo '### RUNG {n}/{m} {name} ###'; x; echo \"### RUNG {n} {name} END rc=$rc ###\"\n"
            f"### RUNG {n}/{m} {name} ###\n{body}\n### RUNG {n} {name} END rc={rc} ###\nDN_{rc}\n")


def run(transcript, root):
    b = boardvm.BoardBatch(Path(root) / f"stage{len(os.listdir(root))}")
    b.put("images/a.dom", text="x")
    b.step("control-clean", "./capstone-vexec ./images/a.dom fixed 0")
    b.step("01_case-buggy", "./capstone-vexec ./images/a.dom buggy 1")
    t = Path(root) / f"t{len(os.listdir(root))}.txt"
    t.write_text(transcript)
    os.environ["BOARDVM_STUB_TRANSCRIPT"] = str(t)
    try:
        return b, b.execute(kit_dir, Path(root) / f"work{len(os.listdir(root))}", 60)
    finally:
        del os.environ["BOARDVM_STUB_TRANSCRIPT"]


def main():
    global kit_dir
    bad = 0
    with tempfile.TemporaryDirectory() as root:
        kit_dir = kit(root)
        body = ("@@STEP-BEGIN control-clean\nCONTROL clean mark\nCONTROL clean RETURNED\n@@STEP-END control-clean 0\n"
                "@@HB 1\n@@STEP-BEGIN 01_case-buggy\ncase=1 arm=buggy\n"
                "capstone-exec: domain fault cause=25 pc=0x10 address=0x20 entry=0x1 code=0x0-0x100 last=0x0\n"
                "@@STEP-END 01_case-buggy 139\nVIRTUAL_STAGED_DONE\nRESULT vbatch retval=0")
        replay = "### RUNG 2/2 vbatch ###\n@@STEP-BEGIN control-clean\nSTALE\n@@STEP-END control-clean 1\n"
        clean = replay + seg(1, 2, "b0-stats", "RESULT b0-stats retval=0") + seg(2, 2, "vbatch", body)
        b, (steps, _, done) = run(clean, root)
        checks = [("the guest script parses as sh",
                   subprocess.run(["sh", "-n"], input=b.script(), text=True).returncode == 0),
                  ("the controller parses as sh",
                   subprocess.run(["sh", "-n"], input=b.controller(), text=True).returncode == 0),
                  ("the gate cds into /test-domains/vbatch", "cd /test-domains/vbatch || exit 1" in b.script()),
                  ("put() returns the board path", b.put("x", text="y") == "/test-domains/vbatch/x"),
                  ("the replayed boot's markers are NOT read -- two start markers make the segment unreadable",
                   steps == {} and done is False)]
        clean = seg(1, 2, "b0-stats", "RESULT b0-stats retval=0") + seg(2, 2, "vbatch", body)
        _, (steps, _, done) = run(clean, root)
        checks += [("both steps read", sorted(steps) == ["01_case-buggy", "control-clean"]),
                   ("the control's exit", steps.get("control-clean", ("", {}))[1] == {"kind": "exit", "value": 0}),
                   ("the case's signal and fault line",
                    steps.get("01_case-buggy", ("", {}))[1].get("kind") == "signal"
                    and "cause=25" in steps["01_case-buggy"][1].get("fault", "")),
                   ("completed", done is True)]
        wedged = seg(1, 2, "b0-stats", "RESULT b0-stats retval=0") + \
            body.split("@@STEP-END 01_case-buggy")[0].join(["### RUNG 2/2 vbatch ###\n", ""])
        _, (steps, _, done) = run(wedged, root)
        checks += [("a rung without END is not completed", done is False),
                   ("the step that began and never ended reads kind none",
                    steps.get("01_case-buggy", ("", {}))[1].get("kind") == "none")]
        p = boardvm.platform(kit_dir, None, Path(__file__))
        checks += [("the qemu key names the board", p["qemu"] == "board:caplifive_vp2_bf3db3175"),
                   ("the monitor is recorded", p["monitor"] == "007d81d")]
    for what, ok in checks:
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
    print("PASS" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
