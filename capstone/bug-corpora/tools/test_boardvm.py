#!/usr/bin/env python3
"""boardvm's offline checks: the guest script parses as sh, the controller routes the batch rung, and execute() in
the stub mode reads ONLY the batch rung's own segment: a replayed previous boot's markers, an echoed command line and
another rung's output never reach split(), and a rung without its END marker reads as not completed. A kit's module
parameters are set, read back from that segment and recorded, and a run without their read-back is refused."""
import os
import shutil
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


def params_checks(root, body):
    """The kit's module parameters (adapter/module/params): set and read back in the gate, the run refused unless the
    batch rung's OWN segment reads every one back as set, and recorded in the platform."""
    global kit_dir
    plain = kit_dir
    kit_dir = Path(root) / "pkit"
    shutil.copytree(plain, kit_dir)
    (kit_dir / "adapter/module/params").write_text("# S-25's arena pin\narena_pin=1\n")
    pre = seg(1, 2, "b0-stats", "RESULT b0-stats retval=0")
    good = pre + seg(2, 2, "vbatch", "@@PARAM arena_pin=Y\n" + body)
    checks = []
    try:
        b, (steps, _, done) = run(good, root)
        script = b.script()
        checks += [("params: the gate sets and reads back arena_pin",
                    "[ -f /sys/module/capstone_vm/parameters/arena_pin ] && echo 1 > /sys/module/capstone_vm/parameters/arena_pin "
                    "&& echo \"@@PARAM arena_pin=" in script),
                   ("params: the gate still parses as sh",
                    subprocess.run(["sh", "-n"], input=script, text=True).returncode == 0),
                   ("params: a read-back of Y for 1 is accepted, and the steps read", done is True and len(steps) == 2),
                   ("params: the platform records them",
                    boardvm.platform(kit_dir, None, Path(__file__)).get("module_params") == "arena_pin=1")]
        refused = []
        for what, t in [("no read-back", pre + seg(2, 2, "vbatch", body)),
                        ("a read-back of N", pre + seg(2, 2, "vbatch", "@@PARAM arena_pin=N\n" + body)),
                        ("a read-back outside the batch rung's segment",
                         "@@PARAM arena_pin=Y\n" + pre + seg(2, 2, "vbatch", body))]:
            try:
                run(t, root)
                refused.append((what, False))
            except RuntimeError:
                refused.append((what, True))
        checks += [(f"params: {w} is REFUSED (negative control)", ok) for w, ok in refused]
        (kit_dir / "adapter/module/params").write_text("arena_pin = 1; rm -rf /\n")
        try:
            boardvm.virtualvm.module_params(kit_dir)
            checks.append(("params: a line that is not name=value is refused", False))
        except ValueError:
            checks.append(("params: a line that is not name=value is refused", True))
        # the guest lines themselves, under sh, against a scratch sysfs directory
        sysfs = Path(root) / "sysfs"
        sysfs.mkdir()
        (sysfs / "arena_pin").write_text("0\n")
        old, boardvm.virtualvm.SYSFS = boardvm.virtualvm.SYSFS, str(sysfs)
        try:
            ok = subprocess.run(["sh"], input="\n".join(boardvm.virtualvm.param_lines([("arena_pin", "1")])) + "\n",
                                text=True, capture_output=True)
            missing = subprocess.run(["sh"], input="\n".join(boardvm.virtualvm.param_lines([("no_such", "1")])) + "\n",
                                     text=True, capture_output=True)
        finally:
            boardvm.virtualvm.SYSFS = old
        checks += [("params: under sh the write lands and its read-back prints",
                    ok.returncode == 0 and ok.stdout.strip() == "@@PARAM arena_pin=1"),
                   ("params: under sh a parameter that does not exist ends the batch (negative control)",
                    missing.returncode == 1 and "@@PARAM no_such REFUSED" in missing.stdout
                    and "VIRTUAL_STAGED_DONE" in missing.stdout)]
    finally:
        kit_dir = plain
    return checks


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
        checks += [("a kit without params records no module_params", "module_params" not in p)]
        checks += params_checks(root, body)
        # run-virtual-cases.py records a board run under its own arm, so the QEMU bundle stays the virtual arm's record
        import argparse
        import importlib.util
        spec = importlib.util.spec_from_file_location("rvc", Path(__file__).with_name("run-virtual-cases.py"))
        rvc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rvc)
        ns = argparse.Namespace
        checks += [("a QEMU run records the virtual arm",
                    rvc.record_arm(ns(arm="virtual-malloc", arm_name=None, board_kit=None)) == "virtual-malloc"),
                   ("a board run records vp2-<arm>",
                    rvc.record_arm(ns(arm="virtual-malloc", arm_name=None, board_kit="k")) == "vp2-malloc"),
                   ("--arm-name wins", rvc.record_arm(ns(arm="virtual-malloc", arm_name="x", board_kit="k")) == "x")]
        # the helpers the corpus-specific runners (postgres c-repros, mmgr, sql) use
        checks += [("boardvm.record_arm on a board run", boardvm.record_arm("virtual-pg-pools", "k") == "vp2-pg-pools"),
                   ("boardvm.record_arm off the board", boardvm.record_arm("virtual-malloc", None) == "virtual-malloc"),
                   ("boardvm.select picks BoardBatch for a board kit",
                    boardvm.select(None, "bk") == (boardvm.BoardBatch, "bk")),
                   ("boardvm.select picks virtualvm.Batch for a QEMU kit",
                    boardvm.select("vk", None) == (boardvm.virtualvm.Batch, "vk"))]
    for what, ok in checks:
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
    print("PASS" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
