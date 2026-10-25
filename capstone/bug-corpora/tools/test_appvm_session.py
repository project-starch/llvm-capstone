#!/usr/bin/env python3
"""appvm's batch-session mode, offline: collect -> boot (a canned board transcript) -> read, with the checks that make
it safe to run a corpus runner unchanged through it."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import appvm  # noqa: E402
from test_boardvm import kit, seg  # noqa: E402

SESSION_CLI = HERE / "board" / "appvm-session.py"


def runner(state, raw, img1, img2, args1=("a", "b c")):
    """What a corpus runner does: stage a share file, then two run_app calls."""
    share = Path(json.loads((Path(state) / "config.json").read_text())["share"])
    (share / "d").mkdir(exist_ok=True)
    (share / "d" / "fixture.txt").write_text("fixture\n")
    one = appvm.run_app(state, img1, list(args1), raw / "case-one", run_args=("--cwd", "/mnt/host/d", "-e", "K=V w"))
    two = appvm.run_app(state, img2, ["9"], raw / "case-two")
    return one, two


def main():
    checks = []
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        k = kit(root)
        img1, img2 = root / "one.dom", root / "two.dom"
        img1.write_bytes(b"\x7fELF-one")
        img2.write_bytes(b"\x7fELF-two")
        s = root / "session"
        raw = root / "raw"
        raw.mkdir()
        subprocess.run([sys.executable, str(SESSION_CLI), "init", str(s), "--board-kit", str(k)], check=True,
                       capture_output=True)
        state = s / "state"
        one, two = runner(state, raw, img1, img2)
        calls = [json.loads(line) for line in (s / "plan.jsonl").read_text().splitlines()]
        checks += [("collect runs nothing", "COLLECT" in one[0] and one[1] == {} and two[1] == {}),
                   ("collect records both calls with cwd and env",
                    [c["name"] for c in calls] == ["case-one", "case-two"] and calls[0]["cwd"] == "/mnt/host/d"
                    and calls[0]["env"] == ["K=V w"] and calls[0]["args"] == ["a", "b c"]),
                   ("collect stages the images and the share", len(list((s / "collect/images").iterdir())) == 2
                    and (s / "collect/share/d/fixture.txt").read_text() == "fixture\n"),
                   ("a session needs no running VM", appvm.require_up(state) is None),
                   ("a refused run_app argument is refused, not ignored",
                    _raises(lambda: appvm.run_app(state, img1, [], raw / "x", run_args=("--user", "1:1"))))]
        body = ("@@STEP-BEGIN session-share\nSESSION-SHARE-OK\n@@STEP-END session-share 0\n"
                "@@STEP-BEGIN case-one\nhello from one\n@@STEP-END case-one 0\n"
                "@@STEP-BEGIN case-two\ncapstone-exec: domain fault cause=25 pc=0x10 address=0x20 entry=0x1 "
                "code=0x0-0x100 last=0x0\n@@STEP-END case-two 139\nVIRTUAL_STAGED_DONE\nRESULT vbatch retval=0")
        t = root / "t.txt"
        t.write_text(seg(1, 2, "b0-stats", "RESULT b0-stats retval=0") + seg(2, 2, "vbatch", body))
        env = dict(os.environ, BOARDVM_STUB_TRANSCRIPT=str(t))
        boot = subprocess.run([sys.executable, str(SESSION_CLI), "boot", str(s)], env=env, capture_output=True,
                              text=True)
        script = (s / "stage" / "gate.sh").read_text()
        checks += [("boot completes and reads every step", boot.returncode == 0 and "2/2 steps ran" in boot.stdout),
                   ("the step runs in its cwd with its env", "cd /mnt/host/d && exec env 'K=V w' "
                    "/test-domains/vbatch/capstone-vexec /test-domains/vbatch/images/one-" in script),
                   ("the share is linked at /mnt/host, checked through it",
                    "ln -sfn /test-domains/vbatch/share /mnt/host" in script
                    and "mount -o bind /test-domains/vbatch/share /mnt/host" in script
                    and "[ -f /mnt/host/.session-share ] && echo SESSION-SHARE-OK" in script
                    and (s / "stage/share/.session-share").is_file()),
                   ("the phase moves to read", appvm.session(state)["phase"] == "read")]
        one, two = runner(state, raw, img1, img2)
        checks += [("read: the recorded exit, versioned, with the image hash",
                    one[1] == {"kind": "exit", "value": 0, "version": 1, "image_sha256": appvm.sha256(img1)}
                    and "hello from one" in one[0]),
                   ("read: the recorded signal and fault line",
                    two[1].get("kind") == "signal" and two[1].get("value") == 11 and "cause=25" in two[1].get("fault", "")),
                   ("read: a call that differs from what was collected is refused",
                    _raises(lambda: runner(state, raw, img1, img2, args1=("a", "changed")))),
                   ("the platform record names the board", appvm.platform(state, None, __file__)["qemu"]
                    == "board:caplifive_vp2_bf3db3175"),
                   ("on_board says board", appvm.on_board(state))]
    bad = 0
    for what, ok in checks:
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
    print("PASS" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


def _raises(f):
    try:
        f()
    except (ValueError, KeyError):
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
