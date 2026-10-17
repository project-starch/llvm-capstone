#!/usr/bin/env python3
"""The board backend's round trip on synthetic steps, under busybox (the board's shell): stage a Batch, bake it,
run the controller on the host in a scratch root, slice, judge. Each reading the judge must produce is one the
QEMU path produces for the same step; each refusal must fire. Skipped without busybox.

    python3 -m unittest test_t3board        (from capstone/bug-corpora/tools)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import virtualvm  # noqa: E402

STUB = """#!/bin/sh
# a launcher stand-in: one domain fault, as the real launcher reports it with diagnostics on
echo "out of $1"
echo "capstone-exec: domain fault cause=25 pc=0x1000 address=0x0" >&2
kill -11 $$
"""


def batch(stage):
    b = virtualvm.Batch(stage)
    b.put("images/a.txt", text="payload\n")
    b.step("plain", "cat ./images/a.txt; exit 3")
    b.step("fault", "./capstone-vexec ./images/a.txt")
    b.step("hang", "sleep 60")
    b.step("after", "echo after-the-hang")
    return b


class RoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="t3board-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.env = dict(os.environ)
        self.addCleanup(os.environ.clear)
        self.addCleanup(os.environ.update, self.env)
        os.environ["CAPSTONE_VIRTUAL_BOARD_BUNDLE"] = "syn"
        kit = self.tmp / "kit"
        kit.mkdir()
        (kit / "capstone-vexec").write_text(STUB)
        (kit / "capstone-vexec").chmod(0o755)
        (kit / "capstone_vm.ko").write_text("stub\n")
        self.kit = kit

    def stage_bake_run(self, step_limit=5):
        board = self.tmp / "board"
        os.environ["CAPSTONE_VIRTUAL_BOARD"] = f"stage:{board}"
        with self.assertRaises(SystemExit) as e:
            batch(self.tmp / "s1").execute(None, None, 0)
        self.assertEqual(e.exception.code, 0)
        t3 = [sys.executable, str(TOOLS / "t3board.py")]
        root = self.tmp / "root"
        subprocess.run(t3 + ["bake", "--dir", str(board), "--bundles", "syn", "--kit", str(self.kit),
                             "--out", str(root), "--step-limit", str(step_limit), "--dry-run-root", str(root)],
                       check=True, stdout=subprocess.DEVNULL)
        run = subprocess.run(["busybox", "sh", str(root / "t3/t3run.sh"), "t3-syn", "x"],
                             env={**os.environ, "T3_ROOT": str(root)}, capture_output=True, text=True, timeout=60)
        transcript = self.tmp / "transcript.txt"
        transcript.write_text(run.stdout.replace("\n", "\r\n"))       # the board console's line ends
        subprocess.run(t3 + ["slice", "--dir", str(board), "--bundles", "syn", "--transcript", str(transcript)],
                       check=True, stdout=subprocess.DEVNULL)
        (board / "board-platform.json").write_text(json.dumps({"qemu": "none", "firmware": "f", "kernel": "k"}))
        return board

    def judge(self, board):
        os.environ["CAPSTONE_VIRTUAL_BOARD"] = f"judge:{board}"
        return batch(self.tmp / "s2").execute(None, None, 0)

    def test_round_trip(self):
        runs, _, completed = self.judge(self.stage_bake_run())
        self.assertTrue(completed)
        self.assertEqual(set(runs), {"plain", "fault", "hang", "after"})
        self.assertEqual(runs["plain"][1], {"kind": "exit", "value": 3})
        self.assertIn("payload", runs["plain"][0])
        self.assertEqual(runs["fault"][1]["kind"], "signal")
        self.assertEqual(runs["fault"][1]["value"], 11)
        self.assertIn("cause=25", runs["fault"][1]["fault"])
        self.assertEqual(runs["hang"][1], {"kind": "none"})               # the per-step limit, read as QEMU's
        self.assertIn("[runner] TIMEOUT", runs["hang"][0])
        self.assertNotIn(virtualvm.BOARD_TIMEOUT, runs["hang"][0])
        self.assertEqual(runs["after"][1], {"kind": "exit", "value": 0})  # the step after a kill still ran
        self.assertIn("after-the-hang", runs["after"][0])

    def test_judge_refuses_other_bytes(self):
        board = self.stage_bake_run()
        os.environ["CAPSTONE_VIRTUAL_BOARD"] = f"judge:{board}"
        b = batch(self.tmp / "s3")
        (b.stage / "images/a.txt").write_text("other\n")
        with self.assertRaises(SystemExit) as e:
            b.execute(None, None, 0)
        self.assertIn("REFUSED", str(e.exception.code))

    def test_judge_refuses_other_steps(self):
        board = self.stage_bake_run()
        os.environ["CAPSTONE_VIRTUAL_BOARD"] = f"judge:{board}"
        b = batch(self.tmp / "s4")
        b.step("extra", "true")
        with self.assertRaises(SystemExit) as e:
            b.execute(None, None, 0)
        self.assertIn("REFUSED", str(e.exception.code))

    def test_unset_switch_is_the_qemu_path(self):
        os.environ.pop("CAPSTONE_VIRTUAL_BOARD", None)
        self.assertIsNone(virtualvm.board())


# busybox is the board's shell; without it there is nothing to run the controller with
RoundTrip = unittest.skipUnless(shutil.which("busybox"), "busybox not installed")(RoundTrip)


if __name__ == "__main__":
    unittest.main()
