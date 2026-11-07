#!/usr/bin/env python3
"""Short process-level checks for the bounded quick-gate supervisor."""

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

RUNNER = Path(__file__).with_name("run-quick-gate.py")


class QuickGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_plan(self, steps, budget=3):
        path = self.root / "plan.json"
        path.write_text(json.dumps(steps))
        before = time.monotonic()
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--plan", str(path), "--out", str(self.root / "out"),
             "--budget-seconds", str(budget)], capture_output=True, text=True, timeout=budget + 3,
        )
        summary = self.root / "out/summary.json"
        return result, json.loads(summary.read_text()) if summary.exists() else None, time.monotonic() - before

    def test_pass_and_literal_argv(self):
        literal = '$(touch forbidden); * "quoted"'
        result, summary, _ = self.run_plan([
            {"name": "literal", "argv": [sys.executable, "-c", "import sys;print(sys.argv[1])", literal]},
            {"name": "second", "argv": [sys.executable, "-c", "print('done')"]},
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual((self.root / "out/step-000/stdout.log").read_text().strip(), literal)
        self.assertEqual([r["status"] for r in summary["steps"]], ["PASS", "PASS"])
        self.assertIn("not full SPEC", summary["scope"])

    def test_early_failure_stops_later_step(self):
        marker = self.root / "must-not-exist"
        result, summary, _ = self.run_plan([
            {"name": "fails", "argv": [sys.executable, "-c", "raise SystemExit(7)"]},
            {"name": "later", "argv": [sys.executable, "-c", f"open({str(marker)!r},'w').close()"]},
        ])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(summary["status"], "FAIL")
        self.assertEqual(summary["steps"][0]["returncode"], 7)
        self.assertEqual(summary["steps"][1]["status"], "NOT_RUN")
        self.assertFalse(marker.exists())

    def test_shared_deadline_covers_all_steps(self):
        result, summary, elapsed = self.run_plan([
            {"name": str(i), "argv": [sys.executable, "-c", "import time;time.sleep(.45)"]}
            for i in range(3)
        ], budget=1.8)
        self.assertEqual(result.returncode, 124)
        self.assertEqual(summary["status"], "TIMEOUT")
        self.assertEqual(summary["steps"][0]["status"], "PASS")
        self.assertEqual(summary["steps"][1]["status"], "TIMEOUT")
        self.assertEqual(summary["steps"][2]["status"], "NOT_RUN")
        self.assertLess(elapsed, 2.3)

    def test_hung_term_ignoring_descendant_is_killed(self):
        pidfile = self.root / "pids"
        code = """import os,signal,time
signal.signal(signal.SIGTERM,signal.SIG_IGN)
pid=os.fork()
if pid==0:
 while True:time.sleep(1)
open(%r,'w').write(str(os.getpid())+' '+str(pid))
while True:time.sleep(1)
""" % str(pidfile)
        result, summary, elapsed = self.run_plan([{"name": "hung", "argv": [sys.executable, "-c", code]}], budget=1.4)
        self.assertEqual(result.returncode, 124)
        self.assertEqual(summary["status"], "TIMEOUT")
        self.assertTrue(summary["steps"][0]["cleanup"]["kill_sent"])
        self.assertTrue(summary["steps"][0]["cleanup"]["leader_reaped"])
        self.assertLess(elapsed, 1.9)
        pids = [int(x) for x in pidfile.read_text().split()]
        try:
            for pid in pids:
                path = Path(f"/proc/{pid}/stat")
                for _ in range(50):
                    if not path.exists() or path.read_text().split()[2] == "Z":
                        break
                    time.sleep(.01)
                else:
                    self.fail(f"owned process {pid} survived group cleanup")
        finally:
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_lock_wait_uses_same_budget(self):
        lock = self.root / "held.lock"
        marker = self.root / "lock-acquired"
        with lock.open("w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            code = (f"import fcntl; f=open({str(lock)!r},'w');"
                    f"fcntl.flock(f,fcntl.LOCK_EX);open({str(marker)!r},'w').close()")
            result, summary, elapsed = self.run_plan(
                [{"name": "blocked-lock", "argv": [sys.executable, "-c", code]}], budget=1)
        self.assertEqual(result.returncode, 124)
        self.assertEqual(summary["status"], "TIMEOUT")
        self.assertFalse(marker.exists())
        self.assertLess(elapsed, 1.5)

    def test_invalid_plan_runs_nothing(self):
        marker = self.root / "must-not-exist"
        command = [sys.executable, "-c", f"open({str(marker)!r},'w').close()"]
        for value in ([], {}, [{"name": "first", "argv": command}, {"name": "first", "argv": command}],
                      [{"name": "first", "argv": command}, {"name": "bad", "argv": [1]}],
                      [{"name": "bad", "argv": "echo bad"}], [{"name": "", "argv": command}]):
            with self.subTest(value=value):
                result, summary, _ = self.run_plan(value)
                self.assertEqual(result.returncode, 2)
                self.assertIsNone(summary)
                self.assertFalse(marker.exists())
                self.assertFalse((self.root / "out").exists())

    def test_invalid_budget_and_existing_output(self):
        steps = [{"name": "ok", "argv": [sys.executable, "-c", "pass"]}]
        for budget in (301, 0, -1, "nan", "inf"):
            with self.subTest(budget=budget):
                plan = self.root / "plan.json"
                plan.write_text(json.dumps(steps))
                result = subprocess.run([sys.executable, str(RUNNER), "--plan", str(plan), "--out", str(self.root / "out"), "--budget-seconds", str(budget)], capture_output=True, timeout=2)
                self.assertEqual(result.returncode, 2)
        (self.root / "out").mkdir()
        result, _, _ = self.run_plan(steps)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
