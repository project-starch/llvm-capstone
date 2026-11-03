#!/usr/bin/env python3
"""Exercise detached children, timeout, interruption and unrelated-PID safety."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

WATCHDOG = Path(__file__).with_name('profile-watchdog.py')


class WatchdogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('CAPSTONE_TMP_ROOT'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'run'

    def command(self, code, seconds=1):
        return [sys.executable, str(WATCHDOG), '--seconds', str(seconds),
                '--out', str(self.out), '--', sys.executable, '-c', code]

    def detached(self, exit_parent=False):
        # Like VM up: its child starts a new session and outlives the CLI.
        return f'''import os, signal, time
from pathlib import Path
pid = os.fork()
if pid == 0:
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path({str(self.root / 'vm.pid')!r}).write_text(str(os.getpid()))
    time.sleep(60)
else:
    {"time.sleep(0.2)" if exit_parent else "time.sleep(60)"}
'''

    def assert_clean(self):
        pid = int((self.root / 'vm.pid').read_text())
        self.assertFalse(Path(f'/proc/{pid}').exists(), f'VM {pid} survived or was not reaped')
        record = json.loads((self.out / 'watchdog.json').read_text())
        self.assertEqual(record['remaining'], [])
        self.assertTrue(any(s['pid'] == pid for s in record['signals']))
        self.assertEqual(json.loads((self.out / 'result.json').read_text())['verdict'], 'FAIL')
        return record

    def test_timeout_reaps_detached_vm_and_leaves_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            result = subprocess.run(self.command(self.detached()), capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('wall-time limit exceeded', self.assert_clean()['reason'])
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait()

    def test_normal_exit_with_orphan_is_failure_and_reaped(self):
        result = subprocess.run(self.command(self.detached(True), 10), capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('live descendants', self.assert_clean()['reason'])

    def test_interruption_reaps_vm(self):
        proc = subprocess.Popen(self.command(self.detached(), 60), stdout=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not (self.root / 'vm.pid').exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue((self.root / 'vm.pid').exists())
            proc.send_signal(signal.SIGTERM)
            proc.communicate(timeout=15)
            self.assertEqual(proc.returncode, 1)
            self.assertIn('interrupted', self.assert_clean()['reason'])
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.communicate(timeout=15)

    def test_timeout_replaces_interrupted_result(self):
        code = f"from pathlib import Path; import time; p=Path({str(self.out)!r}); p.mkdir(); (p/'result.json').write_text('{{'); time.sleep(60)"
        result = subprocess.run(self.command(code), capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads((self.out / 'result.json').read_text())['verdict'], 'FAIL')
        self.assertEqual((self.out / 'result-interrupted.json').read_text(), '{')

    def test_clean_exit_and_nonzero_exit(self):
        result = subprocess.run(self.command('pass'), capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.out = self.root / 'failed'
        result = subprocess.run(self.command('raise SystemExit(7)'), capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads((self.out / 'result.json').read_text())['verdict'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
