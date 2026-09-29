import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "delegated", Path(__file__).with_name("run-libc-test-delegated.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TimeoutTest(unittest.TestCase):
    def test_timeout_requests_guest_cleanup(self):
        calls = []

        class Process:
            def __init__(self, command, **kwargs):
                self.result = Path(command[command.index("--result") + 1])
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def communicate(self, timeout):
                calls.append(("wait", timeout))
                if len(calls) == 1:
                    raise subprocess.TimeoutExpired("cli", timeout)
                self.result.write_text(json.dumps({"kind": "signal", "value": 15}))
                return "", ""
            def send_signal(self, signum): calls.append(("signal", signum))

        with patch.object(runner.subprocess, "Popen", Process):
            verdict, detail = runner.run_one(["cli"], "test", 1, {})
        self.assertEqual(verdict, "HUNG")
        self.assertIn("guest reaped", detail)
        self.assertEqual(calls, [("wait", 1), ("signal", signal.SIGTERM), ("wait", 20)])

    def test_unconfirmed_timeout_stops_suite(self):
        with patch.object(runner.subprocess, "Popen") as popen:
            process = popen.return_value.__enter__.return_value
            process.communicate.side_effect = [
                subprocess.TimeoutExpired("cli", 1),
                subprocess.TimeoutExpired("cli", 20), ("", "")]
            with self.assertRaisesRegex(RuntimeError, "cancellation unconfirmed"):
                runner.run_one(["cli"], "test", 1, {})
            process.send_signal.assert_called_once_with(signal.SIGTERM)
            process.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
