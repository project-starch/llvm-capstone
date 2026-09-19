"""The QEMU console must not block while the suite is using SSH."""

import importlib.util
import io
from pathlib import Path
import sys
import threading
import unittest


class GuestConsoleTests(unittest.TestCase):
    def test_output_larger_than_pty_buffer_is_consumed(self):
        if importlib.util.find_spec("pexpect") is None:
            self.skipTest("pexpect not installed")
        path = Path(__file__).resolve().parents[1] / "host/cheribsd/guest.py"
        spec = importlib.util.spec_from_file_location("cheribsd_guest", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        guest = module.Guest.__new__(module.Guest)
        guest.log = io.StringIO()
        guest.reader_stop = threading.Event()
        guest.process = module.pexpect.spawn(
            sys.executable,
            ["-c", "print('x' * (1024 * 1024)); print('CONSOLE_COMPLETE')"],
            encoding="utf-8",
        )
        guest.process.logfile_read = guest.log
        guest.reader = threading.Thread(target=guest.drain_console, daemon=True)
        try:
            guest.reader.start()
            guest.reader.join(timeout=10)
            self.assertFalse(guest.reader.is_alive(), "console producer blocked")
            self.assertIn("CONSOLE_COMPLETE", guest.log.getvalue())
            self.assertEqual(guest.log.getvalue().count("x"), 1024 * 1024)
        finally:
            guest.close()


if __name__ == "__main__":
    unittest.main()
