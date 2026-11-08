import signal
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from host_profile import HostProfile

class ProfileTests(unittest.TestCase):
    def profiler(self, running, rc):
        p=HostProfile(Path('/unused'),Path('/unused'),123)
        proc=Mock();proc.poll.return_value=None if running else rc;proc.wait.return_value=rc
        p.process=proc;p.log=Mock()
        return p,proc
    def test_own_sigint_is_normal(self):
        p,proc=self.profiler(True,-signal.SIGINT);p.stop()
        proc.send_signal.assert_called_once_with(signal.SIGINT)
        p.log.close.assert_called_once();self.assertIsNone(p.process)
        p.stop();proc.wait.assert_called_once()
    def test_preexisting_sigint_is_failure(self):
        p,proc=self.profiler(False,-signal.SIGINT)
        with self.assertRaises(RuntimeError):p.stop()
        proc.send_signal.assert_not_called();p.log.close.assert_called_once()
    def test_permission_failure_not_hidden(self):
        p,proc=self.profiler(False,255)
        with self.assertRaises(RuntimeError):p.stop()
        proc.send_signal.assert_not_called()
    def test_clean_exit(self):
        p,proc=self.profiler(False,0);p.stop();proc.send_signal.assert_not_called()

if __name__=='__main__':unittest.main()
