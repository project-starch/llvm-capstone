import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from host_window import HostWindow

class WindowTests(unittest.TestCase):
    def window(self):
        values=iter([(1000000000,10),(3000000000,50)])
        return HostWindow(0,lambda pid:next(values))
    def test_fragmented_crlf(self):
        w=self.window();w.feed(b'noise\nHOST_WINDOW_BE');w.feed(b'GIN\r\n');w.feed(b'HOST_WINDOW_END\r\n')
        self.assertEqual(w.result()['wall_seconds'],2)
    def test_coalesced_rejected(self):
        with self.assertRaises(ValueError):self.window().feed(b'HOST_WINDOW_BEGIN\nHOST_WINDOW_END\n')
    def test_wrong_order(self):
        with self.assertRaises(ValueError):self.window().feed(b'HOST_WINDOW_END\n')
    def test_duplicate(self):
        w=self.window();w.feed(b'HOST_WINDOW_BEGIN\n')
        with self.assertRaises(ValueError):w.feed(b'HOST_WINDOW_BEGIN\n')
    def test_incomplete(self):
        w=self.window();w.feed(b'HOST_WINDOW_BEGIN\n')
        with self.assertRaises(ValueError):w.result()

if __name__=='__main__':unittest.main()
