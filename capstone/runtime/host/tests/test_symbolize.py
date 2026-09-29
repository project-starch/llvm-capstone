import subprocess
import unittest

from capstone_vm import symbolize

LINE = ("capstone-exec: domain fault cause=24 pc=0xe0204764 address=0x0 entry=0xe02006c8 "
        "code=0xe0200000-0xe0268000 last=0x40 preparing=0x19 image=/mnt/host/contract-v2.dom")


def fake_run(args, **_):
    if args[0].endswith("llvm-nm"):
        return subprocess.CompletedProcess(args, 0, stdout=
            "00000000000106c8 0000000000000418 T domain_main\n"
            "0000000000013f24 0000000000000854 T main\n", stderr="")
    assert args[0].endswith("llvm-symbolizer") and args[-1] == "0x14764", args
    return subprocess.CompletedProcess(args, 0, stdout="main at contract.c:101:14\n", stderr="")


class SymbolizeTest(unittest.TestCase):
    def test_parse(self):
        record = symbolize.parse(LINE)
        self.assertEqual(record["cause"], 24)
        self.assertEqual(record["pc"], 0xe0204764)
        self.assertEqual(record["entry"], 0xe02006c8)
        self.assertEqual(record["image"], "/mnt/host/contract-v2.dom")
        self.assertIsNone(symbolize.parse("capstone-exec: delegate rounds=3"))

    def test_slide_and_lookup(self):
        record = symbolize.parse(LINE)
        text = symbolize.symbolize(record, "image.dom", run=fake_run)
        self.assertIn("link 0x14764", text)
        self.assertIn("main+0x840 (main at contract.c:101:14)", text)

    def test_release_image_names_the_function(self):
        def no_lines(args, **_):
            if args[0].endswith("llvm-nm"):
                return fake_run(args)
            return subprocess.CompletedProcess(args, 0, stdout="?? at ??:0:0\n", stderr="")
        text = symbolize.symbolize(symbolize.parse(LINE), "image.dom", run=no_lines)
        self.assertIn(": main+0x840;", text)

    def test_missing_anchor(self):
        def no_anchor(args, **_):
            return subprocess.CompletedProcess(args, 0, stdout="0000000000013f24 0000000000000854 T main\n", stderr="")
        with self.assertRaises(LookupError):
            symbolize.symbolize(symbolize.parse(LINE), "image.dom", run=no_anchor)


if __name__ == "__main__":
    unittest.main()
