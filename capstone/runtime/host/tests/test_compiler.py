import tempfile
from pathlib import Path
import unittest

from capstone_vm.compiler import expand, link_arguments


class CompilerTests(unittest.TestCase):
    def test_sources_keep_link_order_and_receive_late_compile_flags(self):
        flags, sources, inputs, output = link_arguments([
            "one/main.c", "libone.a", "two/main.c", "-D", "MESSAGE=two words",
            "-lm", "-lcustom", "-Xlinker", "--no-undefined", "-o", "app.dom"])
        self.assertEqual(flags, ["-D", "MESSAGE=two words"])
        self.assertEqual(sources, ["one/main.c", "two/main.c"])
        self.assertEqual(inputs, ["@source:0", "libone.a", "@source:1", "-lcustom", "--no-undefined"])
        self.assertEqual(output, "app.dom")

    def test_response_file_preserves_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arguments"
            path.write_text("'a b.o' -D 'VALUE=a b' -o 'output file.dom'")
            self.assertEqual(expand(["@" + str(path)]),
                             ["a b.o", "-D", "VALUE=a b", "-o", "output file.dom"])

    def test_unknown_and_dynamic_options_fail(self):
        for option in ("--imaginary", "-shared", "-pie"):
            with self.assertRaises(ValueError):
                link_arguments(["main.o", option])

    def test_response_cycles_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle"
            path.write_text("@" + str(path))
            with self.assertRaises(ValueError):
                expand(["@" + str(path)])
