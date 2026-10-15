"""One entry, one selector: the virtual builder's dispatch.

build-virtual.sh is the only way in to a virtual application build, and its
--dry-run resolves the whole pipeline without running a recipe -- which is what
makes the dispatch testable on a host with no Capstone toolchain. What is
checked is the part a caller gets wrong: that one selector reaches the right
port variable for all eight ports, that a contradicting port variable is
refused instead of silently losing, and that memcached, whose recipe builds
every arm in one output, says so rather than accepting a selector it ignores.
"""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

COMMON = Path(__file__).resolve().parents[1]
BUILDER = COMMON / "application/build-virtual.sh"
# The port variable each application's recipe reads, for lifetimes 0 and 1.
DISPATCH = {
    "cpython": ("CPY_SUBLET=0", "CPY_SUBLET=1"),
    "postgres": ("PGSU_NESTED=none", "PGSU_NESTED=sublet"),
    "ffmpeg": ("FFAPP_SUBLET=0", "FFAPP_SUBLET=1"),
    "tshark": ("TSAPP_SUBLET=0", "TSAPP_SUBLET=1"),
    "perl": ("PERLD_SUBLET=0", "PERLD_SUBLET=1"),
    "mruby": ("MRBD_SUBLET=0", "MRBD_SUBLET=1"),
    "sqlite": ("SQLITE_HEAP=memsys5", "SQLITE_HEAP=memsys5-sublet"),
    "memcached": (None, None),
}
# The two ports whose recipe takes an SDK rather than building one.
SHARED_SDK = {"mruby", "sqlite"}


class Dispatch(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="build-virtual-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def resolve(self, application, lifetimes=None, output=None, environment=None):
        environment = dict(environment or {})
        if lifetimes is not None:
            environment["CAPSTONE_VIRTUAL_LIFETIMES"] = str(lifetimes)
        command = ["bash", str(BUILDER), "--dry-run", application,
                   str(output or self.root / application)]
        return subprocess.run(command, capture_output=True, text=True,
                              env={"PATH": "/usr/bin:/bin", "HOME": str(self.root),
                                   **environment})

    def pipeline(self, *args, **kwargs):
        done = self.resolve(*args, **kwargs)
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout)

    def test_one_selector_reaches_every_port_variable(self):
        for application, (off, on) in DISPATCH.items():
            for lifetimes, expected in ((0, off), (1, on)):
                if application == "memcached" and lifetimes == 1:
                    continue
                with self.subTest(application=application, lifetimes=lifetimes):
                    resolved = self.pipeline(application, lifetimes,
                                             output=self.root / f"{application}-{lifetimes}")
                    self.assertEqual(resolved["port_variable"], expected)
                    self.assertEqual(resolved["lifetimes"], lifetimes)
                    self.assertEqual(resolved["shared_sdk"], application in SHARED_SDK)
                    self.assertTrue(resolved["dry_run"])

    def test_the_default_is_the_released_allocator(self):
        for application, (off, _) in DISPATCH.items():
            with self.subTest(application=application):
                resolved = self.pipeline(application, None,
                                         output=self.root / f"default-{application}")
                self.assertEqual(resolved["lifetimes"], 0)
                self.assertEqual(resolved["port_variable"], off)

    def test_a_contradicting_port_variable_is_refused(self):
        done = self.resolve("cpython", 1, environment={"CPY_SUBLET": "0"})
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("contradicts CAPSTONE_VIRTUAL_LIFETIMES=1", done.stderr)

    def test_a_matching_port_variable_is_accepted(self):
        resolved = self.pipeline("cpython", 1, environment={"CPY_SUBLET": "1"})
        self.assertEqual(resolved["port_variable"], "CPY_SUBLET=1")

    def test_memcached_builds_every_arm_and_takes_no_selector(self):
        done = self.resolve("memcached", 1)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("builds every arm in one output", done.stderr)
        resolved = self.pipeline("memcached", 0, output=self.root / "memcached-0")
        self.assertIsNone(resolved["port_variable"])
        self.assertFalse(resolved["relinked_by_build_py"])

    def test_the_four_relinked_applications_are_the_ones_build_py_takes(self):
        relinked = {application for application in DISPATCH
                    if self.pipeline(application, 0, output=self.root / f"r-{application}")
                    ["relinked_by_build_py"]}
        self.assertEqual(relinked, {"cpython", "postgres", "ffmpeg", "tshark"})

    def test_an_unknown_application_and_a_used_output_are_refused(self):
        done = self.resolve("redis")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("unsupported application: redis", done.stderr)
        used = self.root / "already-there"
        used.mkdir()
        done = self.resolve("cpython", 0, output=used)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("fresh output directory", done.stderr)

    def test_an_invalid_selector_is_refused(self):
        done = self.resolve("cpython", "yes")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("CAPSTONE_VIRTUAL_LIFETIMES=yes", done.stderr)


if __name__ == "__main__":
    unittest.main()
