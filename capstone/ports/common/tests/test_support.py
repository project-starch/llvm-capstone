"""Host controls for immutable inputs, QEMU failure propagation and release pins."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON / "host"))
from port_support import digest, run_guest, stage_run, write_checksums


class HostSupport(unittest.TestCase):
    def test_staged_inputs_are_independent_and_fingerprinted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "input"
            trace.write_bytes(b"trace")
            trace.chmod(0o755)
            first, hashes = stage_run(root / "results", "replay-", {"trace.bin": trace})
            trace.write_bytes(b"changed")
            second, _ = stage_run(root / "results", "replay-", {"trace.bin": trace})
            self.assertNotEqual(first, second)
            self.assertEqual((first / "share/trace.bin").read_bytes(), b"trace")
            self.assertEqual((first / "share/trace.bin").stat().st_mode & 0o777, 0o755)
            self.assertEqual(hashes["trace.bin"], digest(first / "share/trace.bin"))
            self.assertNotEqual(hashes["trace.bin"], digest(second / "share/trace.bin"))

    def test_guest_failure_is_returned_under_the_shared_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            env = {"CAPSTONE_QEMU_LOCK": str(run / "qemu.lock")}
            with patch("port_support.subprocess.run") as execute:
                execute.return_value = subprocess.CompletedProcess([], 75)
                result = run_guest(
                    run, "run workload", "DONE", env=env, timeout_multiplier=60
                )
            self.assertEqual(result.returncode, 75)
            command = execute.call_args.args[0]
            self.assertEqual(
                command[:5], ["flock", "-x", "-w", "60", env["CAPSTONE_QEMU_LOCK"]]
            )
            self.assertEqual(
                command[command.index("--guest-command") + 1], "run workload"
            )
            self.assertEqual(command[command.index("--success-marker") + 1], "DONE")
            self.assertEqual(command[command.index("--timeout-multiplier") + 1], "60")

    def test_bundle_checksums_are_repeatable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.csv").write_bytes(b"measurement\n")
            write_checksums(root)
            original = (root / "SHA256SUMS").read_bytes()
            write_checksums(root)
            self.assertEqual((root / "SHA256SUMS").read_bytes(), original)
            self.assertEqual(
                original.decode(), f"{digest(root / 'summary.csv')}  summary.csv\n"
            )


class SourcePins(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="port cache ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "release.tar"
        self.source.write_bytes(b"pinned release")
        self.archive = self.root / "download/release.tar"
        self.pin = dict(
            name="fixture",
            version="1",
            url=self.source.as_uri(),
            sha256=digest(self.source),
        )
        self.script = self.root / "check.cmake"
        self.script.write_text(
            f"cmake_minimum_required(VERSION 3.25)\n"
            f'include("{COMMON / "cmake/Upstream.cmake"}")\n'
            f'set(PROJECT_SOURCE_DIR "{self.root}")\n'
            f'port_read_upstream()\nport_download("{self.archive}")\n'
        )

    def download(self):
        (self.root / "upstream.json").write_text(json.dumps(self.pin))
        return subprocess.run(
            ["cmake", "-P", self.script], capture_output=True, text=True
        )

    def test_download_cache_and_corruption(self):
        result = self.download()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.archive.read_bytes(), b"pinned release")
        self.source.unlink()
        self.assertEqual(self.download().returncode, 0)
        self.archive.write_bytes(b"corrupted cache")
        result = self.download()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cached archive checksum mismatch", result.stderr)

    def test_bad_download_is_not_published(self):
        self.pin["sha256"] = hashlib.sha256(b"different release").hexdigest()
        result = self.download()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Downloaded archive checksum mismatch", result.stderr)
        self.assertFalse(self.archive.exists())
        self.assertFalse(Path(str(self.archive) + ".part").exists())


if __name__ == "__main__":
    unittest.main()
