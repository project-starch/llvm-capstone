#!/usr/bin/env python3
"""Keep shell builds, corpus and versioned patches on the CMake release pin."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

PORT = Path(__file__).resolve().parents[1]
PIN = json.loads((PORT / "upstream.json").read_text())


class PinTests(unittest.TestCase):
    def shell(self, command, **env):
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1" || exit; ' + command,
                "pin-test",
                str(PORT.parent / "upstream.sh"),
            ],
            env=dict(os.environ, PG_VERSION="", **env),
            text=True,
            capture_output=True,
        )

    def test_shared_pin(self):
        result = self.shell('printf "%s\\n" "$PG_VERSION" "$PG_URL" "$PG_SHA256"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(), [PIN[k] for k in ("version", "url", "sha256")]
        )

    def test_wrong_override(self):
        # Build the environment separately so an inherited PG_VERSION cannot win.
        result = subprocess.run(
            ["bash", str(PORT.parent / "build-mmgr-host.sh")],
            env=dict(os.environ, PG_VERSION="0.0"),
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PostgreSQL version mismatch", result.stderr)

    def test_corrupt_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "bad.tar.bz2"
            archive.write_bytes(b"not PostgreSQL")
            result = self.shell(
                'pg_verify_archive "$PIN_TEST_ARCHIVE"', PIN_TEST_ARCHIVE=str(archive)
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("archive checksum mismatch", result.stderr)

    def test_patch_metadata(self):
        patches = sorted((PORT / "patches").glob("*.patch"))
        self.assertEqual(len(patches), 4)
        for patch in patches:
            self.assertTrue(patch.name.startswith(f"postgresql-{PIN['version']}-"))
            text = patch.read_text()
            self.assertIn(f"Upstream: postgresql {PIN['version']}\n", text)
            self.assertIn(f"Upstream-Archive: {PIN['url']}\n", text)
            self.assertIn(PIN["sha256"], text)

    def test_entry_points_share_manifest(self):
        for name in (
            "build-mmgr-host.sh",
            "build-mmgr-domain.sh",
            "build-mmgr-sublet.sh",
            "census-capstone.sh",
        ):
            text = (PORT.parent / name).read_text()
            self.assertIn('source "$HERE/upstream.sh"', text)
            self.assertNotIn("PG_VERSION=${", text)
        corpus = PORT.parents[2] / "bug-corpora/postgres/mmgr-repros/run-host-repros.sh"
        self.assertIn('source "$PORT/upstream.sh"', corpus.read_text())


if __name__ == "__main__":
    unittest.main()
