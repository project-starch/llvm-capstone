#!/usr/bin/env python3
"""A resumed campaign must reject changed execution artifacts and parameters."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "host/memory"))
from measure import check_resume, digest


class ResumeControls(unittest.TestCase):
    def test_binary_and_matrix_changes_rejected_collector_change_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, collector = root / "binary", root / "port/host/memory/collector"
            collector.parent.mkdir(parents=True)
            binary.write_text("original")
            collector.write_text("old collector")
            old = dict(
                schema="test",
                scope="test",
                node_capacity=100,
                workloads=[["one", "1", "320x180"]],
                repetitions=3,
                arms=["spatial", "sublet"],
                reservations_bytes={"payload": 64},
                fingerprints={str(p): digest(p) for p in (binary, collector)},
            )
            new = dict(old)
            collector.write_text("explicitly resumed collector")
            check_resume(old, new, collector)
            new["node_capacity"] = 101
            with self.assertRaisesRegex(ValueError, "node_capacity"):
                check_resume(old, new, collector)
            new["node_capacity"] = 100
            alternative = root / "different-binary"
            alternative.write_text("different selection")
            new["fingerprints"] = old["fingerprints"] | {
                str(alternative): digest(alternative)
            }
            with self.assertRaisesRegex(ValueError, "newly selected"):
                check_resume(old, new, collector)
            new["fingerprints"] = old["fingerprints"]
            binary.write_text("changed binary")
            with self.assertRaisesRegex(ValueError, "artifact"):
                check_resume(old, new, collector)


if __name__ == "__main__":
    unittest.main()
