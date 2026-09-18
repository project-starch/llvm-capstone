"""Accounting controls: nested lifetimes and rejection of incomplete/changed data."""

import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))

from analyze import HEADER, RECORD, analyze, trace_memory
from collect import loader_stopped_before_sharing, retryable_boot_failure


class MemoryAccounting(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        (self.run / "share").mkdir()
        records = [
            (6, 1, 0, 0, 0, 0, 0),  # Root context.
            (6, 2, 0, 1, 0, 0, 0),  # Child context.
            (1, 2, 1, 0, 100, 0, 0),
            (1, 1, 2, 0, 30, 0, 0),
            (3, 2, 1, 3, 120, 0, 0),  # Replace child object: 150 bytes live.
            (4, 1, 0, 0, 0, 0, 0),  # Reset deletes children and both objects.
            (1, 1, 4, 0, 50, 0, 0),
            (5, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 2, 4),
        ]
        raw = HEADER.pack(b"A11TRACE", 1, RECORD.size, 9, 0, 0, 0, 0, 0, b"")
        raw += b"".join(RECORD.pack(*r) for r in records)
        self.trace = self.run / "share/trace.a11"
        self.trace.write_bytes(raw)
        (self.run / "manifest.json").write_text(
            json.dumps(
                dict(
                    program="replay-spatial",
                    memory_profile=True,
                    sha256={"trace.a11": hashlib.sha256(raw).hexdigest()},
                    regions={"arena": 2048},
                    node_capacity=100,
                )
            )
        )
        (self.run / "verdict.json").write_text('{"passed": true}')
        self.expected = {
            1: (0, 0, 1, 6),
            2: (0, 0, 2, 6),
            3: (100, 1, 2, 1),
            4: (130, 2, 2, 1),
            5: (150, 2, 2, 3),
            6: (0, 0, 1, 4),
            7: (50, 1, 1, 1),
            8: (0, 0, 0, 5),
            9: (0, 0, 0, 0),
        }
        self.rows = []
        for event, (payload, objects, contexts, op) in self.expected.items():
            self.rows.append(
                dict(
                    kind="event",
                    event=event,
                    op=op,
                    live_payload_bytes=payload,
                    live_objects=objects,
                    live_contexts=contexts,
                    backing_bytes=1024,
                    assigned_backing_bytes=1024,
                    block_bytes=1024,
                    stranded_block_bytes=0,
                    metadata_records_live_bytes=0,
                    metadata_records_reserved_bytes=0,
                    arena_capacity_bytes=2048,
                )
            )
        for kind, index in (
            ("payload_peak", 4),
            ("backing_peak", 0),
            ("metadata_peak", 0),
            ("tracked_peak", 0),
        ):
            self.rows.append(dict(self.rows[index], kind=kind))

    def write_report(self):
        text = io.StringIO(newline="")
        writer = csv.writer(text)
        writer.writerow(["PGMEM_HEADER", *self.rows[0]])
        for row in self.rows:
            writer.writerow(["PGMEM", *row.values()])
        text.write("__CAPSTONE_PG_MEMORY_DONE__\n")
        (self.run / "share/payload.log").write_text(text.getvalue())

    def test_nested_reset_and_delete(self):
        snapshots, oracle = trace_memory(self.trace, set(self.expected))
        self.assertEqual(snapshots, self.expected)
        self.assertEqual(oracle["payload_peak_bytes"], 150)
        self.assertEqual(oracle["final_payload_bytes"], 0)
        self.write_report()
        self.assertEqual(analyze(self.run)["oracle"], oracle)

    def test_changed_payload_is_rejected(self):
        self.rows[4]["live_payload_bytes"] += 1
        self.write_report()
        with self.assertRaisesRegex(ValueError, "payload oracle mismatch"):
            analyze(self.run)

    def test_impossible_backing_is_rejected(self):
        self.rows[4]["stranded_block_bytes"] = 1
        self.write_report()
        with self.assertRaisesRegex(ValueError, "carved blocks exceed"):
            analyze(self.run)

    def test_incomplete_run_is_rejected(self):
        self.write_report()
        (self.run / "verdict.json").write_text('{"passed": false}')
        with self.assertRaisesRegex(ValueError, "not a completed"):
            analyze(self.run)

    def test_changed_trace_is_rejected(self):
        self.write_report()
        self.trace.write_bytes(self.trace.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "trace hash changed"):
            analyze(self.run)


class BootRetryControl(unittest.TestCase):
    def test_retry_stops_at_workload_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            run = directory / "run"
            run.mkdir()
            (directory / "runner.log").write_text(
                "__CAPSTONE_INFRA_FLAKE__ phase=boot-login\n"
            )
            (run / "verdict.json").write_text('{"passed": false, "runner_exit": 75}')
            (run / "serial.log").write_text("kernel boot output\n")
            self.assertTrue(retryable_boot_failure(directory))
            (run / "serial.log").write_text("PG: dom=1\n")
            self.assertFalse(retryable_boot_failure(directory))
            (run / "serial.log").write_text("kernel boot output\n")
            (run / "verdict.json").write_text('{"passed": false, "runner_exit": 1}')
            self.assertFalse(retryable_boot_failure(directory))

    def test_explicit_setup_retry_rejects_shared_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            run = directory / "run"
            run.mkdir()
            (directory / "runner.log").write_text("pexpect.exceptions.TIMEOUT\n")
            (run / "verdict.json").write_text('{"passed": false, "runner_exit": 1}')
            (run / "serial.log").write_text("PG: dom=0\nPG: r3=14\n")
            self.assertTrue(loader_stopped_before_sharing(directory))
            self.assertFalse(retryable_boot_failure(directory))
            (run / "serial.log").write_text("PG: r3=14\nPG: shared\n")
            self.assertFalse(loader_stopped_before_sharing(directory))
            (run / "serial.log").write_text("PG: shared\nPG: r3=14\n")
            self.assertFalse(loader_stopped_before_sharing(directory))


if __name__ == "__main__":
    unittest.main()
