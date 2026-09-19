#!/usr/bin/env python3
"""Controls for measurement acceptance and synchronized payload accounting."""

from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "host/memory"))
from measure import observations

ROW = struct.Struct("<16Q")


def event(op, kind=0, live=0, retained=0):
    row = [0] * 16
    row[0], row[1], row[11], row[12] = op, kind, live, retained
    return row


class MeasurementControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, rows, status=0, mode=0):
        header = [0x4650465452433032, len(rows), status] + [0] * 13
        header[11] = mode
        path = self.root / name
        path.write_bytes(ROW.pack(*header) + b"".join(ROW.pack(*r) for r in rows))
        return path

    def test_synchronized_peak_not_sum_of_category_peaks(self):
        rows = [
            event(5, 1, retained=100),
            event(130, 1, 100, 100),
            event(3, 1, retained=100),
            event(6, 1),
            event(5, 2, retained=80),
            event(130, 2, 80, 80),
            event(3, 2, retained=80),
            event(6, 2),
            event(8),
        ]
        result, _ = observations(self.write("trace", rows))
        self.assertEqual(result["combined"]["peak_live_requested_bytes"], 100)
        self.assertEqual(result["combined"]["peak_pool_backing_requested_bytes"], 100)

    def test_cross_kind_state_is_carried_forward(self):
        rows = [
            event(5, 1, retained=100),
            event(5, 2, retained=80),
            event(6, 1),
            event(6, 2),
            event(8),
        ]
        result, _ = observations(self.write("trace", rows))
        self.assertEqual(result["combined"]["peak_idle_pool_requested_bytes"], 180)

    def test_changed_outcome_rejected(self):
        reference = self.write(
            "reference", [event(5, 1, retained=100), event(6, 1), event(8)]
        )
        changed = self.write(
            "changed", [event(5, 1, retained=200), event(6, 1), event(8)]
        )
        with self.assertRaisesRegex(ValueError, "differ"):
            observations(changed, reference)

    def test_failed_or_wrong_mode_rejected(self):
        with self.assertRaises(ValueError):
            observations(self.write("failed", [event(8)], status=110))
        with self.assertRaisesRegex(ValueError, "mode"):
            observations(self.write("wrong", [event(8)], mode=2), mode=0)

    def test_unbalanced_or_inconsistent_accounting_rejected(self):
        for name, rows in (
            ("live", [event(5, 1, retained=100), event(8)]),
            ("bad", [event(130, 1, live=100), event(6, 1), event(8)]),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                observations(self.write(name, rows))


if __name__ == "__main__":
    unittest.main()
