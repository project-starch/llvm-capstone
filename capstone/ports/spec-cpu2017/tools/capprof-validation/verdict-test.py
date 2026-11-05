#!/usr/bin/env python3
"""Exercise the verdict's evidence gates without any licensed SPEC inputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

VERDICT = Path(__file__).resolve().parents[1] / "verdict.py"


class VerdictTest(unittest.TestCase):
    def evaluate(self, route="guest", status=0, guest=None, stub=None, stub_log=False):
        if guest is None:
            guest = {"exit": 0} if route == "guest-plain" else {"kind": "exit", "value": 0}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "invocations").mkdir()
            (out / "runcpu.log").write_text("Success: 1x502.gcc_r\n")
            records = [
                {"route": route, "status": status, "exe": "/run/cpugcc_r_base.test",
                 "detail": {"guest": guest, "perl_pack_stub": stub}},
                {"route": "host-spec-tool", "status": 0, "exe": "/spec/bin/specdiff",
                 "detail": None},
            ]
            for i, record in enumerate(records):
                (out / "invocations" / f"{i:04d}.json").write_text(json.dumps(record))
            if stub_log:
                (out / "perl-pack-stub.jsonl").write_text('{"checks_stubbed": 4}\n')
            return json.loads(subprocess.check_output(
                [sys.executable, str(VERDICT), str(out), "502.gcc_r", "test", "0",
                 "start", "end", "test"], text=True))

    def test_successful_routes(self):
        for route in ("guest", "guest-plain"):
            with self.subTest(route=route):
                self.assertEqual(self.evaluate(route)["verdict"], "PASS")

    def test_failed_guest_cannot_be_overridden_by_success_banner(self):
        cases = [
            {"route": "guest-plain", "status": 1},
            {"route": "guest-plain", "guest": {"exit": 2}},
            {"route": "guest-plain", "guest": {}},
            {"guest": {"kind": "fault", "value": 0}},
            {"guest": {"kind": "exit", "value": 0, "fault": {"cause": 24}}},
            {"status": 1},
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(self.evaluate(**case)["verdict"], "FAIL")

    def test_input_intervention_is_rejected(self):
        for route in ("guest", "guest-plain"):
            for case in ({"stub": {"checks_stubbed": 4}}, {"stub_log": True}):
                with self.subTest(route=route, case=case):
                    result = self.evaluate(route, **case)
                    self.assertEqual(result["verdict"], "FAIL")
                    self.assertTrue(any("inputs were stubbed" in r for r in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
