"""Tests for tools/verdicts.py: every rule of the judge, the registry, the parser and the bundle.

    python3 -m unittest discover -s capstone/bug-corpora/tools -p 'test_*.py'

Each rule is tested from the side where it must FIRE, because a judge whose NO-READING branches
never trigger would still pass a suite of happy-path rows.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verdicts as v  # noqa: E402

ARMS = v.load_arms()
SPEC = ARMS["app-level0"]          # bounds-malloc must fault, uaf-malloc must complete
OK_CONTROLS = [v.Control("bounds-malloc", "fault"), v.Control("uaf-malloc", "complete")]
FAULT = v.Fault(cause=7, pc=0xc02020f0, symbol="pqGetnchar", offset=0x10)


def obs(**kw):
    base = dict(case="00_x", arm="spatial", controls=list(OK_CONTROLS))
    base.update(kw)
    return v.Observation(**base)


class Judge(unittest.TestCase):
    def test_infra_is_never_a_verdict(self):
        verdict, reason, _ = v.judge(obs(infra="infra", fault=FAULT, reached=True,
                                         attribution="probe"), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "infra"))

    def test_unknown_reason_is_an_error(self):
        with self.assertRaises(ValueError):
            v.judge(obs(infra="flaky"), SPEC)

    def test_failed_control_voids_even_an_attributed_fault(self):
        bad = [v.Control("bounds-malloc", "complete"), v.Control("uaf-malloc", "complete")]
        verdict, reason, evidence = v.judge(obs(controls=bad, fault=FAULT, reached=True,
                                                attribution="probe"), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "control-failed"))
        self.assertIn("bounds-malloc: expected fault, observed complete", evidence)

    def test_control_not_declared_for_the_configuration_is_an_error(self):
        with self.assertRaises(ValueError):
            v.judge(obs(controls=[v.Control("uaf-chunk", "complete")]), SPEC)

    def test_fault_before_the_marker_is_setup(self):
        verdict, reason, _ = v.judge(obs(fault=FAULT, reached=False, attribution="function"), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "setup-fault"))

    def test_fault_without_attribution_is_not_a_catch(self):
        verdict, reason, _ = v.judge(obs(fault=FAULT, reached=True), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "unattributed"))

    def test_attributed_fault_is_caught(self):
        for how in v.ATTRIBUTIONS:
            verdict, reason, evidence = v.judge(obs(fault=FAULT, reached=True, attribution=how,
                                                    attribution_evidence="x"), SPEC)
            self.assertEqual((verdict, reason), (v.CAUGHT, None), how)
            self.assertIn(f"by {how}", evidence)

    def test_silence_without_the_marker_is_not_missed(self):
        verdict, reason, _ = v.judge(obs(reached=False, completed=True), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "not-reached"))

    def test_reached_but_never_ended(self):
        verdict, reason, _ = v.judge(obs(reached=True, completed=False), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "inconclusive"))

    def test_silence_needs_the_arms_controls(self):
        verdict, reason, evidence = v.judge(obs(reached=True, completed=True,
                                                controls=[OK_CONTROLS[0]]), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "control-failed"))
        self.assertIn("uaf-malloc", evidence)

    def test_missed(self):
        verdict, reason, _ = v.judge(obs(reached=True, completed=True), SPEC)
        self.assertEqual((verdict, reason), (v.MISSED, None))


class Registry(unittest.TestCase):
    def test_every_configuration_is_well_formed(self):
        self.assertTrue(ARMS)
        for name, spec in ARMS.items():
            self.assertTrue(set(spec["controls_for_missed"]) <= set(spec["controls"]), name)
            self.assertTrue(set(spec["controls"].values()) <= {"fault", "complete"}, name)
            self.assertTrue(spec["controls_for_missed"], f"{name}: a silence would need no proof")

    def test_corpus_arm_configurations_exist(self):
        for path in sorted(HERE.parent.glob("*/*/corpus.json")):
            mapping = json.loads(path.read_text()).get("arm_configurations", {})
            for arm, config in mapping.items():
                self.assertIn(config, ARMS, f"{path}: {arm}")


class Parser(unittest.TestCase):
    # A committed raw result from the c-repros sublet run of 2026-10-06.
    C00 = (HERE.parent / "postgres/c-repros/results/sublet-20261006-161014"
           / "00_CVE-2026-6477_pqfn_unbounded_result_copy.json")

    def test_domain_fault_line(self):
        fault = v.domain_fault(json.loads(self.C00.read_text())["fault"])
        self.assertEqual((fault.cause, fault.pc), (7, 0xc02020f0))

    def test_no_fault_line(self):
        self.assertIsNone(v.domain_fault("capstone-vm: application exited 0"))


class Bundle(unittest.TestCase):
    PLATFORM = {k: "unrecorded" for k in v.PLATFORM_KEYS}

    def test_round_trip_and_tally(self):
        rows = [(obs(fault=FAULT, reached=True, attribution="probe"), None),
                (obs(reached=False), None)]
        rows = [(o, v.judge(o, SPEC)) for o, _ in rows]
        with tempfile.TemporaryDirectory() as d:
            record = v.write_bundle(d, "postgres/c-repros", "spatial", rows,
                                    {"platform": self.PLATFORM, "configuration": "app-level0"})
            self.assertEqual(record["tally"], {"CAUGHT": 1, "NO-READING:not-reached": 1})
            back = v.read_bundle(d)
        self.assertEqual(back[0][0].fault.pc, FAULT.pc)
        self.assertEqual([r[1] for r in back], [v.CAUGHT, v.NO_READING])

    def test_platform_must_be_said(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            v.write_bundle(d, "c", "spatial", [], {"platform": {"compiler": "x"}})


if __name__ == "__main__":
    unittest.main()
