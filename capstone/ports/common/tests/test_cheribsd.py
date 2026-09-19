import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parents[1] / "host/cheribsd/run.py"
spec = importlib.util.spec_from_file_location("cheribsd_runner", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CheriBSDOutcomeTests(unittest.TestCase):
    def test_fault_requires_marker_and_sigprot(self):
        case = dict(name="fault", expect="READY", exit=162)
        for code, output, expected in (
            (162, "READY\n", True),
            (162, "boot failed\n", False),
            (0, "READY\n", False),
            (139, "READY\n", False),
            (162, "command echo READY\n", False),
        ):
            self.assertEqual(
                runner.outcome_matches(
                    case, SimpleNamespace(returncode=code, stdout=output)
                ),
                expected,
            )

    def test_explicit_rejection_requires_marker_and_failure_exit(self):
        case = dict(
            name="rejection", expect="REJECT code=719", exit=1, also_expect=["READY"]
        )
        runner.validate_case(case)
        for code, output, expected in (
            (1, "READY\nREJECT code=719\n", True),
            (1, "REJECT code=719\n", False),
            (0, "REJECT code=719\n", False),
            (162, "REJECT code=719\n", False),
            (1, "REJECT code=718\n", False),
            (1, "", False),
        ):
            self.assertEqual(
                runner.outcome_matches(
                    case, SimpleNamespace(returncode=code, stdout=output)
                ),
                expected,
            )

    def test_regex_must_match_whole_line_and_success_exit(self):
        case = dict(name="replay", expect_regex=r"done=\d+ status=0")
        for output, code, expected in (
            ("done=12 status=0\n", 0, True),
            ("done=12 status=1\n", 0, False),
            ("echo done=12 status=0\n", 0, False),
            ("done=12 status=0\n", 1, False),
        ):
            self.assertEqual(
                runner.outcome_matches(
                    case, SimpleNamespace(returncode=code, stdout=output)
                ),
                expected,
            )

    def test_unsafe_paths_and_ambiguous_oracles_rejected(self):
        for patch in (
            {"name": "../escape"},
            {"name": "case;command"},
            {"inputs": {"file with spaces": "source"}},
            {"inputs": {"../input": "source"}},
            {"outputs": ["/tmp/output"]},
            {"exit": 139},
            {"also_expect": "READY"},
            {"also_expect": [""]},
            {"expect_regex": "READY"},
            {"expect": ""},
        ):
            with self.assertRaises(ValueError):
                runner.validate_case(dict(name="case", expect="READY") | patch)

    def test_runtime_policy_mismatch_is_not_a_pass(self):
        case = dict(
            name="abi", expect="CHERI_ABI pointer_bytes=16 runtime_revocation=0"
        )
        self.assertFalse(
            runner.outcome_matches(
                case,
                SimpleNamespace(
                    returncode=0,
                    stdout="CHERI_ABI pointer_bytes=16 runtime_revocation=1\n",
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
