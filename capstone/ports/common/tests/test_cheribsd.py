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


class ContinueOnFailureTests(unittest.TestCase):
    """--continue-on-failure only changes WHEN the suite stops, never the verdicts."""

    def test_failed_names_are_in_run_order_and_exclude_passes(self):
        results = [
            dict(name="cheribsd-abi", passed=True),
            dict(name="defect-0-mode0", passed=False),
            dict(name="defect-0-mode1", passed=True),
            dict(name="defect-1-mode0", passed=False),
        ]
        self.assertEqual(
            runner.failed_case_names(results), ["defect-0-mode0", "defect-1-mode0"]
        )

    def test_no_failures_is_an_empty_list_not_a_missing_answer(self):
        self.assertEqual(runner.failed_case_names([]), [])
        self.assertEqual(
            runner.failed_case_names([dict(name="cheribsd-abi", passed=True)]), []
        )

    def test_every_failure_is_reported_not_only_the_first(self):
        results = [dict(name=f"defect-{i}-mode1", passed=False) for i in range(40)]
        self.assertEqual(len(runner.failed_case_names(results)), 40)


class DefectFaultOracleTests(unittest.TestCase):
    """The shape the pymalloc defect corpus asks the runner to enforce: one
    complete line whose two PC fields are the same text, plus the setup marker
    and the CheriBSD SIGPROT status. Exit 162 alone must not pass."""

    case = dict(
        name="defect-5-mode1",
        expect_regex=r"PYC_DEFECT_FAULT case=5 signal=34 code=2 "
        r"pc=(0x[0-9a-f]+) expected=\1 exact=1",
        also_expect=["PYC_DEFECT case=5 ready"],
        exit=162,
    )
    ready = "PYC_DEFECT case=5 ready\n"
    fault = (
        "PYC_DEFECT_FAULT case=5 signal=34 code=2 pc=0x10a5c expected=0x10a5c exact=1\n"
    )

    def test_case_is_well_formed(self):
        runner.validate_case(self.case)

    def test_only_an_exact_tag_fault_at_the_probe_passes(self):
        for output, code, expected in (
            (self.ready + self.fault, 162, True),
            # A tag fault somewhere else: the PC fields disagree.
            (
                self.ready + "PYC_DEFECT_FAULT case=5 signal=34 code=2 "
                "pc=0x10a90 expected=0x10a5c exact=0\n",
                162,
                False,
            ),
            # A bounds fault at the right instruction is not this result.
            (
                self.ready + "PYC_DEFECT_FAULT case=5 signal=34 code=1 "
                "pc=0x10a5c expected=0x10a5c exact=1\n",
                162,
                False,
            ),
            # The handler's own verdict must agree with the printed PCs.
            (
                self.ready + "PYC_DEFECT_FAULT case=5 signal=34 code=2 "
                "pc=0x10a5c expected=0x10a5c exact=0\n",
                162,
                False,
            ),
            # Another case's fault line.
            (
                self.ready + "PYC_DEFECT_FAULT case=4 signal=34 code=2 "
                "pc=0x10a5c expected=0x10a5c exact=1\n",
                162,
                False,
            ),
            # SIGPROT with no fault line at all: an arbitrary crash.
            (self.ready, 162, False),
            # The case never reached its critical access.
            (self.fault, 162, False),
            # Right output, wrong platform outcome.
            (self.ready + self.fault, 0, False),
            (self.ready + self.fault, 139, False),
            # Echoed rather than produced.
            (self.ready + "echo " + self.fault, 162, False),
        ):
            with self.subTest(output=output, code=code):
                self.assertEqual(
                    runner.outcome_matches(
                        self.case, SimpleNamespace(returncode=code, stdout=output)
                    ),
                    expected,
                )

    def test_completion_arm_needs_both_markers_and_a_clean_exit(self):
        case = dict(
            name="defect-5-mode0",
            expect="PYC_DEFECT case=5 completed",
            also_expect=["PYC_DEFECT case=5 ready"],
            exit=0,
        )
        runner.validate_case(case)
        done = "PYC_DEFECT case=5 completed\n"
        for output, code, expected in (
            (self.ready + done, 0, True),
            (self.ready, 0, False),
            (done, 0, False),
            (self.ready + done, 1, False),
            (self.ready + done, 162, False),
        ):
            with self.subTest(output=output, code=code):
                self.assertEqual(
                    runner.outcome_matches(
                        case, SimpleNamespace(returncode=code, stdout=output)
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
