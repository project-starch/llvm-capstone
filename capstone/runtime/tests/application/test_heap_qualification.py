"""Host regressions for false-positive heap qualification results; no VM needed."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("heap_gate", Path(__file__).with_name("run-heap.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

DIGEST = "a" * 64
EVIDENCE = {"sha256": DIGEST, "entry": 0x1000, "probes": {"capstone_heap_fault_load": 0x1200}}
BEFORE = {"nodes_allocated_total": 100}
AFTER = {"nodes_allocated_total": 200100, "live_domains": 0, "live_regions": 0, "live_bytes": 0}


def outcome(mode, control=False, cause=5, pc=0x8200, digest=DIGEST, status=None,
            ready=1, survived=None, fault=True):
    if status is None:
        status = 90 << 8 if control else 11
    if survived is None:
        survived = int(control)
    text = (f"application {mode}: {'FAIL' if control else 'PASS'} (wait status={status})\n"
            f"heap evidence {mode}: ready={ready} survived={survived} stderr=1\n")
    if not control and fault:
        text += (f"capstone-exec: domain fault cause={cause} pc={pc:#x} address=0x9000 "
                 f"entry=0x8000 code=0x8000-0xa000 sha256={digest} image=/contract.dom\n")
    return SimpleNamespace(stdout=text, stderr="", returncode=int(control))


class HeapOracleTests(unittest.TestCase):
    def check(self, mode, result, control=False, after=None):
        return gate.check_case("control" if control else "level0", mode, result,
                               BEFORE, AFTER if after is None else after, EVIDENCE)

    def test_control_runs_every_fault_case_through(self):
        for mode in gate.FAULTS:
            with self.subTest(mode=mode):
                self.check(mode, outcome(mode, control=True), control=True)

    def test_level0_faults_on_spatial_cases_and_runs_through_the_others(self):
        for mode in gate.FAULTS:
            with self.subTest(mode=mode):
                self.check(mode, outcome(mode, control=mode not in gate.SPATIAL))

    def test_level0_spatial_survival_is_rejected(self):
        # a shrinking realloc that kept the old bounds would survive here
        for mode in gate.SPATIAL:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.check(mode, outcome(mode, control=True))

    def test_level0_temporal_fault_is_rejected(self):
        # level0 does not revoke: a fault on a stale pointer is some other defect
        for mode in ("fault-stale", "fault-double-free"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.check(mode, outcome(mode, cause=25))

    def test_a_spatial_case_needs_a_bounds_fault(self):
        for mode in ("fault-bounds", "fault-realloc-shrink"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.check(mode, outcome(mode, cause=25))

    def test_control_setup_errors_are_not_survival(self):
        # 63 was accepted by the old gate even though address reuse had failed.
        for exit_code in (0, 46, 63, 68, 124):
            with self.subTest(exit_code=exit_code), self.assertRaises(ValueError):
                self.check("fault-double-free-reused", outcome("fault-double-free-reused",
                           True, status=exit_code << 8), True)

    def test_control_requires_both_markers(self):
        for ready, survived in ((0, 0), (1, 0), (0, 1)):
            with self.subTest(ready=ready, survived=survived), self.assertRaises(ValueError):
                self.check("fault-stale", outcome("fault-stale", True, ready=ready, survived=survived), True)

    def test_unrelated_signal_and_fault_are_rejected(self):
        for changes in ({"status": 9}, {"cause": 30}, {"pc": 0x8204},
                        {"ready": 0}, {"digest": "b" * 64}, {"fault": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check("fault-bounds", outcome("fault-bounds", **changes))

    def test_completion_needs_a_pass_line_and_a_clean_exit(self):
        self.check("churn", SimpleNamespace(stdout="application churn: PASS (wait status=0)\n", returncode=0))
        with self.assertRaises(ValueError):
            self.check("churn", SimpleNamespace(stdout="application churn: PASS (wait status=256)\n",
                                                returncode=0))

    def test_duplicate_evidence_and_incomplete_supervisor_are_rejected(self):
        result = outcome("fault-bounds")
        result.stdout += result.stdout
        with self.assertRaises(ValueError):
            self.check("fault-bounds", result)
        result = outcome("fault-bounds")
        result.returncode = 124
        with self.assertRaises(ValueError):
            self.check("fault-bounds", result)

    def test_resource_leaks_are_rejected(self):
        with self.assertRaises(ValueError):
            self.check("fault-bounds", outcome("fault-bounds"), after=AFTER | {"live_domains": 1})


class LinkOriginTests(unittest.TestCase):
    def setUp(self):
        self.table = {name: (0x1000 + i * 0x100, 0x40) for i, name in enumerate(gate.HEAP_SYMBOLS)}
        self.map = "\n".join(
            f" {address:x} {address:x} 40 4 CMakeFiles/app.dir/level0.c.obj:(.text.{name})\n"
            f" {address:x} {address:x} 40 1 {name}"
            for name, (address, _) in self.table.items())

    def check(self, contents=None, table=None):
        return gate.check_link_map(self.map if contents is None else contents,
                                   self.table if table is None else table, "level0.c.obj")

    def test_matching_map_and_elf(self):
        self.assertEqual(set(self.check()), set(gate.HEAP_SYMBOLS))

    def test_one_wrong_linked_object_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check(self.map.replace("level0.c.obj", "malloc.c.obj", 1))

    def test_decoy_filename_is_not_link_evidence(self):
        # A file named like the allocator used to satisfy the entire origin check.
        with self.assertRaises(ValueError):
            self.check("CMakeFiles/app.dir/level0.c.obj\n")

    def test_stale_map_for_another_elf_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check(table=self.table | {"free": (0x1104, 0x40)})

    def test_missing_and_duplicate_symbols_are_rejected(self):
        for contents in (self.map.rsplit("\n", 1)[0], self.map + "\n" + self.map):
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                self.check(contents)


if __name__ == "__main__":
    unittest.main()
