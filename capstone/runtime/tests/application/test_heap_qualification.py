"""Host regressions for false-positive heap qualification results; no VM needed."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("heap_gate", Path(__file__).with_name("run-heap.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

DIGEST = "a" * 64
EVIDENCE = {"sha256": DIGEST, "entry": 0x1000,
            "probes": {"capstone_heap_fault_load": 0x1200, "sh_free": 0x1800}}
BEFORE = {"nodes_allocated_total": 100}
AFTER = {"nodes_allocated_total": 200100, "live_domains": 0, "live_regions": 0, "live_bytes": 0}


def outcome(mode, control=False, cause=None, pc=None, digest=DIGEST, status=None,
            ready=1, survived=None, fault=True):
    if status is None:
        status = 90 << 8 if control else 11
    if survived is None:
        survived = int(control)
    text = (f"application {mode}: {'FAIL' if control else 'PASS'} (wait status={status})\n"
            f"heap evidence {mode}: ready={ready} survived={survived} stderr=1\n")
    if not control and fault:
        if cause is None:
            cause = 5 if mode in gate.SPATIAL else 25
        if pc is None:
            pc = 0x8800 if mode.startswith("fault-double-free") else 0x8200
        text += (f"capstone-exec: domain fault cause={cause} pc={pc:#x} address=0x9000 "
                 f"entry=0x8000 code=0x8000-0xa000 sha256={digest} image=/contract.dom\n")
    return SimpleNamespace(stdout=text, stderr="", returncode=int(control))


class HeapOracleTests(unittest.TestCase):
    def check(self, mode, result, control=False, after=None):
        return gate.check_case("control" if control else "sublet", mode, result,
                               BEFORE, AFTER if after is None else after, EVIDENCE)

    def level0(self, mode, result):
        return gate.check_case("level0", mode, result, BEFORE, AFTER, EVIDENCE)

    def test_expected_faults_and_completed_controls(self):
        for mode in gate.FAULTS:
            for control in (False, True):
                with self.subTest(mode=mode, control=control):
                    self.check(mode, outcome(mode, control), control)

    def test_level0_faults_on_spatial_cases_and_runs_through_the_others(self):
        for mode in gate.FAULTS:
            with self.subTest(mode=mode):
                self.level0(mode, outcome(mode, control=mode not in gate.SPATIAL))

    def test_level0_spatial_survival_is_rejected(self):
        # a shrinking realloc that kept the old bounds would survive here
        for mode in gate.SPATIAL:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.level0(mode, outcome(mode, control=True))

    def test_level0_temporal_fault_is_rejected(self):
        # level0 does not revoke: a fault on a stale pointer is some other defect
        for mode in ("fault-stale", "fault-double-free"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.level0(mode, outcome(mode))

    def test_realloc_shrink_needs_a_bounds_fault(self):
        with self.assertRaises(ValueError):
            self.check("fault-realloc-shrink", outcome("fault-realloc-shrink", cause=25))

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
        for changes in ({"status": 9}, {"cause": 30}, {"pc": 0x8004},
                        {"ready": 0}, {"digest": "b" * 64}, {"fault": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check("fault-stale", outcome("fault-stale", **changes))

    def test_bounds_cannot_pass_on_a_revoked_pointer(self):
        with self.assertRaises(ValueError):
            self.check("fault-bounds", outcome("fault-bounds", cause=25))

    def test_stale_free_must_fail_at_the_first_probe(self):
        with self.assertRaises(ValueError):
            self.check("fault-double-free", outcome("fault-double-free", pc=0x8804))

    def test_untagged_stale_alias_is_also_rejected_by_hardware(self):
        self.check("fault-stale", outcome("fault-stale", cause=24))

    def test_early_reuse_fault_cannot_pass(self):
        for count in (2, 65228, 199999):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.check("fault-reused", outcome("fault-reused"),
                           after=AFTER | {"nodes_allocated_total": BEFORE["nodes_allocated_total"] + count})

    def test_churn_needs_progress_even_with_a_pass_line(self):
        result = SimpleNamespace(stdout="application churn: PASS (wait status=0)\n", returncode=0)
        self.check("churn", result)
        with self.assertRaises(ValueError):
            self.check("churn", result, after=AFTER | {"nodes_allocated_total": 102})

    def test_duplicate_evidence_and_incomplete_supervisor_are_rejected(self):
        result = outcome("fault-stale")
        result.stdout += result.stdout
        with self.assertRaises(ValueError):
            self.check("fault-stale", result)
        result = outcome("fault-stale")
        result.returncode = 124
        with self.assertRaises(ValueError):
            self.check("fault-stale", result)

    def test_resource_leaks_are_rejected(self):
        with self.assertRaises(ValueError):
            self.check("fault-stale", outcome("fault-stale"), after=AFTER | {"live_domains": 1})


class LinkOriginTests(unittest.TestCase):
    def setUp(self):
        self.table = {name: (0x1000 + i * 0x100, 0x40) for i, name in enumerate(gate.HEAP_SYMBOLS)}
        self.map = "\n".join(
            f" {address:x} {address:x} 40 4 CMakeFiles/app.dir/sublet_heap.c.obj:(.text.{name})\n"
            f" {address:x} {address:x} 40 1 {name}"
            for name, (address, _) in self.table.items())

    def check(self, contents=None, table=None):
        return gate.check_link_map(self.map if contents is None else contents,
                                   self.table if table is None else table, "sublet_heap.c.obj")

    def test_matching_map_and_elf(self):
        self.assertEqual(set(self.check()), set(gate.HEAP_SYMBOLS))

    def test_one_wrong_linked_object_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check(self.map.replace("sublet_heap.c.obj", "level0.c.obj", 1))

    def test_decoy_filename_is_not_link_evidence(self):
        # A file named like the allocator used to satisfy the entire origin check.
        with self.assertRaises(ValueError):
            self.check("CMakeFiles/app.dir/sublet_heap.c.obj\n")

    def test_stale_map_for_another_elf_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check(table=self.table | {"free": (0x1104, 0x40)})

    def test_missing_and_duplicate_symbols_are_rejected(self):
        for contents in (self.map.rsplit("\n", 1)[0], self.map + "\n" + self.map):
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                self.check(contents)


if __name__ == "__main__":
    unittest.main()
