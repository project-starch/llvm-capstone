"""Tests for the corpus adapters' observe() functions, on real run output.

    python3 -m unittest discover -s capstone/bug-corpora/tools -p 'test_*.py'

The c-repros inputs are the committed console and result files of its 2026-10-06 runs; the mmgr
inputs are lines from a 2026-10-10 serial log, quoted as the guest printed them.
"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPORA = HERE.parent
sys.path.insert(0, str(HERE))
import verdicts as v  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSymbols:
    """Resolves every pc to one function, so a test can say what the image's symbols would."""
    def __init__(self, name):
        self.name = name

    def lookup(self, pc, code_start):
        return self.name, 0x10


class CRepros(unittest.TestCase):
    run_arm = load(CORPORA / "postgres/c-repros/shared/run-arm.py", "c_run_arm")
    RESULTS = CORPORA / "postgres/c-repros/results/spatial-20261006-160406"
    CASE0 = "00_CVE-2026-6477_pqfn_unbounded_result_copy"

    def case0(self):
        return ((self.RESULTS / f"{self.CASE0}.out").read_text(),
                json.loads((self.RESULTS / f"{self.CASE0}.json").read_text()))

    def test_fault_outside_the_named_function_is_unattributed(self):
        text, result = self.case0()
        o = self.run_arm.observe("0", text, result, FakeSymbols("memcpy"))
        self.assertTrue(o.reached)
        self.assertEqual((o.fault.cause, o.fault.pc), (7, 0xc02020f0))
        self.assertIsNone(o.attribution)
        self.assertIn("pqGetnchar", o.attribution_evidence)

    def test_fault_in_the_named_function_is_attributed(self):
        text, result = self.case0()
        o = self.run_arm.observe("0", text, result, FakeSymbols("pqGetnchar"))
        self.assertEqual(o.attribution, "function")

    def test_declared_site_attributes(self):
        text, result = self.case0()
        o = self.run_arm.observe("0", text, result, FakeSymbols("memcpy"), sites=["memcpy"])
        self.assertEqual(o.attribution, "function")

    def test_no_symbols_never_attributes(self):
        text, result = self.case0()
        self.assertIsNone(self.run_arm.observe("0", text, result).attribution)

    def test_run_that_never_began_is_infra(self):
        o = self.run_arm.observe("3", "capstone-vm: exec failed\n", {"kind": "exit", "value": 126})
        self.assertEqual(o.infra, "infra")

    def test_exit_75_is_infra_not_a_silence(self):
        o = self.run_arm.observe("3", "case 3 BEGIN\nCONTROL-FAILED 1\n", {"kind": "exit", "value": 75})
        self.assertEqual(o.infra, "infra")

    def test_returned_after_mark_is_completed(self):
        o = self.run_arm.observe("3", "case 3 BEGIN\nPG_DEFECT case=3 mark\ncase 3 RETURNED\n",
                                 {"kind": "exit", "value": 0})
        self.assertTrue(o.reached and o.completed and o.fault is None)

    def test_returned_without_mark_is_not_reached(self):
        o = self.run_arm.observe("3", "case 3 BEGIN\ncase 3 RETURNED\n", {"kind": "exit", "value": 0})
        verdict = v.judge(o, {"controls": {}, "controls_for_missed": []})
        self.assertEqual(verdict[:2], (v.NO_READING, "not-reached"))

    def test_controls(self):
        line = "capstone-exec: domain fault cause=7 pc=0xc0200100 address=0x1 entry=0x2 code=0xc0200000-0xc0300000"
        self.assertEqual(self.run_arm.observe_control("uaf-malloc", "CONTROL uaf-malloc mark\n",
                                                      {"fault": line}), "fault")
        self.assertEqual(self.run_arm.observe_control(
            "uaf-malloc", "CONTROL uaf-malloc mark\nCONTROL uaf-malloc RETURNED\n", {}), "complete")
        self.assertEqual(self.run_arm.observe_control("uaf-malloc", "", {"fault": line}), "none")


class SqlRepros(unittest.TestCase):
    run_arm = load(CORPORA / "postgres/sql-repros/shared/run-arm.py", "sql_run_arm")
    CORPUS = CORPORA / "postgres/sql-repros"
    ALL = {"ltree", "pg_trgm", "fuzzystrmatch", "pgcorpus_reach"}

    def read(self, run, case):
        d = self.CORPUS / "results" / run
        name = next(p.stem for p in d.glob(f"{case}_*.json"))
        return (self.CORPUS / name, (d / f"{name}.out").read_text(),
                json.loads((d / f"{name}.json").read_text()))

    def test_fault_after_the_trigger_started_is_reached_but_not_attributed_by_itself(self):
        case, text, result = self.read("sublet-20261008-080000", "02")
        o = self.run_arm.observe(case, text, result, self.ALL)
        self.assertTrue(o.reached and o.fault and o.fault.cause == 7)
        self.assertIsNone(o.attribution)      # the caller's control.sql run decides

    def test_completed_trigger_is_reached_and_completed(self):
        case, text, result = self.read("spatial-20261006-165125", "04")
        o = self.run_arm.observe(case, text, result, self.ALL)
        self.assertTrue(o.reached and o.completed and o.fault is None)
        self.assertIn("errors", o.reach_evidence)

    def test_missing_extension_is_out_of_denominator(self):
        case, text, result = self.read("sublet-20261008-080000", "03")
        o = self.run_arm.observe(case, text, result, available=set())
        self.assertEqual(o.infra, "out-of-denominator")

    def test_fault_inside_create_extension_is_setup(self):
        case = next(self.CORPUS.glob("03_*"))
        o = self.run_arm.observe(case, "backend> \n", {"kind": "signal", "value": 11,
                                 "fault": "capstone-exec: domain fault cause=7 pc=0x1 address=0x2 code=0x0-"},
                                 self.ALL)
        verdict = v.judge(o, {"controls": {}, "controls_for_missed": []})
        self.assertEqual(verdict[:2], (v.NO_READING, "setup-fault"))

    def test_control_counts_only_a_fault_in_its_own_probe(self):
        line = ("capstone-exec: domain fault cause=24 pc=0x3f88044408 address=0x1 entry=0x2 "
                "code=0x3f88000000-0x3f89000000 last=0x4f")
        result = {"kind": "signal", "value": 11, "fault": line}
        seen = self.run_arm.control_seen
        self.assertEqual(seen("uaf-malloc", "backend> ", result, FakeSymbols("corpus_control_read")), "fault")
        self.assertEqual(seen("uaf-malloc", "backend> ", result, FakeSymbols("AllocSetAlloc")), "none")
        self.assertEqual(seen("uaf-malloc", "backend> ", result), "none")
        self.assertEqual(seen("uaf-malloc", "CONTROL RETURNED\n", {"kind": "exit", "value": 0}), "complete")

    def test_no_prompt_is_infra(self):
        case = next(self.CORPUS.glob("02_*"))
        self.assertEqual(self.run_arm.observe(case, "capstone-vm: boot\n", {}, self.ALL).infra, "infra")


class VirtualSteps(unittest.TestCase):
    import virtualvm as vm

    SERIAL = ("noise\n@@STEP-BEGIN control-uaf-malloc\nCONTROL uaf-malloc mark\n"
              "capstone-exec: domain fault cause=24 pc=0x3f00 address=0x1 entry=0x2 code=0x3f00-0x4f00 last=0\n"
              "@@STEP-END control-uaf-malloc 139\n"
              "@@STEP-BEGIN 03_case\ncase 3 BEGIN\nPG_DEFECT case=3 mark\ncase 3 RETURNED\n@@STEP-END 03_case 0\n"
              "@@STEP-BEGIN 04_case\ncase 4 BEGIN\n")

    def test_fault_step(self):
        text, result = self.vm.split(self.SERIAL)["control-uaf-malloc"]
        self.assertEqual((result["kind"], result["value"]), ("signal", 11))
        self.assertIn("cause=24", result["fault"])

    def test_clean_step(self):
        text, result = self.vm.split(self.SERIAL)["03_case"]
        self.assertEqual(result, {"kind": "exit", "value": 0})
        self.assertIn("case 3 RETURNED", text)

    def test_step_that_never_ended_is_a_timeout_never_a_silence(self):
        text, result = self.vm.split(self.SERIAL)["04_case"]
        self.assertEqual(result["kind"], "none")
        o = CRepros.run_arm.observe("4", text, result)
        self.assertEqual(o.infra, "infra")

    def test_steps_that_never_began_are_absent(self):
        self.assertNotIn("05_case", self.vm.split(self.SERIAL))


class Sqlite(unittest.TestCase):
    sqlite = load(CORPORA / "sqlite/engine-repros/shared/run-virtual.py", "sqlite_run_virtual")
    LINE = ("capstone-exec: domain fault cause=28 pc=0x3fa4a2379c address=0x1 entry=0x2 "
            "code=0x3fa4800000-0x3fa4c00000 last=0x39")

    def test_fault_in_the_asan_function_is_caught(self):
        o = self.sqlite.observe("t", "t BEGIN\n", {"kind": "signal", "value": 11, "fault": self.LINE},
                             FakeSymbols("sqlite3Fts5GetVarint32"), {"sqlite3Fts5GetVarint32"})
        self.assertEqual((o.reached, o.attribution), (True, "function"))

    def test_fault_elsewhere_is_not(self):
        o = self.sqlite.observe("t", "t BEGIN\n", {"kind": "signal", "value": 11, "fault": self.LINE},
                             FakeSymbols("sqlite3Fts5GetTokenizer"), {"sqlite3Fts5GetVarint32"})
        self.assertIsNone(o.attribution)

    def test_own_assert_abort_is_not_a_silence(self):
        o = self.sqlite.observe("t", "t BEGIN\nAssertion failed\n", {"kind": "signal", "value": 6}, None, set())
        verdict = v.judge(o, {"controls": {}, "controls_for_missed": []})
        self.assertEqual(verdict[:2], (v.NO_READING, "inconclusive"))

    def test_returned_is_completed(self):
        o = self.sqlite.observe("t", "t BEGIN\nt RETURNED\n", {"kind": "exit", "value": 0}, None, set())
        self.assertTrue(o.reached and o.completed)


class Mmgr(unittest.TestCase):
    mmgr = load(CORPORA / "postgres/mmgr-repros/shared/run-defects.py", "mmgr_run")
    SERIAL = """[CAPSTONE] Print = Scalar(0xcf18000000000002)
[CAPSTONE] Print = Cap(1, 0x7, 0x101a1280c, 0x101a00000, 0x101ae9800)
[CAPSTONE] Print = Cap(1, 0x7, 0x101a1287c, 0x101a00000, 0x101ae9800)
[CAPSTONE] domain halted by capability fault: cause = 24, pc = 0x101a1280c, tval = 0x0, badaddr = 0x0
"""

    def observe(self, serial, sites=()):
        self.mmgr.SYMBOLS = {"img": FakeSymbols("pg_probe")}
        orig = v.sha256
        self.mmgr.v.sha256 = lambda p: "0" * 64
        try:
            return self.mmgr.observe(serial, 2, 1, "img", set(sites))
        finally:
            self.mmgr.v.sha256 = orig

    def test_fault_at_the_published_probe(self):
        o = self.observe(self.SERIAL)
        self.assertEqual((o.reached, o.attribution, o.fault.cause), (True, "probe", 24))

    def test_fault_elsewhere_is_unattributed(self):
        o = self.observe(self.SERIAL.replace("pc = 0x101a1280c", "pc = 0x101a05cf0"))
        self.assertIsNone(o.attribution)

    def test_delivered_fault_needs_the_launcher_death(self):
        serial = self.SERIAL.replace("domain halted by capability fault", "domain capability fault delivered")
        self.assertIsNone(self.observe(serial).fault)
        self.assertIsNotNone(self.observe(serial + "__EXIT_CODE__139\n").fault)

    def test_no_serial_is_infra(self):
        self.assertEqual(self.observe("").infra, "infra")


if __name__ == "__main__":
    unittest.main()


class VirtualCases(unittest.TestCase):
    """tools/run-virtual-cases.py, the FFmpeg/tshark/memcached case.c corpora on the virtual profile.
    Inputs are the launcher's own line format and the drivers' printf lines."""
    rv = load(HERE / "run-virtual-cases.py", "run_virtual_cases")
    FAULT = ("capstone-exec: domain fault cause=28 pc=0x10234 address=0x55 entry=0x10000 "
             "code=0x10000-0x20000 last=0x4f")

    def run_text(self, n, extra=""):
        return f"case={n} arm=buggy\ncap=16 touched=17 extent=1 crossed=1 damage=0\n{extra}"

    def test_fault_at_a_prefixed_or_bare_probe_is_reached_and_attributed(self):
        for name in ("ffh_read_probe", "write_probe", "wsh_write_probe_u8"):
            o = self.rv.observe("3", self.run_text(3, self.FAULT), {"kind": "signal", "value": 11}, FakeSymbols(name))
            self.assertTrue(o.reached, name)
            self.assertEqual(o.attribution, "probe", name)

    def test_fault_at_a_declared_site_is_function(self):
        o = self.rv.observe("3", self.run_text(3, self.FAULT), {}, FakeSymbols("ff2_case_run"), ("ff2_case_run",))
        self.assertEqual((o.reached, o.attribution), (True, "function"))

    def test_fault_elsewhere_is_neither_reached_nor_attributed(self):
        o = self.rv.observe("3", self.run_text(3, self.FAULT), {}, FakeSymbols("memcpy"), ("ff2_case_run",))
        self.assertFalse(o.reached)
        self.assertIsNone(o.attribution)
        verdict = v.judge(o, {"controls": {}})
        self.assertEqual(verdict[:2], (v.NO_READING, "setup-fault"))

    def test_reproduced_and_exit_0_is_a_completed_reach(self):
        o = self.rv.observe("3", self.run_text(3, "VERDICT DEFECT-REPRODUCED x\n"), {"kind": "exit", "value": 0})
        self.assertEqual((o.reached, o.completed), (True, True))

    def test_exit_75_is_control_failed_not_a_silence(self):
        o = self.rv.observe("3", self.run_text(3, "CONTROL-FAILED 785\n"), {"kind": "exit", "value": 75})
        self.assertEqual(o.infra, "control-failed")

    def test_wrong_case_number_is_infra(self):
        o = self.rv.observe("4", self.run_text(3, "VERDICT DEFECT-REPRODUCED x\n"), {"kind": "exit", "value": 0})
        self.assertEqual(o.infra, "infra")

    def test_step_that_never_ended_is_infra(self):
        o = self.rv.observe("3", self.run_text(3) + "[runner] TIMEOUT\n", {"kind": "none"})
        self.assertEqual(o.infra, "infra")

    def test_fixed_arm_must_print_fixed_and_exit_0(self):
        ok = "case=3 arm=fixed\nVERDICT FIXED y\n"
        self.assertTrue(self.rv.fixed_ok("3", ok, {"kind": "exit", "value": 0}))
        self.assertFalse(self.rv.fixed_ok("3", ok, {"kind": "exit", "value": 1}))
        self.assertFalse(self.rv.fixed_ok("3", "case=3 arm=fixed\nVERDICT DEFECT-REPRODUCED\n", {"kind": "exit", "value": 0}))


class VirtualHosted(unittest.TestCase):
    """run-virtual-cases.py --prebuilt: the nested corpora's port-built programs (wmem first)."""
    rv = load(HERE / "run-virtual-cases.py", "run_virtual_cases_hosted")
    FAULT = VirtualCases.FAULT
    READY, DONE = "WM_DEFECT case={n} ready", "WM_DEFECT case={n} mode={mode} completed"
    CTL = {"controls": {"c": "fault"}, "controls_for_missed": ["c"]}

    def obs(self, text, result, symbol, differential=True, mode="0"):
        o = self.rv.observe_hosted("13", mode, text, result, FakeSymbols(symbol), "wm_probe",
                                   self.READY, self.DONE, differential)
        o.controls = [v.Control("c", "fault", "")]
        return o

    def test_fault_at_the_cases_probe_after_ready_is_caught(self):
        o = self.obs("WM_DEFECT case=13 ready\n" + self.FAULT, {"kind": "signal", "value": 11}, "wm_probe")
        self.assertEqual(v.judge(o, self.CTL)[0], v.CAUGHT)

    def test_planted_fault_off_the_probe_is_unattributed_not_caught(self):
        o = self.obs("WM_DEFECT case=13 ready\n" + self.FAULT, {"kind": "signal", "value": 11}, "wmem_block_free")
        self.assertEqual(v.judge(o, self.CTL)[:2], (v.NO_READING, "unattributed"))

    def test_the_other_probe_function_is_not_this_cases(self):
        o = self.obs("WM_DEFECT case=13 ready\n" + self.FAULT, {"kind": "signal", "value": 11}, "wm_write_probe")
        self.assertEqual(v.judge(o, self.CTL)[:2], (v.NO_READING, "unattributed"))

    def test_fault_before_ready_is_a_setup_fault(self):
        o = self.obs(self.FAULT, {"kind": "signal", "value": 11}, "wm_probe")
        self.assertEqual(v.judge(o, self.CTL)[:2], (v.NO_READING, "setup-fault"))

    def test_differential_reproduced_and_finished_is_missed(self):
        text = "WM_DEFECT case=13 ready\nWM_DEFECT case=13 mode=0 completed\nVERDICT DEFECT-REPRODUCED x\n"
        o = self.obs(text, {"kind": "exit", "value": 0}, None)
        self.assertEqual(v.judge(o, self.CTL)[0], v.MISSED)

    def test_differential_without_reproduced_is_not_a_miss(self):
        text = "WM_DEFECT case=13 ready\nWM_DEFECT case=13 mode=0 completed\nVERDICT INCONCLUSIVE\n"
        o = self.obs(text, {"kind": "exit", "value": 1}, None)
        self.assertEqual(v.judge(o, self.CTL)[:2], (v.NO_READING, "inconclusive"))

    def test_protected_run_that_finished_is_missed_and_one_that_did_not_is_not(self):
        done = "WM_DEFECT case=13 ready\nWM_DEFECT case=13 mode=1 completed\n"
        self.assertEqual(v.judge(self.obs(done, {"kind": "exit", "value": 0}, None, False, "1"), self.CTL)[0], v.MISSED)
        cut = "WM_DEFECT case=13 ready\n"
        self.assertEqual(v.judge(self.obs(cut, {"kind": "exit", "value": 0}, None, False, "1"), self.CTL)[:2],
                         (v.NO_READING, "inconclusive"))

    def test_exit_75_is_control_failed(self):
        o = self.obs("CONTROL-FAILED the virtual heap lent no payload\n", {"kind": "exit", "value": 75}, None)
        self.assertEqual(o.infra, "control-failed")

    def test_fixed_needs_finished_line_verdict_and_exit_0(self):
        ok = "WM_DEFECT case=13 mode=1 completed\nVERDICT FIXED y\n"
        self.assertTrue(self.rv.fixed_hosted("13", "1", ok, {"kind": "exit", "value": 0}, self.DONE))
        self.assertFalse(self.rv.fixed_hosted("13", "0", ok, {"kind": "exit", "value": 0}, self.DONE))
        self.assertFalse(self.rv.fixed_hosted("13", "1", ok, {"kind": "exit", "value": 1}, self.DONE))

    def test_cache_check_refuses_the_wrong_configuration_and_sdk(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            sdk = t / "sdk"
            sdk.mkdir()
            (t / "CMakeCache.txt").write_text(f"WM_SUBLET:BOOL=ON\nWM_CHUNKS:BOOL=ON\nCAPSTONE_SDK:PATH={sdk}\n"
                                              "PORT_PLATFORM:STRING=capstone-application\n")
            want = self.rv.HOSTED["wmem-repros"]["cache"]
            self.assertEqual(self.rv.cache_says(t, want["virtual-nested-pools"], sdk), "")
            self.assertIn("WM_LIBC_SYSTEM", self.rv.cache_says(t, want["virtual-malloc"], sdk))
            self.assertIn("CAPSTONE_SDK", self.rv.cache_says(t, want["virtual-nested-pools"], t))

    def test_stem_matches_the_ports_program_name(self):
        self.assertEqual(self.rv.stem(Path("13_0261fd7da6_http_range_cursor_past_chunk")),
                         "13-http-range-cursor-past-chunk")
