#!/usr/bin/env python3
"""Host regression tests for the SPEC qualification tools; no SPEC or QEMU needed.

Source capstone/tests/capstone-test-env.sh, then run this file with Python.
All data is synthetic and temporary under CAPSTONE_TMP_ROOT. Compiler, harness
setup and guest execution are mocked; the host output comparator is executed.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
BENCHMARK = "525.x264_r"  # Exercises the generated-input path with synthetic bytes.
SERIAL = "CASE_RC:0:0\nCASE_RC:1:0\nIMAGE_RC:0:0\nSPEC_ALL_GATE_EXIT:0\n"
EVIDENCE_FAILURES = (
    "missing_case", "duplicate_case", "reordered_cases", "failed_case",
    "missing_case_receipt", "failed_case_receipt", "missing_image", "duplicate_image",
    "failed_image", "missing_image_receipt", "failed_image_receipt",
    "missing_gate", "repeated_gate", "failed_gate", "missing_serial",
)


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(path, algorithm="sha256"):
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "dont_write_bytecode", True):
        spec.loader.exec_module(module)
    return module


class QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepare = load_helper("prepare-spec-tests")
        cls.setup = load_helper("setup-spec-validation")
        cls.runner = load_helper("run-spec-tests")
        cls.validator = load_helper("validate-spec-tests")

    def setUp(self):
        scratch = Path(os.environ.get("CAPSTONE_TMP_ROOT", str(Path(tempfile.gettempdir()) / "capstone")))
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="spec-qualification-", dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sequence = 0
        self.spec = self.root / "spec"
        self.data = self.spec / "benchspec/CPU" / BENCHMARK / "data"
        (self.data / "test/input").mkdir(parents=True)
        (self.data / "test/compare").mkdir()
        self.control = self.data / "test/input/control"
        self.control.write_text("synthetic.264 BuckBunny.yuv no_md5\n")
        self.reference = self.data / "test/compare/reference.img"
        self.reference.write_bytes(b"synthetic image reference\n")
        self.oracle = self.data / "test/compare/reference.txt"
        self.oracle.write_text("synthetic validator output\n")
        self.builds = self.root / "builds"
        build = self.builds / BENCHMARK / "build_base_current_cap32.0000"
        build.mkdir(parents=True)
        (build / "fixture.c").write_text("int synthetic_fixture;\n")
        (build / "fixture.o").write_bytes(b"synthetic object")
        (build / "make.out").write_text(
            "capstone-cc fixture.o -o x264_r\n"
            "capstone-cc fixture.o -o imagevalidate_525\n"
        )
        self.frozen = self.root / "frozen.json"
        write_json(self.frozen, {BENCHMARK: {"sources": {"fixture.c": digest(build / "fixture.c", "sha512")}}})
        self.existing = self.root / "existing"
        run = self.existing / BENCHMARK / "run/run_base_test_current_cap32.0000"
        run.mkdir(parents=True)
        self.generated = b"synthetic generated video\n"
        (run / "BuckBunny.yuv").write_bytes(self.generated)
        (run / "speccmds.cmd").write_text("".join(
            f"-o old-{i}.out -e old-{i}.err /synthetic/capstone-run.sh "
            f"../x264_r_base.current_cap32 case-{i} > old-{i}.out 2>> old-{i}.err\n"
            for i in range(2)
        ))
        self.sdk, self.decoder = self.root / "compiler", self.root / "decoder"
        self.sdk.write_bytes(b"compiler fixture, never executed")
        self.decoder.write_bytes(b"decoder fixture, never executed")
        self.write_manifest()
        self.prepared = self.prepare_fixture()
        self.official = self.setup_fixture() / "comparators.json"
        self.comparators = read_json(self.official)
        self.images = self.root / "images"
        self.images.mkdir()
        for name in ("Image", "rootfs.ext2"):
            (self.images / name).write_bytes(b"platform fixture")
        self.qemu, self.firmware = self.root / "qemu", self.root / "firmware"
        self.qemu.write_bytes(b"not an executable")
        self.firmware.write_bytes(b"firmware fixture")

    def fresh_path(self, name):
        self.sequence += 1
        return self.root / f"{name}-{self.sequence}"

    def invoke(self, helper, arguments, process=None):
        with patch.object(sys, "argv", [helper.__file__, *arguments]), redirect_stdout(io.StringIO()):
            if process is None:
                return helper.main()
            with patch.object(helper.subprocess, "run", side_effect=process):
                return helper.main()

    def write_manifest(self):
        (self.spec / "MANIFEST").write_text("".join(
            f"{digest(path, 'sha512')} 0 0 {path.relative_to(self.spec)}\n"
            for path in sorted(self.data.rglob("*")) if path.is_file()
        ))

    def prepare_fixture(self):
        output = self.fresh_path("prepared")

        def compile_fixture(command, **kwargs):
            self.assertEqual(command[0], str(self.sdk))
            Path(command[command.index("-o") + 1]).write_bytes(b"synthetic native binary")
            return subprocess.CompletedProcess(command, 0)

        with patch.object(self.prepare, "BENCHMARKS", (BENCHMARK,)):
            self.invoke(self.prepare, [
                "--spec", str(self.spec), "--builds", str(self.builds),
                "--sdk", str(self.sdk), "--frozen", str(self.frozen),
                "--out", str(output), "--run-roots", str(self.existing),
            ], compile_fixture)
        return output

    def setup_fixture(self, generated=None):
        output = self.fresh_path("official")

        def setup_harness(command, **kwargs):
            self.assertIn("--action=setup", command)
            run = output / "harness/benchspec/CPU" / BENCHMARK / "run/run_base_test_native_qemu.0000"
            run.mkdir(parents=True)
            (run / "BuckBunny.yuv").write_bytes(self.generated if generated is None else generated)
            (run / "speccmds.cmd").write_text("-E UNUSED unquoted'value\n" + "".join(
                f"-o new-{i}.out -e new-{i}.err ../x264_r_base.native_qemu case-{i} "
                f"> new-{i}.out 2>> new-{i}.err\n" for i in range(2)
            ))
            comparator = [sys.executable, str(Path(__file__).resolve()), "--compare", str(self.oracle), "validate.out"]
            (run / "compare.cmd").write_text(
                "-o validate.out -e validate.err ../imagevalidate_525_base.native_qemu "
                f"output.img {shlex.quote(str(self.reference))} > validate.out 2>> validate.err\n"
                + "-k failure -c " + shlex.join(comparator) + " > compare.log 2>&1\n"
            )
            return subprocess.CompletedProcess(command, 0)

        self.invoke(self.setup, [
            "--prepared", str(self.prepared), "--spec", str(self.spec),
            "--cc", str(self.sdk), "--ldecod", str(self.decoder), "--out", str(output),
        ], setup_harness)
        return output

    def damage_evidence(self, directory, failure):
        serial = directory / "vm/serial.log"
        replacements = {
            "missing_case": ("CASE_RC:1:0\n", ""),
            "duplicate_case": ("CASE_RC:1:0", "CASE_RC:0:0"),
            "reordered_cases": ("CASE_RC:0:0\nCASE_RC:1:0", "CASE_RC:1:0\nCASE_RC:0:0"),
            "failed_case": ("CASE_RC:1:0", "CASE_RC:1:7"),
            "missing_image": ("IMAGE_RC:0:0\n", ""),
            "duplicate_image": ("IMAGE_RC:0:0", "IMAGE_RC:0:0\nIMAGE_RC:0:0"),
            "failed_image": ("IMAGE_RC:0:0", "IMAGE_RC:0:7"),
            "missing_gate": ("SPEC_ALL_GATE_EXIT:0\n", ""),
            "repeated_gate": ("SPEC_ALL_GATE_EXIT:0", "SPEC_ALL_GATE_EXIT:0\nSPEC_ALL_GATE_EXIT:0"),
            "failed_gate": ("SPEC_ALL_GATE_EXIT:0", "SPEC_ALL_GATE_EXIT:7"),
        }
        if failure in replacements:
            serial.write_text(serial.read_text().replace(*replacements[failure]))
        elif failure == "missing_serial":
            serial.unlink()
        elif failure and failure.endswith("_receipt"):
            name = "case-1.rc" if "case" in failure else "image-0.rc"
            receipt = directory / "output" / name
            if failure.startswith("missing"):
                receipt.unlink()
            else:
                receipt.write_text("7\n")
        elif failure:
            self.fail(f"Unknown evidence mutation: {failure}")

    def execute(self, failure=None, returncode=0, must_refuse=False, shell=None, null_device=False,
                live_output=False):
        output = self.fresh_path("execution")

        def guest(command, **kwargs):
            if must_refuse:
                self.fail("Invalid input reached guest execution")
            self.assertEqual(Path(command[1]).name, "run-staged.py")
            vm = Path(command[command.index("--work") + 1])
            results = Path(command[command.index("--share") + 1])
            vm.mkdir()
            (vm / "serial.log").write_text(SERIAL)
            for name in ("case-0.rc", "case-1.rc", "image-0.rc"):
                (results / name).write_text("0\n")
            for name in ("new-0.out", "new-1.out", "new-0.err", "new-1.err", "validate.err"):
                (results / name).write_text("")
            (results / "validate.out").write_bytes(self.oracle.read_bytes())
            (results / "output.img").write_bytes(self.reference.read_bytes())
            self.damage_evidence(vm.parent, failure)
            return subprocess.CompletedProcess(command, returncode)

        result = self.invoke(self.runner, [
            "--prepared", str(self.prepared), "--qemu", str(self.qemu),
            "--images", str(self.images), "--firmware", str(self.firmware),
            "--comparators", str(self.official), "--out", str(output),
            *(["--shell", str(shell)] if shell else []),
            *(["--null-device"] if null_device else []),
            *(["--live-output"] if live_output else []),
        ], guest)
        return result, output / "runs" / BENCHMARK

    def validate(self, directory):
        report = self.fresh_path("report.json")
        result = self.invoke(self.validator, [
            "--results", str(directory.parent), "--comparators", str(self.official),
            "--spec", str(self.spec), "--out", str(report),
        ])
        return result, read_json(report)

    def update_prepared(self, record):
        write_json(self.prepared / "manifest.json", {BENCHMARK: record})
        write_json(self.prepared / BENCHMARK / "manifest.json", record)

    def bind_comparators(self, directory, comparators):
        # Keep completion metadata consistent to isolate the tested admission
        # condition, rather than failing only on a changed manifest digest.
        write_json(self.official, comparators)
        execution = read_json(directory / "execution.json")
        execution["comparator_manifest_sha256"] = digest(self.official)
        execution["image_commands"] = comparators[BENCHMARK]["image_compares"]
        write_json(directory / "execution.json", execution)

    def test_complete_pipeline_and_retained_evidence(self):
        status, directory = self.execute()
        self.assertEqual(status, 0)
        status, rows = self.validate(directory)
        self.assertEqual(status, 0)
        self.assertTrue(rows[0]["passed"])
        image = rows[0]["image_comparators"][0]
        self.assertEqual(image["stdout"]["text"], self.oracle.read_text())
        self.assertEqual(image["stderr"]["text"], "")
        self.assertEqual(image["reference_sha256"], image["staged_reference_sha256"])
        self.assertEqual(image["image_sha256"], digest(self.reference))
        execution = read_json(directory / "execution.json")
        self.assertEqual(execution["commands"], self.comparators[BENCHMARK]["commands"])
        self.assertEqual(execution["commands"][0]["stdout"], "new-0.out")

    def test_explicit_shell_preserves_workload_and_records_grants(self):
        shell = self.root / "shell.dom"
        shell.write_bytes(b"synthetic capability shell fixture")
        result, directory = self.execute(shell=shell, null_device=True)
        self.assertEqual(result, 0)
        self.assertEqual((directory / "stage/.native-shell").read_bytes(), shell.read_bytes())
        self.assertEqual(read_json(directory / "inputs.json")[".native-shell"], digest(shell))
        platform = read_json(directory.parent.parent / "platform.json")
        self.assertEqual(platform["shell"], digest(shell))
        self.assertTrue(platform["null_device_grant"])
        commands = read_json(directory / "execution.json")["commands"]
        self.assertEqual(commands, self.comparators[BENCHMARK]["commands"])
        launch = read_json(directory / "command.json")
        flags = launch[launch.index("--kernel-arg") + 1]
        self.assertIn("capstone_vm.dom_shell=/tmp/native-shell/hush.dom", flags)
        self.assertIn("capstone_vm.dom_null=1", flags)
        gate = (directory / "stage/gate.sh").read_text()
        self.assertLess(gate.index("rm /tmp/spec-native/.native-shell"), gate.index("CASE_BEGIN:0"))
        self.assertEqual(self.validate(directory)[0], 0)

    def test_live_output_is_optional_and_preserves_workload(self):
        result, ordinary = self.execute()
        self.assertEqual(result, 0)
        platform = read_json(ordinary.parent.parent / "platform.json")
        self.assertFalse(platform["live_output"])
        self.assertNotIn("native_spec_live_watch", (ordinary / "stage/gate.sh").read_text())

        result, observed = self.execute(live_output=True)
        self.assertEqual(result, 0)
        platform = read_json(observed.parent.parent / "platform.json")
        self.assertTrue(platform["live_output"])
        self.assertIn("not performance measurements", platform["timing_note"])
        self.assertEqual(read_json(ordinary / "execution.json")["commands"],
                         read_json(observed / "execution.json")["commands"])
        self.assertEqual(read_json(ordinary / "execution.json")["image_commands"],
                         read_json(observed / "execution.json")["image_commands"])
        gate = (observed / "stage/gate.sh").read_text()
        for marker in ("case-0", "case-1", "image-0"):
            for stream in ("stdout", "stderr"):
                self.assertIn(f"/mnt/results/live/{marker}-{stream}", gate)
        self.assertEqual(self.validate(observed)[0], 0)

    def test_live_watcher_copies_only_outputs_and_reaps(self):
        # Run the actual generated observer in the host shell, using synthetic
        # files. Poll for its first snapshot, then ensure periodic updates occur
        # before stopping it. Its child must not inherit the gate's EXIT trap.
        source_out, source_err = self.root / "out space", self.root / "err'quote"
        source_out.write_text("first\n")
        source_err.write_text("error\n")
        unrelated = self.root / "private-input"
        unrelated.write_text("must not be copied\n")
        live = self.root / "live"
        live.mkdir()
        args = shlex.join([str(source_out), str(source_err),
                           str(live / "stdout"), str(live / "stderr")])
        script = "set -eu\ntrap 'echo PARENT_EXIT' EXIT\n" + self.runner.live_output_functions()
        script += (
            f"native_spec_live_watch {args} &\nnative_spec_live_pid=$!\n"
            "observed_pid=$native_spec_live_pid\n"
            f"while [ ! -f {shlex.quote(str(live / 'stdout'))} ]; do sleep 0.01; done\n"
            f"printf 'second\\n' >> {shlex.quote(str(source_out))}\n"
            f"while [ \"$(wc -l < {shlex.quote(str(live / 'stdout'))})\" -lt 2 ]; do sleep 0.01; done\n"
            "native_spec_live_stop\n"
            "if kill -0 \"$observed_pid\" 2>/dev/null; then exit 91; fi\n"
            "test -z \"$native_spec_live_pid\"\n"
        )
        process = subprocess.Popen(["sh", "-c", script], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            # A broken watcher must not escape the test process tree.
            import signal
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            self.fail("live watcher did not stop")
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(stdout, "PARENT_EXIT\n")
        self.assertEqual((live / "stdout").read_bytes(), source_out.read_bytes())
        self.assertEqual((live / "stderr").read_bytes(), source_err.read_bytes())
        self.assertEqual(sorted(p.name for p in live.iterdir()), ["stderr", "stdout"])
        self.assertEqual(unrelated.read_text(), "must not be copied\n")

    def test_shell_reserved_filename_cannot_replace_workload_input(self):
        shell = self.root / "shell.dom"
        shell.write_bytes(b"synthetic shell")
        (self.prepared / BENCHMARK / "stage/.native-shell").write_bytes(b"original input")
        with self.assertRaisesRegex(ValueError, "Reserved shell"):
            self.execute(must_refuse=True, shell=shell)

    def test_generated_input_and_checksum_policies(self):
        record = read_json(self.prepared / "manifest.json")[BENCHMARK]
        self.assertEqual(record["inputs"]["BuckBunny.yuv"]["control_checksum"], "no_md5")
        self.assertTrue(self.comparators[BENCHMARK]["generated_inputs"]["BuckBunny.yuv"]["matched"])
        valid = hashlib.md5(self.generated).hexdigest()
        for checksum, accepted in ((valid, True), ("0" * 32, False), ("unknown", False)):
            with self.subTest(checksum=checksum):
                self.control.write_text(f"synthetic.264 BuckBunny.yuv {checksum}\n")
                self.write_manifest()
                if accepted:
                    self.prepare_fixture()
                else:
                    with self.assertRaises(ValueError):
                        self.prepare_fixture()

    def test_preparation_rejects_changed_original_input(self):
        self.control.write_text("changed without updating original MANIFEST\n")
        with self.assertRaisesRegex(ValueError, "MANIFEST mismatch"):
            self.prepare_fixture()

    def test_setup_rejects_different_generated_input(self):
        with self.assertRaisesRegex(ValueError, "generated-input mismatch"):
            self.setup_fixture(generated=b"different generated bytes")

    def test_runner_rejects_incomplete_execution(self):
        for failure in EVIDENCE_FAILURES:
            with self.subTest(failure=failure):
                status, directory = self.execute(failure)
                self.assertEqual(status, 1)
                self.assertFalse(read_json(directory / "execution.json")["execution_ok"])

    def test_validator_independently_checks_completion_evidence(self):
        for failure in EVIDENCE_FAILURES:
            with self.subTest(failure=failure):
                _, directory = self.execute()
                self.damage_evidence(directory, failure)
                status, rows = self.validate(directory)
                self.assertEqual(status, 1)
                self.assertFalse(rows[0]["passed"])

    def test_runner_failure_is_nonzero(self):
        status, directory = self.execute(returncode=7)
        self.assertEqual(status, 1)
        self.assertEqual(read_json(directory / "execution.json")["runner_rc"], 7)

    def test_wrong_official_argv_is_rejected_before_execution(self):
        changed = deepcopy(self.comparators)
        changed[BENCHMARK]["commands"][0]["argv"].append("changed-workload")
        write_json(self.official, changed)
        with self.assertRaisesRegex(ValueError, "command mismatch"):
            self.execute(must_refuse=True)

    def test_wrong_executed_argv_cannot_validate(self):
        _, directory = self.execute()
        execution = read_json(directory / "execution.json")
        execution["commands"][0]["argv"].append("changed-workload")
        write_json(directory / "execution.json", execution)
        self.assertEqual(self.validate(directory)[0], 1)

    def test_false_command_match_cannot_validate(self):
        _, directory = self.execute()
        changed = deepcopy(self.comparators)
        changed[BENCHMARK]["command_match"] = False
        self.bind_comparators(directory, changed)
        self.assertEqual(self.validate(directory)[0], 1)

    def test_stale_output_is_rejected_before_execution(self):
        (self.prepared / BENCHMARK / "stage/validate.out").write_bytes(self.oracle.read_bytes())
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.execute(must_refuse=True)

    def test_missing_benchmark_has_explicit_failure_row(self):
        _, directory = self.execute()
        changed = deepcopy(self.comparators)
        changed["999.missing_r"] = deepcopy(changed[BENCHMARK])
        self.bind_comparators(directory, changed)
        status, rows = self.validate(directory)
        self.assertEqual(status, 1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["passed"])
        self.assertFalse(rows[1]["passed"])
        self.assertTrue(rows[1]["execution_errors"])

    def test_missing_or_wrong_output_cannot_validate(self):
        for name, content in (("validate.out", None), ("validate.out", b"wrong output"), ("output.img", None), ("validate.err", None)):
            with self.subTest(name=name, content=content):
                _, directory = self.execute()
                output = directory / "output" / name
                if content is None:
                    output.unlink()
                else:
                    output.write_bytes(content)
                self.assertEqual(self.validate(directory)[0], 1)

    def test_legacy_prepared_references_are_rejected(self):
        record = read_json(self.prepared / "manifest.json")[BENCHMARK]
        del record["references"]
        self.update_prepared(record)
        with self.assertRaisesRegex(ValueError, "image-reference provenance"):
            self.execute(must_refuse=True)
        with self.assertRaisesRegex(ValueError, "image-reference provenance"):
            self.setup_fixture()

    def test_missing_or_wrong_official_image_hash_is_rejected(self):
        for expected in (None, "0" * 64):
            with self.subTest(expected=expected):
                write_json(self.official, self.comparators)
                _, directory = self.execute()
                changed = deepcopy(self.comparators)
                image = changed[BENCHMARK]["image_compares"][0]
                if expected is None:
                    del image["reference_sha256"]
                else:
                    image["reference_sha256"] = expected
                self.bind_comparators(directory, changed)
                status, rows = self.validate(directory)
                self.assertEqual(status, 1)
                self.assertFalse(rows[0]["execution_errors"])
                self.assertFalse(rows[0]["image_comparators"][0]["passed"])
                with self.assertRaisesRegex(ValueError, "image-reference"):
                    self.execute(must_refuse=True)

    def test_actual_staged_reference_hash_is_required(self):
        for fault in ("missing_inventory", "missing_reference", "wrong_hash"):
            with self.subTest(fault=fault):
                _, directory = self.execute()
                path = directory / "inputs.json"
                hashes = read_json(path)
                if fault == "missing_inventory":
                    path.unlink()
                else:
                    if fault == "missing_reference":
                        del hashes["reference.img"]
                    else:
                        hashes["reference.img"] = "0" * 64
                    write_json(path, hashes)
                self.assertEqual(self.validate(directory)[0], 1)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--compare":
        # A deterministic host comparator, not SPEC code or a guest binary.
        equal = Path(sys.argv[2]).read_bytes() == Path(sys.argv[3]).read_bytes()
        print("specdiff run completed" if equal else "synthetic output mismatch")
        raise SystemExit(0 if equal else 1)
    unittest.main()
