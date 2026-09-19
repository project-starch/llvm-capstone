"""Wire corruption, adapter boundaries and all four launcher integrations."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HOST = Path(__file__).resolve().parents[1] / "host"
PORTS = HOST.parents[1]
sys.path.insert(0, str(HOST))
from port_trace import TraceError, TraceReader, inspect_trace


def pym(rows):
    return struct.pack("<12Q", 0x31594C50524D5950, len(rows), *([0] * 10)) + b"".join(
        struct.pack("<4Q", *r) for r in rows
    )


def ggml(rows):
    return struct.pack(
        "<16Q", 0x315854434C4D4747, len(rows), *([0] * 12), 32, 0
    ) + b"".join(struct.pack("<6Q", *r) for r in rows)


def pg(rows, *, parent=0, prefix=0):
    return struct.pack(
        "<8sIIQIIQII32s",
        b"A11TRACE",
        1,
        40,
        0x0102030405060708,
        7,
        parent,
        prefix,
        4,
        0,
        b"test",
    ) + b"".join(struct.pack("<IIIIQQQ", *r) for r in rows)


def ff(rows):
    return struct.pack("<16Q", 0x4650465452433032, len(rows), *([0] * 14)) + b"".join(
        struct.pack("<16Q", *(tuple(r) + (0,) * (16 - len(r)))) for r in rows
    )


FIXTURES = {
    "cpython.pymalloc": (pym, [(1, 0, 17, 9), (3, 0, 33, 7), (5, 1, 0, 0)], 32),
    # Borrowed descriptor dies while its buffer's object can remain live.
    "whisper.ggml-context": (
        ggml,
        [
            (1, 0, 3, 1024, 0, 0),
            (2, 0, 3, 16, 1, 42),
            (4, 0, 3, 0, 0, 0),
            (5, 0, 0, 0, 0, 0),
        ],
        48,
    ),
    "postgres.a11": (
        pg,
        [
            (6, 1, 99, 0, 0, 8192, 8192),
            (1, 1, 1, 0, 32, 0, 0),
            (5, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 3, 1, 1),
        ],
        40,
    ),
    "ffmpeg.buffer-pool": (
        ff,
        [
            (1, 1, 1, 1, 0, 0, 32),
            (129, 1, 1, 1, 0, 0, 32),
            (4, 1, 1, 2),
            (132, 1, 1, 2),
            (8,),
        ],
        128,
    ),
}


class Traces(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "trace.bin"

    def inspect(self, data, **options):
        self.path.write_bytes(data)
        return inspect_trace(self.path, **options)

    def test_all_formats_and_byte_identity(self):
        for name, (encode, rows, size) in FIXTURES.items():
            with self.subTest(format=name):
                data = encode(rows)
                result = self.inspect(data, replay=True, limit=2)
                self.assertEqual(result["trace"]["format"], name)
                self.assertEqual(result["trace"]["records"], len(rows))
                self.assertEqual(result["trace"]["record_bytes"], size)
                self.assertEqual(
                    result["trace"]["sha256"], hashlib.sha256(data).hexdigest()
                )
                self.assertEqual(self.path.read_bytes(), data)
                self.assertEqual(result["roles"]["end"], 1)
                self.assertIsNone(result["recording_provenance"])
                self.assertEqual(
                    result["validation"]["allocator_semantics"], "not-checked"
                )
                self.assertEqual(len(result["preview"]), 2)

    def test_corruption_matrix(self):
        for name, (encode, rows, size) in FIXTURES.items():
            variants = {
                "empty": b"",
                "short-magic": b"A11",
                "short-header": encode(rows)[:16],
                "partial-record": encode(rows)[:-1],
                "trailing-byte": encode(rows) + b"x",
                "missing-end": encode(rows[:-1]),
                "early-end": encode([rows[-1]] + rows),
                "duplicate-end": encode(rows + [rows[-1]]),
                "unknown-op": encode([(999, *rows[0][1:]), *rows[1:]]),
            }
            for fault, data in variants.items():
                with self.subTest(format=name, fault=fault), self.assertRaises(
                    TraceError
                ):
                    self.inspect(data)

    def test_wrong_port_and_huge_count(self):
        data = pym([(5, 0, 0, 0)])
        with self.assertRaisesRegex(TraceError, "expected postgres.a11"):
            self.inspect(data, expected_format="postgres.a11")
        data = data[:8] + struct.pack("<Q", (1 << 64) - 1) + data[16:]
        with self.assertRaisesRegex(TraceError, "count"):
            self.inspect(data)

    def test_postgres_version_endian_size_and_footer(self):
        data = pg([(0, 0, 0, 0, 0, 0, 0)])
        for offset, fmt, value in [
            (8, "<I", 2),
            (12, "<I", 80),
            (16, "<Q", 42),
            (40, "<I", 33),
            (44, "<I", 1),
            (96, "<Q", 99),
        ]:
            bad = bytearray(data)
            struct.pack_into(fmt, bad, offset, value)
            with self.subTest(offset=offset), self.assertRaises(TraceError):
                self.inspect(bad)

    def test_process_prefix_is_preserved_and_requires_flattening(self):
        data = pg([(0, 0, 0, 0, 0, 0, 0)], parent=5, prefix=20)
        self.assertEqual(self.inspect(data)["header"]["prefix"], 20)
        with self.assertRaisesRegex(TraceError, "flatten"):
            self.inspect(data, replay=True)

    def test_ffmpeg_outcomes_do_not_become_commands(self):
        data = ff([(2, 1, 1, 4, 3, 12, 32), (130, 1, 1, 4, 3, 12, 32, 0, 8, 1), (8,)])
        result = self.inspect(data, limit=3)
        outcome = result["preview"][1]
        self.assertEqual(outcome["role"], "observation")
        self.assertEqual(outcome["fields"]["object"], 12)
        self.assertEqual(outcome["fields"]["backing"], 8)
        self.assertEqual(outcome["fields"]["parent"], 3)
        with self.assertRaisesRegex(TraceError, "strip measured"):
            self.inspect(data, replay=True)

    def test_borrowed_release_and_realloc_keep_their_original_meaning(self):
        for name in ("whisper.ggml-context", "cpython.pymalloc"):
            encode, rows, _ = FIXTURES[name]
            result = self.inspect(encode(rows), replay=True, limit=10)
            self.assertEqual(len(result["preview"]), len(rows))
            self.assertEqual(
                [e["fields"]["op"] for e in result["preview"]], [r[0] for r in rows]
            )
        self.assertEqual(result["preview"][1]["fields"]["id"], 0)
        self.assertEqual(result["preview"][-1]["fields"]["id"], 1)

    def test_input_does_not_accept_result_header(self):
        for name in ("cpython.pymalloc", "whisper.ggml-context", "ffmpeg.buffer-pool"):
            encode, rows, _ = FIXTURES[name]
            data = bytearray(encode(rows))
            struct.pack_into("<Q", data, 16, 1)
            with self.subTest(format=name), self.assertRaisesRegex(
                TraceError, "report fields"
            ):
                self.inspect(data, replay=True)

    def test_footer_and_payload_field_rules(self):
        for data in [
            pym([(5, 0, 1, 0)]),
            pym([(1, 0, 8, 256), (5, 1, 0, 0)]),
            ggml([(1, 0, 0, 16, 0, 2), (5, 1, 0, 0, 0, 0)]),
            ggml([(5, 0, 1, 0, 0, 0)]),
            ff([(8, 1)]),
            ff([(1, 3), (8,)]),
        ]:
            with self.subTest(data=data[:8]), self.assertRaises(TraceError):
                self.inspect(data)

    def test_stream_requires_full_consumption_and_detects_change(self):
        data = pym(FIXTURES["cpython.pymalloc"][1])
        self.path.write_bytes(data)
        with TraceReader(self.path) as reader:
            it = reader.events()
            self.assertEqual(next(it).index, 0)
            with self.assertRaisesRegex(TraceError, "fully validated"):
                _ = reader.sha256
            with self.path.open("ab") as stream:
                stream.write(b"x")
            with self.assertRaisesRegex(TraceError, "grew"):
                list(it)
        self.assertFalse(reader.complete)

    def test_cli_validates_beyond_preview_and_emits_no_partial_success(self):
        self.path.write_bytes(pym([(1, 0, 8, 0), (999, 0, 0, 0), (5, 1, 0, 0)]))
        args = [
            sys.executable,
            "-m",
            "port_trace",
            "inspect",
            str(self.path),
            "--limit",
            "1",
        ]
        env = dict(os.environ, PYTHONPATH=str(HOST), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("record 1", json.loads(result.stderr)["error"])
        self.path.write_bytes(pym([(5, 0, 0, 0)]))
        for command in ("validate", "summary", "inspect"):
            args[3] = command
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["trace"]["records"], 1)

    def test_four_launchers_validate_staged_bytes_before_guest(self):
        cases = [
            ("cpython/pymalloc", "cpython.pymalloc"),
            ("whisper/ggml-context", "whisper.ggml-context"),
            ("ffmpeg/buffer-pool", "ffmpeg.buffer-pool"),
            ("postgres/memory-contexts", "postgres.a11"),
        ]
        for component, name in cases:
            with self.subTest(component=component):
                encode, rows, _ = FIXTURES[name]
                data = encode(rows)
                self.path.write_bytes(data)
                domain, linux = self.root / "domain", self.root / "linux"
                for build in (domain, linux):
                    (build / "bin").mkdir(parents=True, exist_ok=True)
                    (build / "regions.json").write_text(
                        json.dumps({"trace": 65536, "scratch": 65536})
                    )
                (domain / "bin/replay.dom").write_bytes(b"domain")
                (domain / "bin/replay-spatial.dom").write_bytes(b"domain")
                (linux / "bin/domain-loader").write_bytes(b"loader")
                qemu = self.root / "qemu"
                qemu.write_bytes(b"emulator")
                tools = self.root / "tools/bin"
                tools.mkdir(parents=True, exist_ok=True)
                (tools / "clang").write_bytes(b"compiler")
                output = self.root / component.replace("/", "-")

                def guest(run, *args, **kwargs):
                    metadata = json.loads((run / "trace.json").read_text())
                    self.assertEqual(
                        metadata["trace"]["sha256"], hashlib.sha256(data).hexdigest()
                    )
                    self.assertEqual(metadata["trace"]["format"], name)
                    if name in ("cpython.pymalloc", "whisper.ggml-context"):
                        words = 12 if name == "cpython.pymalloc" else 16
                        magic = struct.unpack_from("<Q", data)[0]
                        (run / "share/report.bin").write_bytes(
                            struct.pack(
                                f"<{words}Q",
                                magic,
                                len(rows),
                                0,
                                0,
                                len(rows),
                                *([0] * (words - 5)),
                            )
                        )
                    elif name == "ffmpeg.buffer-pool":
                        (run / "share/capstone.bin").write_bytes(data)
                    else:
                        (run / "serial.log").write_text(
                            "__CAPSTONE_PG_REPLAY_DONE__\n__CAPSTONE_PG_REPLAY_BALANCED__\n"
                        )
                    return subprocess.CompletedProcess([], 0)

                argv = [
                    str(PORTS / component / "host/run-qemu.py"),
                    str(self.path),
                    str(output),
                    "--domain-build",
                    str(domain),
                    "--linux-build",
                    str(linux),
                ]
                env = {
                    "CAPSTONE_QEMU_BINARY": str(qemu),
                    "CAPSTONE_LLVM_BUILD_DIR": str(tools.parent),
                }
                with patch.dict(os.environ, env), patch.object(
                    sys, "argv", argv
                ), patch(
                    "port_support.run_guest", side_effect=guest
                ) as mock_guest, redirect_stdout(
                    io.StringIO()
                ):
                    runpy.run_path(argv[0], run_name="__main__")
                    self.assertEqual(mock_guest.call_count, 1)
                verdict = json.loads(next(output.glob("*/verdict.json")).read_text())
                self.assertEqual(verdict["schema"], "capstone.replay-verdict/v1")
                self.assertTrue(verdict["passed"])
                self.assertEqual(
                    verdict["trace"]["sha256"], hashlib.sha256(data).hexdigest()
                )
                self.path.write_bytes(data[:-1])
                with patch.dict(os.environ, env), patch.object(
                    sys, "argv", argv
                ), patch("port_support.run_guest") as mock_guest, self.assertRaises(
                    TraceError
                ):
                    runpy.run_path(argv[0], run_name="__main__")
                mock_guest.assert_not_called()

    def test_failed_validation_cannot_be_promoted_to_passing_verdict(self):
        from port_support import write_replay_verdict

        (self.root / "trace.json").write_text(
            json.dumps({"schema": "capstone.trace-error/v1", "error": "partial record"})
        )
        with self.assertRaisesRegex(ValueError, "cannot pass"):
            write_replay_verdict(self.root, passed=True, runner_exit=0)
        self.assertFalse((self.root / "verdict.json").exists())
        write_replay_verdict(
            self.root, passed=False, runner_exit=1, required=["oracle"]
        )
        verdict = json.loads((self.root / "verdict.json").read_text())
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["required"], ["oracle"])
        self.assertIsNone(verdict["trace"])
