#!/usr/bin/env python3
"""Validate PoisonCap pymalloc lifetimes and an optional native recording."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parents[4] / "common/host/cheribsd/run.py"
MAGIC = 0x31594C50524D5950


def validate_replay(result, native, mode):
    if len(result) != 96 or len(native) != 96 or mode not in (0, 1):
        raise ValueError("invalid replay header size or mode")
    h, n = struct.unpack("<12Q", result), struct.unpack("<12Q", native)
    # Arena layout, retention and metadata are ABI/adapter measurements.
    fields = (0, 1, 3, 4, 5, 6, 7, 11)
    if (
        h[0] != MAGIC
        or h[2] != mode
        or h[3]
        or not h[1]
        or h[1] != h[4]
        or any(h[i] != n[i] for i in fields)
    ):
        raise ValueError("pymalloc replay differs from native logical oracle")
    return dict(
        mode=mode,
        logical_fields_match=True,
        events=h[4],
        output_sha256=hashlib.sha256(result).hexdigest(),
        native_sha256=hashlib.sha256(native).hexdigest(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("build", type=Path)
    p.add_argument("output", type=Path)
    for name in ("sdk", "rootfs", "image"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--port", type=int, default=10430)
    p.add_argument("--disable-default-revocation", action="store_true")
    p.add_argument("--stage", choices=("platform", "pools", "replay"), default="pools")
    p.add_argument("--recording", type=Path)
    p.add_argument("--native-report", type=Path)
    p.add_argument("--reuse-rounds", type=int, default=8)
    p.add_argument("--replay-timeout", type=int, default=7200)
    p.add_argument("--case", action="append", default=[])
    a = p.parse_args()
    if not 1 <= a.reuse_rounds <= 2000 or a.replay_timeout < 1:
        p.error("reuse rounds must be 1..2000 and timeout must be positive")
    if a.stage == "replay" and (not a.recording or not a.native_report):
        p.error("replay requires --recording and --native-report")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    bins = a.build.resolve() / "bin"
    cases = []
    for name in ("live", "read", "write", "reuse", "reused-read"):
        fault = name in ("read", "write", "reused-read")
        cases.append(
            dict(
                name="poison-" + name,
                program=str(bins / "poisoncap-probe"),
                args=[name],
                exit=162 if fault else 0,
                expect=f"POISONCAP {name} " + ("READY" if fault else "PASS"),
            )
        )
    if a.stage != "platform":
        cases.append(
            dict(
                name="allocator-example",
                program=str(bins / "allocator-example"),
                expect="ALLOCATOR_EXAMPLE cpython PASS pointer_bytes=16",
                timeout=600,
            )
        )
        for name in (
            "sizes",
            "realloc",
            "snapshot-failure",
            "unwritten-reuse",
            "arena-turnover",
        ):
            cases.append(
                dict(
                    name="check-" + name,
                    program=str(bins / "allocator-checks"),
                    args=[name],
                    expect=f"PYM_CHECK {name} PASS pointer_bytes=16",
                    timeout=7200,
                )
            )
        for mode in (0, 1):
            for test in range(9):
                fixture = a.output / f"case-{test}.bin"
                fixture.write_bytes(
                    struct.pack("<12Q", MAGIC, 1, *([0] * 10))
                    + struct.pack("<4Q", 0, test, a.reuse_rounds if test == 8 else 0, 0)
                )
                fault = test == 5 or (mode == 1 and test not in (0, 3))
                rejected = mode == 1 and test == 3
                case = dict(
                    name=f"pool-{mode}-{test}",
                    program=str(bins / "pool-security"),
                    args=["input.bin", "output.bin", str(mode)],
                    inputs={"input.bin": str(fixture)},
                    timeout=1800,
                    exit=1 if rejected else 162 if fault else 0,
                )
                if rejected:
                    case["expect"] = "PYM_REJECT code=719"
                    case["also_expect"] = ["PYM_PROBE case=3 ready"]
                elif fault:
                    case["expect"] = f"PYM_PROBE case={test} ready"
                else:
                    case["expect_regex"] = (
                        r"PYM completed=1 alloc=0 free=0 realloc=0 arenas=\d+ released=\d+"
                    )
                    case["outputs"] = ["output.bin"]
                cases.append(case)
    if a.stage == "replay":
        for mode in (0, 1):
            cases.append(
                dict(
                    name=f"recording-{mode}",
                    program=str(bins / "replay"),
                    args=["input.bin", "output.bin", str(mode)],
                    inputs={"input.bin": str(a.recording.resolve())},
                    outputs=["output.bin"],
                    timeout=a.replay_timeout,
                    expect_regex=r"PYM completed=\d+ alloc=\d+ free=\d+ realloc=\d+ arenas=\d+ released=\d+",
                )
            )
    if a.case:
        unknown = set(a.case) - {c["name"] for c in cases}
        if unknown:
            p.error("unknown cases: " + ", ".join(sorted(unknown)))
        cases = [c for c in cases if c["name"] in a.case]
    (a.output / "selection.json").write_text(
        json.dumps(
            dict(
                stage=a.stage,
                complete_suite=not a.case,
                reuse_rounds=a.reuse_rounds,
                cases=[c["name"] for c in cases],
            ),
            indent=2,
        )
        + "\n"
    )
    casefile = a.output / "cases.json"
    casefile.write_text(json.dumps(cases, indent=2) + "\n")
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(a.output / "guest"),
            "--sdk",
            str(a.sdk),
            "--rootfs",
            str(a.rootfs),
            "--image",
            str(a.image),
            "--port",
            str(a.port),
            "--abi-probe",
            str(bins / "cheribsd-abi-probe"),
            "--runtime-revocation",
            "off",
            "--cases",
            str(casefile),
            *(["--disable-default-revocation"] if a.disable_default_revocation else []),
        ],
        check=True,
    )
    validations = []
    for mode in (0, 1):
        name = f"recording-{mode}"
        if not any(c["name"] == name for c in cases):
            continue
        result = (a.output / "guest" / name / "output.bin").read_bytes()
        native = a.native_report.read_bytes()
        validations.append(validate_replay(result, native, mode))
    if validations:
        (a.output / "replay-validation.json").write_text(
            json.dumps(validations, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
