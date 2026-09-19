#!/usr/bin/env python3
"""Run PoisonCap platform controls before FFmpeg lease tests and a replay."""

import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parents[4] / "common/host/cheribsd/run.py"
MAGIC = 0x4650465452433032


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("build", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--rootfs", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--port", type=int, default=10428)
    p.add_argument("--stage", choices=("platform", "pool", "replay"), default="pool")
    p.add_argument("--recording", type=Path)
    p.add_argument("--native-report", type=Path)
    p.add_argument(
        "--disable-default-revocation",
        action="store_true",
        help="Also disable the guest's automatic libc policy for SSH/helper processes",
    )
    p.add_argument(
        "--case", action="append", default=[], help="Diagnostic subset: exact case name"
    )
    a = p.parse_args()
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
                expect=f"POISONCAP {name} " + ("READY" if fault else "PASS"),
                exit=162 if fault else 0,
            )
        )
    if a.stage != "platform":
        cases.append(
            dict(
                name="allocator-example",
                program=str(bins / "allocator-example"),
                expect="ALLOCATOR_EXAMPLE ffmpeg PASS pointer_bytes=16",
            )
        )
        for mode in (0, 2):
            for test in (0, 1, 2, 3, 4, 5, 6, 8, 9, 13):
                header = [0] * 16
                header[0], header[1], header[12], header[13] = MAGIC, 1, test, 8
                fixture = a.output / f"case-{test}.bin"
                fixture.write_bytes(struct.pack("<16Q", *header) + bytes(128))
                fault = mode == 2 and test not in (0, 13)
                case = dict(
                    name=f"pool-{mode}-{test}",
                    program=str(bins / "pool-security"),
                    args=["input.bin", "output.bin", str(mode)],
                    inputs={"input.bin": str(fixture)},
                    exit=162 if fault else 0,
                )
                if fault:
                    case["expect"] = f"FF2_PROBE case={test} ready"
                else:
                    case["expect_regex"] = (
                        r"FF2 status=0 events=1 metadata=[0-9]+ payload=[0-9]+"
                    )
                    case["outputs"] = ["output.bin"]
                cases.append(case)
    if a.stage == "replay":
        cases.append(
            dict(
                name="recording",
                program=str(bins / "replay"),
                timeout=600,
                args=["input.bin", "output.bin", "2"],
                inputs={"input.bin": str(a.recording.resolve())},
                outputs=["output.bin"],
                expect_regex=r"FF2 status=0 events=[0-9]+ metadata=[0-9]+ payload=[0-9]+",
            )
        )
    casefile = a.output / "cases.json"
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
                cases=[c["name"] for c in cases],
            ),
            indent=2,
        )
        + "\n"
    )
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
    if any(c["name"] == "recording" for c in cases):
        result = (a.output / "guest/recording/output.bin").read_bytes()
        native = a.native_report.read_bytes()
        if len(result) < 128 or len(native) < 128:
            raise ValueError("short replay report")
        h, n = struct.unpack("<16Q", result[:128]), struct.unpack("<16Q", native[:128])
        if h[11] != 2 or h[15] != 16:
            raise ValueError(
                "replay report is not PoisonCap mode 2 with purecap pointers"
            )
        if h[:3] != n[:3] or h[0] != MAGIC or h[2] != 0 or result[128:] != native[128:]:
            raise ValueError("PoisonCap replay differs from native event oracle")
        if len(result) != 128 * (1 + h[1]):
            raise ValueError("invalid replay report length")
        (a.output / "replay-validation.json").write_text(
            json.dumps(
                dict(
                    events=h[1],
                    exact_event_match=True,
                    sweeps=h[12],
                    poison_bytes=h[13],
                    snapshot_bytes=h[14],
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
