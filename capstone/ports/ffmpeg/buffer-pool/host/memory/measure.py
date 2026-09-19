#!/usr/bin/env python3
"""Collect fresh, paired allocator replays; report resources, never QEMU time."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from analyze import read, summary

PORT = Path(__file__).resolve().parents[2]
REPO = PORT.parents[3]
sys.path.insert(0, str(PORT.parents[1] / "common/host"))
from port_support import digest, write_json
from port_trace import record_trace


def observations(path, reference=None, mode=None):
    """Require complete, balanced observations and compare outcomes, not headers."""
    head, rows = read(path)
    if head[2] or not rows or rows[-1][0] != 8:
        raise ValueError("failed or incomplete replay")
    if mode is not None and head[11] != mode:
        raise ValueError("wrong replay protection mode")
    if reference is not None:
        expected_head, expected = read(reference)
        if expected_head[2] or rows != expected:
            raise ValueError("replay observations differ from native recording")
    current = {1: (0, 0), 2: (0, 0)}
    peak_live = peak_retained = peak_idle = 0
    timeline = []
    for index, row in enumerate(rows, 1):
        if row[1] in current:
            live, retained = row[11:13]
            if live > retained:
                raise ValueError("live payload exceeds pool backing")
            current[row[1]] = live, retained
        live = sum(v[0] for v in current.values())
        retained = sum(v[1] for v in current.values())
        peak_live = max(peak_live, live)
        peak_retained = max(peak_retained, retained)
        peak_idle = max(peak_idle, retained - live)
        timeline.append((index, live, retained, retained - live))
    if any(live or retained for live, retained in current.values()):
        raise ValueError("recording did not release its pool payload")
    result = summary(head, rows)
    # The wire's *_used fields are monotonic carve watermarks, not live memory.
    result["metadata_carved_bytes"] = result.pop("metadata_used")
    result["payload_carved_bytes"] = result.pop("payload_used")
    result["combined"] = {
        "peak_live_requested_bytes": peak_live,
        "peak_pool_backing_requested_bytes": peak_retained,
        "peak_idle_pool_requested_bytes": peak_idle,
        "final_live_requested_bytes": live,
        "final_pool_backing_requested_bytes": retained,
    }
    result["reuse_gap_histogram"] = {
        name: dict(
            sorted(
                Counter(
                    str(r[10]) for r in rows if r[1] == kind and r[0] == 130
                ).items()
            )
        )
        for kind, name in ((1, "buffer"), (2, "refstruct"))
    }
    return result, timeline


def check_resume(previous, current, collector):
    """Only the collector may change; retain its old identity in the snapshot."""
    for key in (
        "schema",
        "scope",
        "node_capacity",
        "workloads",
        "repetitions",
        "arms",
        "reservations_bytes",
    ):
        if json.dumps(previous[key]) != json.dumps(current[key]):
            raise ValueError(f"cannot resume with changed {key}")
    for path, expected in previous["fingerprints"].items():
        if path != str(collector) and digest(path) != expected:
            raise ValueError(f"cannot resume with changed artifact: {path}")
    # Checking that the old binary still exists is insufficient: a caller
    # could select a different build or emulator while leaving the old one intact.
    for path in current["fingerprints"].keys() - previous["fingerprints"].keys():
        candidate = Path(path)
        if not candidate.is_relative_to(
            collector.parent
        ) and not candidate.is_relative_to(collector.parents[2] / "tests/native"):
            raise ValueError(f"cannot resume with newly selected artifact: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native-build", required=True, type=Path)
    parser.add_argument("--domain-build", required=True, type=Path)
    parser.add_argument("--linux-build", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-reason", help="Required diagnosis authorizing a new attempt"
    )
    parser.add_argument(
        "--workload",
        action="append",
        required=True,
        help="LABEL=SECONDS,WIDTHxHEIGHT; records a fresh native workload",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    if args.resume != bool(args.resume_reason):
        parser.error("--resume and --resume-reason must be supplied together")
    import re

    workloads = []
    for item in args.workload:
        match = re.fullmatch(
            r"([a-z0-9-]+)=([1-9][0-9]*),([1-9][0-9]*x[1-9][0-9]*)", item
        )
        if not match or match[1] in [w[0] for w in workloads]:
            parser.error("invalid or duplicate workload")
        workloads.append(match.groups())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    builds = [
        p.resolve() for p in (args.native_build, args.domain_build, args.linux_build)
    ]
    for build in builds:
        cache = (build / "CMakeCache.txt").read_text()
        if f"CMAKE_HOME_DIRECTORY:INTERNAL={PORT}\n" not in cache:
            raise ValueError(f"build belongs to another source checkout: {build}")
    images = Path(os.environ["CAPSTONE_BUILDROOT_DIR"]) / "build/images"
    files = [
        builds[0] / "bin/replay",
        builds[0] / "ffmpeg/stock/ffmpeg",
        builds[0] / "ffmpeg/traced/ffmpeg",
        builds[1] / "bin/replay.dom",
        builds[2] / "bin/domain-loader",
        Path(os.environ["CAPSTONE_QEMU_BINARY"]),
        *(images / name for name in ("fw_jump.elf", "Image", "rootfs.ext2")),
        *(
            Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"]) / "bin" / name
            for name in ("clang", "ld.lld")
        ),
        *(build / "CMakeCache.txt" for build in builds),
    ]
    # Include sources and helpers, retaining an identity even before a commit.
    for directory in (PORT, PORT.parents[1] / "common", REPO / "capstone/runtime"):
        files.extend(
            p
            for p in directory.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and "results" not in p.parts
        )
    files.extend(
        [
            REPO / "capstone/tests/runtime-qemu/run-domain-smoke.py",
            REPO / "capstone/tests/capstone-test-env.sh",
        ]
    )
    fingerprints = {str(p): digest(p) for p in sorted(set(files))}
    manifest = {
        "schema": "capstone.ffmpeg-measurement/v1",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "scope": "serialized allocator replay; no application timing or full memory ledger",
        "node_capacity": int(os.environ["CAPSTONE_REV_NODES"]),
        "workloads": workloads,
        "repetitions": args.repetitions,
        "arms": ["spatial", "sublet"],
        "fingerprints": fingerprints,
        "results": [],
        "status": "running",
        "unknown": [
            "physical hardware node/tag bytes",
            "live node occupancy",
            "minimum payload/metadata capacity",
            "application cache and DRAM traffic",
        ],
    }
    header = (PORT / "src/shared/trace.h").read_text()
    manifest["reservations_bytes"] = {}
    for key, macro in (
        ("payload", "PAYLOAD"),
        ("metadata", "META"),
        ("trace_input", "FILE"),
        ("trace_output", "FILE"),
    ):
        match = re.search(
            rf"#define FF2_{macro}_BYTES \(([0-9]+)UL \* 1024 \* 1024\)", header
        )
        if not match:
            raise ValueError(
                "unsupported region definition; update reservation accounting"
            )
        manifest["reservations_bytes"][key] = int(match[1]) * 2**20
    commands = []
    if args.resume:
        previous = json.loads((out / "manifest.json").read_text())
        if previous["status"] != "failed":
            raise ValueError("only a stopped failed campaign may be resumed")
        check_resume(previous, manifest, Path(__file__).resolve())
        number = len(previous.get("resumptions", [])) + 1
        snapshot = out / f"manifest.before-resume-{number}.json"
        with snapshot.open("x") as stream:
            json.dump(previous, stream, indent=2)
        previous.setdefault("resumptions", []).append(
            {
                "reason": args.resume_reason,
                "previous_manifest_sha256": digest(snapshot),
                "collector_sha256": digest(__file__),
            }
        )
        fingerprints = previous["fingerprints"] | fingerprints
        manifest = previous
        manifest["fingerprints"] = fingerprints
        manifest["status"] = "running"
        manifest.pop("error", None)
        commands = json.loads((out / "commands.json").read_text())
    write_json(out / "manifest.json", manifest)

    def execute(command, log, env=None):
        commands.append([str(v) for v in command])
        write_json(out / "commands.json", commands)
        with log.open("w") as stream:
            subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True
            )

    def verify_identity():
        if any(digest(path) != value for path, value in fingerprints.items()):
            raise ValueError("campaign artifact changed during execution")

    try:
        for label, seconds, dimensions in workloads:
            recording = out / "recordings" / label
            recording.parent.mkdir(exist_ok=True)
            native = recording / "native.bin"
            if not recording.exists():
                execute(
                    ["bash", PORT / "host/record.sh", recording, seconds, dimensions],
                    out / f"record-{label}.log",
                    dict(os.environ, FFPOOL_NATIVE_BUILD_DIR=str(builds[0])),
                )
                record_trace(
                    recording, recording / "commands.bin", "ffmpeg.buffer-pool"
                )
                execute(
                    [builds[0] / "bin/replay", recording / "commands.bin", native, "0"],
                    recording / "native.log",
                )
            elif not args.resume:
                raise ValueError("unexpected existing recording")
            native_summary, _ = observations(native, recording / "recorded.bin", 0)
            write_json(recording / "native-summary.json", native_summary)
            for name in ("recorded.bin", "commands.bin", "native.bin", "workload.json"):
                path = recording / name
                fingerprints[str(path)] = digest(path)
            for repetition in range(1, args.repetitions + 1):
                for arm in ("spatial", "sublet"):
                    verify_identity()
                    earlier = [
                        r
                        for r in manifest["results"]
                        if (r["workload"], r["arm"], r["repetition"])
                        == (label, arm, repetition)
                    ]
                    passed = [r for r in earlier if r["status"] == "passed"]
                    if passed:
                        if len(passed) != 1:
                            raise ValueError("duplicate accepted measurement")
                        old = passed[0]
                        output = out / old["directory"] / "capstone.bin"
                        result, _ = observations(
                            output,
                            recording / "recorded.bin",
                            0 if arm == "spatial" else 2,
                        )
                        if (
                            digest(output) != old["output_sha256"]
                            or result != old["measurement"]
                        ):
                            raise ValueError("completed measurement changed")
                        print(f"Verified {label}-{arm}-{repetition}", flush=True)
                        continue
                    run = out / "runs" / f"{label}-{arm}-{repetition}"
                    if earlier:
                        run = run.with_name(run.name + f"-attempt-{len(earlier) + 1}")
                    run.mkdir(parents=True)
                    entry = dict(
                        workload=label,
                        arm=arm,
                        repetition=repetition,
                        directory=str(run.relative_to(out)),
                        status="running",
                    )
                    manifest["results"].append(entry)
                    write_json(out / "manifest.json", manifest)
                    print(f"Running {run.name}", flush=True)
                    execute(
                        [
                            sys.executable,
                            PORT / "host/run-qemu.py",
                            recording / "commands.bin",
                            run,
                            "--protection",
                            arm,
                            "--domain-build",
                            builds[1],
                            "--linux-build",
                            builds[2],
                        ],
                        run / "runner.log",
                    )
                    result, timeline = observations(
                        run / "capstone.bin",
                        recording / "recorded.bin",
                        0 if arm == "spatial" else 2,
                    )
                    write_json(run / "measurement.json", result)
                    with (run / "payload.csv").open("w") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(
                            [
                                "event",
                                "live_requested_bytes",
                                "pool_backing_requested_bytes",
                                "idle_pool_requested_bytes",
                            ]
                        )
                        writer.writerows(timeline)
                    entry.update(
                        status="passed",
                        native_events_exact=True,
                        measurement=result,
                        output_sha256=digest(run / "capstone.bin"),
                    )
                    write_json(out / "manifest.json", manifest)
        verify_identity()
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        if manifest["results"] and manifest["results"][-1]["status"] == "running":
            manifest["results"][-1]["status"] = "failed"
        raise
    finally:
        write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
