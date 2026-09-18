#!/usr/bin/env python3
"""Collect paired spatial/Sublet QEMU profiles and matched uninstrumented controls."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from analyze import analyze


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tables(run):
    rows = [
        line.strip()
        for line in (
            run
            / (
                "share/payload.log"
                if (run / "share/payload.log").exists()
                else "serial.log"
            )
        )
        .read_text()
        .splitlines()
        if line.startswith("|")
    ]
    if not rows:
        raise ValueError(f"Missing replay counts: {run}")
    return rows


def retryable_boot_failure(directory):
    verdicts = list(directory.glob("*/verdict.json"))
    if len(verdicts) != 1:
        return False
    verdict = json.loads(verdicts[0].read_text())
    serial = (verdicts[0].parent / "serial.log").read_text(errors="replace")
    log = (directory / "runner.log").read_text(errors="replace")
    return (
        verdict["runner_exit"] == 75
        and "__CAPSTONE_INFRA_FLAKE__ phase=boot-login" in log
        and "PG: dom=" not in serial
    )


def loader_stopped_before_sharing(directory):
    # This permits an explicit retry, never an automatic one. The loader has
    # created regions, but its trace-loaded and shared markers are absent.
    verdicts = list(directory.glob("*/verdict.json"))
    if len(verdicts) != 1:
        return False
    verdict = json.loads(verdicts[0].read_text())
    serial = (verdicts[0].parent / "serial.log").read_text(errors="replace")
    log = (directory / "runner.log").read_text(errors="replace")
    lines = serial.splitlines()
    return (
        verdict["runner_exit"] == 1
        and "pexpect.exceptions.TIMEOUT" in log
        and bool(lines)
        and lines[-1].startswith("PG: r3=")
        and "PG: trace bytes=" not in serial
        and "PG: shared" not in serial
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain_build", type=Path)
    parser.add_argument("linux_build", type=Path)
    parser.add_argument("--reference-build", type=Path, required=True)
    parser.add_argument("--trace", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse completed runs; retain every failed attempt",
    )
    parser.add_argument(
        "--retry-loader-setup",
        action="store_true",
        help="On resume, explicitly retry a timeout before trace/arena sharing; retain the failed attempt",
    )
    args = parser.parse_args()
    if args.retry_loader_setup and not args.resume:
        parser.error("--retry-loader-setup requires --resume")
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    traces = {}
    for value in args.trace:
        label, path = value.split("=", 1)
        if not re.fullmatch(r"[a-z0-9-]+", label) or label in traces:
            parser.error("trace names must be unique lowercase labels")
        traces[label] = Path(path).resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    port = Path(__file__).resolve().parents[1]
    runner = port / "capstone/run-qemu.py"
    manifest = dict(
        scope="QEMU allocator memory behaviour; no FPGA or security experiments",
        arms={
            "replay-spatial": "spatial Capstone ABI with first-fit backing",
            "replay-sublet": "Sublet AllocSet port with delegated subpools",
        },
        comparison="Complete port comparison; backing policy and layout differ",
        repetitions=args.repetitions,
        traces={
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in traces.items()
        },
        tool_sha256={
            p.name: digest(p)
            for p in (Path(__file__), port / "memory-profile/analyze.py", runner)
        },
        node_capacity=int(os.environ.get("CAPSTONE_REV_NODES", "1048576")),
        results=[],
    )

    if args.resume:
        previous = json.loads((args.output / "manifest.json").read_text())
        for key in (
            "scope",
            "arms",
            "comparison",
            "repetitions",
            "traces",
            "node_capacity",
        ):
            if previous[key] != manifest[key]:
                raise SystemExit(f"Cannot resume with changed {key}")
        previous.setdefault("resume_tool_sha256", []).append(manifest["tool_sha256"])
        manifest = previous
        for entry in manifest["results"]:
            paths = list((args.output / entry["directory"]).glob("*/manifest.json"))
            if len(paths) == 1:
                entry["artifacts"] = str(paths[0].parent.relative_to(args.output))

    def save():
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    def run(build, label, trace, program, repetition, profile):
        previous = [
            entry
            for entry in manifest["results"]
            if (
                entry["workload"],
                entry["program"],
                entry["repetition"],
                entry["profile"],
            )
            == (label, program, repetition, profile)
        ]
        expected = {
            "domain.dom": digest(build / "bin" / (program + ".dom")),
            "loader.user": digest(args.linux_build / "bin/domain-loader"),
            "trace.a11": digest(trace),
        }
        for entry in previous:
            if entry["runner_exit"] != 0:
                directory = args.output / entry["directory"]
                if retryable_boot_failure(directory):
                    entry["failure_phase"] = "boot-login"
                elif args.retry_loader_setup and loader_stopped_before_sharing(
                    directory
                ):
                    entry["failure_phase"] = "loader-setup-before-share"
                    entry["explicit_setup_retry"] = True
                else:
                    raise SystemExit(
                        "Cannot resume past this failure; inspect it and use a new campaign"
                    )
                continue
            artifacts = args.output / entry["artifacts"]
            identity = json.loads((artifacts / "manifest.json").read_text())
            verdict = json.loads((artifacts / "verdict.json").read_text())
            if (
                not verdict["passed"]
                or identity["sha256"] != expected
                or identity["qemu_sha256"]
                != digest(Path(os.environ["CAPSTONE_QEMU_BINARY"]))
                or any(
                    digest(artifacts / "share" / name) != value
                    for name, value in expected.items()
                )
            ):
                raise SystemExit(f"Cannot reuse changed or incomplete run: {artifacts}")
            if profile:
                analyze(artifacts)
                if digest(artifacts / "memory.csv") != entry["memory_csv_sha256"]:
                    raise SystemExit(f"Completed profile changed: {artifacts}")
            print(f"Verified completed run: {artifacts.name}", flush=True)
            return artifacts

        # Only a boot-login failure may retry automatically. A replay failure
        # remains a campaign failure, even if a later attempt might pass.
        for attempt in range(len(previous) + 1, 4):
            name = f"{label}-{program}-{repetition}-{'profile' if profile else 'reference'}-attempt-{attempt}"
            directory = args.output / "runs" / name
            directory.mkdir(parents=True)
            command = [
                sys.executable,
                str(runner),
                str(build.resolve()),
                str(args.linux_build.resolve()),
                program,
                str(trace),
                "--results",
                str(directory.resolve()),
                "--separate-scratch",
            ]
            if profile:
                command.append("--memory-profile")
            print(f"Running {name}", flush=True)
            with (directory / "runner.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            paths = list(directory.glob("*/manifest.json"))
            entry = dict(
                workload=label,
                program=program,
                repetition=repetition,
                profile=profile,
                runner_exit=result.returncode,
                directory=str(directory.relative_to(args.output)),
            )
            manifest["results"].append(entry)
            if len(paths) == 1:
                entry["artifacts"] = str(paths[0].parent.relative_to(args.output))
            save()
            if result.returncode or len(paths) != 1:
                if retryable_boot_failure(directory):
                    entry["failure_phase"] = "boot-login"
                    save()
                    continue
                raise SystemExit(f"Run failed: {directory}")
            artifacts = paths[0].parent
            if profile:
                summary = analyze(artifacts)
                entry["memory_csv_sha256"] = digest(artifacts / "memory.csv")
                print(
                    f"  payload peak={summary['oracle']['payload_peak_bytes']}, "
                    f"backing peak={summary['peaks']['backing_peak']['backing_bytes']}, "
                    f"nodes={summary['final']['nodes_created_cumulative']}",
                    flush=True,
                )
            save()
            return artifacts
        raise SystemExit(f"Boot retry budget exhausted: {label}/{program}/{repetition}")

    save()
    for label, trace in traces.items():
        references = {
            program: run(args.reference_build, label, trace, program, 0, False)
            for program in ("replay-spatial", "replay-sublet")
        }
        hashes = {}
        for repetition in range(1, args.repetitions + 1):
            programs = ["replay-spatial", "replay-sublet"]
            if repetition % 2 == 0:
                programs.reverse()
            for program in programs:
                artifacts = run(
                    args.domain_build, label, trace, program, repetition, True
                )
                if tables(artifacts) != tables(references[program]):
                    raise SystemExit(
                        f"Instrumentation changed replay/allocator counts: {artifacts}"
                    )
                current = digest(artifacts / "memory.csv")
                if program in hashes and hashes[program] != current:
                    raise SystemExit(f"Repeated memory profile differs: {artifacts}")
                hashes[program] = current
    manifest["complete"] = True
    manifest["instrumentation_counts_match"] = True
    manifest["repeated_profiles_identical"] = True
    save()
    print(f"Completed QEMU memory campaign: {args.output}", flush=True)


if __name__ == "__main__":
    main()
