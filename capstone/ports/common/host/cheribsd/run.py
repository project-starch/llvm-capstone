#!/usr/bin/env python3
"""Run direct-link allocator examples and optional replay cases in CheriBSD."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
COMPONENTS = json.loads((HERE / "components.json").read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_case(case):
    if not case.get("name") or bool(case.get("expect")) == bool(
        case.get("expect_regex")
    ):
        raise ValueError("case needs a name and a success marker")
    if case.get("expect_regex"):
        re.compile(case["expect_regex"])
    if not isinstance(case.get("args", []), list):
        raise ValueError("case arguments must be an array")
    for name in [case["name"], *case.get("inputs", {}), *case.get("outputs", [])]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ValueError("case and guest file names must be single path components")
    if case.get("exit", 0) not in (0, 162):
        raise ValueError("supported outcomes are completion or CheriBSD SIGPROT")


def outcome_matches(case, result):
    lines = result.stdout.splitlines()
    marker = (
        case["expect"] in lines
        if "expect" in case
        else any(re.fullmatch(case["expect_regex"], line) for line in lines)
    )
    return result.returncode == case.get("exit", 0) and marker


def main():
    from guest import Guest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--rootfs", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument(
        "--build", action="append", default=[], metavar="COMPONENT=DIRECTORY"
    )
    p.add_argument(
        "--cases", type=Path, help="Additional JSON list of replay/client cases"
    )
    p.add_argument("--abi-probe", type=Path, help="Required for a cases-only suite")
    p.add_argument("--port", type=int, default=10427)
    p.add_argument("--runtime-revocation", choices=("off", "on"), default="off")
    a = p.parse_args()
    cases = []
    for item in a.build:
        component, path = item.split("=", 1)
        if a.abi_probe is None:
            a.abi_probe = Path(path).resolve() / "bin/cheribsd-abi-probe"
        for binary, marker in COMPONENTS[component]["examples"].items():
            cases.append(
                dict(
                    name=f"{component}-{binary}",
                    program=str(Path(path).resolve() / "bin" / binary),
                    expect=marker,
                )
            )
    if a.cases:
        cases.extend(json.loads(a.cases.read_text()))
    if a.abi_probe is None:
        p.error("provide --build or --abi-probe to verify the target ABI/runtime")
    cases[:0] = [
        dict(
            name="cheribsd-abi",
            program=str(a.abi_probe),
            expect="CHERI_ABI pointer_bytes=16 runtime_revocation="
            + str(int(a.runtime_revocation == "on")),
        ),
        dict(
            name="cheribsd-bounds",
            program=str(a.abi_probe),
            args=["oob"],
            expect="CHERI_BOUNDARY_READY",
            exit=162,
        ),
    ]
    if not cases or len({c["name"] for c in cases}) != len(cases):
        p.error("provide cases with unique names")
    for case in cases:
        validate_case(case)
        if not Path(case["program"]).is_file():
            p.error("missing program: " + case["program"])
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema="cheribsd-allocator-cases-v1",
        status="running",
        runtime_revocation=a.runtime_revocation,
        results=[],
        cases=[c["name"] for c in cases],
        binaries={c["name"]: digest(c["program"]) for c in cases},
    )
    report["platform"] = {
        name: digest(path)
        for name, path in {
            "qemu": a.sdk / "bin/qemu-system-riscv64cheri",
            "firmware": a.sdk / "share/qemu/bbl-riscv64cheri-virt-fw_jump.bin",
            "kernel": a.rootfs / "boot/kernel/kernel",
            "libc": a.rootfs / "lib/libc.so.7",
            "image": a.image,
        }.items()
    }

    def save():
        (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    lock = Path(
        os.environ.get("CAPSTONE_QEMU_LOCK", Path.home() / ".capstone-locks/qemu.lock")
    )
    lock.parent.mkdir(parents=True, exist_ok=True)
    guest = None
    with lock.open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            guest = Guest(
                a.sdk.resolve(), a.rootfs.resolve(), a.image.resolve(), a.output, a.port
            )
            (a.output / "qemu-command.json").write_text(
                json.dumps(guest.argv, indent=2) + "\n"
            )
            print("BOOT CheriBSD allocator suite", flush=True)
            guest.start()
            for case in cases:
                name = case["name"]
                run = a.output / name
                run.mkdir()
                remote = "/tmp/allocator-tests/" + name
                if guest.ssh("mkdir -p " + shlex.quote(remote)).returncode:
                    raise RuntimeError("cannot prepare guest case directory")
                guest.copy(case["program"], f"root@127.0.0.1:{remote}/program")
                inputs = {}
                for target, source in case.get("inputs", {}).items():
                    guest.copy(source, f"root@127.0.0.1:{remote}/{target}")
                    inputs[target] = digest(source)
                policy = (
                    "_RUNTIME_REVOCATION_"
                    + ("ENABLE" if a.runtime_revocation == "on" else "DISABLE")
                    + "=1"
                )
                command = (
                    "cd "
                    + shlex.quote(remote)
                    + "; ulimit -c 0; env "
                    + policy
                    + " "
                    + shlex.join(["./program", *case.get("args", [])])
                )
                # Keep a shell waiting for the child. OpenSSH cannot transport
                # CheriBSD's SIGPROT as a portable SSH exit-signal name.
                command += '; program_status=$?; exit "$program_status"'
                result = guest.ssh(command, timeout=case.get("timeout", 120))
                (run / "stdout.txt").write_text(result.stdout)
                (run / "stderr.txt").write_text(result.stderr)
                passed = outcome_matches(case, result)
                row = dict(
                    name=name,
                    passed=passed,
                    exit=result.returncode,
                    expected_exit=case.get("exit", 0),
                    marker=case.get("expect", case.get("expect_regex")),
                    stdout_sha256=digest(run / "stdout.txt"),
                    inputs=inputs,
                    outputs={},
                )
                if passed:
                    for filename in case.get("outputs", []):
                        guest.copy(
                            f"root@127.0.0.1:{remote}/{filename}", run / filename
                        )
                        row["outputs"][filename] = digest(run / filename)
                report["results"].append(row)
                save()
                print(("PASS " if passed else "FAIL ") + name, flush=True)
                if not passed:
                    raise RuntimeError("case failed: " + name)
            report["status"] = "complete"
        except BaseException as exc:
            report["status"] = "failed"
            report["error"] = type(exc).__name__
            raise
        finally:
            if guest is not None:
                guest.close()
            save()


if __name__ == "__main__":
    main()
