#!/usr/bin/env python3
"""Validate every expected benchmark using official SPEC comparator commands.

--results is the runner's runs/ directory. Missing cases, guest image-validator
evidence, or official output files fail validation. No QEMU is launched here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def output_path(directory, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a relative output filename: {name!r}")
    return directory / path


def execution_errors(bench, execution, official, directory, comparator_hash):
    errors = []
    if execution.get("benchmark") != bench or execution.get("execution_ok") is not True:
        errors.append("Benchmark execution is missing or failed")
    if execution.get("runner_rc") != 0 or execution.get("execution_errors"):
        errors.append("Runner reported a failure")
    if official.get("command_match") is not True:
        errors.append("Official benchmark command match was not established")
    if not official["commands"] or not official["compares"]:
        errors.append("Empty official command or comparator inventory")
    # Older records did not include the command bodies. Their exact indexed
    # serial markers and status receipts are still required below.
    for key, expected in (("commands", official["commands"]), ("image_commands", official["image_compares"])):
        if key in execution and execution[key] != expected:
            errors.append(f"Executed {key} differ from official commands")
    if execution.get("comparator_manifest_sha256") not in (None, comparator_hash):
        errors.append("Execution used a different comparator manifest")
    serial = (directory / "vm/serial.log").read_text(errors="replace")
    for key, count_key, status_key, marker, stem in (
        ("commands", "expected_commands", "exit_statuses", "CASE_RC", "case"),
        ("image_compares", "expected_image_commands", "image_exit_statuses", "IMAGE_RC", "image"),
    ):
        count = len(official[key])
        if execution.get(count_key, 0) != count or execution.get(status_key, []) != [0] * count:
            errors.append(f"Missing or failed {marker} execution statuses")
        actual = re.findall(rf"^{marker}:(\d+):(\d+)\r?$", serial, re.MULTILINE)
        if [(int(index), int(status)) for index, status in actual] != [(i, 0) for i in range(count)]:
            errors.append(f"Missing, repeated, reordered, or failed {marker} serial markers")
        for index in range(count):
            receipt = directory / "output" / f"{stem}-{index}.rc"
            if not receipt.is_file() or receipt.read_text().strip() != "0":
                errors.append(f"Missing or failed status receipt: {receipt.name}")
    if re.findall(r"^SPEC_ALL_GATE_EXIT:(\d+)\r?$", serial, re.MULTILINE) != ["0"]:
        errors.append("Missing, repeated, or failed gate completion")
    if re.search(r"BUG:|Oops|Kernel panic", serial):
        errors.append("Kernel failure in serial log")
    return errors


def image_evidence(command, index, directory, staged_hashes):
    result = {"command": command, "index": index, "passed": False}
    try:
        receipt = directory / f"image-{index}.rc"
        result["returncode"] = int(receipt.read_text().strip())
        for stream in ("stdout", "stderr"):
            path = output_path(directory, command[stream])
            result[stream] = {
                "path": command[stream], "sha256": digest(path),
                "text": path.read_text(errors="replace"),
            }
        image = output_path(directory, command["argv"][-2])
        result["image_sha256"] = digest(image)
        result["reference_sha256"] = digest(Path(command["argv"][-1]))
        expected = command.get("reference_sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Missing official image-reference hash; fresh setup evidence is required")
        reference = command["argv"][-1]
        marker = "/data/test/compare/"
        name = reference.split(marker, 1)[1] if marker in reference else Path(reference).name
        output_path(directory, name)  # Validate the staged relative name too.
        result["staged_reference"] = name
        result["staged_reference_sha256"] = staged_hashes.get(name)
        if result["reference_sha256"] != expected or staged_hashes.get(name) != expected:
            raise ValueError("Official and actually staged image-reference hashes do not match")
        result["passed"] = result["returncode"] == 0
    except (OSError, ValueError, KeyError, IndexError) as error:
        result["reason"] = str(error)
    return result


def compare(command, directory, environment, timeout):
    result = {"command": command, "passed": False}
    try:
        if len(command) < 3:
            raise ValueError("Incomplete official comparator command")
        output = output_path(directory, command[-1])
        # Check both files before launching specdiff; missing references must
        # never be interpreted as successful empty diagnostic output.
        result["output_sha256"] = digest(output)
        result["reference_sha256"] = digest(Path(command[-2]))
        completed = subprocess.run(
            command, cwd=directory, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors="replace", timeout=timeout,
        )
        result.update(
            returncode=completed.returncode, output=completed.stdout,
            passed=completed.returncode == 0
            and completed.stdout.strip() in ("", "specdiff run completed"),
        )
    except subprocess.TimeoutExpired as error:
        diagnostic = error.stdout or ""
        if isinstance(diagnostic, bytes):
            diagnostic = diagnostic.decode(errors="replace")
        result.update(reason=f"Comparator timed out after {timeout}s", output=diagnostic)
    except (OSError, ValueError, KeyError, IndexError) as error:
        result["reason"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "comparators", "spec", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300, help="Seconds per host comparator")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    spec, results = args.spec.resolve(), args.results.resolve()
    comparators = json.loads(args.comparators.read_text())
    comparator_hash = digest(args.comparators)
    environment = dict(
        os.environ, SPEC=str(spec), SPECPERLLIB=str(spec / "bin/lib") + ":" + str(spec / "bin"),
    )
    rows = []
    for bench, official in comparators.items():
        row = {
            "benchmark": bench, "execution": None, "command_match": False,
            "execution_errors": [], "image_comparators": [], "comparators": [], "passed": False,
        }
        try:
            if Path(bench).name != bench:
                raise ValueError(f"Invalid benchmark name: {bench}")
            directory = results / bench
            execution = json.loads((directory / "execution.json").read_text())
            row["execution"] = execution
            row["command_match"] = official.get("command_match") is True
            row["execution_errors"] = execution_errors(
                bench, execution, official, directory, comparator_hash,
            )
            staged_hashes = json.loads((directory / "inputs.json").read_text()) if official["image_compares"] else {}
            row["image_comparators"] = [
                image_evidence(command, index, directory / "output", staged_hashes)
                for index, command in enumerate(official["image_compares"])
            ]
            row["comparators"] = [
                compare(command, directory / "output", environment, args.timeout)
                for command in official["compares"]
            ]
            row["passed"] = (
                not row["execution_errors"] and bool(row["comparators"])
                and all(check["passed"] for check in row["comparators"])
                and all(check["passed"] for check in row["image_comparators"])
            )
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            row["execution_errors"].append(str(error))
        rows.append(row)
        print(bench, "PASS" if row["passed"] else "FAIL", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    return 0 if rows and all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
