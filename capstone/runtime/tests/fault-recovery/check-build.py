#!/usr/bin/env python3
"""Check resource accounting and reject invalid shared-domain build arguments."""

import json
from pathlib import Path
import subprocess
import tempfile

runtime = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix="capstone-domain-build-controls-") as tmp:
    root = Path(tmp)
    source = root / "source"
    source.mkdir()
    (source / "main.c").write_text("int main(void) { return 0; }\n")
    for case, enabled, args, valid in (
        ("default", None, "DATA_BYTES 4096 STACK_BYTES 2048", True),
        ("off", "OFF", "DATA_BYTES 4096 STACK_BYTES 2048", True),
        ("on", "ON", "DATA_BYTES 4096 STACK_BYTES 2048", True),
        ("missing", "ON", "STACK_BYTES 2048", False),
        ("short-data", "ON", "DATA_BYTES 1024 STACK_BYTES 2048", False),
        ("short-stack", "ON", "DATA_BYTES 4096 STACK_BYTES 16", False),
        ("typo", "ON", "DATA_BYTES 4096 STACK_BYTES 2048 TYPO 1", False),
        ("duplicate", "ON", "DATA_BYTES 4096 STACK_BYTES 2048", False),
    ):
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.25)\nproject(control C ASM)\n"
            "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)\n"
            f'add_subdirectory("{runtime}" runtime)\n'
            "add_executable(control main.c)\n"
            f"capstone_configure_domain(control {args})\n"
            + (
                f"capstone_configure_domain(control {args})\n"
                if case == "duplicate"
                else ""
            )
        )
        build = root / case
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
            ]
            + ([f"-DCAPSTONE_DOMAIN_FAULT_RECOVERY={enabled}"] if enabled else []),
            text=True,
            capture_output=True,
        )
        if (result.returncode == 0) != valid:
            raise SystemExit(
                f"Unexpected {case} configure result:\n{result.stdout}\n{result.stderr}"
            )
        if valid:
            commands = json.loads((build / "compile_commands.json").read_text())
            entry = [c for c in commands if c["file"].endswith("/start.S")]
            if len(entry) != 1:
                raise SystemExit("Expected exactly one shared entry")
            command = entry[0]["command"]
            expected = 4352 if enabled == "ON" else 4096
            if (
                f"-DCAPSTONE_DOMREQ_DATA={expected} " not in command
                or "-DCAPSTONE_DOMREQ_STACK=2048 " not in command
                or ("-DCAPSTONE_DOMAIN_FAULT_RECOVERY " in command) != (enabled == "ON")
            ):
                raise SystemExit(f"Incorrect resource/recovery definitions: {command}")
print("Shared domain build controls passed")
