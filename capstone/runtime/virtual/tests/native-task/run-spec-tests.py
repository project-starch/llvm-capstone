#!/usr/bin/env python3
"""Execute every prepared SPEC test invocation on the native Linux task backend.

Official output validation is a separate step. Supply --comparators to execute
the official guest image validators too. Source the project test environment
first and keep prepared artifacts and the fresh output directory external.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def relative_path(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a relative staged filename: {name!r}")
    return path


def image_arguments(command):
    argv = list(command["argv"])
    argv[0] = "./" + Path(argv[0]).name.split("_base.")[0]
    reference = argv[-1]
    marker = "/data/test/compare/"
    argv[-1] = reference.split(marker, 1)[1] if marker in reference else Path(reference).name
    relative_path(argv[-1])
    relative_path(argv[-2])
    return argv


def status_records(log, marker, expected):
    matches = re.findall(rf"^{marker}:(\d+):(\d+)\r?$", log, re.MULTILINE)
    ids = [int(index) for index, _ in matches]
    statuses = [int(status) for _, status in matches]
    errors = []
    if ids != list(range(expected)):
        errors.append(f"{marker} IDs {ids} differ from expected {list(range(expected))}")
    if any(statuses):
        errors.append(f"{marker} contains failed commands: {statuses}")
    return statuses, errors


def receipt_errors(output, stem, statuses, expected):
    errors = []
    for index in range(expected):
        path = output / f"{stem}-{index}.rc"
        try:
            receipt = path.read_text().strip()
            if not receipt.isdecimal() or index >= len(statuses) or int(receipt) != statuses[index]:
                errors.append(f"Invalid or inconsistent status receipt: {path.name}")
        except OSError as error:
            errors.append(f"Missing status receipt {path.name}: {error}")
    return errors


def live_output_functions():
    # This observer runs in the scalar guest shell. It receives only the two
    # current output filenames; it never executes inside the native task.
    return """native_spec_live_pid=
native_spec_live_copy() {
    [ ! -f "$1" ] || cp -- "$1" "$3" 2>/dev/null || :
    [ ! -f "$2" ] || cp -- "$2" "$4" 2>/dev/null || :
}
native_spec_live_watch() {
    trap - EXIT
    native_spec_live_sleep_pid=
    trap 'trap - TERM INT HUP; if [ -n "$native_spec_live_sleep_pid" ]; then kill "$native_spec_live_sleep_pid" 2>/dev/null || :; fi; wait 2>/dev/null || :; exit 0' TERM INT HUP
    while :; do
        native_spec_live_copy "$@"
        sleep 1 &
        native_spec_live_sleep_pid=$!
        wait "$native_spec_live_sleep_pid" || :
        native_spec_live_sleep_pid=
    done
}
native_spec_live_stop() {
    if [ -n "$native_spec_live_pid" ]; then
        kill "$native_spec_live_pid" 2>/dev/null || :
        wait "$native_spec_live_pid" 2>/dev/null || :
        native_spec_live_pid=
    fi
}
"""


def gate_script(commands, images, compares, shell=False, live_output=False, local_tmp=False, host_window_case=None):
    if host_window_case is not None and not 0 <= host_window_case < len(commands):
        raise ValueError("Host window case is outside the command inventory")
    if host_window_case is not None and live_output:
        raise ValueError("Live output observer cannot run in a timing window")
    lines = [
        "#!/bin/sh", "set -eu",
        "trap 'echo SPEC_ALL_GATE_EXIT:$?; dmesg; echo VIRTUAL_STAGED_DONE' EXIT",
        "mkdir -p /mnt/results",
        "mount -t 9p -o trans=virtio,version=9p2000.L gate9p /mnt/results",
        "cp -a /mnt/vm /tmp/spec-native", "cd /tmp/spec-native",
        "chmod +x *_r *_fr *_ir imagevalidate_* 2>/dev/null || true",
        "[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status",
        "echo 0 > /proc/sys/kernel/randomize_va_space",
    ]
    if live_output:
        lines[2] = (
            "trap 'gate_rc=$?; native_spec_live_stop; "
            "echo SPEC_ALL_GATE_EXIT:$gate_rc; dmesg; echo VIRTUAL_STAGED_DONE' EXIT"
        )
        # Define cleanup before installing the EXIT trap, including early setup
        # failures. The background observer explicitly clears that parent trap.
        lines.insert(2, live_output_functions().rstrip())
        lines.append("mkdir -p /mnt/results/live")

    if shell:
        # Provision only the selected executable outside the benchmark's cwd
        # grant; no additional directory authority is implied by /bin/sh.
        lines.extend([
            "mkdir -p /tmp/native-shell",
            "cp /mnt/vm/.native-shell /tmp/native-shell/hush.dom",
            "chmod +x /tmp/native-shell/hush.dom",
            "rm /tmp/spec-native/.native-shell",
        ])

    if local_tmp:
        lines.append("mkdir -m 0700 .capstone-tmp")

    def retain(name, required=True):
        path = relative_path(name)
        target = Path("/mnt/results") / path
        lines.append("mkdir -p " + shlex.quote(str(target.parent)))
        copy = "cp " + shlex.quote(str(path)) + " " + shlex.quote(str(target))
        lines.append(copy if required else "[ ! -f " + shlex.quote(str(path)) + " ] || " + copy)

    def execute(command, index, marker, stem):
        for stream in ("stdout", "stderr"):
            parent = relative_path(command[stream]).parent
            lines.append("mkdir -p " + shlex.quote(str(parent)))
        live_arguments = shlex.join([
            command["stdout"], command["stderr"],
            f"/mnt/results/live/{stem}-{index}-stdout",
            f"/mnt/results/live/{stem}-{index}-stderr",
        ])
        if live_output:
            lines.extend([
                "native_spec_live_watch " + live_arguments + " &",
                "native_spec_live_pid=$!",
            ])
        measured = stem == "case" and index == host_window_case
        if measured:
            lines.append("echo HOST_WINDOW_BEGIN")
        lines.extend([
            "rc=0",
            shlex.join(command["argv"]) + " >" + shlex.quote(command["stdout"])
            + " 2>" + shlex.quote(command["stderr"]) + " || rc=$?",
        ])
        if measured:
            lines.append("echo HOST_WINDOW_END")
        if live_output:
            lines.extend([
                "native_spec_live_stop",
                "native_spec_live_copy " + live_arguments,
            ])
        lines.extend([
            f"echo {marker}:{index}:$rc",
            f'echo "$rc" > /mnt/results/{stem}-{index}.rc',
        ])
        retain(command["stdout"])
        retain(command["stderr"])

    for index, command in enumerate(commands):
        lines.append(f"echo CASE_BEGIN:{index}")
        execute(command, index, "CASE_RC", "case")
        lines.append(f"echo CASE_END:{index}")
    for index, image in enumerate(images):
        command = dict(image, argv=image_arguments(image))
        execute(command, index, "IMAGE_RC", "image")
        # Preserve the compared image as well as the validator's two streams.
        retain(command["argv"][-2], required=False)
    for command in compares:
        retain(command[-1], required=False)
    # Keep assembly and incidental diagnostics for execution-only runs too.
    lines.extend([
        'for f in *.out *.err *.s *.264 frame_*.yuv; do [ ! -f "$f" ] || cp "$f" /mnt/results/; done',
        "sync",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "out", "qemu", "images", "firmware"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--slots", type=int, default=1, choices=range(1, 33))
    parser.add_argument("--memory", default="8G")
    parser.add_argument("--comparators", type=Path, help="Official setup comparator manifest")
    parser.add_argument("--shell", type=Path, help="Explicit capability shell executable grant for native spawn")
    parser.add_argument("--null-device", action="store_true", help="Explicit pinned /dev/null grant")
    parser.add_argument("--live-output", action="store_true",
                        help="Diagnostic scalar watcher mirrors current stdout/stderr every guest second; not for performance measurements")
    parser.add_argument("--local-tmp", action="store_true", help="Map /tmp to a fresh private directory inside the existing workload grant")
    parser.add_argument("--no-icount", action="store_true", help="Use wall-clock virtual time for functional tests; no deterministic instruction counts")
    parser.add_argument("--host-window-case", type=int, help="Measure QEMU host CPU/wall time around one zero-based workload command, excluding validators and output collection")
    parser.add_argument("--host-cpu", type=int, help="Pin a measured run to one allowed host CPU")
    args = parser.parse_args()
    if args.host_window_case is not None and (args.host_window_case < 0 or args.live_output):
        parser.error("A timing window needs a nonnegative case and no live output observer")
    if args.host_cpu is not None:
        if args.host_window_case is None or args.host_cpu not in os.sched_getaffinity(0):
            parser.error("--host-cpu requires a timing window and an allowed CPU")
        os.sched_setaffinity(0, {args.host_cpu})
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    prepared, output = args.prepared.resolve(), args.out.resolve()
    records = json.loads((prepared / "manifest.json").read_text())
    comparators = json.loads(args.comparators.read_text()) if args.comparators else {}
    benchmarks = args.benchmarks or list(records)
    if not benchmarks or len(set(benchmarks)) != len(benchmarks):
        parser.error("Select a nonempty set of distinct benchmarks")
    if any(bench not in records or Path(bench).name != bench for bench in benchmarks):
        parser.error("Unknown or invalid benchmark name")
    if args.comparators and any(bench not in comparators for bench in benchmarks):
        parser.error("A selected benchmark is missing from --comparators")
    output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[5]
    platform = {
        name: digest(path) for name, path in (
            ("qemu", args.qemu), ("Image", args.images / "Image"),
            ("rootfs", args.images / "rootfs.ext2"), ("firmware", args.firmware),
        )
    }
    if args.shell:
        platform["shell"] = digest(args.shell)
    platform["null_device_grant"] = args.null_device
    platform["local_tmp_grant"] = args.local_tmp
    platform["live_output"] = args.live_output
    platform["icount"] = None if args.no_icount else 0
    platform["host_window_case"] = args.host_window_case
    if args.host_window_case is not None:
        platform["host_affinity"] = sorted(os.sched_getaffinity(0))
    if args.live_output:
        platform["timing_note"] = "Live output observer enabled; elapsed times are diagnostic, not performance measurements."
        print("run-spec-tests: --live-output is diagnostic; elapsed times are not performance measurements", file=sys.stderr)
    write_json(output / "platform.json", platform)
    rows = []
    for bench in benchmarks:
        directory = output / "runs" / bench
        directory.mkdir(parents=True, exist_ok=False)
        stage, result = directory / "stage", directory / "output"
        shutil.copytree(prepared / bench / "stage", stage)
        result.mkdir()
        record = json.loads((prepared / bench / "manifest.json").read_text())
        if record != records[bench]:
            raise ValueError(f"Prepared manifests disagree: {bench}")
        for entries in (record["inputs"], record.get("references", {}), record["binaries"]):
            for name, entry in entries.items():
                expected = entry if isinstance(entry, str) else entry["sha256"]
                if digest(stage / relative_path(name)) != expected:
                    raise ValueError(f"Prepared artifact hash mismatch: {bench}/{name}")
        official = comparators.get(bench)
        commands = official["commands"] if official else record["commands"]
        images = official["image_compares"] if official else []
        compares = official["compares"] if official else []
        if not commands:
            raise ValueError(f"No commands for {bench}")
        if official and (
            official.get("command_match") is not True
            or [item["argv"] for item in commands] != [item["argv"] for item in record["commands"]]
        ):
            raise ValueError(f"Official command mismatch: {bench}")
        for image in images:
            name = image_arguments(image)[-1]
            reference = stage / name
            expected = image.get("reference_sha256")
            provenance = record.get("references", {}).get(name, {})
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(f"Missing official image-reference hash: {bench}/{name}")
            if provenance.get("sha256") != expected:
                raise ValueError(f"Missing or mismatched prepared image-reference provenance: {bench}/{name}")
            if not reference.is_file():
                raise ValueError(f"Missing staged image reference: {reference}")
            if digest(reference) != expected:
                raise ValueError(f"Staged image reference mismatch: {reference}")
        # A fresh run must not inherit an old successful output from staging.
        expected_outputs = [command[-1] for command in compares]
        for command in commands + images:
            expected_outputs.extend((command["stdout"], command["stderr"]))
        expected_outputs.extend(image["argv"][-2] for image in images)
        for name in expected_outputs:
            if (stage / relative_path(name)).exists():
                raise ValueError(f"Output already exists in prepared stage: {bench}/{name}")
        if args.local_tmp and ((stage / ".capstone-tmp").exists() or (stage / ".capstone-tmp").is_symlink()):
            raise ValueError("Reserved private temporary directory is already in the workload")
        if args.shell:
            if (stage / ".native-shell").exists():
                raise ValueError("Reserved shell staging filename is already in the workload")
            shutil.copy2(args.shell, stage / ".native-shell")
        write_json(directory / "inputs.json", {
            str(path.relative_to(stage)): digest(path)
            for path in sorted(stage.rglob("*")) if path.is_file()
        })
        (stage / "gate.sh").write_text(gate_script(
            commands, images, compares, shell=bool(args.shell), live_output=args.live_output, local_tmp=args.local_tmp,
            host_window_case=args.host_window_case,
        ))
        flags = (
            "capstone_vm.native_task=1 capstone_vm.native_io=1 capstone_vm.dom_binfmt=1 "
            "capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 "
            "capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1 "
            f"capstone_vm.supervisor_slots={args.slots} capstone_vm.native_refused_trace=1"
        )
        if args.shell:
            flags += " capstone_vm.dom_shell=/tmp/native-shell/hush.dom"
        if args.null_device:
            flags += " capstone_vm.dom_null=1"
        if args.local_tmp:
            flags += " capstone_vm.dom_local_tmp=1"
        command = [
            "python3", str(repository / "capstone/runtime/virtual/run-staged.py"),
            "--qemu", str(args.qemu.resolve()), "--images", str(args.images.resolve()),
            "--firmware", str(args.firmware.resolve()), "--stage", str(stage),
            "--work", str(directory / "vm"), "--share", str(result), "--memory", args.memory,
            "--disk-mib", "512", "--timeout", str(args.timeout),
            "--kernel-arg", flags,
        ]
        if not args.no_icount:
            command += ["--icount", "0"]
        if args.host_window_case is not None:
            command += ["--host-window"]
        write_json(directory / "command.json", command)
        environment = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA=(
            f"x-capstone-supervisor-slots={args.slots},x-pcrel=false,"
            "x-capstone-raw-moves=true,x-capstone-test-csrs=false,"
            "x-capstone-checked-copy=true,x-capstone-context-transport=true"
        ))
        print("RUN", bench, len(commands), flush=True)
        start, errors, returncode = time.monotonic(), [], None
        with (directory / "runner.log").open("w") as log:
            try:
                returncode = subprocess.run(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                ).returncode
            except OSError as error:
                errors.append(f"Runner launch failed: {error}")
        try:
            serial = (directory / "vm/serial.log").read_text(errors="replace")
        except OSError as error:
            serial = ""
            errors.append(f"Missing serial log: {error}")
        statuses, case_errors = status_records(serial, "CASE_RC", len(commands))
        image_statuses, image_errors = status_records(serial, "IMAGE_RC", len(images))
        errors.extend(case_errors + image_errors)
        errors.extend(receipt_errors(result, "case", statuses, len(commands)))
        errors.extend(receipt_errors(result, "image", image_statuses, len(images)))
        if returncode != 0:
            errors.append(f"Runner failed: {returncode}")
        if re.findall(r"^SPEC_ALL_GATE_EXIT:(\d+)\r?$", serial, re.MULTILINE) != ["0"]:
            errors.append("Missing, repeated, or failed gate completion")
        if re.search(r"BUG:|Oops|Kernel panic", serial):
            errors.append("Kernel failure in serial log")
        row = {
            "benchmark": bench, "runner_rc": returncode,
            "icount": None if args.no_icount else 0,
            "host_elapsed_seconds": time.monotonic() - start,
            "commands": commands, "expected_commands": len(commands), "exit_statuses": statuses,
            "image_commands": images, "expected_image_commands": len(images),
            "image_exit_statuses": image_statuses, "execution_errors": errors,
            "execution_ok": not errors, "spec_validation": "PENDING",
            "comparator_manifest_sha256": digest(args.comparators) if args.comparators else None,
        }
        if args.host_window_case is not None:
            try:
                row["host_window"] = json.loads((directory / "vm/host-window.json").read_text())
                row["host_window"]["case"] = args.host_window_case
                row["host_window"]["affinity"] = sorted(os.sched_getaffinity(0))
                row["host_window"]["scope"] = "One workload invocation including exec/exit; no warm-up, boot, validators or host output collection. Host time, not guest cycles."
            except (OSError, ValueError) as error:
                errors.append(f"Missing or invalid timing window: {error}")
                row["execution_ok"] = False
        rows.append(row)
        write_json(directory / "execution.json", row)
        write_json(output / "execution.json", rows)
        print(json.dumps(row), flush=True)
    return 0 if all(row["execution_ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
