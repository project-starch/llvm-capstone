#!/usr/bin/env python3
"""Relink external licensed SPEC objects and stage verified original test inputs.

Source capstone/tests/capstone-test-env.sh first. Keep all benchmark artifacts
and the fresh output directory outside the repository.
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


BENCHMARKS = (
    "502.gcc_r", "519.lbm_r", "538.imagick_r", "544.nab_r",
    "997.specrand_fr", "999.specrand_ir", "505.mcf_r", "557.xz_r",
    "500.perlbench_r", "525.x264_r",
)


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def stage_path(stage, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe staged path: {name!r}")
    return stage / path


def parse_commands(path, label=None):
    commands = []
    for line in path.read_text().splitlines():
        if not line.lstrip().startswith("-o "):
            continue
        tokens = shlex.split(line)
        if (len(tokens) < 9 or tokens[0] != "-o" or tokens[2] != "-e"
                or tokens[-4:] not in ([">", tokens[1], "2>>", tokens[3]],
                                      [">", tokens[1], "2>", tokens[3]])):
            raise ValueError(f"Unexpected SPEC invocation in {path}: {line}")
        argv = tokens[4:-4]
        if argv and Path(argv[0]).name == "capstone-run.sh":
            argv = argv[1:]
        if not argv:
            raise ValueError(f"Missing SPEC executable in {path}")
        executable = re.fullmatch(r"([A-Za-z0-9_+-]+)_base\.([A-Za-z0-9_-]+)", Path(argv[0]).name)
        if not executable or (label is not None and executable[2] != label):
            raise ValueError(f"Unexpected SPEC executable in {path}: {argv[0]}")
        argv[0] = "./" + executable[1]
        commands.append({
            "argv": argv,
            "stdout": tokens[1], "stderr": tokens[3],
        })
    if not commands:
        raise ValueError(f"No benchmark invocations found in {path}")
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "builds", "sdk", "frozen", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--run-roots", type=Path, nargs="+", required=True,
        help="Existing official SPEC setup roots, searched in order",
    )
    parser.add_argument("--label", default="current_cap32")
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS,
                        help="Explicit subset; the default remains all ten entries")
    parser.add_argument("--require-frozen-objects", action="store_true",
                        help="Require and verify fresh object inventories before relinking")
    args = parser.parse_args()
    benchmarks = args.benchmarks if args.benchmarks is not None else BENCHMARKS
    if len(set(benchmarks)) != len(benchmarks):
        raise ValueError("Duplicate benchmark selection")
    spec, sdk, builds = args.spec.resolve(), args.sdk.resolve(), args.builds.resolve()
    compiler_environment = dict(os.environ)
    # The explicitly selected compiler driver must use its adjacent SDK;
    # an inherited override must not silently select another runtime.
    compiler_environment.pop("CAPSTONE_SDK", None)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for line in (spec / "MANIFEST").read_text().splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) == 4:
            manifest[fields[3]] = fields[0]

    def verify_original(path):
        relative = str(path.relative_to(spec))
        if relative not in manifest or digest(path, "sha512") != manifest[relative]:
            raise ValueError(f"SPEC MANIFEST mismatch: {relative}")

    def copy_original_tree(source, stage):
        copied = {}
        if source.exists():
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    verify_original(path)
                    relative = str(path.relative_to(source))
                    destination = stage_path(stage, relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
                    copied[relative] = {"source": str(path), "sha256": digest(path)}
        return copied

    frozen = json.loads(args.frozen.read_text())
    records = {}
    for bench in benchmarks:
        directory = output / bench
        stage = directory / "stage"
        stage.mkdir(parents=True, exist_ok=False)
        source_bench = "999.specrand_ir" if bench == "997.specrand_fr" else bench
        data = spec / "benchspec/CPU" / source_bench / "data"
        inputs = {}
        for subset in ("all/input", "test/input"):
            inputs.update(copy_original_tree(data / subset, stage))
        commands, links, objects, source_audit, binaries = [], [], {}, {}, {}

        if bench.startswith(("997.", "999.")):
            source = spec / "benchspec/CPU/999.specrand_ir/src"
            executable = bench.split(".", 1)[1]
            for name in ("main.c", "specrand-common/specrand.c", "specrand-common/specrand.h"):
                verify_original(source / name)
            command = [
                str(sdk), "-O2", "-DSPEC", "-DSPEC_CPU", "-DSPEC_LP64",
                "-I" + str(source / "specrand-common"), str(source / "main.c"),
                str(source / "specrand-common/specrand.c"), "-o", str(stage / executable),
            ]
            subprocess.run(command, check=True, env=compiler_environment)
            links.append(command)
            binaries[executable] = digest(stage / executable)
            for line in (stage / "control").read_text().splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    seed, count = line.split()
                    commands.append({
                        "argv": ["./" + executable, seed, count],
                        "stdout": f"rand.{count}.out", "stderr": f"rand.{count}.err",
                    })
        else:
            candidates = sorted([*(builds / bench).glob(f"build_base_{args.label}.*"),
                                 *(builds / bench / "build").glob(f"build_base_{args.label}.*")])
            if len(candidates) != 1:
                raise ValueError(f"Expected one build directory for {bench}: {candidates}")
            build = candidates[0]
            entry = frozen[bench]
            if entry.get("label", args.label) != args.label:
                raise ValueError(f"Frozen build label mismatch: {bench}")
            sources = entry["sources"]
            if not sources:
                raise ValueError(f"Missing frozen source inventory: {bench}")
            for name, expected in sources.items():
                if digest(stage_path(build, name), "sha512") != expected:
                    raise ValueError(f"Frozen source mismatch: {bench}/{name}")
            source_audit = {"files_checked": len(sources), "mismatches": []}
            expected_objects = entry.get("objects")
            if args.require_frozen_objects and not expected_objects:
                raise ValueError(f"Missing frozen object inventory: {bench}")
            if expected_objects is not None:
                if not expected_objects:
                    raise ValueError(f"Empty frozen object inventory: {bench}")
                for name, expected in expected_objects.items():
                    if digest(stage_path(build, name)) != expected:
                        raise ValueError(f"Frozen object mismatch: {bench}/{name}")
                source_audit["objects_checked"] = len(expected_objects)
            recorded_links = entry.get("link_commands")
            if recorded_links is None:
                recorded_links = []
                for log in sorted(build.glob("make*.out")):
                    for line in log.read_text().splitlines():
                        if "capstone-cc" in line and " -o " in line and " -c " not in line:
                            recorded_links.append(shlex.split(line))
            # Validate every recorded link's object set before any compiler call.
            for tokens in recorded_links:
                if (not tokens or not all(isinstance(t, str) for t in tokens)
                        or Path(tokens[0]).name != "capstone-cc" or "-o" not in tokens
                        or tokens.index("-o") + 1 >= len(tokens) or "-c" in tokens):
                    raise ValueError(f"Malformed frozen link command: {bench}")
                for name in tokens:
                    if name.endswith(".o"):
                        stage_path(build, name)
                        if expected_objects is not None and name not in expected_objects:
                            raise ValueError(f"Linked object missing from frozen inventory: {bench}/{name}")
            for tokens in recorded_links:
                executable = tokens[tokens.index("-o") + 1]
                if executable.endswith(".o") or executable == "ldecod_r":
                    continue
                destination = stage_path(stage, executable)
                destination.parent.mkdir(parents=True, exist_ok=True)
                command = [str(sdk), *tokens[1:]]
                command[command.index("-o") + 1] = str(destination)
                subprocess.run(command, cwd=build, check=True, env=compiler_environment)
                links.append(command)
                binaries[executable] = digest(destination)
                for name in tokens:
                    if name.endswith(".o"):
                        objects[name] = digest(build / name)
            if not links:
                raise ValueError(f"No link commands found: {bench}")
            runs = []
            for root in args.run_roots:
                runs = sorted(root.resolve().glob(
                    f"**/{bench}/run/run_base_test_{args.label}.*/speccmds.cmd"
                ))
                if runs:
                    break
            if len(runs) != 1:
                raise ValueError(f"Expected one official run setup for {bench}: {runs}")
            commands = parse_commands(runs[0], args.label)
            shutil.copy2(runs[0], directory / "speccmds.original")
            if bench == "525.x264_r":
                generated = runs[0].parent / "BuckBunny.yuv"
                control = stage / "control"
                expected = control.read_text().splitlines()[0].split()[2]
                if expected != "no_md5":
                    if not re.fullmatch(r"[0-9a-fA-F]{32}", expected):
                        raise ValueError(f"Unknown x264 input checksum policy: {expected!r}")
                    if digest(generated, "md5") != expected.lower():
                        raise ValueError("x264 generated-input MD5 mismatch")
                shutil.copy2(generated, stage / generated.name)
                inputs[generated.name] = {
                    "source": str(generated), "sha256": digest(generated),
                    "generated_input": True, "control_checksum": expected,
                    "control_sha256": digest(control), "official_run": str(runs[0].parent),
                    "official_command_sha256": digest(runs[0]),
                    "fresh_setup_validation": "PENDING",
                }

        references = copy_original_tree(data / "test/compare", stage)
        if not commands:
            raise ValueError(f"No test commands: {bench}")
        for command in commands:
            for stream in ("stdout", "stderr"):
                stage_path(stage, command[stream])
            if command["argv"][0].removeprefix("./") not in binaries:
                raise ValueError(f"Missing benchmark binary: {command['argv'][0]}")
        record = {
            "benchmark": bench, "commands": commands, "links": links,
            "objects": objects, "source_audit": source_audit, "inputs": inputs,
            "references": references, "binaries": binaries,
        }
        if not bench.startswith(("997.", "999.")) and entry.get("source_alternative"):
            record["source_alternative"] = entry["source_alternative"]
            record["frozen_inventory_sha256"] = digest(args.frozen)
        write_json(directory / "manifest.json", record)
        records[bench] = record
        print("PREPARED", bench, len(commands), flush=True)
    write_json(output / "manifest.json", records)


if __name__ == "__main__":
    main()
