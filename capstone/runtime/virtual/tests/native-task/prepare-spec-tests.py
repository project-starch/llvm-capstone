#!/usr/bin/env python3
"""Relink external licensed SPEC objects and stage verified original test inputs.

Source capstone/tests/capstone-test-env.sh first. Keep all benchmark artifacts
and the fresh output directory outside the repository.
"""

import argparse
import hashlib
import json
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


def parse_commands(path):
    commands = []
    for line in path.read_text().splitlines():
        if not line.startswith("-o "):
            continue
        if "capstone-run.sh " not in line:
            raise ValueError(f"Unexpected SPEC invocation in {path}: {line}")
        before, after = line.split("capstone-run.sh ", 1)
        prefix, argv = shlex.split(before), shlex.split(after)
        argv = argv[:argv.index(">")]
        argv[0] = "./" + Path(argv[0]).name.split("_base.")[0]
        commands.append({
            "argv": argv,
            "stdout": prefix[prefix.index("-o") + 1],
            "stderr": prefix[prefix.index("-e") + 1],
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
    args = parser.parse_args()
    spec, sdk, builds = args.spec.resolve(), args.sdk.resolve(), args.builds.resolve()
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
    for bench in BENCHMARKS:
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
            subprocess.run(command, check=True)
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
            candidates = sorted((builds / bench).glob(f"build_base_{args.label}.*"))
            if len(candidates) != 1:
                raise ValueError(f"Expected one build directory for {bench}: {candidates}")
            build = candidates[0]
            sources = frozen[bench]["sources"]
            if not sources:
                raise ValueError(f"Missing frozen source inventory: {bench}")
            for name, expected in sources.items():
                if digest(build / name, "sha512") != expected:
                    raise ValueError(f"Frozen source mismatch: {bench}/{name}")
            source_audit = {"files_checked": len(sources), "mismatches": []}
            for log in sorted(build.glob("make*.out")):
                for line in log.read_text().splitlines():
                    if "capstone-cc" not in line or " -o " not in line or " -c " in line:
                        continue
                    tokens = shlex.split(line)
                    executable = tokens[tokens.index("-o") + 1]
                    if executable.endswith(".o") or executable == "ldecod_r":
                        continue
                    destination = stage_path(stage, executable)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    command = [str(sdk), *tokens[1:]]
                    command[command.index("-o") + 1] = str(destination)
                    subprocess.run(command, cwd=build, check=True)
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
            commands = parse_commands(runs[0])
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
        write_json(directory / "manifest.json", record)
        records[bench] = record
        print("PREPARED", bench, len(commands), flush=True)
    write_json(output / "manifest.json", records)


if __name__ == "__main__":
    main()
