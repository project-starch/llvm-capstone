#!/usr/bin/env python3
"""Obtain original test commands and comparators from the licensed SPEC harness.

Source the project test environment and SPEC's shrc first. The only benchmark
code executed here is SPEC's ordinary host input generator. Keep the fresh
output directory outside the repository.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def invocation(tokens):
    if tokens[:1] != ["-o"] or tokens[2:3] != ["-e"]:
        raise ValueError(f"Unexpected official command layout: {tokens}")
    return {
        "argv": tokens[4:tokens.index(">")],
        "stdout": tokens[1], "stderr": tokens[3],
    }


def command_lines(path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.split()[0] in ("-E", "-r", "-N", "-C"):
            continue
        yield shlex.split(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "spec", "cc", "ldecod", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepared, spec, output = args.prepared.resolve(), args.spec.resolve(), args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = json.loads((prepared / "manifest.json").read_text())
    if not records:
        raise ValueError("The prepared benchmark inventory is empty")
    harness = output / "harness"
    for bench, record in records.items():
        destination = harness / "benchspec/CPU" / bench / "exe"
        destination.mkdir(parents=True)
        if not record["binaries"]:
            raise ValueError(f"Missing binaries: {bench}")
        for name, expected in record["binaries"].items():
            if Path(name).name != name:
                raise ValueError(f"Expected a binary filename: {name}")
            source = prepared / bench / "stage" / name
            if digest(source) != expected:
                raise ValueError(f"Prepared binary hash mismatch: {bench}/{name}")
            target = destination / (name + "_base.native_qemu")
            shutil.copy2(source, target)
            target.chmod(0o755)
        if bench == "525.x264_r":
            target = destination / "ldecod_r_base.native_qemu"
            shutil.copy2(args.ldecod, target)
            target.chmod(0o755)

    config = output / "setup.cfg"
    config.write_text(f"""label = native_qemu
reportable = 0
tune = base
iterations = 1
verify_binaries = 0
output_format = txt
output_root = {harness}
command_add_redirect = 1
intrate,fprate:
 copies = 1
default:
 CC = {args.cc.resolve()}
 CC_VERSION_OPTION = --version
 sw_base_ptrsize = 128-bit capabilities, 64-bit long
""")
    command = [
        str(spec / "bin/runcpu"), "--config=" + str(config),
        "--action=setup", "--nobuild", "--size=test", *records,
    ]
    write_json(output / "command.json", command)
    with (output / "setup.log").open("w") as log:
        subprocess.run(command, cwd=spec, stdout=log, stderr=subprocess.STDOUT, check=True)

    result = {}
    for bench, record in records.items():
        runs = sorted((harness / "benchspec/CPU" / bench / "run").glob(
            "run_base_test_native_qemu.*"
        ))
        if len(runs) != 1:
            raise ValueError(f"Expected one fresh official setup: {bench}: {runs}")
        run = runs[0]
        commands, compares, images = [], [], []
        for tokens in command_lines(run / "speccmds.cmd"):
            command = invocation(tokens)
            command["argv"][0] = "./" + Path(command["argv"][0]).name.split("_base.")[0]
            commands.append(command)
        # SPEC may include the configuration label in stdout/stderr filenames.
        # The benchmark argv must match exactly; the runner uses these fresh
        # official output names when a comparator manifest is supplied.
        if [item["argv"] for item in commands] != [item["argv"] for item in record["commands"]]:
            raise ValueError(f"Official benchmark command mismatch: {bench}")
        for tokens in command_lines(run / "compare.cmd"):
            if tokens[0] == "-k":
                compares.append(tokens[3:tokens.index(">")])
            elif tokens[0] == "-o":
                image = invocation(tokens)
                reference = image["argv"][-1]
                marker = "/data/test/compare/"
                name = reference.split(marker, 1)[1] if marker in reference else Path(reference).name
                expected = record.get("references", {}).get(name, {}).get("sha256")
                actual = digest(Path(reference))
                if expected != actual or digest(prepared / bench / "stage" / name) != actual:
                    raise ValueError(f"Missing or mismatched image-reference provenance: {bench}/{name}")
                image["reference_sha256"] = actual
                images.append(image)
            else:
                raise ValueError(f"Unknown official comparison command: {tokens}")
        if not commands or not compares or any(len(command) < 3 for command in compares):
            raise ValueError(f"Incomplete official command/comparator inventory: {bench}")

        generated_inputs = {}
        for name, entry in record["inputs"].items():
            if not entry.get("generated_input"):
                continue
            prepared_hash = digest(prepared / bench / "stage" / name)
            fresh_hash = digest(run / name)
            if prepared_hash != entry["sha256"] or prepared_hash != fresh_hash:
                raise ValueError(f"Fresh official generated-input mismatch: {bench}/{name}")
            generated_inputs[name] = {
                "prepared_sha256": prepared_hash, "official_sha256": fresh_hash,
                "matched": True,
            }
        if bench == "525.x264_r":
            if "BuckBunny.yuv" not in generated_inputs:
                raise ValueError("Missing x264 generated-input provenance")
            generated_inputs["BuckBunny.yuv"]["generator_sha256"] = digest(args.ldecod)
        result[bench] = {
            "commands": commands, "compares": compares, "image_compares": images,
            "command_match": True, "generated_inputs": generated_inputs,
            "official_command_sha256": digest(run / "speccmds.cmd"),
            "official_compare_sha256": digest(run / "compare.cmd"),
        }
    write_json(output / "comparators.json", result)


if __name__ == "__main__":
    main()
