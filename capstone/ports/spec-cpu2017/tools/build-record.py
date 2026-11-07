#!/usr/bin/env python3
"""Select source packages and freeze fresh SPEC builds; never edit the base lock.

These records describe build inputs, not a native platform qualification.
Licensed sources and outputs remain in the caller's external workspace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def relative(name):
    p = Path(name)
    if p.is_absolute() or not p.parts or ".." in p.parts or any(c.isspace() for c in name):
        raise ValueError(f"Unsafe relative build path: {name!r}")
    return p


def identifier(name):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        raise ValueError(f"Unsafe identifier: {name!r}")
    return name


def sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a SHA-256 digest")
    return value


def select(manifest, overlay=None, benchmarks=None):
    base = json.loads(manifest.read_text())
    benchmarks = base["workloads"]["benchmarks"] if benchmarks is None else benchmarks
    if (not benchmarks or len(set(benchmarks)) != len(benchmarks)
            or any(b not in base["workloads"]["benchmarks"] for b in benchmarks)):
        raise ValueError("Invalid or duplicate benchmark selection")
    selected = {b: {"name": "capstone", "package": p,
                    "sha256": base["srcalt_sha256"][b]}
                for b, p in base["srcalt"].items()}
    result = {"schema_version": 1, "base_manifest_sha256": digest(manifest),
              "overlay_sha256": None, "benchmarks": list(benchmarks),
              "source_alternatives": selected}
    if overlay:
        change = json.loads(overlay.read_text())
        if set(change) != {"schema_version", "base_manifest_sha256", "source_alternatives"} or change["schema_version"] != 1:
            raise ValueError("Unsupported source-only overlay schema")
        if change["base_manifest_sha256"] != result["base_manifest_sha256"]:
            raise ValueError("Source overlay base-manifest mismatch")
        # The config exposes only the Perl selection. Do not silently install
        # an alternative which the config will never select.
        if set(change["source_alternatives"]) != {"500.perlbench_r"}:
            raise ValueError("The source overlay must select only 500.perlbench_r")
        item = change["source_alternatives"]["500.perlbench_r"]
        if set(item) != {"name", "package", "sha256", "replacement_sha256", "object_count"}:
            raise ValueError("Unsupported source-alternative fields")
        if type(item["object_count"]) is not int or item["object_count"] <= 0 or not item["replacement_sha256"]:
            raise ValueError("Source overlay requires replacement hashes and an object count")
        for name, value in item["replacement_sha256"].items():
            relative(name)
            sha256(value)
        selected["500.perlbench_r"] = item
        result["overlay_sha256"] = digest(overlay)
    for item in selected.values():
        identifier(item["name"])
        relative(item["package"])
        sha256(item["sha256"])
    result["source_alternatives"] = {b: item for b, item in selected.items() if b in benchmarks}
    return result


def sdk_inputs(directory):
    directory = directory.resolve()
    config = json.loads((directory / "sdk.json").read_text())
    files = {key: Path(config[key]).resolve() for key in
             ("cc", "ld", "runtime", "libc", "builtins", "script")}
    files.update(driver=directory / "capstone-cc", sdk_json=directory / "sdk.json")
    headers = {}
    for part in ("arch/capstone64", "arch/generic", "obj/include", "include"):
        root = Path(config["musl"]) / part
        if not root.is_dir():
            raise ValueError(f"SDK header directory missing: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                headers[str(path.relative_to(config["musl"]))] = digest(path)
    return {"configuration": config,
            "artifacts": {key: {"path": str(path), "sha256": digest(path)}
                          for key, path in files.items()}, "headers_sha256": headers}


def freeze(spec, label, benchmarks, selection, sdk, sdk_before):
    identifier(label)
    inputs = sdk_inputs(sdk)
    if inputs != sdk_before:
        raise ValueError("SDK inputs changed during the build")
    result = {}
    for bench in benchmarks:
        candidates = sorted((spec / "benchspec/CPU" / bench / "build").glob(f"build_base_{label}.*"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one fresh build for {bench}: {candidates}")
        build = candidates[0]
        compiles, links, logs = {}, [], {}
        for log in sorted(build.glob("make*.out")):
            logs[log.name] = digest(log)
            for line in log.read_text().splitlines():
                if "capstone-cc" not in line or " -o " not in line:
                    continue
                command = shlex.split(line)
                if Path(command[0]).name != "capstone-cc":
                    continue
                target = command[command.index("-o") + 1]
                relative(target)
                if "-c" in command:
                    compiles[target] = command
                else:
                    links.append(command)
        if not compiles or not links:
            raise ValueError(f"Missing fresh compile/link commands: {bench}")
        objects = {name: digest(build / relative(name)) for name in compiles}
        for command in links:
            for name in command:
                if name.endswith(".o") and name not in objects:
                    raise ValueError(f"Linked object has no recorded fresh compilation: {bench}/{name}")
        binaries = {command[command.index("-o") + 1]: digest(build / command[command.index("-o") + 1])
                    for command in links}
        source = selection["source_alternatives"].get(bench)
        if source:
            if "object_count" in source and len(objects) != source["object_count"]:
                raise ValueError(f"Unexpected fresh object count for {bench}: {len(objects)}")
            for name, expected in source.get("replacement_sha256", {}).items():
                if digest(build / relative(name)) != expected:
                    raise ValueError(f"Selected source replacement mismatch: {bench}/{name}")
        extensions = {".c", ".h", ".inc", ".S", ".s", ".cc", ".cpp", ".f", ".f90", ".f95", ".xs", ".tab", ".act"}
        sources = {str(path.relative_to(build)): digest(path, "sha512")
                   for path in sorted(build.rglob("*")) if path.is_file()
                   and (path.suffix in extensions or path.name.startswith("Makefile"))}
        if not sources:
            raise ValueError(f"No sources found for {bench}")
        result[bench] = {"sources": sources, "objects": objects, "binaries": binaries,
                         "compile_commands": compiles, "link_commands": links,
                         "make_log_sha256": logs, "source_alternative": source,
                         "source_selection": selection, "sdk_inputs": inputs,
                         "build_directory": str(build.relative_to(spec)), "label": label}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("select")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--overlay", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--plan", type=Path, required=True)
    s.add_argument("--benchmarks", nargs="+")
    s = sub.add_parser("sdk")
    s.add_argument("--sdk", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("freeze")
    for name in ("spec", "selection", "sdk", "sdk-before", "out"):
        s.add_argument("--" + name, type=Path, required=True)
    s.add_argument("--label", required=True)
    s.add_argument("--benchmarks", nargs="+", required=True)
    a = p.parse_args()
    if a.action == "select":
        result = select(a.manifest, a.overlay, a.benchmarks)
        write_json(a.out, result)
        with a.plan.open("x") as plan:
            for bench, item in result["source_alternatives"].items():
                plan.write(f"{bench}\t{item['package']}\t{item['sha256']}\t{item['name']}\n")
    elif a.action == "sdk":
        write_json(a.out, sdk_inputs(a.sdk))
    else:
        write_json(a.out, freeze(a.spec.resolve(), a.label, a.benchmarks,
                                json.loads(a.selection.read_text()), a.sdk,
                                json.loads(a.sdk_before.read_text())))


if __name__ == "__main__":
    main()
