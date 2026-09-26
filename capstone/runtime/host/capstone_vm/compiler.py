#!/usr/bin/env python3
"""Compiler driver for a CMake-built Capstone application SDK.

Use the generated SDK's capstone-cc as CC in an upstream build. Its adjacent
sdk.json records the compiler, headers and runtime that CMake actually built.
CAPSTONE_SDK may select another SDK directory. No port names or entry adapters.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

MUSL_LIBS = {"c", "m", "pthread", "rt", "dl", "util", "crypt", "xnet", "resolv"}
PAIRED = {"-D", "-U", "-I", "-isystem", "-iquote", "-include", "-imacros",
          "-idirafter", "-Xclang", "-mllvm", "-isysroot"}


def expand(arguments: list[str], depth: int = 0) -> list[str]:
    if depth > 8:
        raise ValueError("response files nest too deeply")
    result = []
    for arg in arguments:
        result.extend(expand(shlex.split(Path(arg[1:]).read_text()), depth + 1)
                      if arg.startswith("@") else [arg])
    return result


def link_arguments(arguments: list[str]) -> tuple[list[str], list[str], list[str], str]:
    """Separate compiler options, source files and ordered linker inputs."""
    compile_flags, sources, inputs = [], [], []
    output = "a.out"
    words = iter(arguments)
    for arg in words:
        if arg == "-o":
            output = next(words)
        elif arg.startswith("-o") and len(arg) > 2:
            output = arg[2:]
        elif arg in PAIRED:
            compile_flags.extend((arg, next(words)))
        elif arg == "-Xlinker":
            inputs.append(next(words))
        elif arg in ("-L", "-l"):
            value = next(words)
            if arg != "-l" or value not in MUSL_LIBS:
                inputs.extend((arg, value))
        elif arg.startswith("-l"):
            if arg[2:] not in MUSL_LIBS:
                inputs.append(arg)
        elif arg.startswith("-Wl,"):
            inputs.extend(arg[4:].split(","))
        elif arg.startswith("-L") or arg.endswith((".o", ".a")):
            inputs.append(arg)
        elif arg.endswith((".c", ".s", ".S")):
            # Preserve source/library ordering, including repeated basenames.
            inputs.append("@source:" + str(len(sources)))
            sources.append(arg)
        elif arg in ("-shared", "-pie"):
            raise ValueError("application domains require a static executable")
        elif arg in ("-static", "-no-pie", "-rdynamic", "-nostdlib", "-nodefaultlibs"):
            continue
        elif arg.startswith(("-D", "-U", "-I", "-std=", "-O", "-g", "-f", "-W", "-m")) or arg in ("-pthread", "-v"):
            compile_flags.append(arg)
        else:
            raise ValueError(f"unsupported compiler driver argument: {arg}")
    return compile_flags, sources, inputs, output


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = expand(sys.argv[1:] if argv is None else argv)
        sdk = Path(os.environ.get("CAPSTONE_SDK", Path(sys.argv[0]).absolute().parent))
        config = json.loads((sdk / "sdk.json").read_text())
        if config["version"] != 1:
            raise ValueError("unsupported SDK version; rebuild the SDK")
        target = [config["cc"], "-target", "capstone64-unknown-elf",
                  "-Xclang", "-target-feature", "-Xclang", "+m",
                  "-Xclang", "-target-feature", "-Xclang", "+a",
                  "-ffreestanding", "-fno-builtin", "-fno-jump-tables", "-nostdinc"]
        for suffix in ("arch/capstone64", "arch/generic", "obj/include", "include"):
            target.extend(("-isystem", str(Path(config["musl"]) / suffix)))
        if any(a in ("-c", "-S", "-E", "-M", "-MM", "--version", "-dumpmachine",
                     "-dumpversion", "--help") or a.startswith("-print-") for a in arguments) or arguments == ["-v"]:
            return subprocess.call([*target, *arguments])
        flags, sources, inputs, output = link_arguments(arguments)
        with tempfile.TemporaryDirectory(prefix="capstone-cc-") as scratch:
            for index, source in enumerate(sources):
                obj = str(Path(scratch) / f"{index}.o")
                result = subprocess.call([*target, *flags, "-c", source, "-o", obj])
                if result:
                    return result
                inputs[inputs.index("@source:" + str(index))] = obj
            return subprocess.call([config["ld"], "--gc-sections", "-T", config["script"],
                                    "-o", output, "--whole-archive", config["runtime"],
                                    "--no-whole-archive", "--start-group", *inputs,
                                    config["libc"], config["builtins"], "--end-group"])
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f"capstone-cc: {error or 'missing option argument'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
