#!/usr/bin/env python3
"""Build one allocator and its link example for CheriBSD purecap."""

import argparse
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PORTS = HERE.parents[2]
COMPONENTS = json.loads((HERE / "components.json").read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("component", choices=COMPONENTS)
    p.add_argument("build", type=Path)
    p.add_argument(
        "--sdk",
        default=os.environ.get("CHERI_SDK"),
        required="CHERI_SDK" not in os.environ,
    )
    p.add_argument(
        "--sysroot",
        default=os.environ.get("CHERI_SYSROOT"),
        required="CHERI_SYSROOT" not in os.environ,
    )
    p.add_argument("--archive", type=Path)
    p.add_argument(
        "--client", type=Path, help="Additional main.c linked as allocator-client"
    )
    p.add_argument("--jobs", type=int, default=2)
    a = p.parse_args()
    if a.jobs < 1:
        p.error("--jobs must be positive")
    component = COMPONENTS[a.component]
    command = [
        "cmake",
        "-S",
        str(PORTS / component["path"]),
        "-B",
        str(a.build),
        "-G",
        "Ninja",
        "-DCMAKE_TOOLCHAIN_FILE="
        + str(PORTS / "common/cmake/toolchains/cheribsd.cmake"),
        "-DCHERI_SDK=" + str(Path(a.sdk).resolve()),
        "-DCHERI_SYSROOT=" + str(Path(a.sysroot).resolve()),
        "-DCMAKE_BUILD_TYPE=Debug",
    ]
    if a.archive:
        command.append(
            "-D" + component["archive_option"] + "=" + str(a.archive.resolve())
        )
    if a.client:
        command.append("-DPORT_CLIENT_SOURCE=" + str(a.client.resolve()))
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(a.build), "-j", str(a.jobs)], check=True)


if __name__ == "__main__":
    main()
