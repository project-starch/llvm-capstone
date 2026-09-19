#!/usr/bin/env python3
"""Reconstruct or verify the published PoisonCap sources outside the repository."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "platform.json").read_text())


def archive(spec, path):
    if not path.exists():
        repo = spec["repository"].removeprefix("https://github.com/")
        url = f"https://api.github.com/repos/{repo}/tarball/{spec['commit']}"
        with urllib.request.urlopen(url) as src, path.open("wb") as dst:
            while data := src.read(1 << 20):
                dst.write(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"archive hash mismatch: {path}")


def files(path):
    with tarfile.open(path) as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts[1:]
            if not parts or not member.isfile():
                continue
            if "__pycache__" in parts or parts[-1].endswith(".pyc"):
                continue
            if ".." in parts or member.name.startswith("/"):
                raise ValueError("unsafe archive path")
            yield parts, tar.extractfile(member).read(), member.mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--cap-library", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if root.is_relative_to(HERE.parents[6]):
        parser.error("source preparation must be outside the repository")
    root.mkdir(parents=True, exist_ok=True)
    artifact = args.artifact or root / "artifact.tar.gz"
    caps = args.cap_library or root / "compressed-cap.tar.gz"
    if args.verify and (not artifact.is_file() or not caps.is_file()):
        parser.error("verification requires the two downloaded archives")
    archive(MANIFEST["artifact"], artifact)
    archive(MANIFEST["compressed_cap"], caps)
    for name, base in MANIFEST["bases"].items():
        source = root / "source" / name
        if not source.exists() and not args.verify:
            subprocess.run(["git", "init", str(source)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "remote",
                    "add",
                    "origin",
                    base["repository"],
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    base["commit"],
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(source), "switch", "--detach", "FETCH_HEAD"],
                check=True,
            )
        head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != base["commit"]:
            raise ValueError(f"unexpected upstream base for {name}: {head}")

    checked = 0

    def apply(path, data, mode):
        nonlocal checked
        if args.verify:
            if not path.is_file() or path.read_bytes() != data:
                raise ValueError(f"source mismatch: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(mode)
        checked += 1

    for parts, data, mode in files(artifact):
        if parts[0] in MANIFEST["bases"]:
            apply(root.joinpath("source", *parts), data, mode)
    capdir = root / "source/qemu/target/cheri-common/cheri-compressed-cap"
    for parts, data, mode in files(caps):
        # The artifact supplies its version-aware replacement beside the
        # dependency directory. It must replace the older pinned header.
        if parts == ("cheri_compressed_cap_128.h",):
            data = (capdir.parent / parts[0]).read_bytes()
        apply(capdir.joinpath(*parts), data, mode)
    print(
        f"PoisonCap source files {'verified' if args.verify else 'prepared'}: {checked}"
    )


if __name__ == "__main__":
    main()
