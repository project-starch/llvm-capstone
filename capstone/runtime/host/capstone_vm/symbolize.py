"""Turn a capstone-exec fault record into a symbol and line.

The record carries the runtime address of domain_main (entry=), so the slide
between link and runtime addresses is exact for the image that produced it.
The image must be the one named in the record, built with debug information
for a line; without it the function name still resolves from the symbol table.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import os
import re
import subprocess
import sys

RECORD = re.compile(
    r"capstone-exec: domain fault cause=(?P<cause>\d+) pc=0x(?P<pc>[0-9a-f]+) "
    r"address=0x(?P<address>[0-9a-f]+) entry=0x(?P<entry>[0-9a-f]+) "
    r"code=0x(?P<base>[0-9a-f]+)-0x(?P<end>[0-9a-f]+)"
    r"(?: last=0x[0-9a-f]+ preparing=0x[0-9a-f]+)?(?: sha256=(?P<sha256>[0-9a-f]{64}))?"
    r"(?: image=(?P<image>.*))?$")


def parse(line: str) -> dict | None:
    match = RECORD.search(line)
    if not match:
        return None
    fields = match.groupdict()
    return {key: (int(value, 16) if key not in ("image", "sha256") and value is not None else value)
            for key, value in fields.items()} | {"cause": int(fields["cause"])}


def tool(name: str) -> str:
    build = os.environ.get("CAPSTONE_LLVM_BUILD_DIR")
    return os.path.join(build, "bin", name) if build else name


def symbols(image: str, run=subprocess.run) -> list[tuple[int, int, str]]:
    """(address, size, name) for every defined symbol, from llvm-nm -S."""
    output = run([tool("llvm-nm"), "-S", "--defined-only", image],
                 capture_output=True, text=True, check=True).stdout
    table = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 4:
            table.append((int(parts[0], 16), int(parts[1], 16), parts[3]))
        elif len(parts) == 3:
            table.append((int(parts[0], 16), 0, parts[2]))
    return sorted(table)


def link_address(table: list[tuple[int, int, str]], symbol: str) -> int:
    for address, _, name in table:
        if name == symbol:
            return address
    raise LookupError(f"{symbol} is not in the image")


def containing(table: list[tuple[int, int, str]], address: int) -> str:
    """The sized symbol that covers the address, as name+offset."""
    for start, size, name in table:
        if size and start <= address < start + size:
            return f"{name}+0x{address - start:x}"
    return "??"


def symbolize(record: dict, image: str, run=subprocess.run) -> str:
    if record.get("sha256"):
        with Path(image).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError("Image SHA-256 does not match the fault record")
    table = symbols(image, run)
    slide = record["entry"] - link_address(table, "domain_main")
    link_pc = record["pc"] - slide
    where = containing(table, link_pc)
    # A debug image also gives the line; a release image answers ?? here.
    output = run([tool("llvm-symbolizer"), f"--obj={image}", "--functions=short", "-p",
                  f"0x{link_pc:x}"], capture_output=True, text=True, check=False).stdout
    line = output.strip().splitlines()[0] if output.strip() else "??"
    if not line.startswith("??"):
        where += f" ({line})"
    return (f"cause {record['cause']} at 0x{record['pc']:x} (link 0x{link_pc:x}): {where}"
            f"; address 0x{record['address']:x}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Override the image named in the record")
    parser.add_argument("record", nargs="?", help="A fault record line; default: read stdin")
    args = parser.parse_args(argv)
    lines = [args.record] if args.record else sys.stdin.read().splitlines()
    found = 0
    for line in lines:
        record = parse(line)
        if not record:
            continue
        image = args.image or record.get("image")
        if not image:
            print("symbolize: the record names no image; pass --image", file=sys.stderr)
            return 2
        print(symbolize(record, image))
        found += 1
    if not found:
        print("symbolize: no fault record in the input", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
