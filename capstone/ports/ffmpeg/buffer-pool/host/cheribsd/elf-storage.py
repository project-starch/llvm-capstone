#!/usr/bin/env python3
"""Inventory allocated ELF sections; excludes shared libraries and guest OS."""

import argparse
import hashlib
import json
from pathlib import Path
import struct


def inventory(path):
    data = path.read_bytes()
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    if header[0][:6] != b"\x7fELF\x02\x01" or header[11] != 64:
        raise ValueError("expected little-endian ELF64")
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, header[6] + i * header[11])
        for i in range(header[12])
    ]
    table = sections[header[13]]
    strings = data[table[4] : table[4] + table[5]]
    allocated = []
    for s in sections:
        if s[2] & 2:  # SHF_ALLOC
            name = strings[s[0] :].split(b"\0")[0].decode()
            allocated.append(
                dict(
                    section=name, bytes=s[5], writable=bool(s[2] & 1), nobits=s[1] == 8
                )
            )
    return dict(
        sha256=hashlib.sha256(data).hexdigest(),
        allocated_sections=allocated,
        zero_initialized_bytes=sum(s["bytes"] for s in allocated if s["nobits"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("binary", nargs="+", help="ROLE=PATH")
    args = parser.parse_args()
    records = {}
    for item in args.binary:
        role, path = item.split("=", 1)
        if role in records:
            parser.error("duplicate role")
        records[role] = inventory(Path(path))
    with args.output.open("x") as f:
        f.write(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
