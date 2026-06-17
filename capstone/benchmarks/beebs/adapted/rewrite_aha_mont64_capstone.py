#!/usr/bin/env python3
"""Generate the Capstone-adapted aha-mont64 source.

The Capstone backend currently crashes on some 64-bit constants whose lower
32 bits require both lui and addi materialization. This helper hoists those
constants into static globals so the values are emitted as data bytes instead
of instruction-level constant materialization sequences.
"""

import re
import sys


def needs_hoist(value):
    """Return True if this 64-bit constant triggers the known backend bug."""
    lo32 = value & 0xFFFFFFFF
    if lo32 == 0:
        return False
    return (lo32 >> 12) != 0 and (lo32 & 0xFFF) != 0


def main():
    if len(sys.argv) != 3:
        print("usage: rewrite_aha_mont64_capstone.py <src> <dst>", file=sys.stderr)
        return 2

    src_path, dst_path = sys.argv[1], sys.argv[2]
    with open(src_path, encoding="utf-8") as src_file:
        text = src_file.read()

    text = re.sub(
        r"^#include <(stdio|stdlib|stdint)\.h>\n",
        "",
        text,
        flags=re.MULTILINE,
    )

    const_decls = []
    const_map = {}

    def replace_const(match):
        raw = match.group(0)
        value = int(raw[:-2], 16)
        if not needs_hoist(value):
            return raw

        key = f"0x{value:016x}"
        if key not in const_map:
            name = f"_mont64_const_{len(const_map)}"
            const_map[key] = name
            const_decls.append(f"static uint64_t {name} = {key}UL;")
        return const_map[key]

    text = re.sub(r"0x[0-9a-fA-F]{5,}(?:LL|UL)", replace_const, text)

    with open(dst_path, "w", encoding="utf-8") as dst_file:
        dst_file.write("typedef unsigned long uint64_t;\n")
        dst_file.write("typedef long int64_t;\n")
        if const_decls:
            dst_file.write("\n".join(const_decls))
            dst_file.write("\n")
        dst_file.write(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

