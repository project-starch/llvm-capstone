#!/usr/bin/env python3
"""Turn ffcensus's per-process counter files (census.<pid>) into slotcensus's JSON lines.

usage: ffcensus-dump.py DIR > census.jsonl
The layout is ffcensus.c's struct census. A file without the magic number is reported on stderr and skipped; no
file at all is an error (exit 1), never an empty result."""
import glob
import json
import os
import struct
import sys

MAGIC = 0x53554E4543534646
HEAD = struct.Struct("<Q4i9q64q")
NAMES = ("pid", "ppid", "forked", "pad", "live", "peak", "allocs", "frees", "moved", "inplace", "failed",
         "live_bytes", "peak_bytes")


def main():
    files = sorted(glob.glob(os.path.join(sys.argv[1], "census.[0-9]*")))
    if not files:
        print(f"ffcensus-dump: no census files in {sys.argv[1]}", file=sys.stderr)
        return 1
    for path in files:
        raw = open(path, "rb").read()
        if len(raw) < HEAD.size + 32 + 1024:
            print(f"ffcensus-dump: {path} is short", file=sys.stderr)
            continue
        fields = HEAD.unpack_from(raw)
        if fields[0] != MAGIC:
            print(f"ffcensus-dump: {path} has no magic", file=sys.stderr)
            continue
        v = dict(zip(NAMES, fields[1:14]))
        comm = raw[HEAD.size:HEAD.size + 32].split(b"\0")[0].decode(errors="replace")
        cmdline = raw[HEAD.size + 32:HEAD.size + 32 + 1024].split(b"\0")[0].decode(errors="replace")
        print(json.dumps({"pid": v["pid"], "ppid": v["ppid"], "cmdline": cmdline, "comm": comm,
                          "forked": v["forked"], "peak_live": v["peak"], "end_live": v["live"],
                          "allocs": v["allocs"], "frees": v["frees"], "realloc_moved": v["moved"],
                          "realloc_inplace": v["inplace"], "failed": v["failed"],
                          "peak_live_bytes": v["peak_bytes"], "size_log2": list(fields[14:78])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
