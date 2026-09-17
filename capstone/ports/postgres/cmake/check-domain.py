#!/usr/bin/env python3
"""Reject ELF images exceeding the loader's two-MiB loadable-image ceiling."""

from pathlib import Path
import struct
import sys

raw = Path(sys.argv[1]).read_bytes()
if raw[:6] != b"\x7fELF\x02\x01":
    raise SystemExit("Expected a little-endian ELF64 domain")
header = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
segments = [
    struct.unpack_from("<IIQQQQQQ", raw, header[5] + i * header[9])
    for i in range(header[10])
]
loaded = [s for s in segments if s[0] == 1]
if not loaded:
    raise SystemExit("Domain has no loadable segment")
span = max(s[3] + s[6] for s in loaded) - min(s[3] for s in loaded)
if span > 2 * 1024 * 1024:
    raise SystemExit(f"Domain needs {span} bytes; the loader permits at most 2 MiB")
print(f"Domain loadable span: {span} bytes")
