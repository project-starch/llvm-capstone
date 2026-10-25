#!/usr/bin/env python3
# scan-csmint.py <file|dir>... : count 32-bit words matching CSMINT's encoding ((w & 0xfe00707f) == 0xa000105b) at every
# 2-byte offset (RVC code places 4-byte instructions at 2-byte alignment). Prints per-file hits; exit 2 if a path is
# missing or nothing was scanned ("no data" is an error, not a zero).
import os, sys, struct
MASK, MATCH = 0xfe00707f, 0xa000105b
files = []
for p in sys.argv[1:]:
    if not os.path.exists(p): sys.exit(f"missing: {p}")
    if os.path.isdir(p):
        for d, _, fs in os.walk(p):
            files += [os.path.join(d, f) for f in fs if os.path.isfile(os.path.join(d, f)) and not os.path.islink(os.path.join(d, f))]
    else: files.append(p)
if not files: sys.exit(2)
tot, nbytes = 0, 0
for f in sorted(files):
    b = open(f, 'rb').read(); nbytes += len(b); n = 0
    for off in range(0, len(b) - 3, 2):
        w = struct.unpack_from('<I', b, off)[0]
        if (w & MASK) == MATCH: n += 1
    tot += n
    if n: print(f"{n:6d}  {f}")
print(f"TOTAL {tot} CSMINT-shaped words in {len(files)} files, {nbytes} bytes")
