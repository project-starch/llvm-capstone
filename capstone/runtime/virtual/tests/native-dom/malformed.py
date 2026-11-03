#!/usr/bin/env python3
"""Make bounded-loader refusal cases beside an SDK image."""
import pathlib
import struct
import sys

path = pathlib.Path(sys.argv[1])
original = path.read_bytes()

def save(suffix, data, mode=0o755):
    target = path.with_name(path.name + '.' + suffix)
    target.write_bytes(data)
    target.chmod(mode)

save('truncated', original[:64])
data = bytearray(original)
marker = data.index(b'CPONVVM7')
data[marker] ^= 1
save('marker', data)
data = bytearray(original)
phoff = struct.unpack_from('<Q', data, 32)[0]
struct.pack_into('<Q', data, phoff + 8, (1 << 64) - 4096)
save('overflow', data)
data = bytearray(original)
struct.pack_into('<I', data, phoff + 4, 7)
save('wx', data)
save('setuid', original, 0o4755)
