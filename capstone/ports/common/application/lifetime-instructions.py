#!/usr/bin/env python3
"""Check lifetime opcodes in an allocator object's executable ELF sections."""
import argparse
from pathlib import Path
import struct


def counts(path):
    data = path.read_bytes()
    if len(data) < 64 or data[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', data, 18)[0] != 259:
        raise ValueError('expected a little-endian Capstone ELF64 object (machine 259)')
    offset = struct.unpack_from('<Q', data, 40)[0]
    stride, count = struct.unpack_from('<HH', data, 58)
    found = {0x51: 0, 0x52: 0}
    for i in range(count):
        header = offset + i * stride
        flags, _, start, size = struct.unpack_from('<QQQQ', data, header + 8)
        if not flags & 4:  # SHF_EXECINSTR
            continue
        if start + size > len(data) or size % 2:
            raise ValueError('truncated executable section')
        pos, end = start, start + size
        while pos + 2 <= end:
            half = struct.unpack_from('<H', data, pos)[0]
            if half & 3 != 3:
                pos += 2
                continue
            if pos + 4 > end:
                raise ValueError('truncated instruction')
            word = struct.unpack_from('<I', data, pos)[0]
            for opcode in found:
                if word & 0xfe00707f == (opcode << 25) | (1 << 12) | 0x5b:
                    found[opcode] += 1
            pos += 4
    return found[0x51], found[0x52]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', type=Path)
    parser.add_argument('--expect', choices=('present', 'absent'), required=True)
    args = parser.parse_args()
    try:
        derive, revoke = counts(args.object)
    except (OSError, ValueError, struct.error) as error:
        parser.error(str(error))
    ok = derive > 0 and revoke > 0 if args.expect == 'present' else derive == revoke == 0
    print(f'{"PASS" if ok else "FAIL"} {args.object.name}: CDERIVE={derive} CREVOKE={revoke}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
