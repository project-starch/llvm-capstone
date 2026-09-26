"""Deterministic valid DNS request/response PCAPs for tshark's real dissectors.

Vary flow count independently from packet count to distinguish per-packet wmem
from per-conversation/file lifetime. Zero UDP checksums are valid in IPv4.
"""
import argparse
import struct
from pathlib import Path

def checksum(data):
    data += b'\0' * (len(data) % 2)
    n = sum(struct.unpack('!%dH' % (len(data)//2), data))
    while n >> 16: n = (n & 65535) + (n >> 16)
    return (~n) & 65535

def packet(index, flows, answer):
    flow = index % flows
    src = bytes((10, 0, 1 + flow//250, 1 + flow%250))
    dst = bytes((10, 1, 0, 1))
    question = b'\x04test\x07example\0' + struct.pack('!HH', 1, 1)
    dns = struct.pack('!HHHHHH', index % 65536, 0x8180 if answer else 0x100, 1, int(answer), 0, 0) + question
    if answer:
        dns += b'\xc0\x0c' + struct.pack('!HHIH', 1, 1, 60, 4) + bytes((192, 0, 2, 1))
        src, dst = dst, src
    udp = struct.pack('!HHHH', 53 if answer else 20000+flow, 20000+flow if answer else 53, len(dns)+8, 0) + dns
    ip = struct.pack('!BBHHHBBH', 0x45, 0, 20+len(udp), index%65536, 0, 64, 17, 0) + src+dst
    ip = ip[:10] + struct.pack('!H', checksum(ip)) + ip[12:]
    return b'\x02\0\0\0\0\x01\x02\0\0\0\0\x02\x08\0' + ip + udp

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', type=int, required=True)
    p.add_argument('--flows', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.flows <= 16000 or a.requests < 1: p.error('invalid request/flow count')
    with a.out.open('xb') as f:
        f.write(struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
        for i in range(a.requests):
            for reply in (False, True):
                data = packet(i, a.flows, reply)
                f.write(struct.pack('<IIII', 1700000000+i//1000, (i%1000)*1000+int(reply), len(data), len(data)))
                f.write(data)

if __name__ == '__main__': main()
