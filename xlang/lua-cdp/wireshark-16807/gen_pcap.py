#!/usr/bin/env python3
# Build a minimal "FTP" pcap for Wireshark #16807: a handful of TCP packets on
# port 21, each carrying >=16 bytes of payload so SomeProto.dissector (which
# registers on tcp.port 21 and needs buffer:len() >= 16) fires on every frame.
# The payload bytes are irrelevant to the bug; only "TCP/21 with >=16 bytes"
# matters. Usage: gen_pcap.py <out.pcap>
import struct, sys

def cksum(b):
    if len(b) % 2:
        b += b'\x00'
    s = sum(struct.unpack('!%dH' % (len(b) // 2), b))
    s = (s >> 16) + (s & 0xffff)
    s += s >> 16
    return (~s) & 0xffff

def frame(src_ip, dst_ip, sport, dport, seq, payload):
    tcp = struct.pack('!HHIIBBHHH', sport, dport, seq, 1, (5 << 4), 0x18, 65535, 0, 0) + payload
    ph = struct.pack('!4s4sBBH', src_ip, dst_ip, 0, 6, len(tcp))
    tcp = tcp[:16] + struct.pack('!H', cksum(ph + tcp)) + tcp[18:]
    total = 20 + len(tcp)
    ip = struct.pack('!BBHHHBBH4s4s', 0x45, 0, total, 0, 0x4000, 64, 6, 0, src_ip, dst_ip)
    ip = ip[:10] + struct.pack('!H', cksum(ip)) + ip[12:]
    eth = b'\x00\x0c\x29\x00\x00\x01' + b'\x00\x0c\x29\x00\x00\x02' + b'\x08\x00'
    return eth + ip + tcp

def main(out):
    gh = struct.pack('<IHHiIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1)  # DLT_EN10MB
    A = bytes(map(int, '192.168.0.10'.split('.')))
    B = bytes(map(int, '192.168.0.20'.split('.')))
    payload = b'220 FTP ready\r\nUSER anonymous\r\n'  # 30 bytes, >= 16
    recs = b''
    for i in range(6):
        if i % 2 == 0:
            f = frame(B, A, 12345 + i, 21, 1000 + i, payload)   # -> server port 21
        else:
            f = frame(A, B, 21, 12345 + i, 2000 + i, payload)   # <- server port 21
        recs += struct.pack('<IIII', i, 0, len(f), len(f)) + f
    with open(out, 'wb') as fh:
        fh.write(gh + recs)

if __name__ == '__main__':
    main(sys.argv[1])
