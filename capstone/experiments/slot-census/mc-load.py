#!/usr/bin/env python3
"""Load a memcached server over its text protocol: set KEYS keys of VALUE bytes, get every key once, delete a tenth,
then print the server's own item and memory statistics.

usage: mc-load.py PORT KEYS [VALUE]
Requests are pipelined in batches of 1000. Prints "stored N", "hits N", "deleted N" and the stats lines curr_items,
total_items, bytes, limit_maxbytes, total_malloced (from "stats slabs"); exit 1 if any reply is not the expected
one."""
import socket
import sys

BATCH = 1000


def main():
    port, keys = int(sys.argv[1]), int(sys.argv[2])
    value = b"v" * (int(sys.argv[3]) if len(sys.argv) > 3 else 100)
    s = socket.create_connection(("127.0.0.1", port))
    f = s.makefile("rb")

    def batch(lines, replies):
        s.sendall(b"".join(lines))
        return [replies(f) for _ in lines]

    def store(f):
        return f.readline() == b"STORED\r\n"

    def get(f):
        line = f.readline()
        if line == b"END\r\n":
            return False
        f.read(int(line.split()[3]) + 2)
        return f.readline() == b"END\r\n"

    def delete(f):
        return f.readline() == b"DELETED\r\n"

    stored = hits = deleted = 0
    for lo in range(0, keys, BATCH):
        r = range(lo, min(lo + BATCH, keys))
        stored += sum(batch([b"set key:%d 0 0 %d\r\n%s\r\n" % (i, len(value), value) for i in r], store))
    for lo in range(0, keys, BATCH):
        r = range(lo, min(lo + BATCH, keys))
        hits += sum(batch([b"get key:%d\r\n" % i for i in r], get))
    for lo in range(0, keys, BATCH * 10):
        r = range(lo, min(lo + BATCH * 10, keys), 10)
        deleted += sum(batch([b"delete key:%d\r\n" % i for i in r], delete))
    print(f"stored {stored}\nhits {hits}\ndeleted {deleted}")
    want = {b"curr_items", b"total_items", b"bytes", b"limit_maxbytes", b"total_malloced"}
    for cmd in (b"stats\r\n", b"stats slabs\r\n"):
        s.sendall(cmd)
        while (line := f.readline()) != b"END\r\n":
            parts = line.split()
            if len(parts) == 3 and parts[1] in want:
                print(parts[1].decode(), parts[2].decode())
    return 0 if stored == keys and hits == keys and deleted == (keys + 9) // 10 else 1


if __name__ == "__main__":
    sys.exit(main())
