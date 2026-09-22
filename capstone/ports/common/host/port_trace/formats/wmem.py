"""Pools are identities; a reset or destroy retires every object of one pool."""

import struct
from ..model import Format, require


class Wmem(Format):
    name, version = "wireshark.wmem", 1
    magic = struct.pack("<Q", 0x31304D454D575357)
    header, record = struct.Struct("<16Q"), struct.Struct("<6Q")
    header_fields = "magic count mode status completed news allocs frees reallocs free_alls gcs destroys checksum live_allocators regions_created regions_peak".split()
    record_fields = "op allocator object size type arg".split()
    operations = {
        1: "new",
        2: "alloc",
        3: "free",
        4: "realloc",
        5: "free_all",
        6: "gc",
        7: "destroy",
        8: "end",
    }
    end = 8

    def read_header(self, data, count, replay):
        h = super().read_header(data, count, replay)
        if replay:
            require(
                not any(self.header.unpack(data)[2:]),
                "replay input contains report fields",
            )
        return h

    def validate_record(self, e, h, count, replay):
        f = e.fields
        op = f["op"]
        if op == 1:
            require(
                f["type"] <= 3 and not (f["object"] or f["size"] or f["arg"]),
                "invalid pool declaration",
            )
        elif op in (2, 4):
            require(
                f["size"] > 0 and f["arg"] <= 255 and not f["type"],
                "invalid object request or payload byte",
            )
        elif op == 3:
            require(
                not (f["size"] or f["type"] or f["arg"]), "free has nonzero payload"
            )
        else:
            require(
                not (f["object"] or f["size"] or f["type"] or f["arg"]),
                "pool operation or end has nonzero payload",
            )
