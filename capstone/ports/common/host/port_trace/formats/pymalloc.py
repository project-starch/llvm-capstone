"""Objects/obmalloc.c API requests; realloc retains a replay-table identity."""

import struct
from ..model import Format, require


class Pymalloc(Format):
    name, version = "cpython.pymalloc", 1
    magic = struct.pack("<Q", 0x31594C50524D5950)
    header, record = struct.Struct("<12Q"), struct.Struct("<4Q")
    header_fields = "magic count mode status completed allocations frees reallocations arenas arena_frees metadata checksum".split()
    record_fields = "op id size value".split()
    operations = {1: "alloc", 2: "calloc", 3: "realloc", 4: "free", 5: "end"}
    end = 5

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
        if f["op"] in (4, 5):
            require(not f["size"] and not f["value"], "free/end has nonzero payload")
        if e.role != "end":
            require(f["value"] <= 255, "payload fill value is not a byte")
