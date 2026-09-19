"""Pool calls and nested callback effects; lease and backing IDs remain separate."""

import struct
from ..model import Format, require


class Ffmpeg(Format):
    name, version = "ffmpeg.buffer-pool", 2
    magic = struct.pack("<Q", 0x4650465452433032)
    header = record = struct.Struct("<16Q")
    header_fields = "magic count status metadata_used payload_used split mrev delin revoke init init_bytes mode reserved0 reserved1 reserved2 reserved3".split()
    record_fields = "op kind pool call parent object size flags backing allocations gap live retained reserved0 reserved1 reserved2".split()
    operations = {
        1: "create",
        2: "get",
        3: "return",
        4: "close",
        5: "new_backing",
        6: "drop_backing",
        7: "callback",
        8: "end",
        129: "create_end",
        130: "get_end",
        131: "return_end",
        132: "close_end",
        135: "callback_end",
    }
    end = 8

    def role(self, op):
        if op == 7:
            return "callback"
        if op in (5, 6) or op & 128:
            return "observation"
        return super().role(op)

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
        if e.role == "end":
            require(
                not any(v for k, v in f.items() if k != "op"), "end has nonzero fields"
            )
        else:
            require(f["kind"] in (1, 2), "unknown pool kind")
        if replay:
            require(
                not any(f[k] for k in self.record_fields[8:]),
                "strip measured outcomes before replaying FFmpeg commands",
            )
