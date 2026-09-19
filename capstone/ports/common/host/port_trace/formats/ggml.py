"""Descriptor identity is distinct from buffer identity and buffer lifetime."""

import struct
from ..model import Format, require


class Ggml(Format):
    name, version = "whisper.ggml-context", 1
    magic = struct.pack("<Q", 0x315854434C4D4747)
    header, record = struct.Struct("<16Q"), struct.Struct("<6Q")
    header_fields = "magic count mode status completed inits objects resets owned_frees borrowed_frees rebinds peak_used checksum live_contexts object_header_size layout_checksum".split()
    record_fields = "op ctx buffer size type arg".split()
    operations = {1: "init", 2: "alloc", 3: "reset", 4: "free", 5: "end"}
    end = 5

    def read_header(self, data, count, replay):
        h = super().read_header(data, count, replay)
        if replay:
            v = self.header.unpack(data)
            require(
                not any(v[2:14]) and not v[15], "replay input contains report fields"
            )
            require(
                h["object_header_size"] == 32,
                "unsupported recorded object-header geometry",
            )
        return h

    def validate_record(self, e, h, count, replay):
        f = e.fields
        if f["op"] == 1:
            require(
                f["arg"] in (0, 1) and not f["type"] and f["size"] > 0,
                "invalid owned/borrowed buffer declaration",
            )
        elif f["op"] == 2:
            require(
                f["arg"] <= 255 and f["type"] <= 2,
                "invalid object type or payload byte",
            )
        else:
            require(
                not any(f[k] for k in ("size", "type", "arg")),
                "reset/free/end has nonzero payload",
            )
            if e.role == "end":
                require(not f["buffer"], "end has nonzero buffer identity")
