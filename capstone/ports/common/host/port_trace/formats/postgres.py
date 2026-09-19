"""A11 context operations; process-prefix references are retained, never invented."""

import struct
from ..model import Format, require


class Postgres(Format):
    name, version = "postgres.a11", 1
    magic = b"A11TRACE"
    header, record = struct.Struct("<8sIIQIIQII32s"), struct.Struct("<IIIIQQQ")
    header_fields = (
        "magic version recsize endian pid ppid prefix nphase pad phase".split()
    )
    record_fields = "op ctx ptr aux s1 s2 s3".split()
    operations = dict(
        enumerate(
            (
                "end",
                "alloc",
                "free",
                "realloc",
                "reset",
                "delete",
                "create_allocset",
                "create_generation",
                "create_slab",
                "create_bump",
                "blocks",
            )
        )
    )
    end = 0

    def read_header(self, data, count, replay):
        h = super().read_header(data, count, replay)
        require(h["version"] == self.version, "unsupported A11 version")
        require(h["recsize"] == self.record.size, "unsupported A11 record size")
        require(h["endian"] == 0x0102030405060708, "unsupported A11 byte order")
        require(
            h["nphase"] <= 32 and not h["pad"], "invalid A11 phase length or padding"
        )
        if replay:
            require(
                not h["ppid"] and not h["prefix"],
                "flatten the A11 process prefix before replay",
            )
        h["magic"] = h["magic"].decode("ascii")
        # Hex preserves arbitrary phase bytes without lossy Unicode replacement.
        h["phase"] = h["phase"][: h["nphase"]].hex()
        return h

    def role(self, op):
        return "observation" if op == 10 else super().role(op)

    def validate_record(self, e, h, count, replay):
        f = e.fields
        if e.role == "end":
            require(
                not any(f[k] for k in ("ctx", "ptr", "aux")),
                "end has nonzero identity fields",
            )
            require(
                f["s1"] == count - 1, "A11 footer record count differs from file size"
            )
