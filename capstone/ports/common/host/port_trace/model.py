"""Wire-format adapters: identities and operation semantics belong to each port."""

from dataclasses import dataclass
import struct


class TraceError(ValueError):
    """An input is not a supported, complete trace of the requested kind."""


def require(condition, message):
    if not condition:
        raise TraceError(message)


@dataclass(frozen=True)
class Event:
    index: int
    operation: str
    role: str
    fields: dict


class Format:
    """A fixed-record legacy schema, explicitly selected by its wire magic.

    validate_record checks the schema, not the allocator's implementation.
    Adapters may distinguish commands, observations and end markers; callers
    must not translate a context release into object death on their own.
    """

    name: str
    version: int
    magic: bytes
    header: struct.Struct
    record: struct.Struct
    header_fields: tuple
    record_fields: tuple
    operations: dict
    end: int

    def read_header(self, data, count, replay):
        values = self.header.unpack(data)
        result = dict(zip(self.header_fields, values))
        require(data[:8] == self.magic, "wrong format magic")
        if "count" in result:
            require(
                result["count"] == count, "header record count differs from file size"
            )
        return result

    def decode(self, values, index):
        op = values[0]
        require(op in self.operations, f"unknown operation {op}")
        return Event(
            index,
            self.operations[op],
            self.role(op),
            dict(zip(self.record_fields, values)),
        )

    def role(self, op):
        return "end" if op == self.end else "command"

    def validate_record(self, event, header, count, replay):
        pass
