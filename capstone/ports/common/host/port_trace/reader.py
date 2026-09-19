"""Streaming inspection of existing trace bytes, with bounded preview storage."""

from collections import Counter
from dataclasses import asdict
import hashlib
import os
from pathlib import Path

from .formats import FORMATS
from .model import TraceError, require


class TraceReader:
    """Context-managed, single-pass reader. Exhaust events() to validate completion.

    Closing early only closes the file; it does not assert a valid footer.
    The digest is available only after the iterator reaches the physical EOF.
    No buffers sized from an untrusted count are allocated.
    """

    def __init__(self, path, *, expected_format=None, replay=False):
        self.path = Path(path)
        self.expected_format = expected_format
        self.replay = replay
        self.complete = False
        self._started = False
        self._stream = None

    def __enter__(self):
        require(self._stream is None, "create a new reader for another pass")
        self._stream = self.path.open("rb")
        try:
            stat = os.fstat(self._stream.fileno())
            self._identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            self.size = stat.st_size
            magic = self._stream.read(8)
            self.format = next((f for f in FORMATS if magic == f.magic), None)
            require(self.format is not None, "unknown or truncated trace magic")
            fmt = self.format
            require(
                self.expected_format in (None, fmt.name),
                f"expected {self.expected_format}, found {fmt.name}",
            )
            require(
                self.size >= fmt.header.size + fmt.record.size,
                "missing header or end record",
            )
            payload = self.size - fmt.header.size
            require(payload % fmt.record.size == 0, "partial trailing record")
            self.count = payload // fmt.record.size
            raw = magic + self._stream.read(fmt.header.size - 8)
            require(len(raw) == fmt.header.size, "truncated header")
            self.header = fmt.read_header(raw, self.count, self.replay)
            self._hash = hashlib.sha256(raw)
            return self
        except BaseException:
            self._stream.close()
            raise

    def __exit__(self, *exc):
        self._stream.close()

    def events(self):
        require(
            self._stream is not None and not self._stream.closed, "reader is not open"
        )
        require(not self._started, "events can only be consumed once")
        self._started = True
        for index in range(self.count):
            raw = self._stream.read(self.format.record.size)
            require(
                len(raw) == self.format.record.size,
                f"record {index}: truncated during read",
            )
            self._hash.update(raw)
            try:
                event = self.format.decode(self.format.record.unpack(raw), index)
                require(
                    (event.role == "end") == (index == self.count - 1),
                    "end marker must occur exactly once, as the final record",
                )
                self.format.validate_record(event, self.header, self.count, self.replay)
            except TraceError as error:
                raise TraceError(f"record {index}: {error}") from error
            yield event
        require(not self._stream.read(1), "trace grew during read")
        stat = os.fstat(self._stream.fileno())
        require(
            self._identity == (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),
            "trace changed during read",
        )
        self.complete = True

    @property
    def sha256(self):
        require(self.complete, "trace has not been fully validated")
        return self._hash.hexdigest()


def inspect_trace(path, *, expected_format=None, replay=False, limit=0):
    """Return the common JSON contract only after the whole file validates.

    Counts are per wire operation, not comparable allocation/lifetime metrics.
    Unknown recorder provenance stays unknown; a current checkout cannot supply
    the source revision that produced an arbitrary historical capture.
    """
    require(
        isinstance(limit, int) and 0 <= limit <= 10000, "preview limit must be 0..10000"
    )
    operations, roles, preview = Counter(), Counter(), []
    with TraceReader(path, expected_format=expected_format, replay=replay) as trace:
        for event in trace.events():
            operations[event.operation] += 1
            roles[event.role] += 1
            if len(preview) < limit:
                preview.append(asdict(event))
        result = {
            "schema": "capstone.trace-inspection/v1",
            "trace": {
                "format": trace.format.name,
                "version": trace.format.version,
                "byte_order": "little",
                "bytes": trace.size,
                "sha256": trace.sha256,
                "records": trace.count,
                "record_bytes": trace.format.record.size,
            },
            "validation": {
                "scope": "framing-and-schema",
                "complete": True,
                "replay_input_checks": replay,
                "allocator_semantics": "not-checked",
            },
            "header": trace.header,
            "operations": dict(sorted(operations.items())),
            "roles": dict(sorted(roles.items())),
            "recording_provenance": None,
        }
        if limit:
            result["preview"] = preview
        return result


def record_trace(run, staged_trace, expected_format):
    """Validate the staged input and write the same trace.json for every port."""
    import json

    try:
        result = inspect_trace(
            staged_trace, expected_format=expected_format, replay=True
        )
    except (TraceError, OSError) as error:
        (Path(run) / "trace.json").write_text(
            json.dumps(
                {
                    "schema": "capstone.trace-error/v1",
                    "format": expected_format,
                    "error": str(error),
                    "validation": "failed",
                },
                indent=2,
            )
            + "\n"
        )
        raise
    (Path(run) / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
