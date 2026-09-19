#!/usr/bin/env python3
"""Strip expected outcomes, compare all events, and summarize measured traces."""

import collections
import json
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "common/host"))
from port_trace import TraceReader

ROW = struct.Struct("<16Q")
MAGIC = 0x4650465452433032
FIELDS = "op kind pool call parent object size flags backing allocations gap live retained reserved0 reserved1 reserved2".split()


def read(path):
    with TraceReader(path, expected_format="ffmpeg.buffer-pool") as trace:
        head = tuple(trace.header[name] for name in trace.format.header_fields)
        rows = [
            tuple(e.fields[name] for name in trace.format.record_fields)
            for e in trace.events()
        ]
        return head, rows


def summary(head, events):
    result = dict(
        events=len(events),
        status=head[2],
        metadata_used=head[3],
        payload_used=head[4],
        mode=head[11],
    )
    result["primitives"] = dict(
        zip(["split", "mrev", "delin", "revoke", "init", "init_bytes"], head[5:11])
    )
    for kind, name in [(1, "buffer"), (2, "refstruct")]:
        rows = [e for e in events if e[1] == kind]
        gets = [e for e in rows if e[0] == 130]
        result[name] = dict(
            pools=sum(e[0] == 1 for e in rows),
            allocations=len(gets),
            backings=sum(e[0] == 5 for e in rows),
            reuses=sum(e[10] > 0 for e in gets),
            peak_live=max((e[11] for e in rows), default=0),
            peak_retained=max((e[12] for e in rows), default=0),
            callbacks=dict(collections.Counter(str(e[7]) for e in rows if e[0] == 7)),
            nested_operations=sum(e[0] in (1, 2, 3, 4) and e[4] != 0 for e in rows),
        )
    return result


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    h, rows = read(args[0])
    if action == "commands":
        with open(args[1], "wb") as f:
            f.write(ROW.pack(MAGIC, len(rows), *([0] * 14)))
            for e in rows:
                f.write(ROW.pack(*e[:8], *([0] * 8)))
    elif action == "summary":
        print(json.dumps(summary(h, rows), indent=2))
    elif action == "compare":
        other_h, other = read(args[1])
        if h[2] or other_h[2] or len(rows) != len(other):
            raise SystemExit("status/count mismatch")
        for i, (a, b) in enumerate(zip(rows, other)):
            if a != b:
                raise SystemExit(
                    f"event {i}: "
                    + str({FIELDS[j]: [a[j], b[j]] for j in range(16) if a[j] != b[j]})
                )
        print(
            json.dumps(
                dict(
                    exact=True,
                    events=len(rows),
                    reference=str(args[0]),
                    replay=str(args[1]),
                )
            )
        )
    else:
        raise SystemExit("expected commands, summary, or compare")
