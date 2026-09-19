#!/usr/bin/env python3
"""Record a bounded CPython 3.13.7 standard-library workload (single interpreter/GIL)."""

import argparse
import gc
import json
import os
from pathlib import Path
import re
import sys
import sysconfig

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
p.add_argument("--module-dir", required=True, type=Path)
p.add_argument("--rounds", type=int, default=20)
p.add_argument(
    "--items",
    type=int,
    default=80,
    help="JSON records per round; buffer count scales proportionally",
)
a = p.parse_args()
if sys.version_info[:3] != (3, 13, 7) or sysconfig.get_config_var("Py_GIL_DISABLED"):
    p.error("requires CPython 3.13.7 with the GIL")
if sys.byteorder != "little" or not sysconfig.get_config_var("WITH_PYMALLOC"):
    p.error("requires a little-endian pymalloc build")
if os.environ.get("PYTHONMALLOC", "pymalloc") != "pymalloc":
    p.error("set PYTHONMALLOC=pymalloc for recording")
if not 1 <= a.rounds <= 100 or not 1 <= a.items <= 1000:
    p.error("rounds must be 1..100 and items must be 1..1000")
sys.path.insert(0, str(a.module_dir.resolve()))
import _pymrecord

pattern = re.compile(r"[a-z]+|[0-9]+")
document = json.dumps(
    [{"key": i, "text": "allocator payload " * 4} for i in range(a.items)]
)
a.output.parent.mkdir(parents=True, exist_ok=True)
partial = a.output.with_suffix(a.output.suffix + ".partial")
if partial.exists() or a.output.exists():
    p.error("output already exists")
buffer_count = max(1, a.items * 5 // 2)
expected_sum = a.items * (a.items - 1) // 2
_pymrecord.start(str(partial))
for iteration in range(a.rounds):
    data = json.loads(document)
    words = [pattern.findall(item["text"] + str(item["key"])) for item in data]
    buffers = [bytearray((i % 700) + 1) for i in range(buffer_count)]
    for buffer in buffers:
        buffer.extend(b"resize" * 20)
    assert sum(item["key"] for item in data) == expected_sum
    assert len(words) == a.items and buffers[0][-6:] == b"resize"
    del data, words, buffers, buffer
    gc.collect()
events, live = _pymrecord.stop()
# Import after capture so inspection cannot alter the recorded workload.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common/host"))
from port_trace import inspect_trace

inspection = inspect_trace(partial, expected_format="cpython.pymalloc", replay=True)
if inspection["trace"]["records"] != events:
    raise SystemExit("recorder event count differs from completed trace")
partial.rename(a.output)
print(f"recorded {events} events; {live} allocations live at capture end")
