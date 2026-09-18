#!/usr/bin/env python3
"""Compare a real CPython API recording against the original and adapted allocator."""

from pathlib import Path
import struct
import subprocess
import sys
import tempfile

python, module_dir, replay, reference, results = sys.argv[1:]
Path(results).mkdir(parents=True, exist_ok=True)
run = Path(tempfile.mkdtemp(prefix="recording-", dir=results))
port = Path(__file__).resolve().parents[2]
subprocess.run(
    [
        python,
        port / "host/record.py",
        run / "trace.bin",
        "--module-dir",
        module_dir,
        "--rounds",
        "20",
    ],
    check=True,
)
outputs = []
for name, binary in (("ported", replay), ("reference", reference)):
    outputs.append(
        subprocess.run(
            [binary, run / "trace.bin", run / f"{name}.bin"],
            capture_output=True,
            check=True,
        ).stdout
    )
assert outputs[0] == outputs[1], outputs
assert (run / "ported.bin").read_bytes() == (run / "reference.bin").read_bytes()
events = list(struct.iter_unpack("<4Q", (run / "trace.bin").read_bytes()[96:]))
report = struct.unpack("<12Q", (run / "ported.bin").read_bytes())
assert report[4] == len(events) and report[7] > 0
assert events[-1][0] == 5 and events[-1][1] == report[5] - report[6]
print(outputs[0].decode(), end="")
