#!/usr/bin/env python3
"""Record real decoder activity, compare every replay event, and run live controls."""

import argparse
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("build", type=pathlib.Path)
args = parser.parse_args()
port = pathlib.Path(__file__).resolve().parents[2]
build = args.build.resolve()
results = build / "test-results"
results.mkdir(exist_ok=True)
run = pathlib.Path(tempfile.mkdtemp(prefix="native-", dir=results))
print(f"Native test artifacts: {run}", flush=True)
env = dict(os.environ, FFPOOL_NATIVE_BUILD_DIR=str(build))
subprocess.run(
    ["bash", port / "native/ffmpeg/record-workload.sh", run, "1", "320x180"],
    env=env,
    check=True,
)

# All modes must reproduce the measured outcomes, not merely exit successfully.
for mode in range(3):
    output = run / f"replay-{mode}.bin"
    subprocess.run(
        [build / "bin/replay", run / "commands.bin", output, str(mode)], check=True
    )
    subprocess.run(
        [
            sys.executable,
            port / "analysis/trace-tools.py",
            "compare",
            run / "recorded.bin",
            output,
        ],
        check=True,
    )

# Native tests exercise valid references, callbacks and deferred close only.
# Fault expectations require the Capstone QEMU suite.
control = run / "control.bin"
control.write_bytes(
    struct.pack("<16Q", 0x4650465452433032, 1, *([0] * 14)) + bytes(128)
)
for mode in range(3):
    subprocess.run(
        [build / "bin/pool-security", control, run / f"control-{mode}.bin", str(mode)],
        check=True,
    )

# Publish the complete recording for QEMU tests only after all native checks pass.
# Previous runs remain available, and CTest may be repeated without overwriting them.
link = results / "workload.next"
link.unlink(missing_ok=True)
link.symlink_to(run.name, target_is_directory=True)
link.replace(results / "workload")
