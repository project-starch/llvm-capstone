#!/usr/bin/env python3
"""Run a prebuilt FFmpeg replay in QEMU and retain every attempt's artifacts."""

import argparse
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common/host"))
from port_support import digest, run_guest, stage_run, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("trace", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument(
    "--protection", choices=("spatial", "backing", "sublet"), default="spatial"
)
work = Path(
    os.environ.get(
        "FFPOOL_WORK",
        Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone"))
        / "ffmpeg-buffer-pool",
    )
)
parser.add_argument(
    "--domain-build",
    type=Path,
    default=os.environ.get("FFPOOL_DOMAIN_BUILD_DIR", work / "build/capstone-domain"),
)
parser.add_argument(
    "--linux-build",
    type=Path,
    default=os.environ.get("FFPOOL_LINUX_BUILD_DIR", work / "build/linux-guest"),
)
parser.add_argument("--expected-status", type=int, default=0)
args = parser.parse_args()
if args.expected_status < 0:
    parser.error("expected status must be nonnegative")
mode = {"spatial": 0, "backing": 1, "sublet": 2}[args.protection]
run, hashes = stage_run(
    args.output,
    "qemu-",
    {
        "trace.bin": args.trace,
        "replay.dom": args.domain_build / "bin/replay.dom",
        "host.user": args.linux_build / "bin/domain-loader",
    },
)
share = run / "share"
expected = args.expected_status
(share / "run.sh").write_text(
    f"""#!/bin/sh
set -e
cp /mnt/host/replay.dom /tmp/ff2.dom
cp /mnt/host/host.user /tmp/ff2-host
cp /mnt/host/trace.bin /tmp/ff2-trace.bin
status=0
/tmp/ff2-host /tmp/ff2.dom /tmp/ff2-trace.bin /tmp/ff2-result.bin {mode} > /tmp/ff2.out || status=$?
cat /tmp/ff2.out
cp /tmp/ff2-result.bin /mnt/host/capstone.bin
test "$status" = {1 if expected else 0}
"""
    + (
        f'grep -q "^FF2 return={expected} status={expected} events=" /tmp/ff2.out\n'
        if expected
        else ""
    )
    + "echo FF2_DONE\n"
)
env = dict(os.environ)
env.setdefault("CAPSTONE_REV_NODES", "65536")
write_json(
    run / "manifest.json",
    {
        "program": "replay",
        "protection": args.protection,
        "node_capacity": int(env["CAPSTONE_REV_NODES"]),
        "sha256": hashes,
        "qemu_sha256": digest(env["CAPSTONE_QEMU_BINARY"]),
    },
)
print(f"FFmpeg artifacts: {run}", flush=True)
result = run_guest(run, "sh /mnt/host/run.sh", "FF2_DONE", env=env, lock_timeout=45)
write_json(
    run / "verdict.json",
    {"passed": result.returncode == 0, "runner_exit": result.returncode},
)
if result.returncode:
    raise SystemExit(result.returncode)
shutil.copyfile(share / "capstone.bin", args.output / "capstone.bin")
(args.output / "qemu-run-path.txt").write_text(str(run) + "\n")
