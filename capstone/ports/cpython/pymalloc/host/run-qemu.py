#!/usr/bin/env python3
"""Stage a pymalloc replay and retain inputs, hashes, report and QEMU transcript."""

import argparse
import os
from pathlib import Path
import shutil
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common/host"))
from port_trace import record_trace
from port_support import write_replay_verdict, digest, run_guest, stage_run, write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("trace", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--protection", choices=("spatial", "sublet"), default="spatial")
work = Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone")) / "cpython-pymalloc"
p.add_argument("--domain-build", type=Path, default=work / "build/capstone-domain")
p.add_argument("--linux-build", type=Path, default=work / "build/linux-guest")
a = p.parse_args()
run, hashes = stage_run(
    a.output,
    "qemu-",
    {
        "trace.bin": a.trace,
        "replay.dom": a.domain_build / "bin/replay.dom",
        "host.user": a.linux_build / "bin/domain-loader",
    },
)
share = run / "share"
record_trace(run, share / "trace.bin", "cpython.pymalloc")
(share / "run.sh").write_text(f"""#!/bin/sh
set -e
cp /mnt/host/replay.dom /tmp/pym.dom
cp /mnt/host/host.user /tmp/pym-host
cp /mnt/host/trace.bin /tmp/pym-trace.bin
/tmp/pym-host /tmp/pym.dom /tmp/pym-trace.bin /tmp/pym-report.bin {int(a.protection == 'sublet')}
cp /tmp/pym-report.bin /mnt/host/report.bin
echo PYM_DONE
""")
env = dict(os.environ)
write_json(
    run / "manifest.json",
    {
        "protection": a.protection,
        "sha256": hashes,
        "qemu_sha256": digest(env["CAPSTONE_QEMU_BINARY"]),
        "compiler_sha256": digest(Path(env["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang"),
        "node_capacity": int(env.get("CAPSTONE_REV_NODES", "65536")),
    },
)
print(f"pymalloc artifacts: {run}", flush=True)
result = run_guest(run, "sh /mnt/host/run.sh", "PYM_DONE", env=env)
ok = result.returncode == 0
if ok:
    report = struct.unpack("<12Q", (share / "report.bin").read_bytes())
    ok = report[0] == 0x31594C50524D5950 and report[3] == 0 and report[1] == report[4]
write_replay_verdict(run, passed=ok, runner_exit=result.returncode)
if not ok:
    raise SystemExit(result.returncode or 1)
shutil.copyfile(share / "report.bin", a.output / "report.bin")
