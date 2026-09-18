"""Replay allocator requests under QEMU, retaining every attempted run."""

import argparse
import os
from pathlib import Path
import shutil
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common/host"))
from port_support import digest, run_guest, stage_run, write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("trace", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--protection", choices=("spatial", "sublet"), default="spatial")
work = (
    Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone")) / "whisper-ggml-context"
)
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
(share / "run.sh").write_text(f"""#!/bin/sh
set -e
cp /mnt/host/replay.dom /tmp/wg.dom
cp /mnt/host/host.user /tmp/wg-host
cp /mnt/host/trace.bin /tmp/wg-trace.bin
/tmp/wg-host /tmp/wg.dom /tmp/wg-trace.bin /tmp/wg-report.bin {int(a.protection == 'sublet')}
cp /tmp/wg-report.bin /mnt/host/report.bin
echo WG_DONE
""")
write_json(
    run / "manifest.json",
    {
        "protection": a.protection,
        "sha256": hashes,
        "qemu_sha256": digest(os.environ["CAPSTONE_QEMU_BINARY"]),
        "compiler_sha256": digest(
            Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang"
        ),
        "node_capacity": int(os.environ.get("CAPSTONE_REV_NODES", "65536")),
    },
)
print(f"ggml artifacts: {run}", flush=True)
env = dict(os.environ)
env.setdefault("CAPSTONE_QEMU_LOGIN_TIMEOUT", "60")
env.setdefault("CAPSTONE_GUEST_COMMAND_TIMEOUT", "360")
result = run_guest(run, "sh /mnt/host/run.sh", "WG_DONE", env=env, timeout_multiplier=1)
ok = result.returncode == 0
if ok:
    report = struct.unpack("<16Q", (share / "report.bin").read_bytes())
    ok = report[0] == 0x315854434C4D4747 and report[3] == 0 and report[1] == report[4]
write_json(run / "verdict.json", {"passed": ok, "runner_exit": result.returncode})
if not ok:
    raise SystemExit(result.returncode or 1)
shutil.copyfile(share / "report.bin", a.output / "report.bin")
