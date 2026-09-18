#!/usr/bin/env python3
"""Run one prebuilt PostgreSQL domain; keep all inputs, identities and output."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("domain_build", type=Path)
parser.add_argument("linux_build", type=Path)
parser.add_argument(
    "program",
    choices=(
        "replay-spatial",
        "replay-sublet",
        "subpool-lifetimes",
        "context-hierarchy",
    ),
)
parser.add_argument("trace", type=Path)
parser.add_argument("--results", type=Path)
parser.add_argument("--memory-profile", action="store_true")
parser.add_argument("--separate-scratch", action="store_true")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[4]
regions = json.loads((args.domain_build / "regions.json").read_text())
if regions != json.loads((args.linux_build / "regions.json").read_text()):
    raise SystemExit(
        "Domain and Linux loader region configurations differ; rebuild them with matching settings"
    )
image = args.domain_build / "bin" / (args.program + ".dom")
loader = args.linux_build / "bin/domain-loader"
for path in (image, loader, args.trace):
    if not path.is_file():
        raise SystemExit(
            f"Missing input: {path}; configure/build the matching preset first"
        )
if args.trace.stat().st_size > regions["trace"]:
    raise SystemExit("Trace exceeds the configured input region")
results = args.results or args.domain_build / "test-results"
results.mkdir(parents=True, exist_ok=True)
run = Path(tempfile.mkdtemp(prefix=args.program + "-", dir=results))
share = run / "share"
share.mkdir()
for source, name in (
    (image, "domain.dom"),
    (loader, "loader.user"),
    (args.trace, "trace.a11"),
):
    shutil.copyfile(source, share / name)
markers = {
    "replay-spatial": (
        "__CAPSTONE_PG_REPLAY_DONE__",
        ["__CAPSTONE_PG_REPLAY_BALANCED__"],
    ),
    "replay-sublet": (
        "__CAPSTONE_PG_REPLAY_DONE__",
        ["__CAPSTONE_PG_REPLAY_BALANCED__", "__CAPSTONE_PG_SUBLET_ONE_EACH__"],
    ),
    "subpool-lifetimes": (
        "__CAPSTONE_PG_SUBPOOL_DONE__",
        ["__CAPSTONE_PG_SUBPOOL_GOOD__"],
    ),
    "context-hierarchy": ("__CAPSTONE_PG_HIER_DONE__", ["__CAPSTONE_PG_HIER_GOOD__"]),
}
done, required = markers[args.program]
command = "cp /mnt/host/loader.user /tmp/pg-loader && chmod 0755 /tmp/pg-loader && /tmp/pg-loader /mnt/host/domain.dom /mnt/host/trace.a11"
if args.program != "replay-spatial":
    command += " --linear-arena"
if args.program != "replay-spatial" or args.memory_profile or args.separate_scratch:
    command += f" --scratch {regions['scratch']}"
if args.memory_profile:
    if args.program not in ("replay-spatial", "replay-sublet"):
        raise SystemExit("Memory profiling accepts replay programs only")
    required.extend([done, "__CAPSTONE_PG_MEMORY_DONE__"])
    done = "__CAPSTONE_PG_HOST_DONE__"
    command += " --report-file /mnt/host/payload.log"
else:
    command += " --tail"
env = dict(os.environ, CAPSTONE_REPO_ROOT=str(repo))
env.setdefault("CAPSTONE_GP_NONLIN", "1")
env.setdefault("CAPSTONE_REV_NODES", "1048576")
qemu = Path(env["CAPSTONE_QEMU_BINARY"])
manifest = {
    "program": args.program,
    "memory_profile": args.memory_profile,
    "separate_scratch": args.separate_scratch or args.memory_profile,
    "regions": regions,
    "node_capacity": int(env["CAPSTONE_REV_NODES"]),
    "sha256": {
        name: hashlib.sha256((share / name).read_bytes()).hexdigest()
        for name in ("domain.dom", "loader.user", "trace.a11")
    },
    "qemu_sha256": hashlib.sha256(qemu.read_bytes()).hexdigest(),
}
(run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
lock = Path(
    env.get("CAPSTONE_QEMU_LOCK", str(Path.home() / ".capstone-locks/qemu.lock"))
)
lock.parent.mkdir(parents=True, exist_ok=True)
smoke = [
    sys.executable,
    str(repo / "capstone/tests/runtime-qemu/run-domain-smoke.py"),
    "--share-dir",
    str(share),
    "--log-file",
    str(run / "serial.log"),
    "--kernel-arg",
    "cma=512M",
    "--timeout-multiplier",
    "60",
    "--guest-command",
    command,
    "--success-marker",
    done,
]
print(f"PostgreSQL artifacts: {run}", flush=True)
result = subprocess.run(["flock", "-w", "60", str(lock), *smoke], env=env)
text = (
    (run / "serial.log").read_text(errors="replace")
    if (run / "serial.log").exists()
    else ""
)
if args.memory_profile:
    report = share / "payload.log"
    text += report.read_text(errors="replace") if report.exists() else ""
passed = (
    result.returncode == 0
    and all(m in text for m in required)
    and "_FAILED__" not in text
    and "_BAD__" not in text
)
(run / "verdict.json").write_text(
    json.dumps(
        {"passed": passed, "runner_exit": result.returncode, "required": required},
        indent=2,
    )
    + "\n"
)
if not passed:
    raise SystemExit(f"PostgreSQL {args.program} failed; inspect {run}")
print(f"PostgreSQL {args.program}: passed")
