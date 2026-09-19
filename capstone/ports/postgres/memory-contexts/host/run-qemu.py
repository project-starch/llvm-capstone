#!/usr/bin/env python3
"""Run one prebuilt PostgreSQL domain; keep all inputs, identities and output."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common/host"))
from port_trace import record_trace
from port_support import write_replay_verdict, REPO_ROOT, run_guest, stage_run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("trace", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--protection", choices=("spatial", "sublet"), default="spatial")
work = Path(
    os.environ.get(
        "PG_WORK",
        Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone"))
        / "postgres-memory-contexts",
    )
)
parser.add_argument("--domain-build", type=Path, default=work / "build/capstone-domain")
parser.add_argument(
    "--linux-build",
    type=Path,
    default=os.environ.get("PG_LINUX_BUILD_DIR", work / "build/linux-guest"),
)
parser.add_argument(
    "--program",
    choices=("subpool-lifetimes", "context-hierarchy"),
    help="Select an explicit lifetime fixture instead of replay",
)
parser.add_argument("--memory-profile", action="store_true")
parser.add_argument("--separate-scratch", action="store_true")
args = parser.parse_args()
if args.program and args.protection != "sublet":
    parser.error("Lifetime fixtures require --protection sublet")
args.program = args.program or "replay-" + args.protection
repo = REPO_ROOT
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
run, input_hashes = stage_run(
    args.output,
    args.program + "-",
    {
        "domain.dom": image,
        "loader.user": loader,
        "trace.a11": args.trace,
    },
)
share = run / "share"
if args.program.startswith("replay-"):
    record_trace(run, share / "trace.a11", "postgres.a11")
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
    "sha256": input_hashes,
    "qemu_sha256": hashlib.sha256(qemu.read_bytes()).hexdigest(),
}
(run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"PostgreSQL artifacts: {run}", flush=True)
result = run_guest(run, command, done, env=env, timeout_multiplier=60)
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
write_replay_verdict(
    run, passed=passed, runner_exit=result.returncode, required=required
)

if not passed:
    print(f"PostgreSQL {args.program} failed; inspect {run}", file=sys.stderr)
    raise SystemExit(result.returncode or 1)
print(f"PostgreSQL {args.program}: passed")
