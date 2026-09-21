#!/usr/bin/env python3
"""Run the wmem defects as paired spatial/sublet domain arms.

Each case is booted twice against the same program, which takes its arm from
the loader at run time. The spatial arm is the port's unprotected baseline:
objects are request-bounded, a packet-pool reset revokes nothing, so the stale
access succeeds and the run must COMPLETE. The sublet arm ends the epoch of
every object the reset retires, so the stale access must FAULT, and it must
fault at the labelled probe the case's oracle names -- the read probe, the
write probe, or the allocator's own probe when the stale pointer is handed
back to wmem -- rather than merely somewhere.
"""

import argparse
import json
import os
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "ports/common/host"))
from port_support import digest, run_guest, stage_run, write_json

MAGIC = 0x31304D454D575357
MARKER_BASE = 0xCF16000000000000
SITES = ("read", "write", "allocator")


def discover(domain_build):
    """The corpus contract builds ONE program per case, named NN-slug, and the
    directory names are the authority. Discover them rather than keep a second
    list that can drift out of step with the corpus."""
    found = {}
    for path in sorted((domain_build / "bin").glob("[0-9][0-9]-*.dom")):
        found[int(path.name[:2])] = (path.name[:-4], path)
    return found


def oracle_site(which):
    """Which labelled probe the case's sublet oracle names, or None when the
    oracle says the protected arm completes: a recorded non-detection, which
    is checked as a completion rather than accepted as a fault anywhere."""
    for d in CORPUS.glob(f"{which:02d}_*"):
        text = json.loads((d / "case.json").read_text())["arms"]["sublet"]["oracle"]
        if text.startswith("complete"):
            return None
        if "allocator" in text:
            return 2
        return 1 if "write probe" in text else 0
    raise SystemExit(f"no case.json for case {which}")


def classify(serial, which, mode, runner_exit, report):
    faults = re.findall(
        r"domain (?:halted by capability fault|capability fault delivered): "
        r"cause = (\d+), pc = (0x[0-9a-f]+)",
        serial,
    )
    marker = f"Print = Scalar(0x{MARKER_BASE | which:x})"
    site = oracle_site(which) if mode == "sublet" else None
    row = {
        "case": which,
        "mode": mode,
        "expected": "fault" if mode == "sublet" and site is not None else "complete",
        "runner_exit": runner_exit,
    }
    if mode == "sublet" and site is not None:
        following = serial.split(marker, 1)[-1] if marker in serial else ""
        sites = re.findall(r"Print = Cap\(\d+, 0x[0-9a-f]+, (0x[0-9a-f]+),", following)
        cause, pc = faults[-1] if faults else ("0", "0")
        expected = sites[site] if len(sites) > site else None
        row.update(
            cause=int(cause), pc=pc, site=SITES[site], expected_pc=expected,
            delivered="capability fault delivered" in serial,
        )
        row["passed"] = (
            marker in serial
            and len(faults) == 1
            and int(cause) in (24, 25)
            and expected is not None
            and int(pc, 16) == int(expected, 16)
        )
        if row["delivered"]:
            row["passed"] = row["passed"] and "__EXIT_CODE__139" in serial
    else:
        row["passed"] = (
            runner_exit == 0
            and not faults
            and marker in serial
            and report is not None
            and report[0] == MAGIC
            and report[3] == 0
            and report[4] == 1
        )
    return row


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
p.add_argument("--domain-build", type=Path, required=True)
p.add_argument("--linux-build", type=Path, required=True)
p.add_argument("--cases", default=None)
p.add_argument("--modes", default="spatial,sublet")
p.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parent.parent,
               help="corpus root whose case.json oracles name the expected probe")
p.add_argument(
    "--negative-control", action="store_true",
    help="corrupt the input record so the program refuses it before any case runs; "
         "every selected arm must then FAIL, and the exit status is 0 only if every one did",
)
a = p.parse_args()
CORPUS = a.corpus.resolve()
programs = discover(a.domain_build)
if not programs:
    p.error(f"no NN-*.dom programs in {a.domain_build}/bin; a corpus that built nothing must not look like one that passed")
cases = [int(c) for c in a.cases.split(",")] if a.cases else sorted(programs)
a.output.mkdir(parents=True, exist_ok=True)
print(f"Defect suite artifacts: {a.output}", flush=True)
verdicts = []
status = 0
for which in cases:
    if which not in programs:
        p.error(f"no program for case {which}")
    for mode in a.modes.split(","):
        if mode not in ("spatial", "sublet"):
            p.error("invalid mode")
        stem, image = programs[which]
        run, hashes = stage_run(
            a.output, f"{stem}-{mode}-",
            {"defect.dom": image, "host.user": a.linux_build / "bin/domain-loader"},
        )
        share = run / "share"
        # A count of 2 with one record fails the driver's CHECK(in->count == 1):
        # the program refuses the input before any case runs.
        count = 2 if a.negative_control else 1
        (share / "trace.bin").write_bytes(
            struct.pack("<22Q", MAGIC, count, *([0] * 14), 0, 0, 0, 0, 0, which)
        )
        hashes["trace.bin"] = digest(share / "trace.bin")
        (share / "run.sh").write_text(f"""#!/bin/sh
set -e
cp /mnt/host/defect.dom /tmp/wm.dom
cp /mnt/host/host.user /tmp/wm-host
cp /mnt/host/trace.bin /tmp/wm-trace.bin
/tmp/wm-host /tmp/wm.dom /tmp/wm-trace.bin /tmp/wm-report.bin {int(mode == 'sublet')}
cp /tmp/wm-report.bin /mnt/host/report.bin
echo WM_DEFECT_DONE
""")
        write_json(
            run / "manifest.json",
            {
                "sha256": hashes,
                "qemu_sha256": digest(os.environ["CAPSTONE_QEMU_BINARY"]),
                "compiler_sha256": digest(Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang"),
                "node_capacity": int(os.environ.get("CAPSTONE_REV_NODES", "65536")),
            },
        )
        env = dict(os.environ)
        env.setdefault("CAPSTONE_QEMU_LOGIN_TIMEOUT", "60")
        env.setdefault("CAPSTONE_GUEST_COMMAND_TIMEOUT", "60")
        result = run_guest(run, "sh /mnt/host/run.sh", "WM_DEFECT_DONE", env=env, timeout_multiplier=1)
        serial = (run / "serial.log").read_text(errors="replace") if (run / "serial.log").exists() else ""
        report = struct.unpack("<16Q", (share / "report.bin").read_bytes()) if (share / "report.bin").exists() else None
        row = classify(serial, which, mode, result.returncode, report)
        row.update(name=stem, run=str(run))
        verdicts.append(row)
        write_json(run / "verdict.json", row)
        write_json(a.output / "verdicts.json", verdicts)
        extra = ""
        if row["expected"] == "fault":
            extra = f" cause={row['cause']} pc={row['pc']} expected={row['expected_pc']} site={row['site']}"
        elif mode == "sublet":
            extra = " (recorded non-detection: completion required)"
        print(f"{'OK  ' if row['passed'] else 'FAIL'} case={which} {stem:40} {mode:8}{extra}", flush=True)
        if not row["passed"]:
            status = 1
passed = sum(r["passed"] for r in verdicts)
if a.negative_control:
    print(f"\nnegative control: {len(verdicts) - passed}/{len(verdicts)} oracles reported FAIL as they must")
    sys.exit(0 if passed == 0 else 1)
print(f"\n{passed}/{len(verdicts)} arms passed")
sys.exit(status)
