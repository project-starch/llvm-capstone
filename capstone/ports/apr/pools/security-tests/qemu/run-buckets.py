#!/usr/bin/env python3
"""Run the bucket allocator's lifetime fixtures as paired spatial/sublet arms.

One program (security-tests/capstone-domain/bucket-lifetimes.c, built through
the seam with -DAPRP_BUCKETS=ON), one boot per fixture and mode. The fixture
publishes its number and the three labelled sites -- the read, the write and
the allocator's own probe in apr_bucket_free -- and the oracle compares the
fault PC against what that boot printed. mode 0 must complete where the table
says so; mode 1 must fault at the named site.
"""

import argparse
import os
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "common/host"))
from port_support import digest, run_guest, stage_run, write_json

MAGIC = 0x314C4F4F50525041
MARKER_BASE = 0xCF1B000000000000
SITES = ("read", "write", "allocator")
# fixture: (mode 0 outcome, mode 1 outcome); an outcome is "complete",
# ("fault", site, causes) or ("status", code)
EXPECTED = {
    0: ("complete", "complete"),
    1: ("complete", ("fault", 0, (24, 25))),
    2: ("complete", ("fault", 1, (24, 25))),
    3: ("complete", ("fault", 0, (24, 25))),
    4: ("complete", ("fault", 0, (24, 25))),
    5: (("fault", 0, (5,)), ("fault", 0, (5,))),
    6: (("status", 538), ("fault", 2, (24, 25))),
}


def classify(serial, report, which, mode, runner_exit):
    faults = re.findall(
        r"domain (?:halted by capability fault|capability fault delivered): "
        r"cause = (\d+), pc = (0x[0-9a-f]+)",
        serial,
    )
    delivered = "capability fault delivered" in serial
    marker = f"Print = Scalar(0x{MARKER_BASE | which:x})"
    expected = EXPECTED[which][mode]
    row = {"fixture": which, "mode": mode, "expected": expected, "runner_exit": runner_exit}
    if isinstance(expected, tuple) and expected[0] == "fault":
        following = serial.split(marker, 1)[-1] if marker in serial else ""
        sites = re.findall(r"Print = Cap\(\d+, 0x[0-9a-f]+, (0x[0-9a-f]+),", following)
        cause, pc = faults[-1] if faults else ("0", "0")
        site = expected[1]
        want = sites[site] if len(sites) > site else None
        row.update(cause=int(cause), pc=pc, site=SITES[site], expected_pc=want, delivered=delivered)
        ok = marker in serial and len(faults) == 1 and want is not None and int(pc, 16) == int(want, 16) and int(cause) in expected[2]
        if delivered:
            ok = ok and "__EXIT_CODE__139" in serial
        row["passed"] = ok
    elif isinstance(expected, tuple) and expected[0] == "status":
        row["status"] = report[3] if report else None
        row["passed"] = not faults and marker in serial and report is not None and report[3] == expected[1]
    else:
        row["completed"] = report[4] if report else None
        row["passed"] = runner_exit == 0 and not faults and marker in serial and report is not None and report[4] == 1
    return row


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
p.add_argument("--image", type=Path, required=True, help="bin/defects.dom built from bucket-lifetimes.c")
p.add_argument("--linux-build", type=Path, required=True, help="the port's linux-guest build, for bin/domain-loader")
p.add_argument("--fixtures", default=",".join(str(i) for i in EXPECTED))
p.add_argument("--modes", default="0,1")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
print(f"Bucket fixture artifacts: {a.output}", flush=True)
verdicts = []
status = 0
for which in map(int, a.fixtures.split(",")):
    for mode in map(int, a.modes.split(",")):
        if which not in EXPECTED or mode not in (0, 1):
            p.error("invalid fixture or mode")
        run, hashes = stage_run(a.output, f"fixture-{which}-mode-{mode}-",
                                {"defects.dom": a.image, "host.user": a.linux_build / "bin/domain-loader"})
        share = run / "share"
        (share / "trace.bin").write_bytes(struct.pack("<16Q", MAGIC, 1, *([0] * 10), 0, which, 0, 0))
        hashes["trace.bin"] = digest(share / "trace.bin")
        (share / "run.sh").write_text(f"""#!/bin/sh
cp /mnt/host/defects.dom /tmp/apr-defects.dom
cp /mnt/host/host.user /tmp/apr-host
cp /mnt/host/trace.bin /tmp/apr-trace.bin
status=0
/tmp/apr-host /tmp/apr-defects.dom /tmp/apr-trace.bin /tmp/apr-report.bin {mode} || status=$?
echo __EXIT_CODE__$status
cp /tmp/apr-report.bin /mnt/host/report.bin 2>/dev/null || true
echo APR_DEFECT_DONE
""")
        write_json(run / "manifest.json", {"sha256": hashes,
                   "qemu_sha256": digest(os.environ["CAPSTONE_QEMU_BINARY"]),
                   "compiler_sha256": digest(Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang")})
        env = dict(os.environ)
        env.setdefault("CAPSTONE_QEMU_LOGIN_TIMEOUT", "90")
        env.setdefault("CAPSTONE_GUEST_COMMAND_TIMEOUT", "90")
        result = run_guest(run, "sh /mnt/host/run.sh", "APR_DEFECT_DONE", env=env, timeout_multiplier=1)
        serial = (run / "serial.log").read_text(errors="replace") if (run / "serial.log").exists() else ""
        if not serial:
            print(f"NO SERIAL CAPTURE for fixture={which} mode={mode}: {run}", flush=True)
            sys.exit(75)
        report = struct.unpack("<12Q", (share / "report.bin").read_bytes()) if (share / "report.bin").exists() else None
        row = classify(serial, report, which, mode, result.returncode)
        row["run"] = str(run)
        verdicts.append(row)
        write_json(run / "verdict.json", row)
        write_json(a.output / "verdicts.json", verdicts)
        detail = ""
        if "cause" in row:
            detail = f" cause={row['cause']} pc={row['pc']} expected={row['expected_pc']} site={row['site']}"
        elif "status" in row:
            detail = f" status={row['status']}"
        print(f"{'PASS' if row['passed'] else 'FAIL'} fixture={which} mode={mode}{detail}", flush=True)
        if not row["passed"]:
            status = 1
print(f"\n{sum(r['passed'] for r in verdicts)}/{len(verdicts)} arms passed")
sys.exit(status)
