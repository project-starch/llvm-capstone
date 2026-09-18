#!/usr/bin/env python3
"""Pair live controls with stale accesses; require the precise fault instruction."""

import argparse
import os
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "common/host"))
from port_support import digest, run_guest, stage_run, write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
work = Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone")) / "cpython-pymalloc"
p.add_argument("--domain-build", type=Path, default=work / "build/capstone-domain")
p.add_argument("--linux-build", type=Path, default=work / "build/linux-guest")
p.add_argument("--cases", default="0,1,2,3,4,5,6,7,8")
p.add_argument("--modes", default="0,1")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
verdicts = []
for mode in map(int, a.modes.split(",")):
    for case in map(int, a.cases.split(",")):
        if mode not in (0, 1) or not 0 <= case <= 8:
            p.error("invalid case or mode")
        run, hashes = stage_run(
            a.output,
            f"mode-{mode}-case-{case}-",
            {
                "security.dom": a.domain_build / "bin/pool-security.dom",
                "host.user": a.linux_build / "bin/domain-loader",
            },
        )
        share = run / "share"
        (share / "trace.bin").write_bytes(
            struct.pack("<16Q", 0x31594C50524D5950, 1, *([0] * 10), 0, case, 0, 0)
        )
        hashes["trace.bin"] = digest(share / "trace.bin")
        rejection = mode == 1 and case == 3
        fault = case in (5, 7) or (mode == 1 and case in (1, 2, 4, 6, 8))
        (share / "run.sh").write_text(f"""#!/bin/sh
set -e
cp /mnt/host/security.dom /tmp/pym-security.dom
cp /mnt/host/host.user /tmp/pym-host
cp /mnt/host/trace.bin /tmp/pym-trace.bin
status=0
/tmp/pym-host /tmp/pym-security.dom /tmp/pym-trace.bin /tmp/pym-report.bin {mode} || status=$?
cp /tmp/pym-report.bin /mnt/host/report.bin
test "$status" = {1 if rejection else 0}
echo PYM_SECURITY_DONE
""")
        write_json(
            run / "manifest.json",
            {
                "sha256": hashes,
                "qemu_sha256": digest(os.environ["CAPSTONE_QEMU_BINARY"]),
                "compiler_sha256": digest(
                    Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang"
                ),
                "node_capacity": int(os.environ.get("CAPSTONE_REV_NODES", "65536")),
            },
        )
        result = run_guest(run, "sh /mnt/host/run.sh", "PYM_SECURITY_DONE")
        serial = (run / "serial.log").read_text(errors="replace")
        stage = f"Print = Scalar(0x{0xcf13000000000000 | case:x})"
        following = serial.split(stage, 1)[-1] if stage in serial else ""
        faults = re.findall(
            r"domain halted by capability fault: cause = (\d+), pc = (0x[0-9a-f]+)",
            serial,
        )
        row = {
            "mode": mode,
            "case": case,
            "run": str(run),
            "runner_exit": result.returncode,
            "expected": "fault" if fault else "rejected" if rejection else "completed",
        }
        if fault:
            sites = re.findall(
                r"Print = Cap\(\d+, 0x[0-9a-f]+, (0x[0-9a-f]+),", following
            )
            site = 1 if case == 2 else 0
            cause, pc = faults[-1] if faults else ("0", "0")
            row.update(
                cause=int(cause),
                pc=pc,
                expected_pc=sites[site] if len(sites) > site else None,
            )
            row["passed"] = (
                stage in serial
                and len(sites) > site
                and int(pc, 16) == int(sites[site], 16)
                and int(cause) in ((5,) if case == 5 else (24, 25))
            )
            if case == 5:
                row["passed"] &= "Cap mem access OOB:" in following
        else:
            row["passed"] = result.returncode == 0 and not faults and stage in serial
            if row["passed"]:
                report = struct.unpack("<12Q", (share / "report.bin").read_bytes())
                row["passed"] &= report[3] == (516 if rejection else 0)
        verdicts.append(row)
        write_json(a.output / "verdicts.json", verdicts)
        print(
            f"{'PASS' if row['passed'] else 'FAIL'} mode={mode} case={case} expected={row['expected']}",
            flush=True,
        )
        if not row["passed"]:
            raise SystemExit(1)
