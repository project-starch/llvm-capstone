"""Check live controls and exact-PC epoch/bounds faults in paired modes."""

import argparse
import json
import os
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "common/host"))
from port_support import digest, run_guest, stage_run, write_json

MAGIC = 0x31304D454D575357
# case: (faults unprotected, faults protected, labelled site: 0 read, 1 write, 2 widen)
EXPECTED = {
    0: (False, False, 0),
    1: (False, True, 0),
    2: (False, True, 1),
    3: (False, True, 0),
    4: (False, False, 0),
    5: (True, True, 0),
    6: (False, True, 0),
    7: (False, True, 0),
    8: (False, True, 0),
    9: (False, True, 0),
    10: (False, True, 0),
    11: (False, True, 0),
    12: (False, True, 2),
}
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
work = Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone")) / "wireshark-wmem"
p.add_argument("--domain-build", type=Path, default=work / "build/capstone-domain")
p.add_argument("--linux-build", type=Path, default=work / "build/linux-guest")
p.add_argument("--cases", default=",".join(str(c) for c in EXPECTED))
p.add_argument("--modes", default="0,1")
p.add_argument(
    "--resume",
    action="store_true",
    help="Explicitly retry unfinished cases, preserving prior attempts; passing binaries must match",
)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
binaries = {
    "security.dom": a.domain_build / "bin/wmem-security.dom",
    "host.user": a.linux_build / "bin/domain-loader",
}
verdicts = []
previous = a.output / "verdicts.json"
if previous.exists():
    if not a.resume:
        p.error("existing verdicts: choose a new output or explicitly --resume")
    requested = {
        (int(m), int(c)) for m in a.modes.split(",") for c in a.cases.split(",")
    }
    for row in json.loads(previous.read_text()):
        if (row["mode"], row["case"]) not in requested:
            p.error("resume must include all previously selected cases")
        run = Path(row["run"])
        if not (run / "verdict.json").exists():
            write_json(run / "verdict.json", row)
        if not row["passed"]:
            continue
        manifest = json.loads((run / "manifest.json").read_text())
        if any(manifest["sha256"][n] != digest(path) for n, path in binaries.items()):
            p.error("resume binaries changed; use a new output directory")
        if (
            manifest["qemu_sha256"] != digest(os.environ["CAPSTONE_QEMU_BINARY"])
            or manifest["compiler_sha256"]
            != digest(Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang")
            or manifest["node_capacity"]
            != int(os.environ.get("CAPSTONE_REV_NODES", "65536"))
        ):
            p.error("resume runtime changed; use a new output directory")
        if any(
            digest(run / "share" / n) != sha for n, sha in manifest["sha256"].items()
        ):
            p.error("retained input checksum mismatch")
        verdicts.append(row)
    if len({(r["mode"], r["case"]) for r in verdicts}) != len(verdicts):
        p.error("duplicate passing cases")
    print(
        f"Resuming with {len(verdicts)} verified passing cases; prior failed attempts retained",
        flush=True,
    )
elif a.resume:
    p.error("no previous verdicts to resume")
for mode in map(int, a.modes.split(",")):
    for case in map(int, a.cases.split(",")):
        if mode not in (0, 1) or case not in EXPECTED:
            p.error("invalid case or mode")
        if any(r["mode"] == mode and r["case"] == case for r in verdicts):
            continue
        run, hashes = stage_run(a.output, f"mode-{mode}-case-{case}-", binaries)
        share = run / "share"
        (share / "trace.bin").write_bytes(
            struct.pack("<22Q", MAGIC, 1, *([0] * 14), 0, 0, 0, 0, 0, case)
        )
        hashes["trace.bin"] = digest(share / "trace.bin")
        (share / "run.sh").write_text(f"""#!/bin/sh
set -e
cp /mnt/host/security.dom /tmp/wm.dom
cp /mnt/host/host.user /tmp/wm-host
cp /mnt/host/trace.bin /tmp/wm-trace.bin
/tmp/wm-host /tmp/wm.dom /tmp/wm-trace.bin /tmp/wm-report.bin {mode}
cp /tmp/wm-report.bin /mnt/host/report.bin
echo WM_SECURITY_DONE
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
        env = dict(os.environ)
        env.setdefault("CAPSTONE_QEMU_LOGIN_TIMEOUT", "60")
        env.setdefault("CAPSTONE_GUEST_COMMAND_TIMEOUT", "60")
        result = run_guest(
            run, "sh /mnt/host/run.sh", "WM_SECURITY_DONE", env=env, timeout_multiplier=1
        )
        serial_path = run / "serial.log"
        serial = serial_path.read_text(errors="replace") if serial_path.exists() else ""
        stage = f"Print = Scalar(0x{0xCF15000000000000 | case:x})"
        following = serial.split(stage, 1)[-1] if stage in serial else ""
        faults = re.findall(
            r"domain halted by capability fault: cause = (\d+), pc = (0x[0-9a-f]+)",
            serial,
        )
        unprotected, protected, site = EXPECTED[case]
        fault = protected if mode == 1 else unprotected
        row = {
            "mode": mode,
            "case": case,
            "run": str(run),
            "runner_exit": result.returncode,
            "expected": "fault" if fault else "completed",
        }
        if fault:
            sites = re.findall(
                r"Print = Cap\(\d+, 0x[0-9a-f]+, (0x[0-9a-f]+),", following
            )
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
                report = struct.unpack("<16Q", (share / "report.bin").read_bytes())
                row["passed"] &= report[0] == MAGIC and report[3] == 0 and report[4] == 1
        verdicts.append(row)
        write_json(run / "verdict.json", row)
        write_json(a.output / "verdicts.json", verdicts)
        print(f"{'PASS' if row['passed'] else 'FAIL'} mode={mode} case={case}", flush=True)
        if not row["passed"]:
            raise SystemExit(1)
