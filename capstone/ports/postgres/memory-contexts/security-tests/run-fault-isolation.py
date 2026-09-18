#!/usr/bin/env python3
"""Prove cooperative domain fault recovery and guest-process isolation in QEMU."""

import argparse
import json
import os
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common/host"))
from port_support import digest, run_guest, stage_run, write_json

sys.path.insert(
    0, str(Path(__file__).resolve().parents[4] / "runtime/tests/fault-recovery")
)
from verdicts import FAULT, SITE, clean


def isolation_passed(serial, status):
    faults = re.findall(FAULT, serial)
    if not (
        clean(serial, status)
        and len(faults) == 3
        and serial.count("FAULT_ISOLATION CHILD SIGSEGV") == 3
        and serial.count("FAULT_ISOLATION HEALTHY") == 2
        and serial.count("PG_CLIENT generation RESULT 0") == 2
        and "__CAPSTONE_FAULT_ISOLATION_DONE__" in serial
        and re.findall(r"FAULT_ISOLATION (HEALTHY|CHILD SIGSEGV)", serial)
        == ["HEALTHY", "CHILD SIGSEGV", "CHILD SIGSEGV", "CHILD SIGSEGV", "HEALTHY"]
    ):
        return False
    if serial.rfind("__CAPSTONE_FAULT_ISOLATION_DONE__") < serial.rfind(
        "FAULT_ISOLATION HEALTHY"
    ):
        return False
    # Each intentional stale access must be the actual delivered fault site.
    blocks = serial.split("FAULT_ISOLATION CHILD SIGSEGV")
    for kind in range(3):
        marker = f"Print = Scalar(0x{0xcf17000000000000 | (kind << 8) | 2:x})"
        block = blocks[kind]
        if serial.count(marker) != 1 or block.count(marker) != 1:
            return False
        after = block.split(marker, 1)[1]
        sites = re.findall(SITE, after)
        delivered = re.findall(FAULT, after)
        if len(sites) != 2 or len(delivered) != 1 or delivered[0][1] != sites[0]:
            return False
        if delivered[0][0] not in ("24", "25"):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-build", type=Path, required=True)
    parser.add_argument("--linux-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    regions = json.loads((args.domain_build / "regions.json").read_text())
    if regions != json.loads((args.linux_build / "regions.json").read_text()):
        parser.error("domain and Linux region sizes differ")
    # Current monitor allocations survive close; bound the same-boot footprint.
    if regions["arena"] > 8 * 1024**2 or regions["trace"] > 1024**2:
        parser.error(
            "configure both builds with PG_ARENA_BYTES=8388608 PG_TRACE_BYTES=1048576"
        )
    env = dict(os.environ)
    env.setdefault("CAPSTONE_QEMU_LOGIN_TIMEOUT", "60")
    env.setdefault("CAPSTONE_GUEST_COMMAND_TIMEOUT", "90")
    env.setdefault("CAPSTONE_GP_NONLIN", "1")
    env.setdefault("CAPSTONE_REV_NODES", "8388608")
    inputs = {
        "loader.user": args.linux_build / "bin/domain-loader",
        "supervisor.user": args.linux_build / "bin/fault-supervisor",
        "healthy.dom": args.domain_build / "bin/client-generation-sublet.dom",
        "fault.dom": args.domain_build / "bin/contexts-sublet.dom",
    }
    run, hashes = stage_run(args.output, "fault-allocators-", inputs)
    for kind in range(3):
        (run / f"share/case-{kind}.bin").write_bytes(struct.pack("<II", kind, 2))
    (run / "share/empty.bin").write_bytes(b"")
    command = (
        "cp /mnt/host/loader.user /tmp/fault-loader && "
        "cp /mnt/host/supervisor.user /tmp/fault-supervisor && "
        "chmod 0755 /tmp/fault-loader /tmp/fault-supervisor && "
        "/tmp/fault-supervisor /tmp/fault-loader /mnt/host/healthy.dom "
        "/mnt/host/empty.bin /mnt/host/fault.dom "
        "/mnt/host/case-0.bin /mnt/host/case-1.bin /mnt/host/case-2.bin"
    )
    hashes.update({p.name: digest(p) for p in (run / "share").glob("*.bin")})
    write_json(
        run / "manifest.json",
        {
            "case": "allocators",
            "sha256": hashes,
            "regions": regions,
            "qemu_sha256": digest(env["CAPSTONE_QEMU_BINARY"]),
            "compiler_sha256": digest(
                Path(env["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang"
            ),
            "revocation_nodes": int(env["CAPSTONE_REV_NODES"]),
        },
    )
    print(f"Fault artifacts: {run}", flush=True)
    result = run_guest(
        run, command, "__CAPSTONE_FAULT_ISOLATION_DONE__", env=env, timeout_multiplier=1
    )
    log = run / "serial.log"
    serial = log.read_text(errors="replace") if log.exists() else ""
    ok = isolation_passed(serial, result.returncode)
    write_json(run / "verdict.json", {"passed": ok, "runner_exit": result.returncode})
    if not ok:
        raise SystemExit(f"FAIL allocators; inspect {run}")
    print("PASS allocators", flush=True)


if __name__ == "__main__":
    main()
