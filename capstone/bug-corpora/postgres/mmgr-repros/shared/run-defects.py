#!/usr/bin/env python3
"""Run the eight memory-context defects as paired spatial/sublet domain arms.

Each case is booted twice against the same logic. The spatial arm is this
port's unprotected baseline -- chunks are offsets inside one arena capability,
so the stale access succeeds and the run must COMPLETE. The sublet arm gives
every chunk its own capability and revokes it on free, so the stale access must
FAULT, and for every case but the first it must fault at the labelled probe
instruction rather than merely somewhere.

Case 0 is the exception and says so: its stale access is a second pfree, so the
manager faults reading a revoked chunk header before any bookkeeping runs. Its
oracle accepts a fault anywhere, and it is the only one that does.
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

CASES = [
    ("1f5b6a5e5d", "tuplestore-double-pfree", "aset"),
    ("3549ffb6af", "dead-items-stale-after-reset", "aset"),
    ("83ce20d671", "dead-items-2024-instance", "aset"),
    ("ed394c4bdf", "live-parts-stale-alias", "aset"),
    ("727bc6ac33f6", "child-join-frees-parent-relids", "aset"),
    ("9d5ce4f1a00a", "windowagg-reset-keeps-byref", "aset"),
    ("a61592253e", "pgoutput-ancestor-delete", "aset"),
    ("9e0b4b1ab5", "reorderbuffer-slab-reuse", "slab"),
]

MARKER_BASE = 0xCF18000000000000


def classify(serial, which, mode, runner_exit):
    # Two emulator behaviours, and the difference is the whole point of the
    # delivery work. Without the local-trap-delivery change the fault HALTS the
    # domain and QEMU exits, so the guest never returns; with it the fault is
    # DELIVERED, the launcher dies by SIGSEGV, and the VM keeps running. Accept
    # both, and when the fault was delivered demand the process-death evidence
    # rather than treating a surviving VM as a weaker result.
    faults = re.findall(
        r"domain (?:halted by capability fault|capability fault delivered): "
        r"cause = (\d+), pc = (0x[0-9a-f]+)",
        serial,
    )
    delivered = "capability fault delivered" in serial
    marker = f"Print = Scalar(0x{MARKER_BASE | which:x})"
    row = {
        "case": which,
        "fix": CASES[which][0],
        "name": CASES[which][1],
        "allocator": CASES[which][2],
        "mode": mode,
        "expected": "fault" if mode == "sublet" else "complete",
        "runner_exit": runner_exit,
    }
    if mode == "sublet":
        following = serial.split(marker, 1)[-1] if marker in serial else ""
        sites = re.findall(r"Print = Cap\(\d+, 0x[0-9a-f]+, (0x[0-9a-f]+),", following)
        cause, pc = faults[-1] if faults else ("0", "0")
        # Case 0 faults inside the manager, so it declares no expected PC.
        expected = None if which == 0 else (sites[0] if sites else None)
        row.update(cause=int(cause), pc=pc, expected_pc=expected, delivered=delivered)
        ok = (
            marker in serial
            and len(faults) == 1
            and "_FAILED__" not in serial
            and int(cause) in (24, 25)
        )
        if which != 0:
            ok = ok and expected is not None and int(pc, 16) == int(expected, 16)
        if delivered:
            # The VM survived, so the claim is containment and it has to be
            # evidenced: the launcher process died by SIGSEGV, and the guest got
            # its shell back to say so. That marker can only appear if a command
            # ran AFTER the fault, which is the whole property.
            #
            # runner_exit stays 1 here and that is correct, not a failure: the
            # generic guest runner waits for the normal completion marker, which
            # a killed launcher never prints. Requiring 0 would reject exactly
            # the arms that demonstrate containment.
            ok = ok and "__EXIT_CODE__139" in serial
        row["passed"] = ok
    else:
        row["passed"] = (
            runner_exit == 0
            and not faults
            and marker in serial
            and "__CAPSTONE_PG_DEFECT_COMPLETED__" in serial
            and "_FAILED__" not in serial
        )
    return row


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
p.add_argument("--domain-build", type=Path, required=True)
p.add_argument("--linux-build", type=Path, required=True)
p.add_argument("--cases", default=",".join(str(i) for i in range(len(CASES))))
p.add_argument("--modes", default="spatial,sublet")
a = p.parse_args()

regions = json.loads((a.domain_build / "regions.json").read_text())
if regions != json.loads((a.linux_build / "regions.json").read_text()):
    p.error("domain and guest region configurations differ")
a.output.mkdir(parents=True, exist_ok=True)
print(f"Defect suite artifacts: {a.output}", flush=True)

verdicts = []
status = 0
for which in map(int, a.cases.split(",")):
    if not 0 <= which < len(CASES):
        p.error(f"no such case: {which}")
    for mode in a.modes.split(","):
        if mode not in ("spatial", "sublet"):
            p.error("invalid mode")
        run, hashes = stage_run(
            a.output,
            f"{which}-{mode}-",
            {
                "defects.dom": a.domain_build / f"bin/defects-{mode}.dom",
                "loader.user": a.linux_build / "bin/domain-loader",
            },
        )
        (run / "share/selection.bin").write_bytes(struct.pack("<II", which, 0))
        hashes["selection.bin"] = digest(run / "share/selection.bin")
        write_json(
            run / "manifest.json",
            {
                "sha256": hashes,
                "regions": regions,
                "qemu_sha256": digest(os.environ["CAPSTONE_QEMU_BINARY"]),
                "compiler_sha256": digest(
                    Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang"
                ),
            },
        )
        command = (
            "cp /mnt/host/loader.user /tmp/pg-loader && chmod 0755 /tmp/pg-loader"
            " && /tmp/pg-loader /mnt/host/defects.dom /mnt/host/selection.bin --tail"
        )
        if mode == "sublet":
            command += " --linear-arena"
        env = dict(os.environ)
        env.setdefault("CAPSTONE_QEMU_LOGIN_TIMEOUT", "60")
        env.setdefault("CAPSTONE_GUEST_COMMAND_TIMEOUT", "60")
        result = run_guest(
            run, command, "__CAPSTONE_PG_HOST_DONE__", env=env, timeout_multiplier=1
        )
        serial = (
            (run / "serial.log").read_text(errors="replace")
            if (run / "serial.log").exists()
            else ""
        )
        row = classify(serial, which, mode, result.returncode)
        row["run"] = str(run)
        verdicts.append(row)
        write_json(run / "verdict.json", row)
        write_json(a.output / "verdicts.json", verdicts)
        flag = "OK  " if row["passed"] else "FAIL"
        extra = ""
        if mode == "sublet" and row.get("cause") is not None:
            extra = f" cause={row['cause']} pc={row.get('pc')}"
            if row.get("expected_pc"):
                extra += f" expected={row['expected_pc']}"
        print(f"{flag} case={which} {CASES[which][1]:32} {mode:8}{extra}", flush=True)
        if not row["passed"]:
            status = 1

print(f"\n{sum(r['passed'] for r in verdicts)}/{len(verdicts)} arms passed")
sys.exit(status)
