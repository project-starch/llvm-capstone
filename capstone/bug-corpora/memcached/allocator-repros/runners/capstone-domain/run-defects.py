#!/usr/bin/env python3
"""Run the memcached allocator defects as paired spatial/sublet domain arms.

Each case is booted twice against the same binary; the port chooses the arm at
runtime from the mode argument, so the two arms differ in exactly one thing.

  spatial (mode 0)  a chunk or cache object keeps the alias it was carved
                    with across its allocator's free list, so a pointer held
                    over slabs_free or cache_free still names live storage --
                    the next item's -- and the run must COMPLETE.
  sublet  (mode 1)  release and issue each revoke the unit and mint a fresh
                    alias, so that pointer is dead and the first read through
                    it must FAULT -- at the labelled probe, not merely
                    somewhere.
  sublet-malloc (mode 2)  Sublet only as the system allocator under a stock
                    slabs.c and cache.c: a chunk carries its whole page's bound,
                    an object its own, and nothing is revoked until a page or an
                    object is given back. Its oracle is the case's
                    `sublet-malloc` arm, like the other two.

The expected PC is not hardcoded. The domain publishes both probe addresses
through its marker, and the oracle compares the fault PC against what that boot
printed, so a relink cannot silently turn the check into a tautology.
"""

import argparse
import json
import os
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "ports/common/host"))
from port_support import digest, run_guest, stage_run, write_json

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[1]  # same anchor as the sibling runners


def load_cases():
    """The corpus's own case.json files, indexed by case number.

    This WAS a hardcoded five-entry list, which meant adding a case to the corpus silently
    left this runner measuring the old set -- the drift the contract warns about, and the
    reason cases 5-7 could not be measured when they landed. The two sibling runners
    (cheribsd/, poisoncap/) already discover; this one now does too, by the same pattern.
    The tuple shape (fix, name, shape) is preserved so every existing use site is unchanged.
    """
    found = {}
    for path in sorted(CORPUS.glob("[0-9][0-9]_*/case.json")):
        claim = json.loads(path.read_text())
        number = claim["case"]
        if number in found:
            raise SystemExit(f"two case.json files claim case {number}")
        found[number] = (claim["upstream_fix"],
                         path.parent.name.split("_", 2)[2].replace("_", "-"),
                         claim["shape"])
    if not found or sorted(found) != list(range(len(found))):
        raise SystemExit("the corpus case numbers are not 0..N-1")
    return [found[i] for i in range(len(found))]


CASES = load_cases()

MAGIC = 0x315342414C53434D  # "MCSLABS1"
MARKER_BASE = 0xCF1C000000000000
REPORT_FIELDS = 12  # struct mcp_header: 12 x uint64
COMPLETED = 4  # its index


def arm_oracle(which, mode):
    """What the case's OWN case.json arm says this mode should do.

    This runner used to assume every row is temporal: the `spatial` mode always completes,
    the `sublet` mode always faults, the probe is always the READ one, and the cause is
    always 24 or 25. All four are wrong for a SPATIAL row -- cases 5-7 are spatial, case 5
    faults on BOTH modes with cause 7 (a store), and cases 6-7 complete on both. The
    assumptions are now read from the arm instead, which is what the sibling wireshark
    runner already does.

    Returns (expect_fault, site, causes): site 0 is the read probe, 1 the write probe, in
    the order mark() publishes them; causes is the tuple of acceptable fault causes."""
    for d in CORPUS.glob(f"{which:02d}_*"):
        arms = json.loads((d / "case.json").read_text())["arms"]
        arm = arms.get(mode)
        if arm is None:
            raise SystemExit(f"case {which} has no {mode} arm")
        text = str(arm.get("oracle", ""))
        low = text.lower()
        if "completes" in low or low.startswith("complete"):
            return False, None, ()
        site = 1 if "write probe" in low else 0
        declared = arm.get("cause")
        return True, site, (int(declared),) if declared is not None else (24, 25)
    raise SystemExit(f"no case.json for case {which}")


def classify(serial, report, which, mode, runner_exit):
    # Two emulator behaviours, and both are real results. Without local trap
    # delivery the fault HALTS the domain and QEMU exits; with it the fault is
    # DELIVERED, the launcher dies by SIGSEGV and the VM survives. Accept both
    # and record which happened, rather than treating a surviving VM as weaker.
    faults = re.findall(
        r"domain (?:halted by capability fault|capability fault delivered): "
        r"cause = (\d+), pc = (0x[0-9a-f]+)",
        serial,
    )
    delivered = "capability fault delivered" in serial
    marker = f"Print = Scalar(0x{MARKER_BASE | which:x})"
    fix, name, shape = CASES[which]
    expect_fault, site, causes = arm_oracle(which, mode)
    row = {
        "case": which,
        "fix": fix,
        "name": name,
        "shape": shape,
        "mode": mode,
        "expected": "fault" if expect_fault else "complete",
        "oracle_arm": mode,
        "runner_exit": runner_exit,
    }
    if expect_fault:
        following = serial.split(marker, 1)[-1] if marker in serial else ""
        sites = re.findall(r"Print = Cap\(\d+, 0x[0-9a-f]+, (0x[0-9a-f]+),", following)
        cause, pc = faults[-1] if faults else ("0", "0")
        expected = sites[site] if len(sites) > site else None
        row.update(cause=int(cause), pc=pc, expected_pc=expected, delivered=delivered,
                   site="write" if site else "read")
        ok = (
            marker in serial
            and len(faults) == 1
            and expected is not None
            and int(pc, 16) == int(expected, 16)
            and int(cause) in causes
        )
        if delivered:
            # The VM survived, so the claim is containment and it has to be
            # evidenced: the launcher died by SIGSEGV and the guest got its
            # shell back to say so, which can only happen if a command ran
            # AFTER the fault. runner_exit stays non-zero here and that is
            # correct: the generic runner waits for a marker a killed launcher
            # never prints.
            ok = ok and "__EXIT_CODE__139" in serial
        row["passed"] = ok
    else:
        row["completed"] = report[COMPLETED] if report else None
        row["passed"] = (
            runner_exit == 0
            and not faults
            and marker in serial
            and report is not None
            and report[COMPLETED] == 1
        )
    return row


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
tmp = Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone"))
p.add_argument(
    "--domain-build",
    type=Path,
    default=tmp / "memcached-allocator-repros/domain",
    help="where shared/build-cases.sh capstone-domain put bin/defect-NN.dom",
)
p.add_argument(
    "--linux-build",
    type=Path,
    default=tmp / "memcached-allocators/build/linux-guest",
    help="the port's linux-guest build, for bin/domain-loader",
)
p.add_argument("--cases", default=",".join(str(i) for i in range(len(CASES))))
p.add_argument("--modes", default="spatial,sublet")
MODE_NUMBER = {"spatial": 0, "sublet": 1, "sublet-malloc": 2,
               # mode 1 on images built with MC_CARVE_BOUNDS, judged by the case's `sublet-carve` arm
               "sublet-carve": 1}
p.add_argument(
    "--negative-control",
    action="store_true",
    help="Corrupt the input so the domain refuses to run the case, and require "
    "EVERY arm to be reported failing. A suite whose oracles cannot say FAIL "
    "proves nothing by saying PASS, so this inverts the exit status: 0 means "
    "every arm failed as it must, 1 means an oracle is vacuous.",
)
a = p.parse_args()

a.output.mkdir(parents=True, exist_ok=True)
print(f"Defect suite artifacts: {a.output}", flush=True)

verdicts = []
status = 0
for which in map(int, a.cases.split(",")):
    if not 0 <= which < len(CASES):
        p.error(f"no such case: {which}")
    for mode in a.modes.split(","):
        if mode not in MODE_NUMBER:
            p.error("invalid mode")
        number = MODE_NUMBER[mode]
        run, hashes = stage_run(
            a.output,
            f"{which:02d}-{CASES[which][1]}-{mode}-",
            {
                "defects.dom": a.domain_build / f"bin/defect-{which:02d}.dom",
                "host.user": a.linux_build / "bin/domain-loader",
            },
        )
        share = run / "share"
        # One event, carrying the case number in its id. Under
        # --negative-control the count is wrong, so the case's own CHECK
        # refuses the input and no defect is performed: no fault for the
        # sublet oracle, no completion for the spatial one.
        count = 2 if a.negative_control else 1
        (share / "trace.bin").write_bytes(
            struct.pack("<16Q", MAGIC, count, *([0] * 10), 0, which, 0, 0)
        )
        hashes["trace.bin"] = digest(share / "trace.bin")
        (share / "run.sh").write_text(f"""#!/bin/sh
cp /mnt/host/defects.dom /tmp/mc-defects.dom
cp /mnt/host/host.user /tmp/mc-host
cp /mnt/host/trace.bin /tmp/mc-trace.bin
status=0
/tmp/mc-host /tmp/mc-defects.dom /tmp/mc-trace.bin /tmp/mc-report.bin {number} || status=$?
echo __EXIT_CODE__$status
cp /tmp/mc-report.bin /mnt/host/report.bin 2>/dev/null || true
echo MC_DEFECT_DONE
""")
        write_json(
            run / "manifest.json",
            {
                "sha256": hashes,
                "qemu_sha256": digest(os.environ["CAPSTONE_QEMU_BINARY"]),
                "compiler_sha256": digest(
                    Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang"
                ),
            },
        )
        # Bound every boot: a suite cannot have an unbounded arm in it.
        env = dict(os.environ)
        env.setdefault("CAPSTONE_QEMU_LOGIN_TIMEOUT", "90")
        env.setdefault("CAPSTONE_GUEST_COMMAND_TIMEOUT", "90")
        result = run_guest(
            run, "sh /mnt/host/run.sh", "MC_DEFECT_DONE", env=env, timeout_multiplier=1
        )
        serial = (
            (run / "serial.log").read_text(errors="replace")
            if (run / "serial.log").exists()
            else ""
        )
        if not serial:
            # No capture is an infrastructure failure, never a measurement.
            print(f"NO SERIAL CAPTURE for case={which} mode={mode}: {run}", flush=True)
            sys.exit(75)
        if "MC_DEFECT_DONE" not in serial and not re.search(
            r"domain (?:halted by capability fault|capability fault delivered)", serial
        ):
            # The guest never ran the command and no fault was raised either:
            # this boot produced no information about the case. Exit 75 rather
            # than record a FAIL that would read like the defect failing to
            # reproduce.
            print(f"BOOT PRODUCED NO RESULT for case={which} mode={mode}: {run}", flush=True)
            sys.exit(75)
        blob = share / "report.bin"
        report = None
        if blob.exists() and blob.stat().st_size >= 8 * REPORT_FIELDS:
            report = struct.unpack(f"<{REPORT_FIELDS}Q", blob.read_bytes()[: 8 * REPORT_FIELDS])
        row = classify(serial, report, which, mode, result.returncode)
        row["run"] = str(run)
        verdicts.append(row)
        write_json(run / "verdict.json", row)
        write_json(a.output / "verdicts.json", verdicts)
        flag = "OK  " if row["passed"] else "FAIL"
        if a.negative_control:
            flag = "FIRED" if not row["passed"] else "VACUOUS"
        extra = ""
        if row.get("expected") == "fault":
            extra = f" cause={row.get('cause')} pc={row.get('pc')}"
            if row.get("expected_pc"):
                extra += f" expected={row['expected_pc']}"
            extra += " delivered" if row.get("delivered") else " halted"
        print(f"{flag} case={which} {CASES[which][1]:34} {mode:8}{extra}", flush=True)
        # Normally an arm must pass; under the negative control it must FAIL.
        if row["passed"] == a.negative_control:
            status = 1

passed = sum(r["passed"] for r in verdicts)
if a.negative_control:
    print(
        f"\nnegative control: {len(verdicts) - passed}/{len(verdicts)} oracles "
        f"fired; {passed} reported a pass on an input that never ran the case"
    )
else:
    print(f"\n{passed}/{len(verdicts)} arms passed")
sys.exit(status)
