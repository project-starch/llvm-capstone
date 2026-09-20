#!/usr/bin/env python3
"""Run the twenty CPython pymalloc defects as paired spatial/sublet domain arms.

Each case is booted twice against the same binary; the port chooses the arm at
runtime from the mode argument, so the two arms differ in exactly one thing.

  spatial (mode 0)  pym_issue/pym_release keep the block's alias, so the freed
                    block stays addressable, the stale access succeeds, and the
                    run must COMPLETE.
  sublet  (mode 1)  every issue and release does sublet_give then sublet_take,
                    so the stale access is a revoked alias and must FAULT -- at
                    the labelled probe instruction, not merely somewhere.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "ports/common/host"))
from port_support import digest, run_guest, stage_run, write_json

CASES = [
    ("gh-143543", "groupby-reentrant-eq", "free/reuse/stale read"),
    ("gh-146613", "grouper-parent-key-freed", "free/reuse/stale read"),
    ("gh-142829", "hamt-eq-interior-cursor", "interior pointer"),
    ("gh-142831", "json-encoder-list-item", "stale entry in a live array"),
    ("gh-145244", "json-encoder-dict-key-error-path", "bulk free, error path"),
    ("gh-148660", "odict-copy-stale-link", "pointer load from freed block"),
    ("gh-151295", "bytes-join-reentrant-buffer", "payload buffer, sub-512"),
    ("gh-148395", "decompressor-next-in", "cursor across two API calls"),
    ("gh-112127", "atexit-unregister-borrowed-tuple", "free/reuse/stale read"),
    ("gh-139210", "iterparse-event-name-after-decref", "payload buffer, error path"),
    ("gh-142560", "bytearray-search-realloc-moved", "realloc moved the block"),
    ("gh-142783", "zoneinfo-eager-decref", "free and use on adjacent lines"),
    ("gh-143004", "counter-update-borrowed-value", "free/reuse/stale read"),
    ("gh-144833", "ssl-decref-self-then-read-self", "interior pointer"),
    (
        "gh-146011",
        "decimal-signaldict-outlives-context",
        "parked in a surviving object",
    ),
    ("gh-149449", "unicodedata-capi-freed-under-cache", "bare PyMem block, cached"),
    ("gh-151403", "fork-exec-fspath-mutates-args", "free/reuse/stale read"),
    ("gh-151416", "spawnv-fspath-mutates-argv", "free/reuse/stale read"),
    ("gh-151695", "curses-screen-encoding-global", "parked in a global"),
    ("gh-153539", "textio-tell-reentrant-decoder", "free/reuse/stale read"),
]

MARKER_BASE = 0xCF19000000000000
REPORT_FIELDS = 12  # struct pym_header: 12 x uint64
COMPLETED = 4  # its index


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
    row = {
        "case": which,
        "fix": fix,
        "name": name,
        "shape": shape,
        "mode": mode,
        "expected": "fault" if mode == "sublet" else "complete",
        "runner_exit": runner_exit,
    }
    if mode == "sublet":
        following = serial.split(marker, 1)[-1] if marker in serial else ""
        sites = re.findall(r"Print = Cap\(\d+, 0x[0-9a-f]+, (0x[0-9a-f]+),", following)
        cause, pc = faults[-1] if faults else ("0", "0")
        expected = sites[0] if sites else None  # [0] is the read probe
        row.update(cause=int(cause), pc=pc, expected_pc=expected, delivered=delivered)
        ok = (
            marker in serial
            and len(faults) == 1
            and expected is not None
            and int(pc, 16) == int(expected, 16)
            and int(cause) in (24, 25)
        )
        if delivered:
            # The VM survived, so the claim is containment and it has to be
            # evidenced: the launcher died by SIGSEGV and the guest got its
            # shell back to say so, which can only happen if a command ran
            # AFTER the fault.
            #
            # runner_exit stays non-zero here and that is CORRECT, not a
            # failure: the generic guest runner waits for a completion marker
            # that a killed launcher never prints. Requiring 0 would reject
            # exactly the arms that demonstrate containment. Do not add it back.
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
work = Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone")) / "cpython-pymalloc"
p.add_argument("--domain-build", type=Path, default=work / "build/capstone-domain")
p.add_argument("--linux-build", type=Path, default=work / "build/linux-guest")
p.add_argument("--cases", default=",".join(str(i) for i in range(len(CASES))))
p.add_argument("--modes", default="spatial,sublet")
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
        if mode not in ("spatial", "sublet"):
            p.error("invalid mode")
        number = 0 if mode == "spatial" else 1
        run, hashes = stage_run(
            a.output,
            f"{which:02d}-{CASES[which][1]}-{mode}-",
            {
                "defects.dom": a.domain_build / "bin/defects.dom",
                "host.user": a.linux_build / "bin/domain-loader",
            },
        )
        share = run / "share"
        # One event, carrying the case number in its id -- the same input shape
        # the port's own security tests use. Under --negative-control the count
        # is wrong, so the domain's own CHECK refuses the input and no defect is
        # performed: no fault for the sublet oracle, no completion for the
        # spatial one.
        count = 2 if a.negative_control else 1
        (share / "trace.bin").write_bytes(
            struct.pack("<16Q", 0x31594C50524D5950, count, *([0] * 10), 0, which, 0, 0)
        )
        hashes["trace.bin"] = digest(share / "trace.bin")
        (share / "run.sh").write_text(f"""#!/bin/sh
cp /mnt/host/defects.dom /tmp/pyc-defects.dom
cp /mnt/host/host.user /tmp/pyc-host
cp /mnt/host/trace.bin /tmp/pyc-trace.bin
status=0
/tmp/pyc-host /tmp/pyc-defects.dom /tmp/pyc-trace.bin /tmp/pyc-report.bin {number} || status=$?
echo __EXIT_CODE__$status
cp /tmp/pyc-report.bin /mnt/host/report.bin 2>/dev/null || true
echo PYC_DEFECT_DONE
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
        # BOUND EVERY BOOT. run_guest defaults to timeout_multiplier=12, which on
        # a boot that wedges before login leaves the whole suite sitting there:
        # on 2026-09-18 one such boot stalled at kernel time 0.70s and held the
        # run for eleven minutes before it was killed by hand. A suite of forty
        # arms cannot have an unbounded arm in it -- every run must return.
        env = dict(os.environ)
        env.setdefault("CAPSTONE_QEMU_LOGIN_TIMEOUT", "90")
        env.setdefault("CAPSTONE_GUEST_COMMAND_TIMEOUT", "90")
        result = run_guest(
            run,
            "sh /mnt/host/run.sh",
            "PYC_DEFECT_DONE",
            env=env,
            timeout_multiplier=1,
        )
        serial = (
            (run / "serial.log").read_text(errors="replace")
            if (run / "serial.log").exists()
            else ""
        )
        if not serial:
            # No capture is an infrastructure failure, never a measurement. Say
            # so and stop, rather than letting an empty string classify as a
            # clean "no fault found".
            print(f"NO SERIAL CAPTURE for case={which} mode={mode}: {run}", flush=True)
            sys.exit(75)
        if "PYC_DEFECT_DONE" not in serial and not re.search(
            r"domain (?:halted by capability fault|capability fault delivered)", serial
        ):
            # The guest never ran the command and no fault was raised either, so
            # this boot produced no information about the case. Exit 75 -- the
            # infrastructure code -- rather than record a FAIL that would read
            # like the defect failing to reproduce.
            print(
                f"BOOT PRODUCED NO RESULT for case={which} mode={mode}: {run}",
                flush=True,
            )
            sys.exit(75)
        blob = share / "report.bin"
        report = None
        if blob.exists() and blob.stat().st_size >= 8 * REPORT_FIELDS:
            report = struct.unpack(
                f"<{REPORT_FIELDS}Q", blob.read_bytes()[: 8 * REPORT_FIELDS]
            )
        row = classify(serial, report, which, mode, result.returncode)
        row["run"] = str(run)
        verdicts.append(row)
        write_json(run / "verdict.json", row)
        write_json(a.output / "verdicts.json", verdicts)
        flag = "OK  " if row["passed"] else "FAIL"
        if a.negative_control:
            flag = "FIRED" if not row["passed"] else "VACUOUS"
        extra = ""
        if mode == "sublet":
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
