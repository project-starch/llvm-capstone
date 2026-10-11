#!/usr/bin/env python3
"""Validate the PoisonCap PostgreSQL manager: paired defect arms and a replay.

The defect arms run under `supervise`, which reports the fault from OUTSIDE the
program under test: the signal and trap PC come from the kernel, and the address
`pg_defect_probe` will sit at is resolved from the child's own map plus the target
ELF. That matters here because a mode-1 arm that faults somewhere OTHER than the
labelled access carries the same exit status as one that faults at it, and only
the PC comparison tells the two apart.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parents[4] / "common/host/cheribsd/run.py"
DEFECTS = 8
SIGPROT = 34
MODES = (0, 1)
PROBE = "pg_defect_probe"

# components.json, restated so a changed marker fails loudly here instead of
# silently widening what counts as a passing example.
EXAMPLES = {
    "allocator-example": "ALLOCATOR_EXAMPLE postgres PASS pointer_bytes=16",
    "client-generation-cheribsd": "PG_CLIENT generation RESULT 0",
    "client-slab-cheribsd": "PG_CLIENT slab RESULT 0",
    "client-bump-cheribsd": "PG_CLIENT bump RESULT 0",
}

# docs/plans/poisoncap-postgres.md, "Predicted readings". Case 0's stale access is
# the second free itself, so it has no probe and its oracle accepts a fault anywhere.
NO_PROBE = {0}
UNCERTAIN = {0}

FAULT = re.compile(
    r"SUPERVISE fault signal=(\d+) code=(\S+) addr=(\S+) pc=0x([0-9a-f]+)"
)
EXPECT = re.compile(r"SUPERVISE expect " + PROBE + r" (?:0x([0-9a-f]+)|unavailable)")
EXIT = re.compile(r"SUPERVISE exit (signalled|status)=(\d+)")


def discover(bins):
    """The corpus contract builds ONE program per case, named NN-slug. Fall back
    to the older single program with runtime dispatch while both exist."""
    programs = sorted(
        p for p in bins.glob("[0-9][0-9]-*") if p.is_file() and os.access(p, os.X_OK)
    )
    if programs:
        return [(int(p.name[:2]), p) for p in programs]
    legacy = bins / "defects"
    return [(n, legacy) for n in range(DEFECTS)] if legacy.is_file() else []


def defect_case(program, mode, which, timeout):
    """One arm. Both arms run the SAME binary; only the mode argument differs."""
    per_case = program.name != "defects"
    guest = program.name if per_case else "defects"
    args = ["./" + guest, str(mode), str(which)] if per_case else [
        "./defects", str(which), str(mode)
    ]
    case = dict(
        name=f"{program.name}-mode{mode}" if per_case else f"defect-{which}-mode{mode}",
        program=str(program.parent / "supervise"),
        inputs={guest: str(program)},
        args=args,
        timeout=timeout,
    )
    if mode == 0:
        case["exit"] = 0
        case["expect"] = f"PG_DEFECT case={which} mode=0 completed"
    else:
        case["exit"] = 128 + SIGPROT
        case["expect"] = f"SUPERVISE exit signalled={SIGPROT}"
        case["also_expect"] = [f"PG_DEFECT case={which} ready"]
    return case


def read_arm(directory):
    """What the kernel said about one arm, parsed from the supervisor's report."""
    out = (directory / "stdout.txt").read_text(errors="replace")
    expect = EXPECT.search(out)
    exit_line = EXIT.search(out)
    faults = [
        dict(signal=int(s), code=c, addr=a, pc=int(pc, 16))
        for s, c, a, pc in FAULT.findall(out)
    ]
    counters = None
    for line in out.splitlines():
        if line.startswith("PG_POISONCAP mode="):
            counters = line
    return dict(
        probe_address=int(expect.group(1), 16) if expect and expect.group(1) else None,
        faults=faults,
        exit_kind=exit_line.group(1) if exit_line else None,
        exit_value=int(exit_line.group(2)) if exit_line else None,
        counters=counters,
    )


def verdict(which, arms, protected=1):
    """A pair is only a pair when the unprotected arm reached the same site."""
    zero, one = arms[0], arms[protected]
    completed = zero["exit_kind"] == "status" and zero["exit_value"] == 0
    faulted = one["exit_kind"] == "signalled" and one["exit_value"] == SIGPROT
    at_probe = None
    if which not in NO_PROBE and faulted and one["probe_address"] is not None:
        at_probe = any(f["pc"] == one["probe_address"] for f in one["faults"])
    paired = completed and faulted and (at_probe is not False)
    return dict(
        case=which,
        unprotected_completed=completed,
        protected_faulted=faulted,
        fault_at_probe=at_probe,
        probe_required=which not in NO_PROBE,
        paired=paired,
        protected_fault_pcs=[hex(f["pc"]) for f in one["faults"]],
        probe_address=hex(one["probe_address"]) if one["probe_address"] else None,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("build", type=Path)
    p.add_argument("output", type=Path)
    for name in ("sdk", "rootfs", "image"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--port", type=int, default=10431)
    p.add_argument("--disable-default-revocation", action="store_true")
    p.add_argument("--stage", choices=("platform", "defects", "replay"), default="defects")
    p.add_argument("--recording", type=Path, help="A11 trace for the replay stage")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--modes", default="0,1",
                   help="Which arms to run. The plain CheriBSD build has only 0.")
    # The guest's OWN temporal safety. Off for the PoisonCap arms, because the
    # adapter sweeps itself and the two together panic the guest; ON is what
    # the plain CheriBSD arm exists to measure. The shared runner's ABI probe
    # verifies the setting took effect, so this cannot silently do nothing.
    p.add_argument("--runtime-revocation", choices=("off", "on"), default="off")
    # A synchronous sweep per free makes a 5k-record replay far slower than
    # any defect arm; pymalloc needed hours for a much shorter recording.
    p.add_argument("--replay-timeout", type=int, default=7200)
    p.add_argument("--case", action="append", default=[])
    a = p.parse_args()
    if a.stage == "replay" and not a.recording:
        p.error("replay requires --recording")
    if a.timeout < 1 or a.replay_timeout < 1:
        p.error("timeouts must be positive")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    bins = a.build.resolve() / "bin"

    cases = []
    found = []
    if a.stage != "platform":
        for binary, marker in EXAMPLES.items():
            cases.append(
                dict(name=binary, program=str(bins / binary), expect=marker,
                     timeout=a.timeout)
            )
        found = discover(bins)
        if not found:
            p.error(f"no case programs and no defects binary under {bins}")
        for mode in [int(m) for m in a.modes.split(",")]:
            if mode not in MODES:
                p.error(f"unknown mode: {mode}")
            for which, program in found:
                cases.append(defect_case(program, mode, which, a.timeout))
        arms = {w: prog.name for w, prog in found}
    if a.stage == "replay":
        for mode in (0, 1):
            cases.append(
                dict(
                    name=f"replay-mode{mode}",
                    program=str(bins / "replay"),
                    args=["trace.bin", str(mode)],
                    inputs={"trace.bin": str(a.recording.resolve())},
                    timeout=a.replay_timeout,
                    expect_regex=r"PG_REPLAY PASS creates=\d+ allocs=\d+ frees=\d+ "
                    r"reallocs=\d+ resets=\d+ deletes=\d+ checked=\d+ pointer_bytes=16",
                )
            )
    if a.case:
        unknown = set(a.case) - {c["name"] for c in cases}
        if unknown:
            p.error("unknown cases: " + ", ".join(sorted(unknown)))
        cases = [c for c in cases if c["name"] in a.case]
    (a.output / "arms.json").write_text(json.dumps({str(w): pr.name for w, pr in found}, indent=2) + "\n")
    (a.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (a.output / "selection.json").write_text(
        json.dumps(
            dict(stage=a.stage, complete_suite=not a.case,
                 cases=[c["name"] for c in cases]), indent=2
        )
        + "\n"
    )

    # Every arm must RUN. A suite that stops at the first failing oracle cannot
    # show that the others would have rejected the same input, and a wrong
    # PREDICTION must not cost the remaining fifteen arms of the boot.
    run = subprocess.run(
        [
            sys.executable, str(RUNNER), str(a.output / "guest"),
            "--sdk", str(a.sdk), "--rootfs", str(a.rootfs), "--image", str(a.image),
            "--port", str(a.port),
            "--abi-probe", str(bins / "cheribsd-abi-probe"),
            "--runtime-revocation", a.runtime_revocation,
            "--cases", str(a.output / "cases.json"),
            "--continue-on-failure",
            *(["--disable-default-revocation"] if a.disable_default_revocation else []),
        ],
    )
    report = a.output / "guest" / "summary.json"
    if not report.is_file():
        # The shared runner never got far enough to write a record. That is an
        # infrastructure failure, and it must not be reported as an empty result.
        print(f"NO SUITE RECORD at {report}; the runner exited {run.returncode} "
              "before any case ran", flush=True)
        return 2
    summary = json.loads(report.read_text())
    if not summary.get("ran_all_cases"):
        print("BOOT OR TRANSPORT FAILED before every case ran; no verdict", flush=True)
        (a.output / "matrix.json").write_text(
            json.dumps(dict(complete=False, status=summary.get("status")), indent=2) + "\n"
        )
        return 2

    pairs, arms_present = [], {}
    names = json.loads((a.output / "arms.json").read_text())
    if not names:
        names = {}
    for mode in [int(m) for m in a.modes.split(",")]:
        for which_s, stem in names.items():
            which = int(which_s)
            name = f"{stem}-mode{mode}" if stem != "defects" else f"defect-{which}-mode{mode}"
            directory = a.output / "guest" / name
            if directory.is_dir():
                arms_present.setdefault(which, {})[mode] = read_arm(directory)
    for which in sorted(arms_present):
        if {0, 1} <= set(arms_present[which]):  # a pair needs both arms
            row = verdict(which, arms_present[which])
            row["uncertain_prediction"] = which in UNCERTAIN
            pairs.append(row)
    matrix = dict(
        complete=True,
        oracle_failures=summary.get("failed_cases", []),
        pairs=pairs,
        paired=sorted(r["case"] for r in pairs if r["paired"]),
        unpaired=sorted(r["case"] for r in pairs if not r["paired"]),
    )
    (a.output / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    for row in pairs:
        site = "at probe" if row["fault_at_probe"] else (
            "no probe (fault anywhere accepted)" if not row["probe_required"]
            else "NOT at probe: " + ", ".join(row["protected_fault_pcs"] or ["no fault"])
        )
        print(
            f"case {row['case']}: mode0 "
            + ("completed" if row["unprotected_completed"] else "DID NOT complete")
            + ", mode1 "
            + ("faulted" if row["protected_faulted"] else "DID NOT fault")
            + f" [{site}]",
            flush=True,
        )
    print(f"paired: {matrix['paired']}  unpaired: {matrix['unpaired']}", flush=True)
    return 0 if not matrix["unpaired"] and not matrix["oracle_failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
