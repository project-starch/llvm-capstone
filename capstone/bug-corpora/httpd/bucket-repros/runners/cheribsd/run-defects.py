#!/usr/bin/env python3
"""Run the httpd bucket-allocator defects on stock CheriBSD, against the
platform's own malloc, with libc revocation on or off.

This is the arm that asks the shipping temporal mechanism the question. The
port's CheriBSD build takes every node from the platform's malloc and returns
it through the platform's free, exactly as upstream APR does; there is no
adapter authority and no protected mode. What varies is libc revocation --
CheriBSD's own, kernel-defaulted mechanism -- and the question is whether it
sees a stale bucket. Two recycling levels stand between a bucket and malloc:
apr_bucket_free files a small node on the bucket allocator's own freelist and
a block goes back to APR's size-bucketed list, whose default configuration
frees nothing. So the expected answer, in both configurations, is that the
sequence COMPLETES -- and the port's CheriBSD build, with bucket-pointers.c
over node-malloc.c, is what makes that a measurement of the shipping
allocators rather than of the census build.

A completing arm proves nothing on its own: it is what "no mechanism" would
look like too. So every run also executes revocation-control, a program that
frees a block, sweeps, and reads through the old pointer at the corpus's own
labelled load. With revocation on it must FAULT there -- SIGPROT, si_code
PROT_CHERI_TAG, pc == apr_defect_read -- and with revocation off it must
complete. That control is a platform control: if it does not behave, the run
exits 75 and no case verdict is recorded.

The program under test prints nothing about itself. The supervisor observes
it from outside -- signal, si_code and PC from the kernel, the expected
address from the child's memory map and ELF -- and the arms share one guest
boot through the common CheriBSD runner.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[1]
RUNNER = CORPUS.parents[2] / "ports/common/host/cheribsd/run.py"

MAGIC = 0x314C4F4F50525041  # "APRPOOL1"
REPORT_BYTES = 96  # struct aprp_header: 12 x uint64
SIGPROT = 34
PROT_CHERI_TAG = 2
SIGPROT_EXIT = 128 + SIGPROT
PLATFORM_CONTROLS = ("cheribsd-abi", "cheribsd-bounds")
REVOCATION_CONTROL = "revocation-control"
EXIT_INFRASTRUCTURE = 75

EXPECT_LINE = re.compile(r"SUPERVISE expect (\S+) (0x[0-9a-f]+)")
FAULT_LINE = re.compile(
    r"SUPERVISE fault signal=(\d+) code=(\d+) addr=(0x[0-9a-f]+) pc=(0x[0-9a-f]+)"
)


def load_cases():
    """The corpus's own case.json files, indexed by case number."""
    cases = {}
    for path in sorted(CORPUS.glob("[0-9][0-9]_*/case.json")):
        claim = json.loads(path.read_text())
        number = claim["case"]
        if number in cases:
            raise SystemExit(f"two case.json files claim case {number}")
        cases[number] = dict(
            case=number,
            directory=path.parent.name,
            slug=path.parent.name.split("_", 2)[2],
            upstream_fix=claim["upstream_fix"],
            title=claim["title"],
            shape=claim["shape"],
            allocator_layer=claim["allocator_layer"],
        )
    if not cases or sorted(cases) != list(range(len(cases))):
        raise SystemExit("the corpus case numbers are not 0..N-1")
    return cases


def fixture(number, negative_control):
    """One event, 128 bytes, the case number in the event id.

    Under the negative control the header's count says two events while the
    file carries one, so the program refuses the input before any pool is
    created and no case is performed at all.
    """
    count = 2 if negative_control else 1
    return struct.pack("<16Q", MAGIC, count, *([0] * 10), 0, number, 0, 0)


def fault_oracle():
    """The complete line a faulting program must produce, and nothing weaker."""
    return (
        rf"SUPERVISE fault signal={SIGPROT} code={PROT_CHERI_TAG} "
        r"addr=(0x[0-9a-f]+) pc=\1"
    )


def build_case(number, supervisor, program, input_path, timeout, slug=""):
    label = f"{number:02d}-{slug.replace('_', '-')}" if slug else f"defect-{number}"
    return dict(
        name=f"{label}-mode0",
        program=str(supervisor),
        args=["./target", "input.bin", "output.bin", "0"],
        inputs={"target": str(program), "input.bin": str(input_path)},
        timeout=timeout,
        expect="SUPERVISE exit status=0",
        exit=0,
        outputs=["output.bin"],
    )


def build_revocation_control(supervisor, control, revocation_on, timeout):
    """The instrument's positive control, graded by the same supervisor.

    Its expected outcome flips with the platform setting: revocation on must
    fault at the labelled load, off must complete. Either way it runs before
    the case arms and is required.
    """
    case = dict(
        name=REVOCATION_CONTROL,
        program=str(supervisor),
        args=["./control"],
        inputs={"control": str(control)},
        timeout=timeout,
    )
    if revocation_on:
        case["expect_regex"] = fault_oracle()
        case["exit"] = SIGPROT_EXIT
    else:
        case["expect"] = "SUPERVISE exit status=0"
        case["also_expect"] = ["REVOCATION_CONTROL stale read completed"]
        case["exit"] = 0
    return case


def report_fields(blob):
    if blob is None or len(blob) != REPORT_BYTES:
        return None
    magic, count, mode, status, completed, nodes, reuses, releases = struct.unpack(
        "<8Q", blob[:64]
    )
    return dict(
        magic_ok=magic == MAGIC,
        count=count,
        mode=mode,
        status=status,
        completed=completed,
        nodes=nodes,
        node_reuses=reuses,
        node_releases=releases,
        sha256=hashlib.sha256(blob).hexdigest(),
    )


def observed_fault(stdout):
    """What the supervisor said: the resolved expectation and the FIRST fault."""
    expect = None
    faults = []
    for line in stdout.splitlines():
        found = EXPECT_LINE.fullmatch(line)
        if found:
            expect = found[2]
        found = FAULT_LINE.fullmatch(line)
        if found:
            faults.append(found)
    first = faults[0] if faults else None
    fault = (
        dict(signal=int(first[1]), code=int(first[2]), addr=first[3], pc=first[4])
        if first
        else None
    )
    return expect, fault, len(faults)


def evaluate(number, row, stdout, report):
    """The verdict for one arm: it must COMPLETE, and no fault may have happened."""
    expect, fault, seen = observed_fault(stdout)
    fields = report_fields(report)
    verdict = dict(
        case=number,
        mode=0,
        expected="completed",
        ran=row is not None,
        exit=row["exit"] if row else None,
        expected_exit=0,
        runner_oracle=bool(row and row["passed"]),
        stdout_sha256=row["stdout_sha256"] if row else None,
        report=fields,
        expect=expect,
        fault=fault,
        faults_seen=seen,
    )
    verdict["passed"] = bool(
        verdict["runner_oracle"]
        and verdict["exit"] == 0
        and fault is None
        and fields is not None
        and fields["magic_ok"]
        and fields["status"] == 0
        and fields["completed"] == 1
        and fields["count"] == 1
        and fields["mode"] == 0
    )
    return verdict


def evaluate_revocation_control(row, stdout, revocation_on):
    expect, fault, seen = observed_fault(stdout)
    verdict = dict(
        name=REVOCATION_CONTROL,
        revocation=revocation_on,
        expected="tag fault at apr_defect_read" if revocation_on else "completed",
        ran=row is not None,
        exit=row["exit"] if row else None,
        runner_oracle=bool(row and row["passed"]),
        expect=expect,
        fault=fault,
        faults_seen=seen,
    )
    if revocation_on:
        verdict["passed"] = bool(
            verdict["runner_oracle"]
            and verdict["exit"] == SIGPROT_EXIT
            and expect is not None
            and fault is not None
            and fault["signal"] == SIGPROT
            and fault["code"] == PROT_CHERI_TAG
            and fault["addr"] == fault["pc"] == expect
        )
    else:
        verdict["passed"] = bool(
            verdict["runner_oracle"] and verdict["exit"] == 0 and fault is None
        )
    return verdict


def control_fired(verdict):
    """Did this arm's oracle reject a fixture that never ran its case?"""
    report = verdict.get("report") or {}
    return bool(
        not verdict["passed"] and not verdict.get("fault") and not report.get("completed")
    )


def numbers(text, limit, what):
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item.isdigit() or int(item) not in limit:
            raise SystemExit(f"no such {what}: {item!r}")
        if int(item) not in values:
            values.append(int(item))
    if not values:
        raise SystemExit(f"select at least one {what}")
    return values


def main():
    claims = load_cases()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("build", type=Path, help="where shared/build-cases.sh cheribsd put bin/")
    p.add_argument("output", type=Path, help="Fresh output directory")
    for name in ("sdk", "rootfs", "image"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--port", type=int, default=10442)
    p.add_argument("--disable-default-revocation", action="store_true")
    p.add_argument(
        "--runtime-revocation",
        choices=("off", "on"),
        default="on",
        help="Per-process CheriBSD libc revocation for the programs under test. "
        "'on' is the platform's shipping default and the configuration this arm "
        "exists to measure; the ABI control verifies it in the guest through "
        "malloc_revoke_enabled(), and revocation-control shows it can fire.",
    )
    p.add_argument("--cases", default=",".join(str(i) for i in sorted(claims)))
    p.add_argument(
        "--modes",
        default="0",
        help="Only mode 0 exists on this target: the stock build has no protected arm "
        "and refuses mode 1. Kept for command-line symmetry with the other runners.",
    )
    p.add_argument("--timeout", type=int, default=1800, help="Seconds per arm")
    p.add_argument(
        "--negative-control",
        action="store_true",
        help="Corrupt every fixture so the program refuses it before any pool is "
        "created, execute EVERY selected arm anyway, and require every oracle to "
        "report a failure with no completed report and no fault. Exit 0 means every "
        "selected oracle fired. revocation-control takes no fixture and must still "
        "pass.",
    )
    a = p.parse_args()
    selected = numbers(a.cases, set(claims), "case")
    if numbers(a.modes, {0, 1}, "mode") != [0]:
        p.error("the stock CheriBSD build has only mode 0; there is no PoisonCap build of APR")
    if a.timeout < 1:
        p.error("the per-arm timeout must be positive")
    bins = a.build.resolve() / "bin"
    programs = {n: bins / f"defect-{n:02d}" for n in selected}
    supervisor = bins / "supervise"
    probe = bins / "cheribsd-abi-probe"
    control = bins / REVOCATION_CONTROL
    for path in [*programs.values(), probe, supervisor, control]:
        if not path.is_file():
            p.error(f"missing program: {path} (build with shared/build-cases.sh cheribsd)")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    fixtures = a.output / "fixtures"
    fixtures.mkdir()
    revocation_on = a.runtime_revocation == "on"

    arms_wanted, hashes = [], {}
    for number in selected:
        path = fixtures / f"case-{number}.bin"
        path.write_bytes(fixture(number, a.negative_control))
        hashes[number] = hashlib.sha256(path.read_bytes()).hexdigest()
        arms_wanted.append(
            (
                number,
                build_case(
                    number, supervisor, programs[number], path, a.timeout, claims[number]["slug"]
                ),
            )
        )
    # The revocation control runs first: a case verdict is only worth reading
    # after the instrument has shown it can distinguish on from off.
    cases = [build_revocation_control(supervisor, control, revocation_on, a.timeout)]
    cases += [case for _, case in arms_wanted]
    (a.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (a.output / "selection.json").write_text(
        json.dumps(
            dict(
                schema="httpd-bucket-defect-cheribsd-selection-v1",
                complete_suite=sorted(selected) == sorted(claims),
                negative_control=a.negative_control,
                runtime_revocation=a.runtime_revocation,
                guest_default_revocation=(
                    "off" if a.disable_default_revocation else "preserved"
                ),
                cases=selected,
                modes=[0],
                timeout_seconds=a.timeout,
                arms=[c["name"] for c in cases],
            ),
            indent=2,
        )
        + "\n"
    )

    guest = a.output / "guest"
    if a.negative_control:
        print(
            "negative control: the shared runner is expected to report every arm "
            "FAIL and to exit non-zero; its traceback below is that outcome, not "
            "an infrastructure failure.",
            flush=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(guest),
            "--sdk",
            str(a.sdk),
            "--rootfs",
            str(a.rootfs),
            "--image",
            str(a.image),
            "--port",
            str(a.port),
            "--abi-probe",
            str(probe),
            "--runtime-revocation",
            a.runtime_revocation,
            "--continue-on-failure",
            "--cases",
            str(a.output / "cases.json"),
            *(["--disable-default-revocation"] if a.disable_default_revocation else []),
        ],
        check=False,
    )

    summary_path = guest / "summary.json"
    if not summary_path.is_file():
        print(f"NO GUEST SUMMARY: {summary_path}", flush=True)
        return EXIT_INFRASTRUCTURE
    summary = json.loads(summary_path.read_text())
    rows = {row["name"]: row for row in summary["results"]}
    if not summary.get("ran_all_cases"):
        print("BOOT DID NOT REACH EVERY ARM: " + summary.get("status", "unknown"), flush=True)
        return EXIT_INFRASTRUCTURE
    for name in PLATFORM_CONTROLS:
        if not rows.get(name, {}).get("passed"):
            print(f"PLATFORM CONTROL FAILED: {name}", flush=True)
            return EXIT_INFRASTRUCTURE

    def stdout_of(name):
        path = guest / name / "stdout.txt"
        return path.read_text(errors="replace") if path.is_file() else ""

    control_verdict = evaluate_revocation_control(
        rows.get(REVOCATION_CONTROL), stdout_of(REVOCATION_CONTROL), revocation_on
    )
    detail = ""
    if control_verdict.get("fault"):
        f = control_verdict["fault"]
        detail = (
            f" signal={f['signal']} code={f['code']} addr={f['addr']} pc={f['pc']} "
            f"expect={control_verdict.get('expect')}"
        )
    print(
        f"{'OK  ' if control_verdict['passed'] else 'FAIL'} {REVOCATION_CONTROL} "
        f"revocation={a.runtime_revocation} exit={control_verdict['exit']}{detail}",
        flush=True,
    )
    if not control_verdict["passed"]:
        print(f"PLATFORM CONTROL FAILED: {REVOCATION_CONTROL}", flush=True)
        return EXIT_INFRASTRUCTURE

    arms, status = [], 0
    for number, case in arms_wanted:
        name = case["name"]
        row = rows.get(name)
        if row is None:
            print(f"ARM NEVER RAN: {name}", flush=True)
            return EXIT_INFRASTRUCTURE
        blob = guest / name / "output.bin"
        report = blob.read_bytes() if blob.is_file() else None
        verdict = evaluate(number, row, stdout_of(name), report)
        verdict.update(
            claims[number],
            fixture_sha256=hashes[number],
            program_sha256=summary["binaries"][name],
        )
        if a.negative_control:
            verdict["control_fired"] = control_fired(verdict)
            if not verdict["control_fired"]:
                status = 1
            flag = "FIRED  " if verdict["control_fired"] else "VACUOUS"
        else:
            if not verdict["passed"]:
                status = 1
            flag = "OK  " if verdict["passed"] else "FAIL"
        arms.append(verdict)
        fields = verdict.get("report") or {}
        detail = ""
        if fields:
            detail = (
                f" completed={fields.get('completed')} nodes={fields.get('nodes')} "
                f"reuses={fields.get('node_reuses')}"
            )
        if verdict.get("fault"):
            f = verdict["fault"]
            detail += f" FAULT signal={f['signal']} code={f['code']} pc={f['pc']}"
        print(
            f"{flag} case={number:<2} mode=0 revocation={a.runtime_revocation} "
            f"{claims[number]['upstream_fix']:<10} exit={verdict['exit']}{detail}",
            flush=True,
        )

    (a.output / "verdicts.json").write_text(
        json.dumps(
            dict(
                schema="httpd-apr-defect-cheribsd-v1",
                negative_control=a.negative_control,
                complete_suite=sorted(selected) == sorted(claims),
                runtime_revocation=summary["runtime_revocation"],
                guest_default_revocation=summary["guest_default_revocation"],
                platform=summary["platform"],
                binaries=summary["binaries"],
                platform_controls=[rows[name] for name in PLATFORM_CONTROLS],
                revocation_control=control_verdict,
                arms=arms,
            ),
            indent=2,
        )
        + "\n"
    )
    passed = sum(arm["passed"] for arm in arms)
    if a.negative_control:
        fired = sum(arm["control_fired"] for arm in arms)
        print(
            f"\nnegative control: {fired}/{len(arms)} oracles fired; "
            f"{len(arms) - fired} reported a pass on an input that never ran its case",
            flush=True,
        )
    else:
        print(f"\n{passed}/{len(arms)} arms passed", flush=True)
    print(f"Artifacts: {a.output}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
