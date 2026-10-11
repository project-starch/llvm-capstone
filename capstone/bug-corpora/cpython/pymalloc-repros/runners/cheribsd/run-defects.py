#!/usr/bin/env python3
"""Run the twenty CPython pymalloc defects on PoisonCap/CheriBSD, in pairs.

Every case is executed twice against the SAME `bin/defects` binary, which
chooses its arm from the mode argument, so the two arms differ in exactly one
thing:

  mode 0  request-bounded spatial authority, no per-object invalidation. The
          freed block stays addressable, the stale access succeeds, and the
          case must reach its end and report a completed run.
  mode 1  PoisonCap lifetime invalidation. The stale access is a revoked alias
          and must raise SIGPROT with si_code PROT_CHERI_TAG at the labelled
          `pyc_defect_read` instruction -- not somewhere, and not some other
          kind of capability fault.

WHAT IS DELIBERATELY NOT ACCEPTED AS PROTECTION. Exit 162 is the status of
EVERY SIGPROT on CheriBSD, so it says nothing on its own: an arbitrary crash, a
bounds fault, a permission fault, a tag fault at another instruction, a failure
before the case marker, or the allocator refusing the request would all produce
it. The program therefore prints the trap PC and the address of the probe label
from inside its own handler, and this runner requires a complete fault line
whose two PC fields are the same text, whose signal is 34 and whose code is 2,
preceded by the case's ready marker.

All arms share one guest boot through the common CheriBSD runner; no SSH, QEMU
or copy logic is duplicated here.
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

MAGIC = 0x31594C50524D5950
REPORT_BYTES = 96  # struct pym_header: 12 x uint64
FIXTURE_BYTES = 128  # that header plus one 4 x uint64 event
SIGPROT = 34
PROT_CHERI_TAG = 2
SIGPROT_EXIT = 128 + SIGPROT
PLATFORM_CONTROLS = ("cheribsd-abi", "cheribsd-bounds")
EXIT_INFRASTRUCTURE = 75

# What the supervisor reports, all of it observed from outside the program: the
# signal, si_code and faulting PC come from the kernel (PT_LWPINFO and
# PT_GETCAPREGS), and the expected address from the child's own memory map plus
# the symbol in its ELF. The program under test prints nothing and judges
# nothing about itself.
EXPECT_LINE = re.compile(r"SUPERVISE expect (\S+) (0x[0-9a-f]+)")
FAULT_LINE = re.compile(
    r"SUPERVISE fault signal=(\d+) code=(\d+) addr=(0x[0-9a-f]+) pc=(0x[0-9a-f]+)"
)


def load_cases():
    """The corpus's own case.json files, indexed by case number.

    The metadata is read from the corpus rather than restated here, so a
    verdict cannot describe a case differently from the case's own claims.
    """
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
    file carries one, so the program refuses the input before `defect()` runs
    and no case is performed at all.
    """
    count = 2 if negative_control else 1
    return struct.pack("<16Q", MAGIC, count, *([0] * 10), 0, number, 0, 0)


def fault_oracle():
    """The complete line a protected arm must produce, and nothing weaker.

    The backreference is the point: the faulting address and the PC must be the
    same text. Whether that address is the RIGHT one is settled against the
    supervisor's own `expect` line, which is resolved from the ELF and the
    memory map -- see evaluate().
    """
    return (
        rf"SUPERVISE fault signal={SIGPROT} code={PROT_CHERI_TAG} "
        r"addr=(0x[0-9a-f]+) pc=\1"
    )


def build_case(number, mode, supervisor, program, input_path, timeout, slug=""):
    # The supervisor runs the case binary and observes it; the case binary is
    # copied in as ./target so the supervisor can resolve its path and ELF.
    label = f"{number:02d}-{slug.replace('_', '-')}" if slug else f"defect-{number}"
    case = dict(
        name=f"{label}-mode{mode}",
        program=str(supervisor),
        args=["./target", "input.bin", "output.bin", str(mode)],
        inputs={"target": str(program), "input.bin": str(input_path)},
        timeout=timeout,
    )
    if mode == 0:
        case["expect"] = "SUPERVISE exit status=0"
        case["exit"] = 0
        case["outputs"] = ["output.bin"]
    else:
        case["expect_regex"] = fault_oracle()
        case["exit"] = SIGPROT_EXIT
    return case


def report_fields(blob):
    """The replay report's logical fields, or None if there is no report."""
    if blob is None or len(blob) != REPORT_BYTES:
        return None
    magic, count, mode, status, completed = struct.unpack("<5Q", blob[:40])
    return dict(
        magic_ok=magic == MAGIC,
        count=count,
        mode=mode,
        status=status,
        completed=completed,
        sha256=hashlib.sha256(blob).hexdigest(),
    )


def evaluate(number, mode, row, stdout, report):
    """The verdict for one arm, from what the KERNEL reported about it."""
    lines = stdout.splitlines()
    verdict = dict(
        case=number,
        mode=mode,
        expected="completed" if mode == 0 else "tag fault at pyc_defect_read",
        ran=row is not None,
        exit=row["exit"] if row else None,
        expected_exit=0 if mode == 0 else SIGPROT_EXIT,
        runner_oracle=bool(row and row["passed"]),
        stdout_sha256=row["stdout_sha256"] if row else None,
    )
    if mode == 0:
        fields = report_fields(report)
        verdict["report"] = fields
        verdict["passed"] = bool(
            verdict["runner_oracle"]
            and verdict["exit"] == 0
            and fields is not None
            and fields["magic_ok"]
            and fields["status"] == 0
            and fields["completed"] == 1
            and fields["count"] == 1
            and fields["mode"] == mode
        )
        return verdict

    # The supervisor resolves the probe address from the ELF and the child's
    # memory map. It says "unavailable" rather than guessing when it cannot,
    # and an unavailable expectation is a failed arm, never a match.
    expect = None
    for line in lines:
        found = EXPECT_LINE.fullmatch(line)
        if found:
            expect = found[2]
    faults = [FAULT_LINE.fullmatch(line) for line in lines]
    faults = [f for f in faults if f]
    # The FIRST fault is the one the case produced; anything after it is the
    # aftermath of dying and must not be able to rescue the arm.
    first = faults[0] if faults else None
    verdict["expect"] = expect
    verdict["fault"] = (
        dict(signal=int(first[1]), code=int(first[2]), addr=first[3], pc=first[4])
        if first
        else None
    )
    verdict["faults_seen"] = len(faults)
    verdict["passed"] = bool(
        verdict["runner_oracle"]
        and verdict["exit"] == SIGPROT_EXIT
        and expect is not None
        and first is not None
        and int(first[1]) == SIGPROT
        and int(first[2]) == PROT_CHERI_TAG
        and first[3] == first[4] == expect
    )
    return verdict


# The arm names each case.json declares, so the command line and the corpus
# speak one vocabulary. The Capstone runner names its arms too (spatial,
# sublet); numbers stay accepted because the binary takes a mode argument.
MODE_NAMES = {"spatial": 0, "protected": 1}


def control_fired(verdict):
    """Did this arm's oracle reject a fixture that never ran its case?

    The fixture is refused before any case runs, so a fired oracle shows no
    fault at the probe and no completed report: the arm did not merely fail, it
    produced no measurement at all. Kept as a function because it is the check
    that decides whether a whole control run means anything, and because two
    refactors broke it where only a ten-minute guest run could notice.
    """
    report = verdict.get("report") or {}
    return bool(
        not verdict["passed"]
        and not verdict.get("fault")
        and not report.get("completed")
    )


def numbers(text, limit, what, names=None):
    values = []
    for item in text.split(","):
        item = item.strip()
        value = (names or {}).get(item.lower())
        if value is None:
            if not item.isdigit():
                raise SystemExit(f"no such {what}: {item!r}")
            value = int(item)
        if value not in limit:
            raise SystemExit(f"no such {what}: {item!r}")
        if value not in values:
            values.append(value)
    if not values:
        raise SystemExit(f"select at least one {what}")
    return values


def main():
    claims = load_cases()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("build", type=Path, help="PoisonCap pymalloc build directory")
    p.add_argument("output", type=Path, help="Fresh output directory")
    for name in ("sdk", "rootfs", "image"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--port", type=int, default=10431)
    p.add_argument(
        "--abi-probe",
        type=Path,
        help="cheribsd-abi-probe; defaults to BUILD/bin/cheribsd-abi-probe",
    )
    p.add_argument("--disable-default-revocation", action="store_true")
    p.add_argument(
        "--runtime-revocation",
        choices=("off", "on"),
        default="off",
        help="Per-process CheriBSD libc revocation for the program under test. "
        "'on' keeps the PLATFORM's own temporal mechanism active, which the ABI "
        "control then verifies through malloc_revoke_enabled(); the adapter's "
        "own PoisonCap sweeps are selected by the mode argument and are "
        "independent of this.",
    )
    p.add_argument(
        "--cases",
        default=",".join(str(i) for i in sorted(claims)),
        help="Comma-separated diagnostic subset; the default is the whole corpus",
    )
    p.add_argument(
        "--modes",
        default="0,1",
        help="Comma-separated modes: 0/1, or the arm names spatial/protected "
        "that each case.json declares. Default 0,1",
    )
    p.add_argument("--timeout", type=int, default=1800, help="Seconds per arm")
    p.add_argument(
        "--negative-control",
        action="store_true",
        help="Corrupt every fixture so the program refuses it before defect() runs, "
        "execute EVERY selected arm anyway, and require every oracle to report a "
        "failure with no ready, fault or completed marker. A suite whose oracles "
        "cannot say FAIL proves nothing by saying PASS, so this inverts the exit "
        "status: 0 means every selected oracle fired.",
    )
    a = p.parse_args()
    selected = numbers(a.cases, set(claims), "case")
    modes = numbers(a.modes, {0, 1}, "mode", MODE_NAMES)
    if a.timeout < 1:
        p.error("the per-arm timeout must be positive")
    bins = a.build.resolve() / "bin"
    # One program per defect, built by shared/build-cases.sh through the port's
    # one-source seam.
    programs = {n: bins / f"defect-{n:02d}" for n in selected}
    supervisor = bins / "supervise"
    probe = a.abi_probe.resolve() if a.abi_probe else bins / "cheribsd-abi-probe"
    for path in [*programs.values(), probe, supervisor]:
        if not path.is_file():
            p.error(f"missing program: {path} (build with shared/build-cases.sh)")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    fixtures = a.output / "fixtures"
    fixtures.mkdir()

    arms_wanted, hashes = [], {}
    for number in selected:
        path = fixtures / f"case-{number}.bin"
        path.write_bytes(fixture(number, a.negative_control))
        hashes[number] = hashlib.sha256(path.read_bytes()).hexdigest()
        for mode in modes:
            arms_wanted.append(
                (
                    number,
                    mode,
                    build_case(
                        number,
                        mode,
                        supervisor,
                        programs[number],
                        path,
                        a.timeout,
                        claims[number]["slug"],
                    ),
                )
            )
    cases = [case for _, _, case in arms_wanted]
    (a.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (a.output / "selection.json").write_text(
        json.dumps(
            dict(
                schema="pymalloc-defect-poisoncap-selection-v1",
                complete_suite=sorted(selected) == sorted(claims)
                and sorted(modes) == [0, 1],
                negative_control=a.negative_control,
                runtime_revocation=a.runtime_revocation,
                guest_default_revocation=(
                    "off" if a.disable_default_revocation else "preserved"
                ),
                cases=selected,
                modes=modes,
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
    # One boot for every arm. A failing arm must not end the boot: a negative
    # control has to show that ALL the selected oracles reject the input, and
    # an ordinary run is more useful when one failure does not discard the
    # thirty-nine arms behind it.
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
    # Infrastructure failures are kept apart from measured case failures. A boot
    # that died early, or a platform whose own controls do not hold, has not
    # measured this corpus either way.
    if not summary.get("ran_all_cases"):
        print(
            "BOOT DID NOT REACH EVERY ARM: " + summary.get("status", "unknown"),
            flush=True,
        )
        return EXIT_INFRASTRUCTURE
    for name in PLATFORM_CONTROLS:
        if not rows.get(name, {}).get("passed"):
            print(f"PLATFORM CONTROL FAILED: {name}", flush=True)
            return EXIT_INFRASTRUCTURE

    arms, status = [], 0
    for number, mode, case in arms_wanted:
        name = case["name"]
        row = rows.get(name)
        if row is None:
            print(f"ARM NEVER RAN: {name}", flush=True)
            return EXIT_INFRASTRUCTURE
        stdout_path = guest / name / "stdout.txt"
        stdout = (
            stdout_path.read_text(errors="replace") if stdout_path.is_file() else ""
        )
        blob = guest / name / "output.bin"
        report = blob.read_bytes() if blob.is_file() else None
        verdict = evaluate(number, mode, row, stdout, report)
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
        detail = ""
        if mode == 1 and verdict.get("fault"):
            fault = verdict["fault"]
            detail = (
                f" signal={fault['signal']} code={fault['code']} "
                f"addr={fault['addr']} pc={fault['pc']} "
                f"expect={verdict.get('expect')}"
            )
        print(
            f"{flag} case={number:<2} mode={mode} "
            f"{claims[number]['upstream_fix']:<10} exit={verdict['exit']}{detail}",
            flush=True,
        )

    (a.output / "verdicts.json").write_text(
        json.dumps(
            dict(
                schema="pymalloc-defect-poisoncap-v1",
                negative_control=a.negative_control,
                complete_suite=sorted(selected) == sorted(claims)
                and sorted(modes) == [0, 1],
                runtime_revocation=summary["runtime_revocation"],
                guest_default_revocation=summary["guest_default_revocation"],
                platform=summary["platform"],
                binaries=summary["binaries"],
                platform_controls=[rows[name] for name in PLATFORM_CONTROLS],
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
            f"{len(arms) - fired} reported a pass or a marker on an input that "
            f"never ran its case",
            flush=True,
        )
    else:
        print(f"\n{passed}/{len(arms)} arms passed", flush=True)
    print(f"Artifacts: {a.output}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
