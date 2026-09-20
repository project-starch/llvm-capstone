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
CORPUS = HERE.parent
RUNNER = CORPUS.parents[2] / "ports/common/host/cheribsd/run.py"

MAGIC = 0x31594C50524D5950
REPORT_BYTES = 96  # struct pym_header: 12 x uint64
FIXTURE_BYTES = 128  # that header plus one 4 x uint64 event
SIGPROT = 34
PROT_CHERI_TAG = 2
SIGPROT_EXIT = 128 + SIGPROT
PLATFORM_CONTROLS = ("cheribsd-abi", "cheribsd-bounds")
EXIT_INFRASTRUCTURE = 75

FAULT_LINE = re.compile(
    r"PYC_DEFECT_FAULT case=(\d+) signal=(\d+) code=(\d+) "
    r"pc=(0x[0-9a-f]+) expected=(0x[0-9a-f]+) exact=([01])"
)


def load_cases():
    """The corpus's own case.json files, indexed by case number.

    The metadata is read from the corpus rather than restated here, so a
    verdict cannot describe a case differently from the case's own claims.
    """
    cases = {}
    for path in sorted(CORPUS.glob("gh-*/case.json")):
        claim = json.loads(path.read_text())
        number = claim["case"]
        if number in cases:
            raise SystemExit(f"two case.json files claim case {number}")
        cases[number] = dict(
            case=number,
            directory=path.parent.name,
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


def fault_oracle(number):
    """The complete line a protected arm must print, and nothing weaker.

    The backreference is the point: the trap PC and the probe label address are
    required to be the same text, so a fault at another instruction cannot
    satisfy it however plausible the rest of the line looks.
    """
    return (
        rf"PYC_DEFECT_FAULT case={number} signal={SIGPROT} code={PROT_CHERI_TAG} "
        r"pc=(0x[0-9a-f]+) expected=\1 exact=1"
    )


def build_case(number, mode, program, input_path, timeout):
    case = dict(
        name=f"defect-{number}-mode{mode}",
        program=str(program),
        args=["input.bin", "output.bin", str(mode)],
        inputs={"input.bin": str(input_path)},
        also_expect=[f"PYC_DEFECT case={number} ready"],
        timeout=timeout,
    )
    if mode == 0:
        case["expect"] = f"PYC_DEFECT case={number} completed"
        case["exit"] = 0
        case["outputs"] = ["output.bin"]
    else:
        case["expect_regex"] = fault_oracle(number)
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
    """The verdict for one arm, from what the guest actually printed."""
    lines = stdout.splitlines()
    ready = f"PYC_DEFECT case={number} ready" in lines
    verdict = dict(
        case=number,
        mode=mode,
        expected=(
            "completed" if mode == 0 else "SIGPROT/PROT_CHERI_TAG at pyc_defect_read"
        ),
        ran=row is not None,
        exit=row["exit"] if row else None,
        expected_exit=0 if mode == 0 else SIGPROT_EXIT,
        ready_marker=ready,
        runner_oracle=bool(row and row["passed"]),
        stdout_sha256=row["stdout_sha256"] if row else None,
    )
    if mode == 0:
        fields = report_fields(report)
        verdict["completed_marker"] = f"PYC_DEFECT case={number} completed" in lines
        verdict["report"] = fields
        verdict["passed"] = bool(
            verdict["runner_oracle"]
            and ready
            and verdict["completed_marker"]
            and verdict["exit"] == 0
            and fields is not None
            and fields["magic_ok"]
            and fields["status"] == 0
            and fields["completed"] == 1
            and fields["count"] == 1
            and fields["mode"] == mode
        )
        return verdict
    fault = None
    for line in lines:
        match = FAULT_LINE.fullmatch(line)
        if match:
            fault = dict(
                case=int(match[1]),
                signal=int(match[2]),
                code=int(match[3]),
                pc=match[4],
                expected_pc=match[5],
                exact=int(match[6]),
            )
    verdict["fault"] = fault
    verdict["passed"] = bool(
        verdict["runner_oracle"]
        and ready
        and verdict["exit"] == SIGPROT_EXIT
        and fault is not None
        and fault["case"] == number
        and fault["signal"] == SIGPROT
        and fault["code"] == PROT_CHERI_TAG
        and fault["pc"] == fault["expected_pc"]
        and fault["exact"] == 1
    )
    return verdict


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
    p.add_argument("build", type=Path, help="PoisonCap pymalloc build directory")
    p.add_argument("output", type=Path, help="Fresh output directory")
    for name in ("sdk", "rootfs", "image"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--port", type=int, default=10431)
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
    p.add_argument("--modes", default="0,1", help="Comma-separated modes, default 0,1")
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
    modes = numbers(a.modes, {0, 1}, "mode")
    if a.timeout < 1:
        p.error("the per-arm timeout must be positive")
    program = a.build.resolve() / "bin/defects"
    probe = a.build.resolve() / "bin/cheribsd-abi-probe"
    for path in (program, probe):
        if not path.is_file():
            p.error(f"missing program: {path} (build with -DPY_CORPUS_SRC=...)")
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
                (number, mode, build_case(number, mode, program, path, a.timeout))
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
            verdict["control_fired"] = bool(
                not verdict["passed"]
                and not verdict["ready_marker"]
                and not verdict.get("completed_marker")
                and not verdict.get("fault")
            )
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
                f"pc={fault['pc']} expected={fault['expected_pc']} "
                f"exact={fault['exact']}"
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
