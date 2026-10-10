#!/usr/bin/env python3
"""Run the c-repros cases on one Capstone domain arm.

    run-arm.py --arm spatial|sublet --state <vm state>
               --bindir <shared/build-domain.sh's output> [--out <dir>]

One program per case, as the corpus contract says: a capability fault ends the
domain, so a case that provokes one cannot also report results beside it.

WHAT THIS REFUSES TO DO:

  * Score a case whose BEGIN line never appeared. The program prints
    `case N BEGIN` before anything else, so its absence means the image did
    not run, not that the arm was silent.

  * Score a control failure as a verdict. shared/driver.c exits 75 through
    pgclient_give_up for an infrastructure failure -- a malloc that returned
    NULL, a binary told to run a case it was not built for -- and 75 is never
    a statement about the mechanism.

  * Confuse "the case returned" with "the arm saw nothing" without evidence
    that the defect ran. Each case prints its own PG_DEFECT marker before the
    defective access, and a silent row is only recorded when that marker is in
    the output.

The arm is proven by the image's sha256, not by the label passed in. There is
one image per case here rather than one for the corpus, so every row carries
its own hash.
"""
import argparse
import fcntl
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1]
REPO = CORPUS.parents[3]
PYTHON = sys.executable


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vm(state, *argv, timeout=300):
    return subprocess.run(
        [PYTHON, "-m", "capstone_vm", "--state", str(state), *argv],
        cwd=str(REPO / "capstone/runtime/host"), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)


def score(number, text, result):
    """(verdict, evidence). Controls first, then the mechanism."""
    if "[runner] TIMEOUT" in text:
        return "other", "runner timeout; not a measurement"
    if "CONTROL-FAILED" in text:
        line = next((l for l in text.splitlines() if "CONTROL-FAILED" in l), "")
        return "control-failure", f"the case refused its own setup: {line.strip()}"
    if result.get("kind") == "exit" and result.get("value") == 75:
        return "control-failure", "exit 75: an infrastructure failure, never a verdict"

    fault = result.get("fault")
    if fault:
        expect = re.search(r"expect_fault_in=(\S+)", text)
        where = f"; case expected it in {expect.group(1)}" if expect else ""
        return "detected", f"capability fault: {fault}{where}"
    if result.get("kind") == "signal":
        return "detected", f"terminated by signal {result.get('value')}"

    if f"case {number} BEGIN" not in text:
        return "BADRUN", "no BEGIN line -- the image did not run"
    if "PG_DEFECT" not in text:
        return ("BADRUN",
                "the case ran but printed no PG_DEFECT marker, so there is no "
                "evidence the defective access was reached")
    if f"case {number} RETURNED" in text:
        return "silent", "reached its marker and returned; the mechanism did not report"
    return "other", "reached its marker but neither returned nor faulted"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--bindir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only", help="comma-separated case numbers")
    parser.add_argument("--gate", default="00",
                        help="the case that must be detected for the run to count")
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()

    declared = json.loads((CORPUS / "corpus.json").read_text())["required_arms"]
    if args.arm not in declared:
        sys.exit(f"arm {args.arm!r} is not in corpus.json required_arms: {declared}")
    if args.arm == "cheribsd-revocation":
        sys.exit("cheribsd-revocation is not a domain arm -- use shared/build-cheri.sh's output")
    config = json.loads((args.state / "config.json").read_text())
    share = Path(config["share"])
    if "running" not in vm(args.state, "status").stdout:
        sys.exit("the VM is not up; bring it up first, this runner will not boot it")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or (CORPUS / "results" / f"{args.arm}-{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    lock = open(share / ".c-repros.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another run holds the share's lock")

    found = sorted(d for d in CORPUS.glob("[0-9][0-9]_*") if (d / "case.c").is_file())
    if args.only:
        keep = set(args.only.split(","))
        found = [d for d in found if d.name.split("_")[0] in keep]

    rows, produced, images = [], 0, {}
    for case_dir in found:
        tag = case_dir.name
        number = str(int(tag.split("_")[0]))
        image = args.bindir / f"{tag}.dom"
        if not image.is_file():
            rows.append((tag, "not-applicable",
                         f"no domain image at {image}; shared/build-domain.sh has not "
                         f"built this case for this arm"))
            print(f"{tag:<52} not-applicable", flush=True)
            continue
        images[tag] = sha256(image)
        result_path, log_path = out / f"{tag}.json", out / f"{tag}.out"
        command = [PYTHON, "run.py", "--state", str(args.state), "--cwd", "/tmp",
                   "--result", str(result_path), str(image), "--", number]
        with log_path.open("w") as log:
            try:
                subprocess.run(command,
                               cwd=str(REPO / "capstone/ports/common/application"),
                               stdout=log, stderr=subprocess.STDOUT, timeout=600)
            except subprocess.TimeoutExpired:
                log.write("\n[runner] TIMEOUT\n")
        text = log_path.read_text(errors="replace")
        result = {}
        if result_path.is_file() and result_path.stat().st_size:
            result = json.loads(result_path.read_text())
        if text.strip():
            produced += 1
        verdict, why = score(number, text, result)
        rows.append((tag, verdict, why))
        print(f"{tag:<52} {verdict:<16} {why[:60]}", flush=True)

    if not produced:
        sys.exit("NO CASE PRODUCED OUTPUT -- a harness failure, not a measurement of zero")

    gated = None
    if not args.no_gate:
        hit = [v for t, v, _ in rows if t.split("_")[0] == args.gate]
        if not hit:
            print(f"gate case {args.gate} was not run", file=sys.stderr)
            return 2
        gated = hit[0] == "detected"
        if not gated:
            print(f"MECHANISM GATE FAILED: case {args.gate} is {hit[0]}, not detected. "
                  f"Every silent row is unqualified; not writing a matrix.", file=sys.stderr)
            return 2

    with (out / "matrix.tsv").open("w") as stream:
        stream.write("case\tarm\tverdict\tevidence\n")
        for tag, verdict, why in rows:
            stream.write(f"{tag}\t{args.arm}\t{verdict}\t{why}\n")
    counts = {}
    for _, verdict, _ in rows:
        counts[verdict] = counts.get(verdict, 0) + 1
    scored = sum(n for v, n in counts.items()
                 if v not in ("not-applicable", "control-failure", "BADRUN", "other"))
    (out / "inputs.json").write_text(json.dumps({
        "arm": args.arm,
        "bindir": str(args.bindir),
        "image_sha256": images,
        "runner_sha256": sha256(Path(__file__)),
        "started_utc": stamp,
        # A corpus case doing double duty, which is weaker than a purpose-built
        # control: it says the mechanism reported on SOMETHING here, not that it
        # would have reported on each silent case.
        "mechanism_gate": {"case": args.gate, "passed": gated,
                           "kind": "a corpus case required to be detected"}
        if not args.no_gate else None,
        "cases": len(rows),
        "scored": scored,
        "verdicts": counts,
    }, indent=2) + "\n")
    print(f"\n--- {args.arm}: {scored} scored of {len(rows)} ---")
    for key in sorted(counts):
        print(f"  {key:<16} {counts[key]}")
    print(f"results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
