#!/usr/bin/env python3
"""Run the sql-repros cases on one Capstone domain arm.

    ARM=spatial|sublet run-arm.py --state <vm state> --image <postgres.dom>
                                  --fixture <initdb'd cluster> [--out <dir>]

The arm is proven by the image's sha256, recorded beside the results, never by
the label passed in.

WHAT THIS REFUSES TO DO, each because it has already gone wrong once:

  * Score a run whose stand-alone backend prompt never appeared. "The input was
    accepted" and "nothing executed" look identical otherwise.

  * Score at all when no case produced output. A parse failure is not a
    measurement of zero.

  * Give a verdict to a case whose extension the image cannot create. This is
    the one that cost a result. On 2026-10-05 case 03 was scored `silent` on
    both Capstone arms from an image that did not contain ltree: a domain
    cannot dlopen, the MODULES list of ports/postgres/app/build-domain.sh named
    only dict_snowball and plpgsql, and the guest share held plpgsql.control
    alone. `CREATE EXTENSION ltree` failed, the lquery cast never ran, and the
    backend prompt still appeared -- so nothing above caught it and the row
    read as "the mechanism saw nothing". The preflight below now asks the image
    which extensions it can actually create, and a case needing one it cannot
    is `not-applicable`, outside the denominator, with the reason recorded.

  * Pass an EXPECT-ABSENT pattern to grep without `--`. Several begin with '-'
    and were read as options, which filed a reached case NOT-REACHED.
"""
import argparse
import fcntl
import hashlib
import json
import re
import shutil
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


def backend(args, sql_path, data_name, result_path, log_path, timeout=1800):
    """One stand-alone backend against its own cluster, SQL on stdin."""
    command = [PYTHON, "run.py", "--state", str(args.state), "--cwd", "/tmp",
               "--user", "1000:1000", "--result", str(result_path),
               "--stdin", str(sql_path), str(args.image),
               "--", "--single", "-D", f"/mnt/host/{data_name}",
               "-c", "shared_buffers=4MB", "-c", "max_connections=10",
               "-c", "timezone=GMT", "-c", "log_timezone=GMT",
               "-c", "dynamic_shared_memory_type=sysv", "postgres"]
    with log_path.open("w") as log:
        try:
            subprocess.run(command, cwd=str(REPO / "capstone/ports/common/application"),
                           stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            log.write("\n[runner] TIMEOUT\n")
    text = log_path.read_text(errors="replace")
    result = {}
    if result_path.is_file() and result_path.stat().st_size:
        result = json.loads(result_path.read_text())
    return text, result


def cluster(args, name):
    """A fresh cluster per attempt: a case must not inherit another's damage."""
    data = args.share / name
    shutil.rmtree(data, ignore_errors=True)
    shutil.copytree(args.fixture, data)
    subprocess.run(["chmod", "-R", "u+rwX", str(data)])
    subprocess.run(["chown", "-R", "1000:1000", str(data)], stderr=subprocess.DEVNULL)
    return data


def extensions_needed(case_dir):
    sql = (case_dir / "trigger.sql").read_text()
    return re.findall(r"CREATE\s+EXTENSION\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
                      sql, re.IGNORECASE)


def directives(sql_text):
    errors, absent = None, []
    for line in sql_text.splitlines():
        match = re.search(r"EXPECT-ERRORS:\s*(\d+)", line)
        if match:
            errors = int(match.group(1))
        match = re.search(r"EXPECT-ABSENT:\s*(\S.*?)\s*$", line)
        if match:
            absent.append(match.group(1))
    return errors, absent


def score(case_dir, text, result, available):
    """(verdict, evidence). Controls first, mechanism second, directives last.

    The order is the whole point. A control failure that is scored as a verdict
    is worse than no row at all, because it reads as evidence about the arm.
    """
    meta = json.loads((case_dir / "case.json").read_text())

    limit = meta.get("harness_limit")
    if limit:
        return "not-runnable", f"declared by the case: {limit}"

    missing = [e for e in extensions_needed(case_dir) if e not in available]
    if missing:
        return ("not-applicable",
                "this image cannot create " + ", ".join(missing)
                + " -- outside this arm's denominator, not a verdict about it")

    if "[runner] TIMEOUT" in text:
        return "other", "runner timeout; not a measurement"

    fault = result.get("fault")
    if fault:
        return "detected", f"capability fault: {fault}"
    if result.get("kind") == "signal":
        return "detected", f"terminated by signal {result.get('value')}"

    if "backend>" not in text:
        return "BADRUN", "no stand-alone backend prompt -- nothing executed"

    # An extension the preflight said was available but that failed here is a
    # control failure for this case, not a silent arm.
    for name in extensions_needed(case_dir):
        if re.search(rf'ERROR:.*extension "{re.escape(name)}"', text) or \
           re.search(rf'ERROR:.*could not (open extension control file|load library).*{re.escape(name)}', text):
            return ("control-failure",
                    f"CREATE EXTENSION {name} failed in this run although the "
                    f"preflight created it; the trigger did not reach the defect")

    want_errors, absent = directives((case_dir / "trigger.sql").read_text())
    notes = []
    if want_errors is not None:
        # A full-line match: --single prefixes errors with a timestamp, so an
        # anchored '^ERROR:' matches zero every time.
        got = len(re.findall(r"\bERROR:", text))
        notes.append(f"errors {got}/{want_errors} expected")
        if got != want_errors:
            return "differential", "; ".join(notes)
    for pattern in absent:
        hit = subprocess.run(["grep", "-aoE", "--", pattern], input=text,
                             stdout=subprocess.PIPE, universal_newlines=True)
        if hit.stdout.strip():
            notes.append(f"EXPECT-ABSENT fired: {hit.stdout.splitlines()[0][:60]!r}")
            if meta.get("oracle_is_recording"):
                # The directive was written from this case's own run, so it
                # fires by construction: reachability, not reproduction.
                return "reached-recording", "; ".join(notes)
            return "differential", "; ".join(notes)
        notes.append("EXPECT-ABSENT held")
    return "silent", "; ".join(notes) or "completed with no directive and no fault"


def preflight(args, out):
    """Prove the image runs SQL, and learn which extensions it can create.

    Both answers come from the image itself. Asking the build what it linked
    would test the build script's intention; asking the backend tests what is
    in the binary that is about to be measured.
    """
    if "running" not in vm(args.state, "status").stdout:
        sys.exit("the VM is not up; bring it up first, this runner will not boot it")
    vm(args.state, "exec", "--", "/bin/sh", "-c",
       'grep -q "^pg:" /etc/passwd || echo "pg:x:1000:1000:pg:/tmp:/bin/sh" >> /etc/passwd; '
       'grep -q "^pg:" /etc/group  || echo "pg:x:1000:" >> /etc/group')
    staged = vm(args.state, "exec", "--", "/bin/sh", "-c",
                "mkdir -p /usr/local/pgsql && "
                "[ -d /usr/local/pgsql/share/timezonesets ] || "
                "cp -a /mnt/host/pgshare /usr/local/pgsql/share; "
                "test -d /usr/local/pgsql/share/timezonesets && echo SHARE-OK")
    if "SHARE-OK" not in staged.stdout:
        sys.exit(f"guest share staging failed: {staged.stdout[-400:]}")

    wanted = sorted({e for d in cases(args) for e in extensions_needed(d)})
    probe = out / "preflight.sql"
    probe.write_text("SELECT 1 AS backend_runs_sql;\n"
                     + "".join(f"CREATE EXTENSION {e};\n" for e in wanted))
    cluster(args, "pgdata-preflight")
    text, _ = backend(args, probe, "pgdata-preflight",
                      out / "preflight.json", out / "preflight.out", timeout=600)
    shutil.rmtree(args.share / "pgdata-preflight", ignore_errors=True)

    if "backend>" not in text:
        sys.exit("POSITIVE CONTROL FAILED: the image did not reach a backend prompt")
    if "backend_runs_sql" not in text:
        sys.exit("POSITIVE CONTROL FAILED: the image did not answer SELECT 1")
    available = {e for e in wanted
                 if not re.search(rf'ERROR:.*(extension|library).*"?{re.escape(e)}"?', text)}
    for name in wanted:
        print(f"  extension {name:<16} {'available' if name in available else 'NOT AVAILABLE'}")
    return wanted, available


def cases(args):
    found = sorted(d for d in CORPUS.glob("[0-9][0-9]_*")
                   if (d / "trigger.sql").is_file())
    if args.only:
        keep = set(args.only.split(","))
        found = [d for d in found if d.name.split("_")[0] in keep]
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--share", type=Path, help="defaults to the state's own share")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True,
                        help="an initdb'd cluster, copied fresh for every case")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only", help="comma-separated case numbers")
    parser.add_argument("--gate", default="02",
                        help="the case that must be detected for the run to count")
    parser.add_argument("--no-gate", action="store_true",
                        help="record the run without the mechanism gate, and say so")
    args = parser.parse_args()

    declared = json.loads((CORPUS / "corpus.json").read_text())["required_arms"]
    if args.arm not in declared:
        sys.exit(f"arm {args.arm!r} is not in corpus.json required_arms: {declared}")
    if args.arm == "cheribsd-revocation":
        sys.exit("cheribsd-revocation is not a domain arm")
    if not args.image.is_file():
        sys.exit(f"no image at {args.image}")
    config = json.loads((args.state / "config.json").read_text())
    args.share = args.share or Path(config["share"])
    if Path(config["share"]) != args.share:
        sys.exit(f"the VM state's share is {config['share']} but this run uses {args.share}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or (CORPUS / "results" / f"{args.arm}-{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    # One run at a time: two would share the staging directory under the share.
    lock = open(args.share / ".sql-repros.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another run holds the share's lock")

    digest = sha256(args.image)
    print(f"arm={args.arm} image={digest[:16]}")
    wanted, available = preflight(args, out)

    rows, produced = [], 0
    for case_dir in cases(args):
        tag = case_dir.name
        name = f"pgdata-{tag}"
        cluster(args, name)
        text, result = backend(args, case_dir / "trigger.sql", name,
                               out / f"{tag}.json", out / f"{tag}.out")
        shutil.rmtree(args.share / name, ignore_errors=True)
        if text.strip():
            produced += 1
        verdict, why = score(case_dir, text, result, available)
        rows.append((tag, verdict, why))
        print(f"{tag:<52} {verdict:<16} {why[:64]}", flush=True)

    if not produced:
        sys.exit("NO CASE PRODUCED OUTPUT -- a harness failure, not a measurement of zero")

    gated = None
    if not args.no_gate:
        hit = [v for t, v, _ in rows if t.split("_")[0] == args.gate]
        if not hit:
            print(f"gate case {args.gate} was not run; use --only to include it "
                  f"or --no-gate to record the run without it", file=sys.stderr)
            return 2
        gated = hit[0] == "detected"
        if not gated:
            print(f"MECHANISM GATE FAILED: case {args.gate} is {hit[0]}, not detected. "
                  f"Every silent row in this run is unqualified; not writing a matrix.",
                  file=sys.stderr)
            return 2

    with (out / "matrix.tsv").open("w") as stream:
        stream.write("case\tarm\tverdict\tevidence\n")
        for tag, verdict, why in rows:
            stream.write(f"{tag}\t{args.arm}\t{verdict}\t{why}\n")

    counts = {}
    for _, verdict, _ in rows:
        counts[verdict] = counts.get(verdict, 0) + 1
    scored = sum(n for v, n in counts.items()
                 if v not in ("not-applicable", "not-runnable", "control-failure",
                              "BADRUN", "other"))
    (out / "inputs.json").write_text(json.dumps({
        "arm": args.arm,
        "image": str(args.image),
        "image_sha256": digest,
        "runner_sha256": sha256(Path(__file__)),
        "fixture": str(args.fixture),
        "started_utc": stamp,
        "extensions_wanted": wanted,
        "extensions_available": sorted(available),
        # The gate is a corpus case doing double duty, which is weaker than a
        # purpose-built control: it says the mechanism reported on SOMETHING in
        # this configuration, not that it would have reported on each silent
        # case. It is recorded so a reader can weigh it rather than assume it.
        "mechanism_gate": {"case": args.gate, "passed": gated,
                           "kind": "a corpus case required to be detected"}
        if not args.no_gate else None,
        "cases": len(rows),
        "scored": scored,
        "verdicts": counts,
    }, indent=2) + "\n")
    print(f"\n--- {args.arm}: {scored} scored of {len(rows)} run ---")
    for key in sorted(counts):
        print(f"  {key:<16} {counts[key]}")
    print(f"results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
