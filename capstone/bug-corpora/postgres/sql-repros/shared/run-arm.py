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


def score(case_dir, text, result, available, preinstalled=()):
    verdict, why = _score(case_dir, text, result, available, preinstalled)
    # On every verdict, not only the ones that reach the end of _score. A
    # `detected` obtained against a pre-created extension has to carry that
    # fact where the verdict is read -- in results/matrix.tsv and in the arm's
    # evidence -- and not only in a field beside it.
    pre = [e for e in extensions_needed(case_dir) if e in preinstalled]
    if pre and verdict not in ("not-applicable", "not-runnable"):
        why += ("; ran against a fixture that already carried "
                + ", ".join(pre)
                + ", so this arm did not have to create it and this row says "
                  "nothing about whether it can")
    return verdict, why


def _score(case_dir, text, result, available, preinstalled=()):
    """(verdict, evidence). Controls first, mechanism second, directives last.

    The order is the whole point. A control failure that is scored as a verdict
    is worse than no row at all, because it reads as evidence about the arm.
    """
    meta = json.loads((case_dir / "case.json").read_text())

    limit = meta.get("harness_limit")
    if limit:
        return "not-runnable", f"declared by the case: {limit}"

    # An extension the fixture already carries is not one this arm has to be
    # able to create. The case still says CREATE EXTENSION, which succeeds as a
    # no-op lookup of a row that is already there, and the statement the case
    # is actually about then runs. Every verdict reached this way says so.
    pre = [e for e in extensions_needed(case_dir) if e in preinstalled]
    missing = [e for e in extensions_needed(case_dir)
               if e not in available and e not in preinstalled]
    if missing:
        return ("not-applicable",
                "this image cannot create " + ", ".join(missing)
                + " -- outside this arm's denominator, not a verdict about it")

    if "[runner] TIMEOUT" in text:
        return "other", "runner timeout; not a measurement"

    # A fault is only this case's result if it happened after the setup. The
    # trigger's CREATE EXTENSION statements come first, and the stand-alone
    # backend prints one prompt per statement it is ready for, so the prompts
    # seen before the fault say which statement was running. Measured on the
    # sublet arm on 2026-10-06: case 03 faulted with ONE prompt, so the fault
    # was in CREATE EXTENSION ltree and not in the lquery cast the case is
    # about. Scoring that as detected would credit the mechanism with catching
    # a defect it never reached.
    setup = len(extensions_needed(case_dir))
    fault = result.get("fault")
    faulted = bool(fault) or result.get("kind") == "signal"
    if faulted:
        before = text.count("backend>")
        if setup and before <= setup:
            return ("setup-fault",
                    f"faulted after {before} prompt(s) with {setup} CREATE "
                    f"EXTENSION statement(s) ahead of the trigger, so the fault "
                    f"is in the setup and not at the defect: {fault or 'signal'}")
        if fault:
            return "detected", f"capability fault: {fault}"
        return "detected", f"terminated by signal {result.get('value')}"

    if "backend>" not in text:
        return "BADRUN", "no stand-alone backend prompt -- nothing executed"

    # An extension the preflight said was available but that failed here is a
    # control failure for this case, not a silent arm.
    #
    # EXCEPT "already exists", for one the fixture carries. That error is the
    # expected outcome of the case's own CREATE EXTENSION line when the
    # extension is pre-created, and matching it here made the oracle
    # one-sided: every run that did not fault was scored control-failure, so
    # `silent` was unreachable and a `detected` had nothing to be contrasted
    # with. Any OTHER error naming a pre-created extension is still a control
    # failure.
    for name in extensions_needed(case_dir):
        if name in pre and re.search(
                rf'ERROR:.*extension "{re.escape(name)}" already exists', text):
            continue
        if re.search(rf'ERROR:.*extension "{re.escape(name)}"', text) or \
           re.search(rf'ERROR:.*could not (open extension control file|load library).*{re.escape(name)}', text):
            return ("control-failure",
                    f"CREATE EXTENSION {name} failed in this run although the "
                    f"preflight created it; the trigger did not reach the defect")

    # A syntax error means the backend was handed something other than the
    # trigger as written, so whatever followed is not this case's measurement.
    # `postgres --single` takes one line per statement with no continuation,
    # and a multi-line trigger reaches it in pieces: cases 07 and 09 were
    # scored silent on three arms that way on 2026-10-06. Cases that expect
    # errors say so with EXPECT-ERRORS, and a syntax error is never one of
    # those -- the defect is in the backend, not in the SQL.
    if re.search(r"ERROR:\s*(syntax error|unterminated)", text):
        first = re.search(r"ERROR:\s*(syntax error|unterminated)[^\n]*", text)
        return ("control-failure",
                f"the trigger did not parse as written, so nothing below it is "
                f"this case's result: {first.group(0)[:90]}")

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
    # The extensions the image was built with. build-domain.sh links the module
    # into the image and stages its control and version scripts here, and both
    # halves are needed: CREATE EXTENSION reads the control file before it ever
    # asks dfmgr for the library, so an image carrying the module but a share
    # without the control file fails exactly as one carrying neither.
    if args.extensions.is_dir():
        target = args.share / "pgshare" / "extension"
        target.mkdir(parents=True, exist_ok=True)
        for source in sorted(args.extensions.iterdir()):
            shutil.copy2(source, target / source.name)
        print(f"  staged {len(list(args.extensions.iterdir()))} extension files "
              f"from {args.extensions}")
    else:
        print(f"  no extension directory at {args.extensions}; only what the "
              f"share already holds will be creatable")

    staged = vm(args.state, "exec", "--", "/bin/sh", "-c",
                "mkdir -p /usr/local/pgsql && "
                "[ -d /usr/local/pgsql/share/timezonesets ] || "
                "cp -a /mnt/host/pgshare /usr/local/pgsql/share; "
                # Unconditionally, unlike the rest of the share: a guest that
                # has booted before already has an extension directory, and the
                # stale one would decide what this run can create.
                "cp -f /mnt/host/pgshare/extension/* "
                "       /usr/local/pgsql/share/extension/ 2>/dev/null; "
                "test -d /usr/local/pgsql/share/timezonesets && echo SHARE-OK")
    if "SHARE-OK" not in staged.stdout:
        sys.exit(f"guest share staging failed: {staged.stdout[-400:]}")

    wanted = sorted({e for d in cases(args) for e in extensions_needed(d)})

    # ONE SESSION PER EXTENSION, and each has to say so itself. Putting all of
    # them in one session and reading "no error" as success is wrong in both
    # directions, and both were observed on 2026-10-06: on the sublet arm the
    # session took a capability fault on its second statement, and the three
    # extensions after it were recorded available although they never ran.
    # Absence of an error is not evidence; the marker below is.
    first = out / "preflight-00-backend.sql"
    first.write_text("SELECT 'PREFLIGHT-BACKEND-OK' AS marker;\n")
    cluster(args, "pgdata-preflight")
    text, _ = backend(args, first, "pgdata-preflight",
                      out / "preflight-00.json", out / "preflight-00.out", timeout=600)
    shutil.rmtree(args.share / "pgdata-preflight", ignore_errors=True)
    if "backend>" not in text:
        sys.exit("POSITIVE CONTROL FAILED: the image did not reach a backend prompt")
    if "PREFLIGHT-BACKEND-OK" not in text:
        sys.exit("POSITIVE CONTROL FAILED: the image did not answer a trivial SELECT")

    available, notes = set(), {}
    for index, name in enumerate(wanted, start=1):
        probe = out / f"preflight-{index:02d}-{name}.sql"
        # The marker is read back OUT OF pg_extension, not printed beside the
        # CREATE. `postgres --single` does not abort a session on ERROR, it
        # moves to the next statement, so a marker that merely follows the
        # CREATE prints whether or not the extension was created -- which
        # recorded pgcrypto as available on an image that cannot contain it.
        # ON ONE LINE, for the same reason the triggers are: `postgres --single`
        # takes a line at a time and has no continuation. Written over two
        # lines this probe was itself split in half, every extension came back
        # NOT AVAILABLE, and two arms recorded case 07 as not-applicable on an
        # image that could create pg_trgm perfectly well.
        probe.write_text(
            f"CREATE EXTENSION {name};\n"
            f"SELECT 'PREFLIGHT-OK-' || extname AS marker FROM pg_extension"
            f" WHERE extname = '{name}';\n")
        cluster(args, "pgdata-preflight")
        text, result = backend(args, probe, "pgdata-preflight",
                               out / f"preflight-{index:02d}.json",
                               out / f"preflight-{index:02d}.out", timeout=600)
        shutil.rmtree(args.share / "pgdata-preflight", ignore_errors=True)
        # A syntax error here is this runner's bug, not a missing extension, and
        # the two must never be confused: one is a harness failure and the
        # other is a fact about the image.
        if re.search(r"ERROR:\s*(syntax error|unterminated)", text):
            sys.exit(f"PREFLIGHT IS BROKEN: the probe for {name} did not parse. "
                     f"This is the runner's SQL, not the image: see "
                     f"{out / f'preflight-{index:02d}.out'}")
        if f"PREFLIGHT-OK-{name}" in text:
            available.add(name)
            # The row exists, which is what decides whether a case can run.
            # Whether THIS probe put it there is a different question, and
            # conflating them hid something once: with a fixture that already
            # carried ltree, the probe reported "created" on an arm that in
            # fact cannot create it -- CREATE EXTENSION errored with "already
            # exists" and the extension's script, which is what faults, never
            # ran.
            notes[name] = ("already in the fixture"
                           if re.search(rf'ERROR:.*extension "{re.escape(name)}" already exists', text)
                           else "created")
        elif result.get("fault") or result.get("kind") == "signal":
            # Creating it faults. Not available for measurement, and worth its
            # own word: a case built on it would record a fault that belongs to
            # the extension's own setup rather than to the defect.
            notes[name] = "FAULTS ON CREATE"
        else:
            notes[name] = "NOT AVAILABLE"
        print(f"  extension {name:<16} {notes[name]}")
    return wanted, available


def _fixture_provenance(fixture):
    """What make-fixture.py recorded beside this cluster, if anything."""
    side = fixture.with_suffix(".provenance.json")
    if not side.is_file():
        return None
    try:
        return json.loads(side.read_text())
    except ValueError:
        return {"unreadable": str(side)}


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
    parser.add_argument("--preinstalled", default="",
                        help="comma-separated extensions already created in the "
                             "fixture, which this arm therefore does not have to "
                             "create; recorded with every verdict that used one")
    parser.add_argument("--extensions", type=Path,
                        help="the control and version scripts build-domain.sh "
                             "staged; defaults to <image>/../../share/extension")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only", help="comma-separated case numbers")
    parser.add_argument("--control", action="store_true",
                        help="run each case's control.sql instead of its "
                             "trigger.sql. The control is the same statement "
                             "below the threshold the defect needs, so it must "
                             "COMPLETE; a fault there means the trigger's fault "
                             "was not the defect and the row must be withdrawn")
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
    args.extensions = args.extensions or (args.image.resolve().parents[1]
                                          / "share" / "extension")
    config = json.loads((args.state / "config.json").read_text())
    args.share = args.share or Path(config["share"])
    if Path(config["share"]) != args.share:
        sys.exit(f"the VM state's share is {config['share']} but this run uses {args.share}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or (CORPUS / "results"
                       / f"{args.arm}-{'control-' if args.control else ''}{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    # One run at a time: two would share the staging directory under the share.
    lock = open(args.share / ".sql-repros.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another run holds the share's lock")

    preinstalled = {e.strip() for e in args.preinstalled.split(",") if e.strip()}
    digest = sha256(args.image)
    print(f"arm={args.arm} image={digest[:16]}")
    wanted, available = preflight(args, out)
    if preinstalled:
        print("  pre-created in the fixture: " + ", ".join(sorted(preinstalled)))

    rows, produced = [], 0
    for case_dir in cases(args):
        tag = case_dir.name
        name = f"pgdata-{tag}"
        cluster(args, name)
        sql = case_dir / ("control.sql" if args.control else "trigger.sql")
        if not sql.is_file():
            rows.append((tag, "no-control",
                         f"{sql.name} does not exist for this case"))
            print(f"{tag:<52} no-control", flush=True)
            shutil.rmtree(args.share / name, ignore_errors=True)
            continue
        text, result = backend(args, sql, name,
                               out / f"{tag}.json", out / f"{tag}.out")
        shutil.rmtree(args.share / name, ignore_errors=True)
        if text.strip():
            produced += 1
        verdict, why = score(case_dir, text, result, available, preinstalled)
        if args.control:
            # A control is read by whether it COMPLETED, not by whether the
            # mechanism reported. `detected` here is the bad outcome: it says
            # the fault does not depend on the threshold the defect needs, so
            # whatever the trigger produced was not this defect.
            verdict = {"detected": "control-broken",
                       "silent": "control-held",
                       "differential": "control-held"}.get(verdict, verdict)
            why = ("the control ran below the defect's threshold and "
                   + ("FAULTED ANYWAY, so the trigger's fault is not this "
                      "defect" if verdict == "control-broken"
                      else "completed, as a control must")
                   + "; " + why)
        rows.append((tag, verdict, why))
        print(f"{tag:<52} {verdict:<16} {why[:64]}", flush=True)

    if not produced:
        sys.exit("NO CASE PRODUCED OUTPUT -- a harness failure, not a measurement of zero")

    gated = None
    if args.control:
        args.no_gate = True          # nothing in a control run should fault
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
                              "setup-fault", "BADRUN", "other"))
    (out / "inputs.json").write_text(json.dumps({
        "arm": args.arm,
        "image": str(args.image),
        "image_sha256": digest,
        "runner_sha256": sha256(Path(__file__)),
        "fixture": str(args.fixture),
        "extensions_staged_from": str(args.extensions),
        "started_utc": stamp,
        "extensions_wanted": wanted,
        "extensions_available": sorted(available),
        # Extensions the fixture carried. A verdict on a case needing one of
        # these says nothing about whether this arm could have created it.
        "extensions_preinstalled": sorted(preinstalled),
        # Which image wrote the catalog this run read, when the fixture
        # carries a pre-created extension. Without it the row says an
        # extension was pre-created but not by what.
        "fixture_provenance": _fixture_provenance(args.fixture),
        # The gate is a corpus case doing double duty, which is weaker than a
        # purpose-built control: it says the mechanism reported on SOMETHING in
        # this configuration, not that it would have reported on each silent
        # case. It is recorded so a reader can weigh it rather than assume it.
        "mechanism_gate": {"case": args.gate, "passed": gated,
                           "kind": "a corpus case required to be detected"}
        if not args.no_gate else None,
        "ran": "control.sql" if args.control else "trigger.sql",
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
