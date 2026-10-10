#!/usr/bin/env python3
"""Run the sql-repros cases on one Capstone domain arm, and record what each run showed.

    run-arm.py --arm spatial|sublet --state <vm state> --image <root>/link/postgres.dom
               --fixture <initdb'd cluster> --llvm-bin <compiler>/bin [--out <dir>] [--only 02,03]

This runner REPORTS; tools/verdicts.py decides. Each case is its trigger.sql on a stand-alone
backend against a fresh copy of the fixture, and becomes one Observation:

  reached     the backend got past the case's CREATE EXTENSION lines and ran the trigger
              statement itself: one prompt per statement, counted. A SQL trigger has no marker
              before its access; reaching it is the statement running, and the evidence says so
  completed   the backend ran to the end of the input and exited
  fault       the host's domain fault line, the pc resolved from the image's own symbols
  attribution `control` when the case's control.sql -- the same statement below the defect's
              threshold -- completes on the same image in the same invocation; `function` when
              the fault lies in a function case.json `fault_sites` justifies. A fault with
              neither is not a catch: case 03's sublet fault was in exactly that position, and
              its control faulted at the same instruction

WHAT THE ARM IS, read from the image's build root and never from the label: `domain/nested.mode`
(none, or sublet for PostgreSQL's context pools) and the runtime SDK's heap. corpus.json maps the
arm to a configuration in tools/arms.json, which fixes both; a mismatch is refused.

Before any case the configuration's controls run in the server, on the same image, through
pgcorpus_reach's corpus_control(): a write past and a read after free of a malloc'd object, and
on the nested-sublet configuration the same through palloc. A silence is MISSED only when every
control did what the configuration declares.

WHAT THIS STILL REFUSES, each because it went wrong once: scoring a run whose backend prompt never
appeared; scoring a case whose extension the image cannot create (out of the arm's denominator,
with the reason); scoring a trigger that did not parse (`postgres --single` takes one line per
statement); and calling a fault in the CREATE EXTENSION setup the defect's.

Exit: 0 judged, 75 nothing could be judged.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[1]
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import appvm  # noqa: E402
import virtualvm  # noqa: E402
import verdicts as v  # noqa: E402

# What each configuration's image must be: (nested mode, SDK heap).
BUILD = {"app-level0": ("none", "level0"),
         "app-level0-pg-nested-sublet": ("sublet", "level0"),
         "virtual-mallocng": ("none", "virtual-mallocng"),
         "virtual-mallocng-pg-pools": ("sublet", "virtual-mallocng")}
PG_ARGS = ("--single -D {data} -c shared_buffers=4MB -c max_connections=10 -c timezone=GMT "
           "-c log_timezone=GMT -c dynamic_shared_memory_type=sysv postgres")


def identity(image):
    """(nested, heap, build root) of a PostgreSQL image, from what its build left beside it.
    build-virtual.sh writes image/manifest.json beside image/postgres.dom and the build root at
    source/; build-domain.sh alone leaves the image in <root>/link/."""
    image = image.resolve()
    manifest = image.parent / "manifest.json"
    if manifest.is_file():
        m = json.loads(manifest.read_text())
        root = image.parents[1] / "source"
        nested = {"postgres": "sublet"}.get(m.get("nested"), m.get("nested"))
        heap = "virtual-mallocng" if m.get("profile") == "virtual" else m.get("heap")
        return nested, heap, root
    root = image.parents[1]
    mode = root / "domain/nested.mode"
    return (mode.read_text().strip() if mode.is_file() else "none"), appvm.sdk_identity(root / "runtime")[0], root


def backend(args, sql_path, data_name, out_base, timeout=1800):
    """One stand-alone backend against its own cluster, SQL on stdin."""
    return appvm.run_app(args.state, args.image, [
        "--single", "-D", f"/mnt/host/{data_name}", "-c", "shared_buffers=4MB",
        "-c", "max_connections=10", "-c", "timezone=GMT", "-c", "log_timezone=GMT",
        "-c", "dynamic_shared_memory_type=sysv", "postgres"], out_base,
        run_args=["--user", "1000:1000", "--stdin", str(sql_path)], timeout=timeout)


def statements(path):
    """The file's statements, one per line, with comment-only and blank lines dropped.

    `postgres --single` prints a prompt for every input LINE, comments included, and the setup
    attribution below counts prompts against the number of CREATE EXTENSION lines ahead of the
    trigger. A control file with a fifty-line header therefore looks as though it ran fifty
    statements, so a fault inside CREATE EXTENSION would be attributed to the trigger instead of
    to setup. Raised in review on PR #198, where case 03's control showed 23 prompts for one
    CREATE EXTENSION. Dropping what is not a statement makes the count exact. The corpus already
    requires one statement per line, because this backend has no line continuation, so no
    statement is split by this. What was sent is written out beside the run.
    """
    keep = [ln for ln in path.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("--")]
    return "".join(ln + "\n" for ln in keep)


def cluster(args, name):
    """A fresh cluster per run: a case must not inherit another's damage."""
    data = args.share / name
    shutil.rmtree(data, ignore_errors=True)
    shutil.copytree(args.fixture, data)
    subprocess.run(["chmod", "-R", "u+rwX", str(data)])
    subprocess.run(["chown", "-R", "1000:1000", str(data)], stderr=subprocess.DEVNULL)
    return data


def session(args, sql_text_or_path, name, raw):
    """Write the SQL if given as text, run it on a fresh cluster, clean up. (text, result).

    On the virtual kit nothing runs yet: the session becomes a step of the one batch boot, and
    its (text, result) is read from args.runs after the boot."""
    sql = Path(sql_text_or_path) if isinstance(sql_text_or_path, Path) else raw / f"{name}.sql"
    if not isinstance(sql_text_or_path, Path):
        sql.write_text(sql_text_or_path)
    if args.replay:
        return args.runs.get(name, ("", {"kind": "none"}))
    if args.batch is not None:
        staged = args.batch.put(f"sql/{name}.sql", sql)
        args.batch.step(name, "rm -rf /tmp/pgd && cp -a /mnt/vm/pgcluster /tmp/pgd && chown -R nobody /tmp/pgd "
                        f"&& su nobody -s /bin/sh -c 'cd /mnt/vm; ./capstone-vexec ./bin/postgres "
                        f"{PG_ARGS.format(data='/tmp/pgd')} < {staged}'; rc=$?; rm -rf /tmp/pgd; exit $rc")
        return "", {"kind": "none"}
    data = f"pgdata-{name}"
    cluster(args, data)
    try:
        return backend(args, sql, data, raw / name)
    finally:
        shutil.rmtree(args.share / data, ignore_errors=True)


def extensions_needed(case_dir, name="trigger.sql"):
    return re.findall(r"CREATE\s+EXTENSION\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
                      (case_dir / name).read_text(), re.IGNORECASE)


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


def faulted(text, result, symbols=None):
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if fault is None and result.get("kind") == "signal":
        fault = v.Fault(cause=-1, pc=0)            # a signal with no domain fault line
    return fault


def observe(case_dir, text, result, available, preinstalled=(), symbols=None):
    """What one trigger run showed, as facts. Attribution by control is added by the caller."""
    meta = json.loads((case_dir / "case.json").read_text())
    o = v.Observation(case=case_dir.name, arm="", image_sha256=result.get("image_sha256"))
    needed = extensions_needed(case_dir)
    pre = [e for e in needed if e in preinstalled]
    if meta.get("harness_limit"):
        o.infra, o.notes = "out-of-denominator", f"declared by the case: {meta['harness_limit']}"
        return o
    missing = [e for e in needed if e not in available and e not in preinstalled]
    if missing:
        o.infra, o.notes = "out-of-denominator", "this image cannot create " + ", ".join(missing)
        return o
    if "[runner] TIMEOUT" in text:
        o.infra, o.notes = "infra", "runner timeout"
        return o
    if "backend>" not in text:
        o.infra, o.notes = "infra", "no stand-alone backend prompt: nothing executed"
        return o
    if pre:
        o.notes = (f"ran against a fixture that already carried {', '.join(pre)}; "
                   "this row says nothing about whether this arm can create it")
    setup = len(needed)
    # Exact because the runner sends statements only; see statements().
    prompts = text.count("backend>")
    fault = faulted(text, result, symbols)
    if fault:
        o.fault = fault
        o.reached = prompts > setup
        o.reach_evidence = (f"{prompts} statement(s) reached the backend, against {setup} CREATE "
                            f"EXTENSION line(s) ahead of the trigger; comment and blank lines are "
                            f"not sent, so the count is statements and not input lines")
        sites = set(meta.get("fault_sites", []))
        if fault.symbol and fault.symbol in sites:
            o.attribution, o.attribution_evidence = "function", f"{fault.symbol} is a declared fault site"
        else:
            o.attribution_evidence = (f"in {fault.symbol or 'no known function'}; declared sites "
                                      f"{sorted(sites) or 'none'}")
        return o
    for name in needed:
        if name in pre and re.search(rf'ERROR:.*extension "{re.escape(name)}" already exists', text):
            continue
        if re.search(rf'ERROR:.*extension "{re.escape(name)}"', text) or re.search(
                rf"ERROR:.*could not (open extension control file|load library).*{re.escape(name)}", text):
            o.notes = (o.notes + f"; CREATE EXTENSION {name} failed in this run").lstrip("; ")
            return o                                                   # not reached
    if re.search(r"ERROR:\s*(syntax error|unterminated)", text):
        first = re.search(r"ERROR:\s*(syntax error|unterminated)[^\n]*", text)
        o.infra = "infra"
        o.notes = f"the trigger did not parse as written: {first.group(0)[:90]}"
        return o
    o.reached = prompts > setup
    o.completed = result.get("kind") == "exit"
    evidence = []
    want_errors, absent = directives((case_dir / "trigger.sql").read_text())
    if want_errors is not None:
        got = len(re.findall(r"\bERROR:", text))
        evidence.append(f"errors {got}/{want_errors} expected"
                        + (" -- the defect is visible" if got != want_errors else ""))
    for pattern in absent:
        hit = subprocess.run(["grep", "-aoE", "--", pattern], input=text,
                             stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
        evidence.append(f"EXPECT-ABSENT {'fired' if hit else 'held'}"
                        + (" (written from this case's own run: reachability only)"
                           if hit and meta.get("oracle_is_recording") else ""))
    o.reach_evidence = (f"the trigger statement ran ({prompts} prompts, {setup} setup line(s))"
                        + (f"; {'; '.join(evidence)}" if evidence else ""))
    return o


CONTROL_PROBES = ("corpus_control_write", "corpus_control_read")


def control_seen(name, text, result, symbols=None):
    """'fault' when the control faulted IN its own probe function, 'complete' when it returned.

    Not by its NOTICE mark: the stand-alone backend buffers its output, and a fault kills the
    process before the buffer is written, so the mark of a control that faulted is never seen
    (2026-10-10, both controls on the virtual VM). The pc, resolved from the image's symbols,
    is the stronger evidence anyway."""
    fault = faulted(text, result, symbols)
    if fault:
        return "fault" if fault.symbol in CONTROL_PROBES else "none"
    return "complete" if "CONTROL RETURNED" in text else "none"


def preflight(args, out):
    """Prove the image runs SQL, and learn which extensions it can create -- from the image itself.

    ONE SESSION PER EXTENSION, each read back out of pg_extension: a session that faults on its
    second statement, or a CREATE that errors while a marker beside it still prints, both once
    recorded extensions as available that were not (2026-10-06). One statement per line:
    `postgres --single` has no continuation, and a split probe recorded every extension missing.
    """
    if args.batch is None:
        stage_physical(args)
    planning = args.batch is not None and not args.replay
    text, _ = session(args, "SELECT 'PREFLIGHT-BACKEND-OK' AS marker;\n", "preflight-00", out)
    if not planning and "PREFLIGHT-BACKEND-OK" not in text:
        sys.exit("POSITIVE CONTROL FAILED: the image did not answer a trivial SELECT")
    wanted = sorted({e for d in cases(args) for e in extensions_needed(d)} | {"pgcorpus_reach"})
    available = set()
    for index, name in enumerate(wanted, start=1):
        text, result = session(args, f"CREATE EXTENSION {name};\nSELECT 'PREFLIGHT-OK-' || extname AS "
                                     f"marker FROM pg_extension WHERE extname = '{name}';\n",
                               f"preflight-{index:02d}-{name}", out)
        if planning:
            continue
        if re.search(r"ERROR:\s*(syntax error|unterminated)", text):
            sys.exit(f"PREFLIGHT IS BROKEN: the probe for {name} did not parse (the runner's SQL)")
        state = "NOT AVAILABLE"
        if f"PREFLIGHT-OK-{name}" in text:
            available.add(name)
            state = ("already in the fixture" if re.search(
                rf'ERROR:.*extension "{re.escape(name)}" already exists', text) else "created")
        elif faulted(text, result):
            state = "FAULTS ON CREATE"
        print(f"  extension {name:<16} {state}", flush=True)
    return wanted, available


def stage_physical(args):
    """The persistent VM: the pg user, and the share the image's binary looks for."""
    appvm.require_up(args.state)
    appvm.vm(args.state, "exec", "--", "/bin/sh", "-c",
             'grep -q "^pg:" /etc/passwd || echo "pg:x:1000:1000:pg:/tmp:/bin/sh" >> /etc/passwd; '
             'grep -q "^pg:" /etc/group  || echo "pg:x:1000:" >> /etc/group; '
             # The virtual module's device is root-only; the backend runs as 1000 (run-ports.py
             # opens it the same way).
             '[ -c /dev/capstone-vm ] && chmod 666 /dev/capstone-vm || true')
    # The share the backend reads (timezone sets, catalogs' SQL), from --pg-share. No other script
    # puts it into the VM share, and a guest without it cannot start a backend.
    target = args.share / "pgshare"
    if not (target / "timezonesets").is_dir():
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(args.pg_share, target)
    # The control file and version scripts build-domain.sh staged: CREATE EXTENSION reads the
    # control file before it asks for the library, so a share without it fails like an image
    # without the module.
    if args.extensions.is_dir():
        target = args.share / "pgshare" / "extension"
        target.mkdir(parents=True, exist_ok=True)
        for source in sorted(args.extensions.iterdir()):
            shutil.copy2(source, target / source.name)
    staged = appvm.vm(args.state, "exec", "--", "/bin/sh", "-c",
                      # Replaced whole, every run: on a VM that stays up, a half-staged share from an
                      # earlier attempt would otherwise make `cp -a` copy INTO it (seen 2026-10-10).
                      "mkdir -p /usr/local/pgsql && rm -rf /usr/local/pgsql/share && "
                      "cp -a /mnt/host/pgshare /usr/local/pgsql/share && "
                      "test -d /usr/local/pgsql/share/timezonesets && echo SHARE-OK")
    if "SHARE-OK" not in staged.stdout:
        sys.exit(f"guest share staging failed: {staged.stdout[-400:]}")


def cases(args):
    found = sorted(d for d in CORPUS.glob("[0-9][0-9]_*") if (d / "trigger.sql").is_file())
    if args.only:
        keep = set(args.only.split(","))
        found = [d for d in found if d.name[:2] in keep]
    return found


def measure(args, spec, raw, preinstalled, symbols):
    """Preflight, the configuration's controls, then every case. On the virtual kit this runs
    twice: once to plan every session into the batch (every control.sql included, since whether
    one is needed is only known afterwards), and once over the boot's results."""
    planning = args.batch is not None and not args.replay
    wanted, available = preflight(args, raw)
    controls = []
    for name in spec["controls"]:
        if not planning and "pgcorpus_reach" not in available:
            controls.append(v.Control(name, "none", "pgcorpus_reach is not creatable on this image"))
            continue
        text, result = session(args, f"CREATE EXTENSION pgcorpus_reach;\nSELECT corpus_control('{name}');\n",
                               f"control-{name}", raw)
        if planning:
            continue
        seen = control_seen(name, text, result, symbols)
        where = (faulted(text, result, symbols) or v.Fault(0, 0)).symbol
        controls.append(v.Control(name, seen, f"fault in {where}" if where else (result.get("fault") or "")[:160]))
        print(f"  control {name:<16} {controls[-1].observed:<9} (expected {spec['controls'][name]})", flush=True)

    rows = []
    for case_dir in cases(args):
        text, result = session(args, statements(case_dir / "trigger.sql"), case_dir.name, raw)
        if planning:
            if (case_dir / "control.sql").is_file():
                session(args, statements(case_dir / "control.sql"), f"{case_dir.name}-control", raw)
            continue
        o = observe(case_dir, text, result, available, preinstalled, symbols)
        o.arm, o.image_sha256, o.controls = args.arm, v.sha256(args.image), list(controls)
        if o.fault and o.reached and not o.attribution and (case_dir / "control.sql").is_file():
            ctext, cresult = session(args, statements(case_dir / "control.sql"), f"{case_dir.name}-control", raw)
            if faulted(ctext, cresult):
                o.attribution_evidence += "; its control.sql FAULTS too, so the fault is not the defect's"
            elif "backend>" in ctext and cresult.get("kind") == "exit":
                o.attribution = "control"
                o.attribution_evidence = "control.sql (the statement below the threshold) completed on this image"
            else:
                o.attribution_evidence += "; its control.sql did not run to an answer"
        verdict = v.judge(o, spec)
        rows.append((o, verdict))
        print(f"{case_dir.name:<52} {verdict[0]}{':' + verdict[1] if verdict[1] else ''}  {verdict[2][:80]}",
              flush=True)
    return wanted, available, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--state", type=Path, help="a persistent capstone-vm (physical profile)")
    where.add_argument("--virtual-kit", type=Path, help="the virtual platform kit (tools/virtualvm.py); "
                       "every session then runs in ONE boot")
    parser.add_argument("--share", type=Path, help="physical: defaults to the state's own share")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True,
                        help="an initdb'd cluster, copied fresh for every run")
    parser.add_argument("--pg-share", type=Path,
                        help="the share/postgresql the image reads (timezone sets etc.); "
                             "default <fixture>/../install/share/postgresql")
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--preinstalled", default="",
                        help="comma-separated extensions the fixture already carries; every row "
                             "that used one says so")
    parser.add_argument("--extensions", type=Path,
                        help="control and version scripts build-domain.sh staged; default "
                             "<build root>/share/extension")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--raw", type=Path, help="console logs, SQL and the virtual stage; never inside the bundle")
    parser.add_argument("--only", help="comma-separated case numbers")
    args = parser.parse_args()
    args.batch, args.replay, args.runs = None, False, {}

    decl = json.loads((CORPUS / "corpus.json").read_text())
    config = decl.get("arm_configurations", {}).get(args.arm)
    if config not in BUILD:
        sys.exit(f"arm {args.arm!r} is not a Capstone domain arm of this corpus")
    spec = v.load_arms()[config]
    nested, heap, root = identity(args.image)
    if (nested, heap) != BUILD[config] or (spec["target"] == "capstone-virtual") != bool(args.virtual_kit or appvm.profile(args.state) == "virtual"):
        print(f"CONTROL-FAILED {args.image} is nested={nested} heap={heap} and the run is on "
              f"{'the virtual kit' if args.virtual_kit else 'a physical VM'}; arm {args.arm} is {config}, "
              f"which needs nested={BUILD[config][0]} heap={BUILD[config][1]} on {spec['target']}",
              file=sys.stderr)
        return 75
    args.extensions = args.extensions or root / "share" / "extension"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or CORPUS / "results" / f"{stamp}-qemu" / args.arm
    raw = args.raw or Path("/tmp/capstone/sql-repros-raw") / f"{stamp}-{args.arm}"
    raw.mkdir(parents=True, exist_ok=True)
    preinstalled = {e.strip() for e in args.preinstalled.split(",") if e.strip()}
    symbols = v.Symbols(args.llvm_bin, args.image)
    print(f"arm={args.arm} ({config}) image={v.sha256(args.image)[:16]} nested={nested} heap={heap}")

    pg_share = args.pg_share = args.pg_share or args.fixture.parent / "install/share/postgresql"
    if not (pg_share / "timezonesets").is_dir():
        sys.exit(f"no PostgreSQL share at {pg_share}")
    if args.virtual_kit:
        args.batch = virtualvm.Batch(raw / "stage")
        args.batch.put("bin/postgres", args.image)
        args.batch.put("share", pg_share)
        if args.extensions.is_dir():
            for source in sorted(args.extensions.iterdir()):
                args.batch.put(f"share/extension/{source.name}", source)
        args.batch.put("pgcluster", args.fixture)
        measure(args, spec, raw, preinstalled, symbols)            # plan every session
        args.runs, serial, completed = args.batch.execute(
            args.virtual_kit, raw / "guest", timeout=600 + 300 * len(args.batch.steps))
        print(f"  virtual boot: {'completed' if completed else 'DID NOT COMPLETE'}; "
              f"{len(args.runs)}/{len(args.batch.steps)} sessions ran; serial {serial}", flush=True)
        args.replay = True
        platform = virtualvm.platform(args.virtual_kit, args.llvm_bin / "clang", HERE)
    else:
        vmconfig = json.loads((args.state / "config.json").read_text())
        args.share = args.share or Path(vmconfig["share"])
        if Path(vmconfig["share"]) != args.share:
            sys.exit(f"the VM state's share is {vmconfig['share']} but this run uses {args.share}")
        lock = appvm.share_lock(args.share, "sql-repros")  # noqa: F841 -- held for the run
        platform = appvm.platform(args.state, args.llvm_bin / "clang", HERE)
    wanted, available, rows = measure(args, spec, raw, preinstalled, symbols)

    record = v.write_bundle(out, "postgres/sql-repros", args.arm, rows, {
        "configuration": config, "image": {"sha256": v.sha256(args.image), "nested": nested, "heap": heap},
        "fixture": fixture_record(args.fixture, preinstalled),
        "extensions": {"wanted": wanted, "available": sorted(available)},
        "platform": platform})
    print(f"--- {args.arm} ({config}): {record['tally']}\nresults: {out}")
    return 75 if rows and all(r[1][0] == v.NO_READING for r in rows) else 0


def fixture_record(fixture, preinstalled):
    """What the fixture is, and where it came from.

    The tree hash says WHICH cluster this was, so "the same fixture" no longer rests on a host
    path -- the gap raised in review on PR #198. It does not say which build produced it, and a
    cluster whose extensions were created by one image is being handed to another. make-fixture.py
    writes that beside the fixture as <fixture>.provenance.json, so it is carried here when it
    exists and its absence is recorded rather than passed over.
    """
    prov = Path(str(fixture) + ".provenance.json")
    if not prov.is_file():
        prov = Path(fixture).with_suffix(".provenance.json")
    record = {"cluster_tree_sha256": tree_sha(fixture), "preinstalled": sorted(preinstalled)}
    if prov.is_file():
        record["provenance"] = json.loads(prov.read_text())
        record["provenance_from"] = str(prov)
    else:
        record["provenance"] = None
        record["provenance_note"] = ("no provenance file beside the fixture, so which image "
                                     "created its extensions is not recorded in this run")
    return record


def tree_sha(root):
    """One hash over a directory's relative paths and contents, as run-ports.py records a cluster."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(Path(root).rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(root)).encode() + b"\0")
            h.update(bytes.fromhex(v.sha256(f)))
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
