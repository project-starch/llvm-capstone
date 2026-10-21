#!/usr/bin/env python3
"""Run the parent, sibling and successor controls in the backend, on one arm, in one boot.

    run-controls.py --arm release|protected --virtual-kit KIT --image OUT/image/postgres.dom \
        --fixture CLUSTER --pg-share SHARE --llvm-bin BIN --raw RAW --out RESULT.json [--repeat N]

Pre-registration: capstone/docs/plans/2026-10-11-nested-lifetime-controls.md. Built on
../shared/run-arm.py's virtual path (tools/virtualvm.Batch, one boot, a fresh copy of the cluster
for every session, the backend as `nobody`), which is not changed: this runner is separate because
these are controls, not cases, and their facts are not the judge's Observation fields.

Each session is `CREATE EXTENSION pgcorpus_lifetime; SELECT lifetime_control('<name>');` on a
fresh cluster. What the arm is comes from the image's own manifest (`nested`), never from the
label. Before the controls, two sessions say what the image is: a trivial SELECT, and the
corpus's own arm control `corpus_control('uaf-palloc')` (pgcorpus_reach), which faults on the
protected image and completes on the release image.

The record per session: every `LIFETIME` line the module printed (addresses, bounds, checks), the
accesses it reached (its `mark` lines) with each probe's file:line, whether it returned, and the
host's domain fault line, if any, with cause, pc and the function the pc lies in, resolved from
the image's own symbols. This runner records; the prediction is compared in `verdict`, which
says which check failed.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import sdk_builtins  # noqa: E402  (C-81: refuse a builtins build before anything starts)
import verdicts as v  # noqa: E402
import virtualvm  # noqa: E402

MODULE = HERE.parent / "module" / "pgcorpus_lifetime.c"
NESTED = {"release": "none", "protected": "postgres"}
CONTROLS = ("parent", "sibling", "successor", "successor-stale")
PG_ARGS = ("--single -D /tmp/pgd -c shared_buffers=4MB -c max_connections=10 -c timezone=GMT "
           "-c log_timezone=GMT -c dynamic_shared_memory_type=sysv postgres")
# Which probe each `mark access=` line of a control goes through.
PROBES = {
    ("parent", "read-link"): "lc_probe_parent_read_link",
    ("parent", "read-header"): "lc_probe_parent_read_header",
    ("parent", "write"): "lc_probe_parent_write",
    ("parent", "read"): "lc_probe_parent_read",
    ("sibling", "read"): "lc_probe_sibling_read",
    ("sibling", "write"): "lc_probe_sibling_write",
    ("successor", "write"): "lc_probe_successor_write",
    ("successor", "read"): "lc_probe_successor_read",
    ("successor-stale", "write"): "lc_probe_successor_write",
    ("successor-stale", "read"): "lc_probe_successor_read",
    ("successor-stale", "stale-read"): "lc_probe_stale_read",
}


def probe_lines():
    """{probe function: 'path:line'} of the access statement inside each lc_probe_* function:
    the first line after the function's name that both accesses memory and carries the
    control's comment."""
    rel = MODULE.relative_to(REPO)
    lines = MODULE.read_text().splitlines()
    found = {}
    for i, line in enumerate(lines):
        m = re.match(r"^(lc_probe_\w+)\(", line)
        if not m:
            continue
        for j in range(i + 1, min(i + 15, len(lines))):
            if "/*" in lines[j] and ";" in lines[j] and ("*" in lines[j].split("/*")[0] or "[" in lines[j]):
                found[m.group(1)] = f"{rel}:{j + 1}"
                break
    missing = set(PROBES.values()) - set(found)
    if missing:
        sys.exit(f"probe lines not found for {sorted(missing)}")
    return found


def session(batch, name, sql):
    staged = batch.put(f"sql/{name}.sql", text=sql)
    batch.step(name, "rm -rf /tmp/pgd && cp -a /mnt/vm/pgcluster /tmp/pgd && chown -R nobody /tmp/pgd "
               f"&& su nobody -s /bin/sh -c 'cd /mnt/vm; ./capstone-vexec ./bin/postgres {PG_ARGS} "
               f"< {staged}'; rc=$?; rm -rf /tmp/pgd; exit $rc")


def facts(control, text, result, symbols, probe_at, funcs):
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("LIFETIME")]
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    fault_line = next((l.strip() for l in text.splitlines() if "domain fault cause=" in l), None)
    accesses = []
    for l in lines:
        m = re.match(rf"LIFETIME {re.escape(control)} mark access=(\S+)", l)
        if m:
            probe = PROBES[(control, m.group(1))]
            accesses.append({"access": m.group(1), "probe": probe, "file_line": probe_at[probe],
                             "probe_link_address": funcs.get(probe), "mark": l})
    kv = {}
    for l in lines:
        for k, val in re.findall(r"([A-Za-z_-]+)=(\S+)", l):
            kv.setdefault(k, val)
    returned = next((l for l in lines if l.startswith(f"LIFETIME-RETURNED {control} ")), None)
    failed = next((l for l in lines if l.startswith("LIFETIME-CONTROL-FAILED")), None)
    rec = {
        "exit": result,
        "returned": returned,
        "setup_failed": failed,
        "row_printed": "lifetime_control" in text and returned is not None and
                       re.search(rf'lifetime_control = "{re.escape(control)} ', text) is not None,
        "lines": lines,
        "accesses": accesses,
        "fault": None,
    }
    if fault:
        rec["fault"] = {"cause": fault.cause, "pc": hex(fault.pc), "function": fault.symbol,
                        "offset": fault.offset, "line": fault_line}
    rec["outcome"] = ("fault" if fault else
                      "completed" if returned and result.get("kind") == "exit" and result.get("value") == 0
                      else "none")
    return rec


def check(arm, control, rec):
    """The pre-registered prediction for this control on this arm, against the record."""
    ret = rec["returned"] or ""
    flags = dict(re.findall(r"(\w+)=(\d)", ret))
    bad = []
    if rec["setup_failed"]:
        return "NO-READING", f"setup failed: {rec['setup_failed']}"
    if control == "successor-stale" and arm == "protected":
        f = rec["fault"]
        if not f:
            return "DIFF", f"expected a fault at lc_probe_stale_read, outcome {rec['outcome']}"
        if f["cause"] != 25:
            bad.append(f"cause {f['cause']}, expected 25")
        if f["function"] != "lc_probe_stale_read":
            bad.append(f"fault in {f['function']}, expected lc_probe_stale_read")
        if not any(a["access"] == "stale-read" for a in rec["accesses"]):
            bad.append("no stale-read mark before the fault")
        if not any(re.search(r"successor-stale context=\S+ .* equal=1", l) for l in rec["lines"]):
            bad.append("C's address equal to A's not shown in this run")
        return ("ok" if not bad else "DIFF"), "; ".join(bad) or "fault cause 25 at the probe"
    if rec["outcome"] != "completed":
        return "DIFF", f"expected completion, outcome {rec['outcome']}" + (
            f" (cause {rec['fault']['cause']} in {rec['fault']['function']})" if rec["fault"] else "")
    want = {"parent": ("link_ok", "header_ok", "readback_ok", "allocator_intact"),
            "sibling": ("intact", "readback_ok"),
            "successor": ("same_address", "readback_ok"),
            "successor-stale": ("same_address", "readback_ok", "stale_read_completed", "stale_equals_C")}[control]
    bad = [k for k in want if flags.get(k) != "1"]
    return ("ok" if not bad else "DIFF"), ("checks failed: " + ", ".join(bad)) if bad else "completed, " + ", ".join(want)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=sorted(NESTED))
    p.add_argument("--virtual-kit", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--pg-share", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True, help="stage and console; outside the repository")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=1)
    a = p.parse_args()
    sdk_builtins.gate([a.image], who="postgres:run-controls.py")

    manifest = json.loads((a.image.parent / "manifest.json").read_text())
    if manifest.get("nested") != NESTED[a.arm] or manifest.get("profile") != "virtual":
        print(f"CONTROL-FAILED {a.image} is nested={manifest.get('nested')} profile={manifest.get('profile')}; "
              f"arm {a.arm} needs nested={NESTED[a.arm]} on the virtual profile", file=sys.stderr)
        return 75
    extensions = a.image.parents[1] / "source/share/extension"
    if not (extensions / "pgcorpus_lifetime.control").is_file():
        print(f"CONTROL-FAILED no pgcorpus_lifetime.control in {extensions}", file=sys.stderr)
        return 75
    probe_at = probe_lines()
    symbols = v.Symbols(a.llvm_bin, a.image)
    funcs = {name: hex(addr) for addr, _, name in symbols.funcs if name.startswith("lc_probe_")}
    if set(PROBES.values()) - set(funcs):
        print(f"CONTROL-FAILED the image lacks probes {sorted(set(PROBES.values()) - set(funcs))}", file=sys.stderr)
        return 75

    a.raw.mkdir(parents=True, exist_ok=True)
    batch = virtualvm.Batch(a.raw / "stage")
    batch.put("bin/postgres", a.image)
    batch.put("share", a.pg_share)
    for source in sorted(extensions.iterdir()):
        batch.put(f"share/extension/{source.name}", source)
    batch.put("pgcluster", a.fixture)
    session(batch, "preflight", "SELECT 'PREFLIGHT-BACKEND-OK' AS marker;\n")
    session(batch, "arm-uaf-palloc", "CREATE EXTENSION pgcorpus_reach;\nSELECT corpus_control('uaf-palloc');\n")
    order = []
    for rep in range(1, a.repeat + 1):
        for control in CONTROLS:
            name = f"{control}-{rep}"
            order.append((name, control, rep))
            session(batch, name, f"CREATE EXTENSION pgcorpus_lifetime;\nSELECT lifetime_control('{control}');\n")
    start = time.time()
    runs, serial, completed = batch.execute(a.virtual_kit, a.raw / "guest", timeout=600 + 300 * len(batch.steps))
    wall = round(time.time() - start, 1)

    record = {"arm": a.arm, "image": {"path": str(a.image), "sha256": v.sha256(a.image),
                                      "manifest": manifest},
              "platform": virtualvm.platform(a.virtual_kit, a.llvm_bin / "clang", HERE),
              "module_sha256": v.sha256(MODULE), "probe_link_addresses": funcs, "probe_lines": probe_at,
              "boot": {"completed": completed, "sessions_planned": len(batch.steps),
                       "sessions_ran": len(runs), "wall_seconds": wall},
              "identity": {}, "sessions": []}
    text, result = runs.get("preflight", ("", {"kind": "none"}))
    record["identity"]["preflight"] = "PREFLIGHT-BACKEND-OK" in text and result.get("kind") == "exit"
    text, result = runs.get("arm-uaf-palloc", ("", {"kind": "none"}))
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    seen = ("fault" if fault and fault.symbol == "corpus_control_read" else
            "complete" if "CONTROL RETURNED" in text else "none")
    record["identity"]["uaf_palloc"] = {"observed": seen, "expected": "fault" if a.arm == "protected" else "complete",
                                        "fault": ({"cause": fault.cause, "pc": hex(fault.pc),
                                                   "function": fault.symbol} if fault else None)}
    print(f"arm={a.arm} image={record['image']['sha256'][:16]} boot={'completed' if completed else 'INCOMPLETE'} "
          f"{len(runs)}/{len(batch.steps)} sessions, {wall}s; preflight={record['identity']['preflight']} "
          f"uaf-palloc={seen} (expected {record['identity']['uaf_palloc']['expected']})", flush=True)
    for name, control, rep in order:
        text, result = runs.get(name, ("", {"kind": "none"}))
        rec = facts(control, text, result, symbols, probe_at, funcs)
        rec.update(session=name, control=control, repetition=rep)
        rec["verdict"], rec["verdict_reason"] = check(a.arm, control, rec)
        record["sessions"].append(rec)
        f = rec["fault"]
        print(f"  {name:<20} {rec['outcome']:<9} {rec['verdict']:<5} {rec['verdict_reason']}"
              + (f"  [cause {f['cause']} pc {f['pc']} in {f['function']}+{f['offset']}]" if f else ""), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=1) + "\n")
    ok = (completed and record["identity"]["preflight"]
          and record["identity"]["uaf_palloc"]["observed"] == record["identity"]["uaf_palloc"]["expected"]
          and all(s["verdict"] == "ok" for s in record["sessions"]))
    print(f"--- {a.arm}: {'every prediction held' if ok else 'NOT every prediction held'}; {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
