#!/usr/bin/env python3
"""compare.py CONTROL_OUT ARM_OUT: the landing verdict of two run-arm.sh outputs, failing closed.

Per gate, the FAIL set of each side: the FAIL lines (tmp paths normalized), the JSON record's
status and every test whose value is not true. Two arms land only when every gate's FAIL sets are
identical -- but a FAIL set can only be compared when both sides actually ran what the gate is
about. Before this check existed, the runtime gate's sqlite/mruby/perl rows and the whole memcached
gate exited 126 ("capstone-vexec: image: Exec format error": prebuilt images in an older image ABI)
in the control and in the arm alike, and the two identical FAIL sets read as a match.

So every gate is first split into ROWS, the launches it makes, and each row is RAN or NOT RUN.

  Exit-marker gates (all but memcached). A row is one <NAME>_EXIT[:field...]:<code> line the gate
  script echoes after a launch (VM_EXIT:sqlite_write:0, FP_EXIT:0, PIN_EXIT:16:seq:0,
  HOOK_EXIT:base1:walk:0 ...), read from runtime.log for the runtime gate and from the guest
  serial logs for the others. NOT RUN: exit 125, 126 or 127 (UNRUN_EXIT); a launcher refusal
  ("capstone-vexec: image:"), "Exec format error", VM_CONTRACT_FAIL:argv or a shell "not found" between the row's
  marker and either neighbouring marker (both neighbours, because some modes print their output
  before the marker and some after); or the row is missing -- REQUIRED_ROWS lists every row the
  current runners make, and a row seen on one side is required on the other. No exit marker at all,
  or no result (no PASS/FAIL line, no JSON status), makes the gate NOT RUN.
  memcached. A row is a case: pthread, normal1, normal2, usr1, marker and every fixture the
  runner itself checks (its <case>_harness lines). NOT RUN: no MC_BEGIN or no MC_END, MC_END
  125-127, a refusal in the case's console text or server.err, a missing or unreadable job.json,
  a server that exited 125-127, or no check line for the case. Every attempt of a case is judged.

A gate with a NOT RUN row on either side is INCOMPLETE, never IDENTICAL; one INCOMPLETE gate makes
the whole verdict INCOMPLETE. Exit status: 0 IDENTICAL, 1 DIFFERENT, 2 INCOMPLETE, 3 usage.
The logs are read, never the binaries: libc's strerror table contains "Exec format error".
A row's PASS/FAIL is reported where the runner gives it per row (runtime.json, memcached's checks);
elsewhere a row that ran is RAN and its outcome is in the gate's FAIL set.

Layout read: OUT/gates/<gate>.log, OUT/gates/runtime.json, OUT/gates/<gate>/**/serial.log,
OUT/gates/memcached/<case>/{job.json,server.err}, OUT/gates/SUMMARY.txt. A control directory whose
step-hooks gate was rerun in the documented form as step-hooks2.log is read from that file.
--rows OUT prints the rows one output makes (the input for REQUIRED_ROWS when a runner changes).
--json FILE writes the rows and verdicts as a record.
"""
import argparse
import json
import re
import sys
from pathlib import Path

GATES = ["hosted", "runtime", "nodes", "malloc", "pin-ahead", "fp", "pthread", "step-hooks", "memcached"]
# The rows the runners at dev 976df58996f3 make (compare.py --rows on that commit's gate run; the
# same rows on 724bf0bea11e). A runner that adds a row needs no change here (a row on one side is
# required on the other); one that drops a row makes this table fail closed until it is updated.
REQUIRED_ROWS = {
    # run.py's expected-exit list (run.py: tests[name] = lines.count(f'VM_EXIT:{name}:{code}') == 1),
    # and --perl, which run-arm.sh always passes.
    "runtime": ["VM_EXIT:" + n for n in (
        "normal", "threads", "vm", "vm-large", "vm-mappings", "nested", "nested-stale", "heap-threads",
        "vm-ro-write", "vm-none-read", "vm-guard", "vm-padding", "sparse-churn", "sparse-stale", "stale",
        "bounds", "retired", "preempt", "parallel_one", "parallel_two", "sqlite_write", "sqlite_read",
        "mruby", "cleanup", "perl")],
    "hosted": ["HOSTED_EXIT:" + n for n in ('native', 'virtual', 'noreturn')],
    "nodes": ["NODE_EXIT:" + n for n in ('growth', 'stale', 'budget', 'quota')],
    "malloc": [
        'MALLOC_NATIVE_EXIT', 'MALLOC_BASIC_EXIT', 'MALLOC_POPULATION_EXIT', 'MALLOC_CHURN_EXIT',
        'MALLOC_ALIGN_PROBE_EXIT', 'MALLOC_PREDICATE_EXIT', 'MALLOC_REMAP_ALIGN_EXIT',
        'MALLOC_HOT_EXIT', 'MALLOC_HOT_CONTROL_EXIT', 'MALLOC_DENIAL_EXIT:stale',
        'MALLOC_DENIAL_EXIT:inplace-stale', 'MALLOC_DENIAL_EXIT:remap-stale',
        'MALLOC_DENIAL_EXIT:bounds', 'MALLOC_DENIAL_EXIT:stale-free',
        'MALLOC_DENIAL_EXIT:reuse-stale', 'MALLOC_DENIAL_EXIT:reuse-free',
        'MALLOC_DENIAL_EXIT:interior-free', 'MALLOC_DENIAL_EXIT:nested-ok',
        'MALLOC_DENIAL_EXIT:nested-stale', 'MALLOC_DENIAL_EXIT:nested-grandchild',
        'MALLOC_DENIAL_EXIT:nested-double', 'MALLOC_DENIAL_EXIT:recycle-tags',
        'MALLOC_DENIAL_EXIT:recycle-stale', 'MALLOC_DENIAL_EXIT:recycle-free',
        'MALLOC_DENIAL_EXIT:recycle-header', 'MALLOC_DENIAL_EXIT:recycle-slack',
        'MALLOC_DENIAL_EXIT:recycle-limits', 'MALLOC_DENIAL_EXIT:recycle-root',
        'MALLOC_DENIAL_EXIT:recycle-geometry', 'MALLOC_DENIAL_EXIT:recycle-nested',
        'MALLOC_DENIAL_EXIT:recycle-aligned', 'MALLOC_DENIAL_EXIT:recycle-realloc',
        'MALLOC_DENIAL_EXIT:calloc-clean', 'MALLOC_DENIAL_EXIT:calloc-shrink',
        'MALLOC_DENIAL_EXIT:calloc-geometry', 'MALLOC_DENIAL_EXIT:meta-reuse'],
    "pin-ahead": ["PIN_EXIT:" + n for n in (
        '1:base', '1:seq', '1:stop', '1:vma', '1:vmax', '1:rowrite', '1:retire', '16:base',
        '16:seq', '16:stop', '16:vma', '16:vmax', '16:rowrite', '16:retire', '64:base', '64:seq',
        '64:stop', '64:vma', '64:vmax', '64:rowrite', '64:retire')],
    "fp": ['FP_EXIT'],
    "pthread": ['PTHREAD_EXIT', 'PTHREAD_FALLBACK_EXIT',
                'PTHREAD_IDENTITY_EXIT:identity-exit', 'PTHREAD_IDENTITY_EXIT:identity-jump',
                'PTHREAD_IDENTITY_EXIT:private-tls'],
    "step-hooks": ["HOOK_EXIT:" + n for n in (
        'base1:walk', 'base1:syscalls', 'base16:walk', 'base64:walk', 'base64:seq', 'resched2:walk',
        'resched2:syscalls', 'signal2:walk', 'signal2:syscalls', 'deadline2:walk',
        'deadline2:syscalls', 'resched1:walk', 'signal1:walk', 'deadline1:walk', 'resched3:walk',
        'resched3:syscalls', 'signal3:walk', 'signal3:syscalls', 'kill1:walk', 'kill2:walk',
        'resched4:walk', 'signal4:walk', 'deadline4:walk', 'kill4:walk', 'short64:walk',
        'short64:seq', 'life16:seq', 'life16:stop', 'life16:vma', 'life16:vmax', 'life16:retire',
        'life64:seq', 'life64:stop', 'life64:vma', 'life64:vmax', 'life64:retire', 'ro64:rowrite',
        'adapt:walk', 'adapt:stream', 'adaptdirect:walk', 'adaptdirect:stream',
        'adaptres:walk', 'adaptshort:walk', 'adaptclip:clip',
        'adaptclip:clipx', 'adaptro:clipwrite', 'adaptnonlin:nonlin',
        'adaptgaps:gaps', 'adaptinterleave:interleave')],
}
# The runtime gate's application rows and the runtime.json tests that judge each.
RUNTIME_TESTS = {"VM_EXIT:sqlite_write": ("sqlite_write", "sqlite_results"),
                 "VM_EXIT:sqlite_read": ("sqlite_read", "sqlite_results"),
                 "VM_EXIT:mruby": ("mruby", "mruby_results"),
                 "VM_EXIT:perl": ("perl", "perl_output")}
MEMCACHED_REQUIRED = ["pthread", "normal1", "normal2", "usr1", "marker"]
NOT_EXECUTED = re.compile(r"capstone-vexec: image:|Exec format error|VM_CONTRACT_FAIL:argv|: not found\s*$")
EXIT_LINE = re.compile(r"^([A-Z][A-Z0-9_]*_EXIT(?::[^:\s]+)*):(\d+)\s*$")
# 126/127: the launcher could not open or refused the image, or the shell found no program; 125: the
# launcher failed in its own setup (exec.c die(): the device, the spawner, the image grant ...).
UNRUN_EXIT = {125, 126, 127}
# run.py's --legacy row expects the refusal (legacy_abi_rejected); it is a check, not an omission.
EXPECTED_REFUSALS = {"VM_EXIT:legacy"}


def norm(s):
    s = re.sub(r"/home/[^ ]*/out-[^/ ]+/", "<out>/", s)
    s = re.sub(r"/[^ ]*/gates/tmp/[^ ]*", "<tmp>", s)
    s = re.sub(r"virtual-app\.[A-Za-z0-9_]+", "virtual-app.X", s)
    return s.strip()


def read(path):
    return path.read_text(errors="replace").replace("\r", "") if path.is_file() else ""


def records(text):
    dec = json.JSONDecoder(); out = []; i = 0
    while True:
        j = text.find("{", i)
        if j < 0:
            return out
        try:
            obj, end = dec.raw_decode(text, j)
            if isinstance(obj, dict) and "status" in obj:
                out.append(obj)
            i = end
        except ValueError:
            i = j + 1


def failset(text, extra=None):
    lines = text.splitlines()
    fails = sorted({norm(l) for l in lines if l.startswith("FAIL")})
    recs = records(text) + ([extra] if extra else [])
    jf = set()
    for r in recs:
        jf.add(f"status={r.get('status')}")
        for k, v in (r.get("tests") or {}).items():
            if v is not True:
                jf.add(f"test {k}={v}")
    checks = {}
    for l in lines:
        m = re.match(r"^(PASS|FAIL) (\S+)\s*$", l)
        if m:
            checks[m[2]] = checks.get(m[2], True) and m[1] == "PASS"
    return fails, sorted(jf), recs, checks


def exit_rows(texts, tests=None):
    """{row: (state, evidence)} from the exit markers of the texts (each one guest console)."""
    rows = {}
    for text in texts:
        lines = text.splitlines()
        marks = [(i, m[1], int(m[2])) for i, l in enumerate(lines) for m in [EXIT_LINE.match(l.strip())] if m]
        bad = [i for i, l in enumerate(lines) if NOT_EXECUTED.search(l)]
        for n, (i, name, code) in enumerate(marks):
            lo = marks[n - 1][0] if n else -1
            hi = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
            near = [lines[j].strip() for j in bad if lo < j < hi]
            if name in EXPECTED_REFUSALS:
                state, why = "RAN", f"exit {code} (refusal expected)"
            elif code in UNRUN_EXIT or near:
                state, why = "NOT RUN", f"exit {code}" + (f"; {near[0][:60]}" if near else "")
            elif tests is not None and name in RUNTIME_TESTS:
                known = [tests[k] for k in RUNTIME_TESTS[name] if k in tests]
                state, why = ("PASS" if known and all(v is True for v in known) else "FAIL"), f"exit {code}"
            elif tests is not None and name.split(":", 1)[1] in tests:
                state, why = ("PASS" if tests[name.split(":", 1)[1]] is True else "FAIL"), f"exit {code}"
            else:
                state, why = "RAN", f"exit {code}"
            prev = rows.get(name)
            if prev and prev[0] == "NOT RUN":
                continue          # one refused launch of a row is enough: it stays NOT RUN
            if prev and state != "NOT RUN":
                state, why = ("FAIL" if "FAIL" in (prev[0], state) else state), why + f"; {prev[1]} (repeated)"
            rows[name] = (state, why)
    return rows


def memcached_rows(gdir, checks):
    serial = read(gdir / "guest/serial.log")
    begun = re.findall(r"^MC_BEGIN:(\S+)$", serial, re.M)
    # Every attempt of a case, MC_BEGIN to its MC_END: a case run twice is judged on both.
    attempts = {}
    for m in re.finditer(r"^MC_BEGIN:(\S+)\n(.*?)^MC_END:\1:(\d+)$", serial, re.M | re.S):
        attempts.setdefault(m[1], []).append((int(m[3]), m[2]))
    checked = [k[:-len("_harness")] for k in checks if k.endswith("_harness")]
    if any(k.startswith("pthread_") for k in checks):
        checked.append("pthread")
    rows = {}
    for label in dict.fromkeys(MEMCACHED_REQUIRED + begun + checked):
        mine = {k: v for k, v in checks.items() if k.startswith(label + "_")}
        if label not in begun:
            rows[label] = ("NOT RUN", "no MC_BEGIN"); continue
        if label not in attempts or len(attempts[label]) != begun.count(label):
            rows[label] = ("NOT RUN", "no MC_END"); continue
        why = []
        for code, text in attempts[label]:
            if code in UNRUN_EXIT:
                why.append(f"exit {code}")
            if any(NOT_EXECUTED.search(l) for l in text.splitlines()):
                why.append("image refused")
        code = attempts[label][-1][0]
        if label != "pthread":
            try:
                job = json.loads(read(gdir / label / "job.json"))
            except ValueError:
                job = None
            if not isinstance(job, dict) or "kind" not in job:
                why.append("no job record")
            elif job.get("kind") == "exit" and job.get("value") in UNRUN_EXIT:
                why.append(f"server exit {job.get('value')}")
            if any(NOT_EXECUTED.search(l) for l in read(gdir / label / "server.err").splitlines()):
                why.append("server image refused")
        if not mine:
            why.append("no check line")
        if why:
            rows[label] = ("NOT RUN", "; ".join(why)); continue
        rows[label] = ("PASS" if all(mine.values()) else "FAIL", f"exit {code}, {sum(mine.values())}/{len(mine)} checks")
    return rows


def side(out, gate, ctl):
    g = out / "gates" if (out / "gates").is_dir() else out
    log = g / f"{gate}.log"
    if ctl and gate == "step-hooks" and (g / "step-hooks2.log").exists():
        log = g / "step-hooks2.log"
    text = read(log)
    extra = None
    if gate == "runtime" and (g / "runtime.json").is_file():
        extra = json.loads((g / "runtime.json").read_text())
    fails, jf, recs, checks = failset(text, extra)
    summary = read(g / "SUMMARY.txt")
    rc = re.search(r"^" + re.escape(gate) + r"\s+(rc=\d+|NOT RUN[^\n]*)", summary, re.M)
    gate_why = []
    if not text:
        gate_why.append("no log" + (f" ({rc[1]})" if rc else ""))
    elif not (checks or recs):
        gate_why.append("no result (no PASS/FAIL line, no JSON status)")
    if gate == "memcached":
        rows = memcached_rows(g / "memcached", checks)
    else:
        if gate == "runtime":
            texts = [text]
        else:
            serials = sorted((g / gate).glob("**/serial.log")) if (g / gate).is_dir() else []
            texts = [read(p) for p in serials] or [text]
        rows = exit_rows(texts, (extra or {}).get("tests") if gate == "runtime" else None)
        if text and not rows:
            gate_why.append("no exit marker: the guest launched nothing")
    if gate_why:
        rows = dict(rows, **{f"[{gate}]": ("NOT RUN", "; ".join(gate_why))})
    return dict(log=log.name, rc=rc[1] if rc else "?", fails=fails, json=jf,
                npass=sum(1 for v in checks.values() if v), rows=rows)


def compare_gate(gate, c, r):
    # A row one side makes is required on the other, and so is every REQUIRED_ROWS row.
    for row in dict.fromkeys(REQUIRED_ROWS.get(gate, []) + list(c["rows"]) + list(r["rows"])):
        for x in (c, r):
            x["rows"].setdefault(row, ("NOT RUN", "row missing"))
    unrun = [row for row in c["rows"] if c["rows"][row][0] == "NOT RUN" or r["rows"][row][0] == "NOT RUN"]
    if unrun:
        return "INCOMPLETE", unrun
    if c["fails"] == r["fails"] and c["json"] == r["json"]:
        return "IDENTICAL", []
    return "DIFFERENT", []


def print_rows(out):
    for gate in GATES:
        s = side(out, gate, False)
        print(f"{gate}: " + " ".join(f"{k}={v[0]}" for k, v in s["rows"].items()))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--rows":
        print_rows(Path(sys.argv[2])); return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("control", type=Path)
    ap.add_argument("arm", type=Path)
    ap.add_argument("--gates", default=" ".join(GATES), help="gates to compare (default: all nine)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--all-rows", action="store_true", help="print every row, not only the NOT RUN and application rows")
    a = ap.parse_args()
    for d in (a.control, a.arm):
        if not d.is_dir():
            print(f"compare.py: no directory {d}", file=sys.stderr); return 3
    verdicts = {}; record = {}
    if not a.gates.split():
        print("compare.py: no gates selected", file=sys.stderr); return 3
    for gate in a.gates.split():
        c, r = side(a.control, gate, True), side(a.arm, gate, False)
        v, unrun = compare_gate(gate, c, r)
        verdicts[gate] = v
        ran = sum(1 for k in c["rows"] if c["rows"][k][0] != "NOT RUN" and r["rows"][k][0] != "NOT RUN")
        print(f"{gate:10s} {v:10s} rows ran {ran}/{len(c['rows'])} | control[{c['log']} {c['rc']}] "
              f"FAIL={len(c['fails'])} PASS={c['npass']} | arm[{r['log']} {r['rc']}] FAIL={len(r['fails'])} PASS={r['npass']}")
        for row in c["rows"]:
            (cs, ce), (as_, ae) = c["rows"][row], r["rows"][row]
            show = a.all_rows or row in unrun or row in RUNTIME_TESTS or gate == "memcached" or cs != as_
            if show:
                print(f"    {row:24s} control {cs:7s} {ce[:60]:60s} | arm {as_:7s} {ae[:60]}"
                      f" [{'same' if cs == as_ else 'differs'}]")
        if v != "INCOMPLETE":
            for x in sorted(set(c["fails"]) ^ set(r["fails"])):
                print(f"    {'control-only' if x in c['fails'] else 'arm-only'}: {x}")
            for x in sorted(set(c["json"]) ^ set(r["json"])):
                print(f"    {'control-only' if x in c['json'] else 'arm-only'}: json {x}")
        record[gate] = dict(verdict=v, not_run=unrun, control=c, arm=r)
    if "INCOMPLETE" in verdicts.values():
        overall, status = "INCOMPLETE", 2
    elif "DIFFERENT" in verdicts.values():
        overall, status = "DIFFERENT", 1
    else:
        overall, status = "IDENTICAL", 0
    print(f"VERDICT: {overall}" + ("" if status != 2 else
          " (gates with NOT RUN rows: " + ", ".join(g for g, v in verdicts.items() if v == "INCOMPLETE") + ")"))
    if a.json:
        a.json.write_text(json.dumps(dict(verdict=overall, gates=record), indent=1) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
