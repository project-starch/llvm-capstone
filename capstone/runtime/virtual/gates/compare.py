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
the whole verdict INCOMPLETE. Exit status: 0 IDENTICAL, 1 DIFFERENT, 2 INCOMPLETE, 3 usage, 4 REFUSED (a run's memory mode).

MEMORY MODE. run-arm.sh --mode on|off records the requested mode in OUT/mode, and every gate runner
logs the mode the loaded module reports after each insmod (<GATE>_MEMORY_MODE:Y|N|A, see
memory_mode.py). compare.py refuses (exit 4, nothing compared) an output that has no mode file, a gate
that logged no mode line, fewer or more lines than it makes module loads (MODE_LOADS), or a mode line of
the other mode than OUT/mode asked for: ON = Y only, OFF = N or A (a module without the unpinned
parameter). The refusal does not depend on parity: a gate whose runner reports memory_mode_verified
failed or reports nothing is refused even when both sides fail it the same way. Also refused, for the
same reason: a SUMMARY.txt carrying MODE_CHECK_FAILED or OPTIONAL_GATE_FAILED (run-arm.sh's own
rejection of the arm), an optional qualification gate (unpinned-mm, unpinned-mutants) with a non-zero
rc, and one whose own mode witnesses are missing or not Y. The nine ordinary gates may end non-zero:
their FAIL sets are what is compared. --control-mode / --arm-mode additionally pin
the pair being compared (a control that is not the OFF arm it should be is as wrong as a log that
disagrees). `compare.py --check-mode OUT` runs that check on one output (run-arm.sh does, last).
FAULT PATH. run-arm.sh's FAST_FAULT=on|off (default off) records the unpinned fault path in OUT/fast-fault
and exports it to the runners, which pass unpinned_fast_fault=1 for `on` and log <PREFIX>FAST_FAULT:Y|N|A after
every insmod. An output with the file must log one such line per module load, Y for `on` and N or A for `off`
(otherwise REFUSED); `on` needs mode on. --control-fast / --arm-fast pin the pair (a control without the file
counts as off).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import memory_mode  # noqa: E402

GATES = ["hosted", "runtime", "nodes", "malloc", "pin-ahead", "fp", "pthread", "step-hooks", "memcached"]
# Optional qualification gates (run-arm.sh GATES=..., ON only). Unlike the nine they are not compared
# row by row: each must succeed when it ran, and its own mode witnesses are checked.
OPTIONAL_GATES = ["unpinned-mm", "unpinned-mutants"]
# Module loads (insmod) per gate in a full run, hence the memory-mode lines it must log, one each:
# the runners' memory_mode.verified(log, unpinned, N) counts (nodes: 2; pin-ahead: its three windows;
# step-hooks: one per configuration, 30 with the malloc configurations skipped). A runner that changes
# its number of loads fails closed here until this table is updated. unpinned-mm loads seven.
MODE_LOADS = {"hosted": 1, "runtime": 1, "nodes": 2, "malloc": 1, "pin-ahead": 3, "fp": 1,
              "pthread": 1, "step-hooks": 30, "memcached": 1}
MM_LOADS = 7
# An arm-level rejection run-arm.sh wrote into SUMMARY.txt: no comparison may use that output.
ARM_REJECTION = re.compile(r"^(MODE_CHECK_FAILED|OPTIONAL_GATE_FAILED)\b.*$", re.M)
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
    # run-malloc.py at dev ae47030ceeb6 (compare.py --rows on a dev OFF arm): one MALLOC_EXIT:<case> row per case.
    "malloc": ["MALLOC_EXIT:" + n for n in (
        "native", "ok", "population", "hot", "hot-service-control", "churn", "stale", "stale-free",
        "inplace-stale", "remap-stale", "bounds", "interior-free", "reuse-stale", "reuse-free",
        "nested-ok", "nested-stale", "nested-grandchild", "nested-double")],
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
# Memcached fixture 24 cannot pass as written, in either memory mode or in the dev head, and WHICH of its
# two checks names the failure depends on where the allocator places things. Two outcomes are recorded
# (laneONQ fix rounds, unpinned/README.md; the gatefix lane saw both in both arms of an OFF-versus-OFF
# comparison; the memcached gate run alone on one build and one platform gave the second in some runs of
# the OFF arm and the first in the others):
#   not created   the header-only condition is not created: the fixture prints "header-only=0" and
#                 "triggering condition not created" (or, before it gets that far, "ia not linked": the
#                 expired item was never linked, seen in the first merged-head OFF arm), returns its known
#                 no-op mark 18e0018 and exits 24 with no fault. fx24_trigger FAILS (the no-op is
#                 self-reported), fx24_prediction holds.
#   created+crash the condition is created ("header-only=1", then "touch") and the server dies with a
#                 signal at a domain fault, cause 24 at address 0. fx24_prediction FAILS, fx24_trigger
#                 holds (a non-RETURN outcome is not a self-reported no-op).
# A FAIL set holding exactly one of the pair is folded into one placement-dependent entry ONLY when the
# case's own job/output/fault files show the signature that belongs to the failed check. A single
# failure with any other evidence (an unrelated crash, a different no-op, no fixture output) is not
# folded and makes the fx24 row NOT RUN: the gate is INCOMPLETE, never IDENTICAL. A side where fx24
# passes both checks or fails both is not folded either (it stays DIFFERENT from a folded side). The
# fold still counts as a FAIL; it only removes the placement difference between two sides that both
# fail one of the recorded ways. Remove this when the fixture is fixed.
PLACEMENT_DEPENDENT = {"memcached": ("fx24_prediction", "fx24_trigger")}
PLACEMENT_FOLDED = "FAIL fx24 (one of prediction/trigger: placement dependent)"
# the failed check each recorded signature belongs to
PLACEMENT_SIGNATURES = {"not_created": "FAIL fx24_trigger", "created_crash": "FAIL fx24_prediction"}


def fx24_signature(case):
    """'not_created', 'created_crash' or None from the fx24 case directory's job, output and fault."""
    try:
        job = json.loads(read(case / "job.json"))
    except ValueError:
        return None
    if not isinstance(job, dict):
        return None
    out, fault = read(case / "server.out"), read(case / "fault.txt").strip()
    kind = job.get("kind")
    noop = (re.search(r"^MCAPP-FIX 24 header-only=0$", out, re.M)
            and re.search(r"^MCAPP-FIX 24 triggering condition not created", out, re.M)) or (
            re.search(r"^MCAPP-FIX 24 ia not linked$", out, re.M)
            and not re.search(r"^MCAPP-FIX 24 header-only=", out, re.M))
    if kind == "exit" and not fault and noop and re.search(r"^MCAPP-FIX 24 mark=18e0018$", out, re.M):
        return "not_created"
    if (kind == "signal" and re.search(r"^MCAPP-FIX 24 header-only=1$", out, re.M)
            and re.search(r"^MCAPP-FIX 24 touch$", out, re.M)
            and re.search(r"domain fault cause=24 .*address=0x0(\s|$)", fault)):
        return "created_crash"
    return None


def fold_placement(gate, fails, signature=None):
    """(fails, unclassified): the FAIL set with a recorded fx24 placement outcome folded into one entry.
    `unclassified` names a single fx24 failure whose evidence is not one of the recorded signatures."""
    pair = {"FAIL " + n for n in PLACEMENT_DEPENDENT.get(gate, ())}
    hit = [f for f in fails if f in pair]
    if len(hit) != 1:
        return fails, None
    if PLACEMENT_SIGNATURES.get(signature) != hit[0]:
        return fails, f"{hit[0][5:]} failed alone, evidence is not a recorded fx24 signature ({signature or 'none'})"
    return sorted([f for f in fails if f not in pair] + [PLACEMENT_FOLDED]), None


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
    fails, unclassified = fold_placement(
        gate, fails, fx24_signature(g / "memcached" / "fx24") if gate == "memcached" else None)
    summary = read(g / "SUMMARY.txt")
    rc = re.search(r"^" + re.escape(gate) + r"\s+(rc=\d+|NOT RUN[^\n]*)", summary, re.M)
    gate_why = []
    if not text:
        gate_why.append("no log" + (f" ({rc[1]})" if rc else ""))
    elif not (checks or recs):
        gate_why.append("no result (no PASS/FAIL line, no JSON status)")
    if gate == "memcached":
        rows = memcached_rows(g / "memcached", checks)
        if unclassified:
            rows["fx24"] = ("NOT RUN", unclassified)
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


def gate_dir(out):
    return out / "gates" if (out / "gates").is_dir() else out


def gate_texts(out, gate):
    g = gate_dir(out)
    if gate == "memcached":
        return [read(g / "memcached/guest/serial.log")]
    if gate == "runtime":
        return [read(g / "runtime.log")]
    serials = sorted((g / gate).glob("**/serial.log")) if (g / gate).is_dir() else []
    return [read(p) for p in serials] or [read(g / f"{gate}.log")]


def mode_lines(out, gate):
    """The memory-mode values the gate's guest console logged (every insmod prints one)."""
    return [m for text in gate_texts(out, gate) for m in memory_mode.modes(text)]


def fast_lines(out, gate):
    """The fault-path values (<PREFIX>FAST_FAULT:Y|N|A) the gate's guest console logged, one per insmod."""
    return [m for text in gate_texts(out, gate) for m in memory_mode.fast_modes(text)]


def verified_results(out, gate, ctl=False):
    """Every memory_mode_verified result the gate's runner reported: its PASS/FAIL line, and the value in
    each JSON record (the log's, runtime.json, the gate's own result.json), as booleans."""
    g = gate_dir(out)
    log = g / f"{gate}.log"
    if ctl and gate == "step-hooks" and (g / "step-hooks2.log").exists():
        log = g / "step-hooks2.log"
    text = read(log)
    # The malloc runner (run-malloc.py since the protected-mallocng rewrite) names its check `memory_mode`; every
    # other runner says memory_mode_verified. The alias is accepted for the malloc gate only.
    names = ("memory_mode_verified", "memory_mode") if gate == "malloc" else ("memory_mode_verified",)
    found = [m[1] == "PASS" for m in re.finditer(r"^(PASS|FAIL) (?:%s)\s*$" % "|".join(names), text, re.M)]
    recs = records(text)
    for path in (g / "runtime.json" if gate == "runtime" else None, g / gate / "result.json"):
        if path is not None and path.is_file():
            try:
                recs.append(json.loads(path.read_text()))
            except ValueError:
                found.append(False)      # a result file that cannot be read is not a pass
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        tests = rec.get("tests")
        for name in names:
            if isinstance(tests, dict) and name in tests:
                found.append(tests[name] is True)
        checks = rec.get("checks")
        for name in names:
            if isinstance(checks, dict) and name in checks:
                found.append(checks[name] is True)
            elif isinstance(checks, list):
                found += [c.get("passed") is True for c in checks
                          if isinstance(c, dict) and c.get("name") == name]
    return found


def requested_fast(out):
    """The fault path run-arm.sh was asked for (OUT/fast-fault: on|off), or None for an output without the file
    (a control from a tree that predates the option)."""
    f = out / "fast-fault"
    word = read(f).strip() if f.is_file() else ""
    return word if word in ("on", "off") else None


def requested_mode(out):
    f = out / "mode"
    word = read(f).strip() if f.is_file() else ""
    return word if word in ("on", "off") else None


def optional_problems(out, name):
    """Problems with the mode witnesses of an optional qualification gate that ran (ON only)."""
    g = gate_dir(out)
    problems = []
    if name == "unpinned-mm":
        # run-mm.py logs the mode of every one of its seven module loads: MM_ARM:<arm>:<fixture>:<Y|N>
        arms = re.findall(r"^MM_ARM:\w+:\w+:(\w)\s*$", read(g / name / "guest/serial.log"), re.M)
        if len(arms) != MM_LOADS or any(m != "Y" for m in arms):
            problems.append(f"{name}: {len(arms)} MM_ARM line(s) {sorted(set(arms))}, want {MM_LOADS} of Y")
        try:
            record = json.loads(read(g / name / "result.json"))
        except ValueError:
            record = {}
        checks = record.get("checks") if isinstance(record, dict) else None
        if not isinstance(checks, dict) or checks.get("mode") is not True or checks.get("completed") is not True:
            problems.append(f"{name}: result.json lacks passing completed/mode checks")
    elif name == "unpinned-mutants":
        # every step-hooks run of the negative controls: its own result.json, and the console's mode lines
        runs = sorted(p.parent for p in (g / name).glob("*/result.json")) if (g / name).is_dir() else []
        if not runs:
            problems.append(f"{name}: no run record under {name}/")
        for d in runs:
            lines = [m for p in sorted(d.glob("**/serial.log")) for m in memory_mode.modes(read(p))]
            try:
                checks = json.loads(read(d / "result.json")).get("checks", {})
            except (ValueError, AttributeError):
                checks = {}
            if not lines or any(m != "Y" for m in lines):
                problems.append(f"{name}/{d.name}: mode line(s) {sorted(set(lines))}, want only Y")
            if checks.get("memory_mode_verified") is not True or checks.get("completed") is not True:
                problems.append(f"{name}/{d.name}: memory_mode_verified/completed not true")
    return problems


def check_mode(out, gates=None, ctl=False):
    """Problems (a list of strings) with OUT as evidence of the mode it was asked to run in:
    the mode file; the arm's own rejection lines; a failed optional qualification gate; and per gate
    the number of mode lines (one per module load), their value and the runner's memory_mode_verified."""
    want = requested_mode(out)
    if want is None:
        return [f"{out}: no mode file (written by run-arm.sh --mode on|off)"]
    summary = read(gate_dir(out) / "SUMMARY.txt")
    finished = re.findall(r"^(\S+)\s+rc=(\d+)", summary, re.M)
    names = [n for n, _ in finished]
    ran = [x for x in GATES if x in names]
    problems = [f"arm rejected itself: {m[0].strip()}" for m in ARM_REJECTION.finditer(summary)]
    for n, rc in finished:
        # The nine ordinary gates fail the same way in the control (their FAIL sets are compared); a
        # selected qualification gate is a prerequisite, not a baseline.
        if n in OPTIONAL_GATES and rc != "0":
            problems.append(f"{n}: optional qualification gate failed (rc={rc})")
    ok = ("Y",) if want == "on" else ("N", "A")
    fast = requested_fast(out)
    if fast == "on" and want != "on":
        problems.append(f"fast fault path requested in mode {want} (needs on)")
    for gate in (gates or ran):
        found = mode_lines(out, gate)
        if not found:
            problems.append(f"{gate}: logged no memory-mode line (mode {want} requested)")
        elif any(m not in ok for m in found):
            problems.append(f"{gate}: logged mode {sorted(set(found))} but {want} was requested")
        elif gate in MODE_LOADS and len(found) != MODE_LOADS[gate]:
            problems.append(f"{gate}: {len(found)} memory-mode line(s) for {MODE_LOADS[gate]} module load(s)")
        if fast is not None and found:
            # one fault-path line per module load, all of the requested path: Y only for ON; N (or A: a
            # module without the parameter) for OFF -- an output that asked for a path must prove it ran.
            paths = fast_lines(out, gate)
            fok = ("Y",) if fast == "on" else ("N", "A")
            if len(paths) != len(found):
                problems.append(f"{gate}: {len(paths)} fault-path line(s) for {len(found)} module load(s) (fast {fast} requested)")
            elif any(m not in fok for m in paths):
                problems.append(f"{gate}: logged fault path {sorted(set(paths))} but fast {fast} was requested")
        verified = verified_results(out, gate, ctl)
        if not verified:
            problems.append(f"{gate}: no memory_mode_verified result")
        elif not all(verified):
            problems.append(f"{gate}: memory_mode_verified failed")
    for n in OPTIONAL_GATES:
        if n in names:
            if want != "on":
                problems.append(f"{n}: ran in mode {want} (needs on)")
            problems += optional_problems(out, n)
    if not (gates or finished):
        problems.append(f"{out}: SUMMARY.txt lists no finished gate")
    return problems


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
    if len(sys.argv) == 3 and sys.argv[1] == "--check-mode":
        out = Path(sys.argv[2])
        problems = check_mode(out)
        want = requested_mode(out)
        for gate in GATES:
            found = mode_lines(out, gate)
            if found:
                print(f"MODE {gate}: {len(found)} line(s) {sorted(set(found))} (requested {want}), "
                      f"fault path {sorted(set(fast_lines(out, gate)))} (requested {requested_fast(out) or 'unrecorded'}), "
                      f"memory_mode_verified {verified_results(out, gate)}")
        for x in problems:
            print("MODE_MISMATCH " + x)
        return 1 if problems else 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("control", type=Path)
    ap.add_argument("arm", type=Path)
    ap.add_argument("--gates", default=" ".join(GATES), help="gates to compare (default: all nine)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--all-rows", action="store_true", help="print every row, not only the NOT RUN and application rows")
    ap.add_argument("--control-mode", choices=("on", "off"), help="the memory mode the control must have been run in")
    ap.add_argument("--arm-mode", choices=("on", "off"), help="the memory mode the arm must have been run in")
    ap.add_argument("--control-fast", choices=("on", "off"), help="the fault path (OUT/fast-fault) the control must have been run with; off also accepts a control without the file")
    ap.add_argument("--arm-fast", choices=("on", "off"), help="the fault path the arm must have been run with; off also accepts an arm without the file")
    a = ap.parse_args()
    for d in (a.control, a.arm):
        if not d.is_dir():
            print(f"compare.py: no directory {d}", file=sys.stderr); return 3
    refused = []
    for label, d, pinned, path in (("control", a.control, a.control_mode, a.control_fast), ("arm", a.arm, a.arm_mode, a.arm_fast)):
        refused += [f"{label}: {x}" for x in check_mode(d, a.gates.split() or None, label == "control")]
        if pinned and requested_mode(d) != pinned:
            refused.append(f"{label}: run in mode {requested_mode(d)}, {pinned} required")
        if path and (requested_fast(d) or "off") != path:
            refused.append(f"{label}: run with fast fault path {requested_fast(d) or 'unrecorded'}, {path} required")
    if refused and a.gates.split():
        for x in refused:
            print("REFUSED " + x)
        print("VERDICT: REFUSED (memory mode)")
        return 4
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
