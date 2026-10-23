#!/usr/bin/env python3
"""Measure the ASAN-blind population against REAL PHP on virtual Capstone. Observes; judges.

WHY THIS IS A SECOND RUNNER AND NOT A FLAG ON run-cases.py. That script is the PHASE A runner
and it is tied to Phase A in two ways that do not generalise: its denominator comes from the
PHYSICAL suite's own `reachable` list (24 cases, which excludes the 7 this one exists to
measure), and its `reached` comes from a stage bit printed by app/php_virtual_main.c -- a SAPI
that is not linked here, because real PHP owns main(). Bending run-cases.py to cover both would
put two populations and two reach-derivations behind one set of flags, and its 23 offline tests
exist to keep the Phase A column stable. So: a separate runner, reporting into the SAME judge.

WHAT IS DERIVED RATHER THAN LISTED, and the reason is the same each time -- a list in our repo
drifts from the corpus, and the drift is invisible:

  the population   the ASAN-blind definition, recomputed from the corpus's two result files
                   (outcome != OK in results-asan-default, outcome == OK in -stockalloc).
                   31 cases today. A list here could quietly shrink to the cases we pass.
  the expected site the first PHP source frame in ASAN's OWN stderr. We do not maintain a table
                   of where each fault should land, so we cannot bend one after seeing a result.
  `reached`        the corpus's recorded pristine OUTCOME plus the control arm running the case
                   to its normal end. See below -- this is a DEVIATION from the pre-registration
                   and the reason is recorded there.

HOW `reached` IS ESTABLISHED, since this is the part a reader should check hardest. The judge
tests `reached` BEFORE attribution, exactly so that a fault during startup is refused as
`setup-fault` rather than counted. Two facts together establish it, neither sufficient alone:

  1. `revocation-sanity` COMPLETED on this arm. The arm therefore starts PHP up and runs a
     script to its normal end, so a fault on a case is after startup, not in it. This is the
     gate the exact arm failed in Phase A, when all six cases faulted inside zend_startup and
     the judge refused all six.
  2. the STOCK arm (PHP_CAP_MODE=0, nothing revoked) ran this case to its normal end, and the
     corpus's own pristine record says the case completes (`outcome == "OK"`). The trigger is
     therefore live in these bytes, and the oracle agrees the case is one that completes.

Because one image serves every arm, (2) transfers across arms: the bytes that ran the trigger
under mode 0 are the bytes running it under mode 7.

(2) HAS A SECOND ROUTE, and it is needed for a case the first route structurally cannot
describe: one the platform catches even on the stock arm. Such a case has no completing control
arm by construction, so the rule above makes it un-`reached` on every arm and the judge reports
`setup-fault` -- the instrument blamed for the clearest result the instrument can produce. This
is not hypothetical and it is not hindsight: the pre-registration at 692e9d79a244 predicts
CRASH-022 CAUGHT on all three temporal arms, the weakest included, so the apparatus could not
have expressed its own prediction. CRASH-022 and CRASH-102 both faulted on the stock arm, at
ext/standard/array.c:546 and ext/standard/string.c:1332 -- ASAN's own recorded line for each.

So a fault at the case's OWN ASAN-recorded site also establishes `reached`, because that line
cannot be executed without the trigger having run. That is the same argument this file already
makes below for cases with an environmental precondition, applied where it is just as true, and
each Observation records WHICH route licensed it. The honest cost: for such a case the reading
no longer carries the independent check that a control arm reproduced the oracle's output. It
is named in the results rather than left to be discovered.

DEVIATION FROM THE PRE-REGISTRATION, stated here because it weakens a stated criterion. The
pre-registration said the control arm's stdout would have to equal the corpus's recorded
pristine stdout BYTE FOR BYTE. That cannot be required, and the reason is PHP's own output, not
ours: CRASH-099 makes PHP 5.0.0 print

    Warning: fopen(php://memory): failed to open stream: <strerror> in <path> on line 2

because php://memory did not exist until PHP 5.1 -- native PHP 5.0.0 prints the same warning.
Two parts of that line legitimately differ: the absolute script path (the corpus directory on
the host, /mnt/vm in the guest) and the strerror wording for errno 0 (glibc "Success", musl "No
error information"). Normalising text until it matches is how an oracle stops being one, so the
stdout comparison is kept as a REPORTED note -- printed, and carried into the bundle -- while
`reached` rests on the two facts above. A divergence is therefore visible on every case without
silently deciding any of them.
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TOOLS = REPO / "capstone" / "bug-corpora" / "tools"
sys.path.insert(0, str(TOOLS))
import verdicts                                                          # noqa: E402

ARMS = ("php-real-stock", "php-real-block", "php-real-object", "php-real-exact")
STOCK = "php-real-stock"
SPATIAL = ("php-real-object", "php-real-exact")      # control, fault -- the bound alone
TEMPORAL = ("php-real-stock", "php-real-block", "php-real-object")

SANITY = "revocation-sanity"
BOUND_PROBE = "seam-bound-probe"
SANITY_SCRIPT = "phpb-sanity.php"

MODE_LINE = re.compile(r"ZEND_CAP_MODE (\d+) exact=(\d) revoke=(\d)")
SEAM_LINE = re.compile(r"PHP_VIRTUAL SEAM blocks=(\d+) frees=(\d+) issues=(\d+) "
                       r"retires=(\d+) revokes=(\d+)")
SEAM_FAIL = re.compile(r"ZEND_CAP_FAIL code=(\d+) ([^\n]*)")
STATS_LINE = re.compile(r"CAPSTONE_VM_STATS .*?\bnodes=(\d+).*?\bnode_capacity=(\d+)")
KERNEL_LINE = re.compile(r"^\[\s*\d+\.\d+\]")
# A fault in the seam's own sources is a port artefact and never a catch. The physical column
# learned this the expensive way: with only a pc, a fault in the allocator looks exactly like a
# fault in the program.
SEAM_OWN = ("zend-cap-blocks.c", "zend-cap-backing.c", "php_virtual_lower.c")
# Cases whose ASAN site is PHP's own allocator. A fault there cannot be attributed by site --
# it is indistinguishable from the port breaking -- so these need the diagnostic audit arm that
# run-cases.py drives for CRASH-073, and are `unattributed` until it is wired up here.
ALLOCATOR_SITE = "Zend/zend_alloc.c"

# The per-run watchdog in run-virtual.py fired on this run. A run that was KILLED is not
# an observation of the program at all -- it is the harness giving up -- so it is reported
# as infra rather than as a silence. Keyed on the marker and not on exit 137, because the
# program can produce an exit code and only that loop can produce the marker.
WATCHDOG = re.compile(r"VM_EXIT_TIMEOUT:")

# WHAT THE NODE-TABLE GATE IS FOR, AND WHY IT IS NOT A NODE COUNT. The hazard is real: the
# node table running out produces a cause-30 fault at an arbitrary pc, which is otherwise
# indistinguishable from a catch and would be the most likely way this port invents a false
# positive. The pre-registration at 692e9d79a244 promised to refuse any run whose node
# high-water came within a reserve of capacity.
#
# That gate was implemented against the wrong field and could not have worked. The printed
# `nodes=` is cumulative allocations, not a high-water (see node_pressure), so it is not
# bounded by the capacity -- and the counter the pre-registration named is not printed at all.
# As written the check fired on roughly twenty perfectly good runs of this measurement and
# would have refused them as `infra`, catches included.
#
# The replacement is not a better proxy, it is the thing itself: a cause-30 fault SAYS it is a
# cause-30 fault, on the same line the pc comes from. Keying on it is strictly stronger than
# any threshold over a counter, needs no reserve to tune, and cannot fire on a healthy run.
CAUSE_RESOURCE = 30

ASAN_FRAME = re.compile(
    r"#\d+ 0x[0-9a-f]+ in (\S+) [^ ]*?((?:Zend|main|ext|sapi|TSRM|regex)/[\w./]+):(\d+)")


def oracle_crashes(spec):
    """Did NATIVE PHP die on a SIGNAL for this case?

    This is the premise behind `out-of-denominator`, and it is NOT the same question as
    "did the oracle exit zero". PHP exits 255 on its OWN fatal error -- a parse error, an
    uncaught exception, a failed include -- having printed a deterministic diagnostic and
    shut down cleanly. Such a case has a perfectly good oracle and IS scoreable; only a
    case the program cannot survive has no pair to read.

    The old test was `pristine_outcome != "OK"`, which conflated the two. It happened never
    to misfire on the ASAN-blind 31, where the only non-OK outcomes are two SIGNAL:11 -- a
    condition satisfied by its input rather than by its logic, which is the shape this
    project has been burned by before. It DOES misfire on the temporal cases ASAN already
    sees: CRASH-062 and CRASH-070 are EXIT:255 from a PHP fatal error, and the old test
    would have thrown both out of the denominator as "native PHP crashes on it".
    """
    return spec["pristine_outcome"].startswith("SIGNAL:")


# Which cases to measure. Both are DERIVED from the corpus, never listed here: a list of
# case ids living in our repo would drift from the corpus and the drift would be invisible.
POPULATIONS = ("asan-blind", "temporal-asan-visible")


def population(corpus, kind="asan-blind"):
    """The measured set, recomputed from the corpus's own files.

    asan-blind              detected with PHP's allocator PATCHED, missed with it as shipped.
    temporal-asan-visible   CWE-416 that ASAN finds unaided -- the complement, on the temporal
                            axis. These are not blind BECAUSE `efree` reaches `free()`, which
                            is the same fact that makes them the interesting control: a
                            detector that interposes at `free` should get them.
    """
    default = json.loads((corpus / "results-asan-default.json").read_text())["results"]
    stock = json.loads((corpus / "results-asan-stockalloc.json").read_text())["results"]
    pristine = json.loads((corpus / "results-pristine-default.json").read_text())["results"]
    stock_outcome = {e["id"]: e.get("outcome", "") for e in stock}
    pris = {e["id"]: e for e in pristine}
    cases = {}
    meta = {m["id"]: m for m in
            json.loads((corpus / "metadata" / "corpus.json").read_text())}
    for e in default:
        blind = e.get("outcome") != "OK" and stock_outcome.get(e["id"]) == "OK"
        if kind == "asan-blind":
            if not blind:
                continue
        elif kind == "temporal-asan-visible":
            if blind or meta.get(e["id"], {}).get("cwe") != "CWE-416":
                continue
        else:
            sys.exit(f"unknown population {kind!r}; choose from {POPULATIONS}")
        frames = ASAN_FRAME.findall(e.get("stderr", ""))
        if not frames:
            sys.exit(f"{e['id']}: ASAN's stderr has no PHP source frame; the oracle cannot "
                     f"supply an expected site and a site of our own would be a table to bend")
        fn, path, line = frames[0]
        p = pris.get(e["id"])
        if p is None:
            sys.exit(f"{e['id']}: no entry in results-pristine-default.json; `reached` has no oracle")
        cases[e["id"]] = {
            "cwe": e.get("cwe", ""), "requires": e.get("requires", ""),
            "asan_func": fn, "asan_file": path, "asan_line": int(line),
            "pristine_stdout": p.get("stdout", ""), "pristine_outcome": p.get("outcome", ""),
            "pristine_rc": p.get("rc", 0),
        }
        # TWO INDEPENDENT SOURCES FOR THE SAME FACT, MADE TO AGREE. The corpus publishes
        # `crashes_on_pristine_build` per case AND a recorded pristine run; this runner reads
        # the run. If they ever disagree the premise behind `out-of-denominator` is unsound,
        # and that must stop the measurement rather than be silently resolved in our favour.
        flag = str(meta.get(e["id"], {}).get("crashes_on_pristine_build", ""))
        crashed = cases[e["id"]]["pristine_outcome"].startswith("SIGNAL:")
        if flag.startswith(("True", "False")) and (flag.startswith("True") != crashed):
            sys.exit(f"{e['id']}: the corpus says crashes_on_pristine_build={flag!r} but its "
                     f"recorded pristine run is {cases[e['id']]['pristine_outcome']!r}. The "
                     f"premise behind out-of-denominator cannot be checked -- refusing to run.")
    recorded = {"asan-blind": 31, "temporal-asan-visible": 29}.get(kind)
    if recorded is not None and len(cases) != recorded:
        print(f"NOTE: the {kind} population is {len(cases)} today, not the {recorded} the plan "
              f"records. That is a change in the corpus, not in this runner.", file=sys.stderr)
    if not cases:
        sys.exit("the population came out EMPTY -- refusing to report a denominator of zero")
    return dict(sorted(cases.items()))


# The launcher prints its stats without a preceding newline, so when the application's own
# output does not end in one the two land on the SAME line -- e.g. "999999999999999CAPSTONE_VM_
# STATS arenas=39 ...". A startswith() filter leaves that intact and the oracle comparison then
# fails for a reason that has nothing to do with the program.
STATS_ANYWHERE = re.compile(r"CAPSTONE_VM_STATS .*$")


def app_output(lines):
    """The application's own stdout: kernel console lines and the VM's stats line removed."""
    out = []
    for raw in lines:
        line = STATS_ANYWHERE.sub("", raw.rstrip("\r"))
        if KERNEL_LINE.match(line):
            continue
        if not line and STATS_ANYWHERE.search(raw):
            continue                      # the stats line stood alone
        out.append(line)
    return "\n".join(out)


def strip_seam_noise(text):
    """What PHP itself printed: the seam's own mode and stats lines removed."""
    keep = [l for l in text.splitlines()
            if not l.startswith("ZEND_CAP_MODE") and not l.startswith("PHP_VIRTUAL SEAM")]
    return "\n".join(keep)


def normalise_paths(text, case):
    """Replace any absolute path naming this case's file with its basename.

    The ONLY normalisation applied, and only for the reported comparison -- never to decide a
    verdict. PHP prints the script's path in a warning, and the host path and the guest's
    /mnt/vm path are both correct.
    """
    return re.sub(r"\S*" + re.escape(case) + r"\.php", f"{case}.php", text or "")


def same_as_oracle(observed, expected):
    """Trailing-newline-insensitive comparison; the console does not preserve a final newline."""
    return observed.strip("\n") == expected.strip("\n")


def note(obs, text):
    """Append a note. Notes ACCUMULATE: a run can be worth several remarks at once -- its node
    counters, an unconfirmed precondition, output that diverged from the oracle -- and an
    assignment would keep only whichever happened to be written last."""
    obs.notes = (obs.notes + "; " + text) if obs.notes else text


def node_pressure(text):
    """(nodes allocated, capacity) for a run, or None if the VM printed no stats.

    READ THE NAME CAREFULLY: this is NOT a high-water. runtime/virtual/exec.c:554 prints
    `stats.nodes`, and the module sets that from `c->table[3]`, which it calls
    `nodes_allocated` where it aggregates (module/capstone_vm.c:673, :704) -- a CUMULATIVE
    count of nodes ever minted. The high-water is a different field, `c->table[1] - 2`, and
    so are `nodes_live` and `nodes_retired`. None of those three is printed.

    That matters because a cumulative count is not bounded by the capacity and routinely
    exceeds it: this measurement has runs reading nodes=13480 against node_capacity=13312, and
    one reading 127102 against 103424 with collections=2 and reclaimed=50557. Comparing them
    is meaningless, so this value is REPORTED and never gated on. See the comment on
    CAUSE_RESOURCE for what replaced the gate."""
    best = None
    for m in STATS_LINE.finditer(text or ""):
        nodes, cap = int(m.group(1)), int(m.group(2))
        if best is None or nodes > best[0]:
            best = (nodes, cap)
    return best


# The image's own link base, read from its LOAD segments by verdicts.Symbols rather than
# written down here. It happens to be 0x10000 for this image, and a literal would be a second
# source of truth that goes wrong silently the day the link script changes: every pc would
# relocate to the wrong place and every catch would read `unattributed`.
_SYMBOLS = {}


def image_base(llvm_bin, image):
    key = str(image)
    if key not in _SYMBOLS:
        _SYMBOLS[key] = verdicts.Symbols(llvm_bin, image)
    return _SYMBOLS[key].base


def symbolize(llvm_bin, image, fault, text):
    """(file, line) for a fault, from the image's line tables. ('', 0) when there are none."""
    m = re.search(r"code=(0x[0-9a-f]+)-", text)
    if not m or fault is None:
        return "", 0
    static = fault.pc - int(m.group(1), 16) + image_base(llvm_bin, image)
    out = subprocess.run([str(Path(llvm_bin) / "llvm-symbolizer"), "--obj", str(image),
                          "--functions=short", f"{static:#x}"],
                         capture_output=True, text=True).stdout.splitlines()
    where = out[1].strip() if len(out) > 1 else ""
    m2 = re.match(r"(.+?):(\d+):\d+$", where)
    if not m2:
        return "", 0
    # Returned WHOLE and compared by suffix, not trimmed against a list of known build prefixes.
    # The symbolizer's path depends on where the tree was mirrored and on whether a file came
    # from the read-only corpus mount, and a prefix list silently stops matching when either
    # changes -- which turns a catch into `unattributed` for a reason that has nothing to do
    # with the program. ASAN's own site is repo-relative, so a suffix match is the comparison
    # that is actually being asked for.
    return m2.group(1), int(m2.group(2))


def arm_controls(arm, sanity_ok, probe_ok, arms):
    """The Control list for an arm, naming only what arms.json declares for it."""
    declared = arms[arm]["controls"]
    out = [verdicts.Control(name=SANITY, observed="complete" if sanity_ok else "did-not-complete")]
    if BOUND_PROBE in declared:
        out.append(verdicts.Control(name=BOUND_PROBE,
                                    observed="fault" if probe_ok else "did-not-fault"))
    return out


def observe(case, spec, arm, image_sha, text, exit_code, controls, stock_reached, llvm_bin,
            image, precondition_met=True, site_licensed=False):
    obs = verdicts.Observation(case=case, arm=arm, image_sha256=image_sha, controls=controls)
    # NO PREMISE, SO NO PAIR. The corpus's own pristine record says native PHP does not
    # complete this case -- it crashes with no capability platform anywhere near it -- so there
    # is nothing here for an arm to catch or miss and the cell is out of the denominator.
    #
    # Said in the judge's own vocabulary rather than left to fall through. Without this the two
    # such cases land wherever the later rules put them, and both landings MISREPORT: CRASH-057
    # faults on our arms and so reads `setup-fault`, which blames our instrument for a crash
    # native PHP also has, and CRASH-100 is silent and so reads `not-reached`, which is a claim
    # about our run rather than about the oracle. Both are NO-READING either way, so this moves
    # no case into or out of CAUGHT; it makes the reason true.
    if oracle_crashes(spec):
        obs.infra = "out-of-denominator"
        obs.notes = (f"the corpus's pristine record for this case is "
                     f"{spec['pristine_outcome']!r}: native PHP dies on a signal, "
                     f"so there is no pair to read")
        return obs
    # NO DATA IS AN ERROR, NOT A ZERO. Both of these say the harness never got a reading, and
    # both would otherwise fall through to `completed=False, reached=False` -- which the judge
    # reports as `not-reached`, a statement about the PROGRAM. The first measurement made
    # exactly that mistake available: CRASH-092 wedged the guest and the ten cases after it
    # alphabetically were unobserved on two arms.
    if WATCHDOG.search(text):
        obs.infra = "infra"
        obs.notes = "the per-run watchdog killed this run; it never returned a reading"
        return obs
    if exit_code is None:
        obs.infra = "infra"
        obs.notes = ("no exit status was observed for this run -- the guest session ended "
                     "before it reported, so there is nothing here to judge")
        return obs
    fail = SEAM_FAIL.search(text)
    if fail:
        obs.infra = "setup-fault"
        obs.notes = f"the seam failed: code={fail.group(1)} {fail.group(2)[:90]}"
        return obs
    fault = verdicts.domain_fault(text, None)
    if fault is not None and fault.cause == CAUSE_RESOURCE:
        obs.infra = "infra"
        obs.notes = (f"cause-{CAUSE_RESOURCE} resource fault at pc {fault.pc:#x}: the node table "
                     f"ran out, which lands at an arbitrary pc and is not a reading about the "
                     f"program")
        return obs
    pressure = node_pressure(text)
    if pressure:
        note(obs, f"nodes allocated {pressure[0]}, table capacity {pressure[1]} "
                  f"(cumulative, not a high-water: reported, not gated)")
    said = strip_seam_noise(text)
    # COMPLETED MEANS "ENDED THE WAY THE ORACLE DID", NOT "EXITED ZERO". Hardcoding 0 is wrong
    # for any case whose pristine record is an EXIT, and PHP 5.0.0 has four of them: its own
    # fatal errors leave status 255 after printing a diagnostic, which is the case behaving
    # exactly as the corpus recorded it. With 0 hardcoded such a case can never be `completed`,
    # so it falls through to `inconclusive` on every arm -- the control arm included, which then
    # takes the whole column of that case down with it. This is the SAME generalisation already
    # applied to `ran` below; applying it there and not here is what made CRASH-062 and
    # CRASH-070 unreadable. Inert for an OK oracle, so the published 31 are unaffected.
    want_rc = 0 if spec["pristine_outcome"] == "OK" else spec["pristine_rc"]
    obs.completed = (exit_code == want_rc and fault is None)
    # TWO ROUTES TO `reached`, AND THE SECOND IS NOT A CONCESSION -- IT IS THE RULE THE FIRST
    # ONE IS A PROXY FOR. The question is only ever "did the trigger run", and the control arm
    # completing with the oracle's output is evidence for that, not the definition of it.
    #
    # Derived from the control arm alone, the rule is unable to express a prediction the
    # pre-registration at 692e9d79a244 actually made: CRASH-022 is predicted CAUGHT on ALL
    # THREE temporal arms, the weakest included. A case the platform catches even at mode 0 has
    # no completing control arm by construction, so `reached` would be false on every arm and
    # the judge would report `setup-fault` -- a fault blamed on the instrument -- for the one
    # shape of case the platform handles best. CRASH-022 and CRASH-102 both did exactly that.
    #
    # The second route is the argument already written below for `requires` cases, applied
    # where it is just as true: a fault at the case's OWN ASAN-recorded source line cannot be
    # reached without the trigger having run. Which route licensed a given reading is recorded
    # in reach_evidence rather than left for a reader to reconstruct.
    obs.reached = bool(site_licensed) or (bool(stock_reached) and bool(precondition_met))
    if site_licensed:
        obs.reach_evidence = (f"some arm of this run faulted at {spec['asan_file']}, the case's "
                              f"own ASAN-recorded site, which cannot be reached without the "
                              f"trigger having run")
    elif obs.reached:
        obs.reach_evidence = ("the stock arm ran this case to its normal end, the corpus's "
                              "pristine record says it completes, and revocation-sanity "
                              "completed on this arm")
    else:
        obs.reach_evidence = ""
    if stock_reached and not precondition_met:
        # A case with a declared environmental precondition that NOTHING in this measurement
        # confirmed was met. CRASH-092 is the shape of the problem: it creates its own loopback
        # listener and then echoes "done" WHETHER OR NOT the listener came up, so a silence here
        # cannot be told apart from a platform that caught nothing. Refusing it is the only
        # honest option; a MISSED would be a claim about revocation drawn from a socket.
        note(obs, f"declared precondition {spec['requires']!r} was not confirmed met by any "
                  f"arm in this run, so a silence here is not a reading")

    if fault is not None:
        obs.fault = fault
        path, line = symbolize(llvm_bin, image, fault, text)
        if any(own in path for own in SEAM_OWN):
            obs.attribution = None
            obs.attribution_evidence = (f"the fault is in the seam's own source ({path}:{line}) "
                                        f"-- a port artefact, not the program")
            return obs
        if spec["asan_file"].endswith(ALLOCATOR_SITE):
            obs.attribution = None
            obs.attribution_evidence = (
                f"ASAN's own site for this case is {ALLOCATOR_SITE}:{spec['asan_line']}, so a "
                f"fault there cannot be told from the port breaking; it needs the diagnostic "
                f"audit arm (observed {path}:{line})")
            return obs
        if path.endswith(spec["asan_file"]):
            exact = (line == spec["asan_line"])
            obs.attribution = "function"
            obs.attribution_evidence = (
                f"{spec['asan_file']}:{line} is ASAN's own file for this case"
                + (f", and its own line" if exact else
                   f" (ASAN: line {spec['asan_line']}; a capability fault can land on a "
                   f"neighbouring instruction)"))
        else:
            obs.attribution = None
            obs.attribution_evidence = (f"faulted at {path or '?'}:{line} but ASAN's site is "
                                        f"{spec['asan_file']}:{spec['asan_line']}")
        return obs

    # normalise_paths, for the SAME reason the stock-arm comparison below applies it: PHP prints
    # the script's absolute path inside its own warnings and fatal errors, and the corpus's host
    # path and the guest's /mnt/vm path are both correct. Comparing them literally reports a
    # divergence on every such case -- "2 lines vs 2", differing only in the path.
    if said.strip() and not same_as_oracle(normalise_paths(said, case),
                                           normalise_paths(spec["pristine_stdout"], case)):
        note(obs, f"completed but its output differs from the corpus's pristine record "
                  f"({len(said.splitlines())} lines vs "
                  f"{len(spec['pristine_stdout'].splitlines())})")
    return obs


# ---- the runs ------------------------------------------------------------------------------

def run_arm(args, arm, mode, cases, record, serial, wants_probe):
    """One guest boot: the sanity control, the bound probe, then every case, all at `mode`."""
    img = args.image.name
    runs = [(SANITY, f"PHP_CAP_MODE={mode} {img} -d memory_limit=512M /mnt/vm/{SANITY_SCRIPT}")]
    if wants_probe:
        # probe-seam stage 7 sets its OWN modes internally (rounded, read index 9, then exact,
        # read index 9), so it is not parameterised by the arm. See arms.json on what that does
        # and does not establish.
        runs.append((BOUND_PROBE, f"{args.probe.name} 7"))
    for case in cases:
        runs.append((case, f"PHP_CAP_MODE={mode} {img} -d memory_limit=512M /mnt/vm/{case}.php"))
    cmd = [sys.executable, str(HERE / "run-virtual.py"),
           "--qemu", str(args.qemu), "--images", str(args.images), "--adapter", str(args.adapter),
           "--image", str(args.image), "--record", str(record), "--serial-log", str(serial),
           "--timeout", str(args.timeout), "--run-timeout", str(args.run_timeout)]
    if wants_probe:
        cmd += ["--image", str(args.probe)]
    cmd += ["--extra", str(args.scratch / SANITY_SCRIPT)]
    for case in cases:
        cmd += ["--extra", str(args.scratch / f"{case}.php")]
    for name, argv in runs:
        cmd += ["--run", f"{name}={argv}"]
    rc = subprocess.call(cmd)
    if not record.is_file():
        sys.exit(f"run-virtual.py wrote no record (rc={rc}); nothing to judge")
    return rc, json.loads(record.read_text())


def stage_inputs(args, cases):
    """Copy the corpus's case files VERBATIM, plus the benign workload the sanity control runs."""
    args.scratch.mkdir(parents=True, exist_ok=True)
    for case in cases:
        src = args.corpus / "corpus" / f"{case}.php"
        if not src.is_file():
            sys.exit(f"no case file at {src}")
        (args.scratch / f"{case}.php").write_bytes(src.read_bytes())
    # The sanity workload must allocate and free heavily through emalloc/efree and contain NO
    # defect, so that "revocation is sound on a clean workload" is what it actually measures.
    (args.scratch / SANITY_SCRIPT).write_text(
        "<?php\n"
        "// revocation-sanity: a benign workload. Every arm must run this to its normal end;\n"
        "// an arm that cannot is not a column, and a silence on it is not a MISSED.\n"
        "$a = array();\n"
        "for ($i = 0; $i < 400; $i++) { $a[] = str_repeat('x', ($i % 23) + 1); }\n"
        "for ($i = 0; $i < 400; $i += 2) { unset($a[$i]); }\n"
        "$s = '';\n"
        "foreach ($a as $v) { $s .= substr($v, 0, 3); }\n"
        "$u = unserialize(serialize(array('k' => $a, 'n' => 17)));\n"
        "echo 'sanity ', strlen($s), ' ', count($u['k']), ' ', $u['n'], \"\\n\";\n")


def seam_counters(text):
    m = SEAM_LINE.search(text or "")
    mode = MODE_LINE.search(text or "")
    return (int(mode.group(1)) if mode else None,
            int(m.group(5)) if m else None)


def cross_arm_gates(args, per_arm, llvm_bin):
    """Refuse the whole measurement if the arms are not the arms they claim to be."""
    problems = []
    dis = subprocess.run([str(Path(llvm_bin) / "llvm-objdump"), "-d", str(args.image)],
                         capture_output=True, text=True).stdout
    revokes_in_image = len(re.findall(r"\brevoke\b", dis))
    if revokes_in_image == 0:
        problems.append("the image contains no revoke instruction, so no arm could revoke "
                        "anything and all three temporal columns would read alike")
    modes = {arm: per_arm[arm]["mode_word"] for arm in per_arm}
    if len(set(v for v in modes.values() if v is not None)) != len(modes):
        problems.append(f"the arms did not report distinct mode words: {modes}")
    for arm in ARMS:
        want = per_arm[arm]["declared_mode"]
        got = per_arm[arm]["mode_word"]
        if got != want:
            problems.append(f"{arm}: arms.json declares PHP_CAP_MODE={want} but the image "
                            f"reported {got}")
    rev = {arm: per_arm[arm]["revokes"] for arm in TEMPORAL}
    s, b, o = (rev[a] for a in TEMPORAL)
    if not (s == 0 and b == 0 and (o or 0) > 0):
        problems.append(f"revokes did not order stock==0, block==0, object>0: {rev}. Without "
                        f"that ordering a temporal catch cannot be attributed to the seam "
                        f"rather than to the allocator underneath it")
    return problems, revokes_in_image


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qemu", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True, help="php-real-seam.dom")
    p.add_argument("--probe", type=Path, required=True, help="probe-seam.dom, for the bound control")
    p.add_argument("--corpus", type=Path, required=True, help="the php-5.0.0-bug-corpus root")
    p.add_argument("--population", default="asan-blind", choices=POPULATIONS,
                   help="which set to measure (default: the ASAN-blind cases)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scratch", type=Path, help="where case files are staged (default: --out/stage)")
    p.add_argument("--llvm-bin", type=Path, default=REPO / "llvm/cmake-build-debug/bin")
    p.add_argument("--timeout", type=int, default=4200,
                   help="wall seconds for one arm's whole guest session")
    p.add_argument("--run-timeout", type=int, default=40,
                   help="GUEST seconds for one run before the watchdog kills it")
    p.add_argument("--case", action="append", help="restrict to these cases (debugging)")
    p.add_argument("--from-record", action="store_true",
                   help="re-judge stored runs without booting")
    p.add_argument("--prereg", default="", help="the pre-registration commit this run answers")
    args = p.parse_args()
    args.scratch = args.scratch or (args.out / "stage")
    args.out.mkdir(parents=True, exist_ok=True)

    arms = verdicts.load_arms()
    for arm in ARMS:
        if arm not in arms:
            sys.exit(f"arms.json has no entry for {arm}")

    cases = population(args.corpus, args.population)
    if args.case:
        cases = {c: cases[c] for c in args.case if c in cases}
        if not cases:
            sys.exit("--case selected nothing in the population")
    print(f"population: {len(cases)} ASAN-blind cases, derived from the corpus's own definition")

    image_sha = subprocess.run(["sha256sum", str(args.image)], capture_output=True,
                               text=True).stdout.split()[0]
    # The bound probe is a DIFFERENT image on purpose -- probe-seam.dom, built from the seam's
    # sources without PHP -- so its transcript reports its own sha256 and must be checked
    # against its own file. Exempting it instead would make the guard unable to notice a stale
    # probe transcript, which is the one run in the session whose image can change on its own.
    probe_sha = subprocess.run(["sha256sum", str(args.probe)], capture_output=True,
                               text=True).stdout.split()[0]
    if not args.from_record:
        stage_inputs(args, cases)

    per_arm, texts = {}, {}
    for arm in ARMS:
        mode = arms[arm]["mode"]
        wants_probe = BOUND_PROBE in arms[arm]["controls"]
        record = args.out / f"{arm}-run.json"
        serial = args.out / f"{arm}-serial.log"
        print(f"\n==> {arm}  (PHP_CAP_MODE={mode})", flush=True)
        if args.from_record:
            if not record.is_file():
                sys.exit(f"--from-record: no stored run at {record}")
            rec = json.loads(record.read_text())
        else:
            _, rec = run_arm(args, arm, mode, cases, record, serial, wants_probe)
        # A stored run from a DIFFERENT image must not be re-judged as this one's. Each run is
        # checked against the image it was supposed to be, which for the bound probe is the
        # probe's own.
        for name, lines in rec["output"].items():
            want = probe_sha if name == BOUND_PROBE else image_sha
            got = re.search(r"sha256=([0-9a-f]{64})", "\n".join(lines))
            if got and got.group(1) != want:
                sys.exit(f"{arm}/{name}: the stored transcript is from image {got.group(1)[:12]}, "
                         f"not {want[:12]}")
        texts[arm] = {n: app_output(l) for n, l in rec["output"].items()}
        sanity_text = texts[arm].get(SANITY, "")
        sanity_ok = (rec["exits"].get(SANITY) == 0
                     and verdicts.domain_fault(sanity_text, None) is None)
        probe_ok = True
        if wants_probe:
            probe_text = texts[arm].get(BOUND_PROBE, "")
            probe_ok = (verdicts.domain_fault(probe_text, None) is not None
                        and "rounded arm read index 9" in probe_text)
        mode_word, revokes = seam_counters(sanity_text)
        per_arm[arm] = {"rec": rec, "sanity_ok": sanity_ok, "probe_ok": probe_ok,
                        "mode_word": mode_word, "revokes": revokes,
                        "declared_mode": mode, "wants_probe": wants_probe}
        print(f"    {SANITY}: {'complete' if sanity_ok else 'DID NOT COMPLETE'} "
              f"(mode={mode_word} revokes={revokes})")
        if wants_probe:
            print(f"    {BOUND_PROBE}: {'faulted as declared' if probe_ok else 'DID NOT'}")

    problems, revokes_in_image = cross_arm_gates(args, per_arm, args.llvm_bin)
    print(f"\ncross-arm gates: revoke instructions in image = {revokes_in_image}")
    if problems:
        for t in problems:
            print(f"  GATE FAILED: {t}", file=sys.stderr)
        sys.exit("refusing to report verdicts: the arms are not the arms they claim to be")
    print("  all cross-arm gates pass")

    # `reached` per case, from the stock arm against the corpus's pristine record.
    stock_reached, output_diverged = {}, {}
    for case, spec in cases.items():
        text = texts[STOCK].get(case, "")
        said = strip_seam_noise(text)
        # The control must reproduce the ORACLE'S end, which is exit 0 for a case that
        # completes and 255 for one PHP ends with its own fatal error. For every case whose
        # oracle is "OK" this is exactly the old `== 0`, so the published ASAN-blind
        # measurement is untouched by the generalisation.
        want_rc = 0 if spec["pristine_outcome"] == "OK" else spec["pristine_rc"]
        ran = (per_arm[STOCK]["rec"]["exits"].get(case) == want_rc
               and verdicts.domain_fault(text, None) is None)
        # The oracle must itself say this is a case that completes. CRASH-057 is recorded as
        # SIGNAL:11 on the pristine build, so there is no silence to interpret for it.
        oracle_ok = not oracle_crashes(spec)
        stock_reached[case] = ran and oracle_ok
        if not oracle_ok:
            print(f"    {case}: the corpus's pristine record is "
                  f"{spec['pristine_outcome']!r}, not OK -- no premise for a pair")
        elif ran and not same_as_oracle(normalise_paths(said, case),
                                        normalise_paths(spec["pristine_stdout"], case)):
            output_diverged[case] = (said, spec["pristine_stdout"])
            print(f"    {case}: ran, but its output differs from the oracle's "
                  f"({len(said.splitlines())} lines vs "
                  f"{len(spec['pristine_stdout'].splitlines())})")

    # PASS ONE: where did each run fault? A fault at the case's OWN ASAN site is proof that the
    # case's declared precondition (a writable /tmp, a working loopback) was met -- you cannot
    # reach that line otherwise -- and that proof then licenses reading a SILENCE on the other
    # arms of the same case. Without it, every `requires` case would be refused on every arm,
    # including the ones that plainly caught it.
    precondition_met, site_licensed = {}, {}
    for case, spec in cases.items():
        hit = False
        for arm in ARMS:
            text = texts[arm].get(case, "")
            f = verdicts.domain_fault(text, None)
            if f is None:
                continue
            path, _ = symbolize(args.llvm_bin, args.image, f, text)
            if path.endswith(spec["asan_file"]):
                hit = True
                break
        # THE LICENCE ANSWERS "DID THE TRIGGER RUN", NOT "IS THERE A PAIR TO READ". Those are
        # different questions and folding them together is wrong in a way the offline suite
        # caught: CRASH-057's pristine record is SIGNAL:11, so native PHP crashes on it with no
        # capability platform anywhere near, and a fault of ours is not evidence of a catch
        # whatever line it lands on. The oracle's premise is kept as a separate conjunct.
        site_licensed[case] = hit and not oracle_crashes(spec)
        precondition_met[case] = hit if spec["requires"] else True
        if spec["requires"]:
            print(f"    {case}: precondition {spec['requires']!r} "
                  + ("confirmed by a fault at ASAN's own site" if hit
                     else "NOT confirmed -- a silence on this case is not a reading"))
        elif hit and not stock_reached[case]:
            # The case the control-arm rule cannot describe: caught even on the weakest arm.
            print(f"    {case}: the stock arm did not complete it, but an arm faulted at "
                  f"{spec['asan_file']} -- the trigger ran")

    rows, table = {arm: [] for arm in ARMS}, {}
    for arm in ARMS:
        controls = arm_controls(arm, per_arm[arm]["sanity_ok"], per_arm[arm]["probe_ok"], arms)
        for case, spec in cases.items():
            text = texts[arm].get(case, "")
            obs = observe(case, spec, arm, image_sha, text,
                          per_arm[arm]["rec"]["exits"].get(case),
                          controls, stock_reached[case], args.llvm_bin, args.image,
                          precondition_met[case], site_licensed[case])
            if case in output_diverged:
                note(obs, "the stock arm's output differs from the corpus's pristine record "
                          "(reported, not gated: see the module docstring on the deviation)")
            judged = verdicts.judge(obs, arms[arm])
            rows[arm].append((obs, judged))
            table[(case, arm)] = judged

    width = max(len(c) for c in cases)
    print(f"\n{'case':{width}}  " + "  ".join(f"{a.replace('php-real-',''):>11}" for a in ARMS))
    for case in cases:
        cells = []
        for arm in ARMS:
            v, reason, _ = table[(case, arm)]
            cells.append(f"{(v if v != verdicts.NO_READING else reason):>11}")
        print(f"{case:{width}}  " + "  ".join(cells))

    print()
    for arm in ARMS:
        counts = {}
        for _, (v, reason, _) in rows[arm]:
            key = v if v != verdicts.NO_READING else f"{v}/{reason}"
            counts[key] = counts.get(key, 0) + 1
        print(f"  {arm:18} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    platform = {
        "compiler": subprocess.run([str(args.llvm_bin / "clang"), "--version"],
                                   capture_output=True, text=True).stdout.splitlines()[0],
        "qemu": str(args.qemu), "firmware": str(args.images / "fw_jump.elf"),
        "kernel": str(args.images / "Image"), "runner": "ports/php/app/run-realphp.py",
    }
    for arm in ARMS:
        verdicts.write_bundle(args.out / f"bundle-{arm}", "php-5.0.0-bug-corpus", arm, rows[arm],
                              {"platform": platform, "image_sha256": image_sha,
                               "pre_registration": args.prereg,
                               "population": "ASAN-blind, recomputed from the corpus",
                               "cases": len(cases)})
    print(f"\nbundles written under {args.out}")


if __name__ == "__main__":
    main()
