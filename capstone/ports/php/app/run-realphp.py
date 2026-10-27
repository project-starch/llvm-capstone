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
import argparse, hashlib, importlib.util, io, json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def run_virtual_module():
    """run-virtual.py's platform constants, IMPORTED rather than copied. A second copy of the
    -cpu string in the record is a record that can disagree with the run it describes, which is
    worse than no record at all: it reads as evidence. The filename has a hyphen, so it cannot
    be a plain import."""
    spec = importlib.util.spec_from_file_location("php_run_virtual", HERE / "run-virtual.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cap_insn_counter():
    """count-cap-insns.py's counter, IMPORTED rather than reimplemented.

    There used to be two copies of this gate -- one here and one in build-virtual.sh -- and on
    2026-10-26 they drifted: the build's copy was moved to the instruction ENCODING when the seam
    went to CDERIVE/CREVOKE, and this one was left counting a `revoke` MNEMONIC that
    llvm-objdump does not emit for either instruction. The build passed and this refused the
    whole measurement after all eight guest sessions had run. One implementation, imported."""
    spec = importlib.util.spec_from_file_location("php_cap_insns", HERE / "count-cap-insns.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.count
TOOLS = REPO / "capstone" / "bug-corpora" / "tools"
sys.path.insert(0, str(TOOLS))
import verdicts                                                          # noqa: E402

ARMS = ("php-real-stock", "php-real-block", "php-real-object", "php-real-exact")
STOCK = "php-real-stock"
# The corpus's arm names (bug-corpora/SCHEMA.md "Arms") for the three measured configurations;
# the stock arm is the control the judge reads `reached` from and stays under its own name.
FRAMEWORK_ARM = {"php-real-block": "virtual-malloc", "php-real-object": "virtual-nested-pools",
                 "php-real-exact": "virtual-nested-exact", "php-real-stock": "php-real-stock"}
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
    """The measured set, read from the corpus's own case directories.

    `corpus` is bug-corpora/php/engine-repros. Each NN_<fix>_<id>/ holds trigger.php and
    oracle.json, written by that corpus's tools/build-cases.py from the PHP 5.0.0 bug corpus and
    OUR native runs of it (ASan with PHP's allocator patched, ASan with it as shipped, and the
    plain build). The population is decided there, once, and recorded per case:

    asan-blind              detected with PHP's allocator PATCHED, missed with it as shipped.
    temporal-asan-visible   CWE-416 that ASAN finds unaided -- the complement, on the temporal
                            axis. These are not blind BECAUSE `efree` reaches `free()`, which
                            is the same fact that makes them the interesting control: a
                            detector that interposes at `free` should get them.

    Cases are keyed by their DIRECTORY NAME, which is what tools/derive-verdicts.py keys the
    case.json arms by; the corpus id (CRASH-nnn) is in the spec as `id`.
    """
    if kind not in POPULATIONS:
        sys.exit(f"unknown population {kind!r}; choose from {POPULATIONS}")
    cases = {}
    for d in sorted(Path(corpus).glob("[0-9][0-9]_*")):
        if not (d / "oracle.json").is_file():
            continue
        o = json.loads((d / "oracle.json").read_text())
        if o["population"] != kind:
            continue
        if not (d / "trigger.php").is_file():
            sys.exit(f"{d.name}: no trigger.php")
        # The DECLARED site comes from case.json, the framework's record (SCHEMA.md
        # `fault_sites`): the one function a fault may land in and be this defect's, with the
        # reason it was chosen written from the native run and not from any fault here. The
        # oracle's own frame is carried beside it for the file and line the function was seen
        # at; the two are written by the same builder and must agree.
        c = json.loads((d / "case.json").read_text())
        sites = list(c.get("fault_sites") or [])
        if not sites or not c.get("fault_sites_why"):
            sys.exit(f"{d.name}: case.json declares no fault_sites (with fault_sites_why); nothing "
                     f"can attribute a fault to this case, so it cannot be measured")
        stack = o["asan"].get("stack") or []
        if o["asan"]["func"] not in sites:
            sys.exit(f"{d.name}: fault_sites {sites} do not name ASan's own frame {o['asan']['func']}")
        cases[d.name] = {
            "id": o["id"], "cwe": o["cwe"], "requires": o.get("requires", ""),
            "requires_kind": o.get("requires_kind", "execution" if o.get("requires", "").strip() else ""),
            "asan_func": o["asan"]["func"], "asan_file": o["asan"]["file"], "asan_line": int(o["asan"]["line"]),
            "asan_stack": stack, "fault_sites": sites,
            "pristine_stdout": o["pristine"].get("stdout", ""), "pristine_outcome": o["pristine"]["outcome"],
            "pristine_rc": o["pristine"].get("rc", 0),
        }
        # A CONSISTENCY CHECK, not two independent sources: oracle.json's crash flag is derived
        # from the same pristine run it records, so a disagreement means the case file was
        # edited or generated wrongly, and that must stop the measurement rather than be
        # silently resolved in our favour.
        crashed = cases[d.name]["pristine_outcome"].startswith("SIGNAL:")
        if bool(o.get("crashes_on_pristine_build")) != crashed:
            sys.exit(f"{d.name}: oracle.json says crashes_on_pristine_build={o.get('crashes_on_pristine_build')!r} "
                     f"but its recorded pristine run is {cases[d.name]['pristine_outcome']!r}. The premise "
                     f"behind out-of-denominator cannot be checked -- refusing to run.")
    if not cases:
        sys.exit(f"no case directory under {corpus} is in population {kind!r}")
    recorded = {"asan-blind": 46, "temporal-asan-visible": 20}.get(kind)
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
    """The application's own stdout: kernel console lines, the VM's stats line and the
    launcher's own diagnostics removed.

    The launcher prints one CAPSTONE_VM_ADD_OK line per grant it makes -- thousands on a
    large case -- onto the same console. They are the platform talking, not PHP, and they
    were reaching the oracle comparison: a case whose PHP output matched the pristine record
    exactly was reported as diverging, and the divergence note then stood beside its verdict
    (CRASH-084 and CRASH-139 in the 2026-10-27 run). The stats line is parsed from the RAW
    transcript by the callers that need it, never from this."""
    out = []
    for raw in lines:
        line = STATS_ANYWHERE.sub("", raw.rstrip("\r"))
        if KERNEL_LINE.match(line) or line.startswith("CAPSTONE_VM_"):
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


def normalise_paths(text, case, cid=None):
    """Replace any path naming this case's file -- by its directory key (the guest stages it as
    /mnt/vm/<key>.php) or by its corpus id (native PHP ran corpus/CRASH-nnn.php) -- with one
    token, so a path-only difference is not read as a divergence.

    The ONLY normalisation applied, and only for the reported comparison -- never to decide a
    verdict. PHP prints the script's path in a warning, and both paths are correct.
    """
    names = [re.escape(case)] + ([re.escape(cid)] if cid else [])
    return re.sub(r"\S*(?:" + "|".join(names) + r")\.php", f"{cid or case}.php", text or "")


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


def symbols_of(llvm_bin, image):
    key = str(image)
    if key not in _SYMBOLS:
        _SYMBOLS[key] = verdicts.Symbols(llvm_bin, image)
    return _SYMBOLS[key]


def image_base(llvm_bin, image):
    return symbols_of(llvm_bin, image).base


def symbolize(llvm_bin, image, fault, text):
    """(function, file, line) for a fault, from the image's symbols and line tables. The
    function is '' when the symbolizer has none, and (file, line) is ('', 0) when there are no
    line tables for it -- musl's routines in this image, say, which have a symbol and no
    source. The function used to be discarded here, which left every libc fault a bare
    pc: the judge could not even say "this is strlen"."""
    m = re.search(r"code=(0x[0-9a-f]+)-", text)
    if not m or fault is None:
        return "", "", 0
    code = int(m.group(1), 16)
    static = fault.pc - code + image_base(llvm_bin, image)
    # THE FUNCTION COMES FROM THE SYMBOL TABLE, NOT FROM THE SYMBOLIZER. This image keeps its
    # local symbols, and llvm-symbolizer's function line then names whatever symbol is nearest
    # -- `.Lpcrel_hi9`, an assembler label -- or `??`. The first judge built on that line read
    # every fault as "not the declared function" (corpus-run3, 2026-10-27: zero CAUGHT on 66
    # cases). verdicts.Symbols walks nm's sized function symbols, which is what Fault.symbol
    # was always meant to carry; the symbolizer supplies only the line table's file:line.
    func, _ = symbols_of(llvm_bin, image).lookup(fault.pc, code)
    func = func or ""
    out = subprocess.run([str(Path(llvm_bin) / "llvm-symbolizer"), "--obj", str(image),
                          "--functions=none", f"{static:#x}"],
                         capture_output=True, text=True).stdout.splitlines()
    where = out[0].strip() if out else ""
    m2 = re.match(r"(.+?):(\d+):\d+$", where)
    if not m2:
        return func, "", 0
    # Returned WHOLE and compared by suffix, not trimmed against a list of known build prefixes.
    # The symbolizer's path depends on where the tree was mirrored and on whether a file came
    # from the read-only corpus mount, and a prefix list silently stops matching when either
    # changes -- which turns a catch into `unattributed` for a reason that has nothing to do
    # with the program. ASAN's own site is repo-relative, so a suffix match is the comparison
    # that is actually being asked for.
    return func, m2.group(1), int(m2.group(2))


def arm_controls(arm, sanity_ok, probe_ok, arms):
    """The Control list for an arm, naming only what arms.json declares for it."""
    declared = arms[arm]["controls"]
    out = [verdicts.Control(name=SANITY, observed="complete" if sanity_ok else "did-not-complete")]
    if BOUND_PROBE in declared:
        out.append(verdicts.Control(name=BOUND_PROBE,
                                    observed="fault" if probe_ok else "did-not-fault"))
    return out


def observe(case, spec, arm, image_sha, text, exit_code, controls, stock_reached, llvm_bin,
            image, precondition_met=True, score_oracle_crashes=False, protective_causes=(25, 28),
            raw=None):
    """One case on one arm, as an Observation. `text` is the application's own output
    (app_output); `raw` the transcript before that filter, for the launcher's counters, which
    the filter removes -- read from `text` they were never found, so the node-pressure note
    below reached no bundle until 2026-10-27."""
    obs = verdicts.Observation(case=case, arm=arm, image_sha256=image_sha, controls=controls)
    native = spec["pristine_outcome"]
    note(obs, f"native PHP: {native}")
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
    #
    # --score-oracle-crashes LIFTS THIS, and the reason to want it is narrow but real: a native
    # SIGNAL lands wherever the reused memory happened to break, while a capability fault lands
    # at the offending access. "Turns a nondeterministic crash into a localised fault" is a
    # different claim from "turns silence into a fault", and it is one these cases CAN support.
    #
    # What it does NOT do is grant the cell a catch. Attribution still has to hold, and the
    # stock arm still faults on these by construction -- so the honest use of this scoring is
    # the per-arm ATTRIBUTION rate, not the headline count, and the two scorings must be
    # reported side by side. Default off, so the published denominator is unchanged.
    if oracle_crashes(spec) and not score_oracle_crashes:
        obs.infra = "out-of-denominator"
        obs.notes = (f"the corpus's pristine record for this case is "
                     f"{spec['pristine_outcome']!r}: native PHP dies on a signal, "
                     f"so there is no pair to read")
        return obs
    if oracle_crashes(spec):
        obs.notes = (f"SCORED WITH THE ORACLE CRASHING: the corpus's pristine record is "
                     f"{spec['pristine_outcome']!r}, so native PHP dies on this case too and a "
                     f"fault here is NOT evidence that the protection found something plain "
                     f"execution missed. Read the fault SITE, not the verdict.")
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
    pressure = node_pressure(raw if raw is not None else text)
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
    # ATTRIBUTION FIRST, because reach can rest on it. A fault is this defect's when it lands
    # in the ONE function the case declares (case.json fault_sites: ASan's first PHP frame in
    # our native run, chosen from that run and never from a fault here) AND in the file that
    # frame was seen in, AND with a cause the configuration declares protective. Until
    # 2026-10-27 the rule was "any line in ASan's file", which the second run's reading of the
    # 2026-10-26 corpus showed to be wider than its evidence: zend_execute.c is on every
    # script's path, and a fault anywhere in it read as the case's catch. Now the function
    # has to match, and the evidence names which frame it is.
    # Two facts are read off a fault, and they license different things. SITE: the fault is
    # in the declared function, in its file -- the trigger ran, whatever the cause, so this
    # observation is `reached`. QUALIFIED: site plus a protective cause (and not the seam's
    # own source, not the allocator) -- the mechanism caught the defect, so the fault is
    # attributed. A cause-24 fault at the site (CRASH-064) is reached and unattributed, which
    # is what happened: the program got to its defect and dereferenced its own bad value.
    at_site = False
    if fault is not None:
        obs.fault = fault
        func, path, line = symbolize(llvm_bin, image, fault, text)
        fault.symbol = func or None
        site = next((f for f in spec.get("asan_stack", []) if f.get("func") == spec["asan_func"]), None)
        site_file = (site or {}).get("file") or spec["asan_file"]
        leaf = (spec.get("asan_stack") or [{}])[0]
        at_site = bool(func) and func in spec["fault_sites"] and path.endswith(site_file)
        if any(own in path for own in SEAM_OWN):
            obs.attribution_evidence = (f"the fault is in the seam's own source ({path}:{line}) "
                                        f"-- a port artefact, not the program")
        elif spec["asan_file"].endswith(ALLOCATOR_SITE):
            obs.attribution_evidence = (
                f"ASAN's own site for this case is {ALLOCATOR_SITE}:{spec['asan_line']}, so a "
                f"fault there cannot be told from the port breaking; it needs the diagnostic "
                f"audit arm (observed {func or '?'} at {path}:{line})")
        elif fault.cause not in protective_causes:
            # The configuration says which causes its mechanism produces (arms.json
            # protective_causes). Cause 24 -- an untagged value used as a capability -- is the
            # program dereferencing its own corrupted pointer, the same value native PHP dies
            # on; CRASH-064 carried the identical 0x3f00000001 on every arm. Its SITE may well
            # be the declared one; it is still not the mechanism catching anything.
            obs.attribution_evidence = (
                f"cause {fault.cause} in {func or '?'} at {path or '?'}:{line} is not one of this "
                f"configuration's protective causes {list(protective_causes)}: an untagged or "
                f"otherwise invalid value the program itself formed, not a revoked (25) or "
                f"out-of-bounds (28) access"
                + (" -- and it did land in the declared function" if func in spec["fault_sites"] else ""))
        elif func in spec["fault_sites"] and path.endswith(site_file):
            obs.attribution = "function"
            obs.attribution_evidence = (
                f"{func} at {site_file}:{line} is the function this case declares (fault_sites, "
                f"ASan frame #{(site or {}).get('index', 0)} of our native run)"
                + (", at ASan's own line" if line == spec["asan_line"] else
                   f" (ASan: line {spec['asan_line']}; a capability fault can land on a "
                   f"neighbouring instruction)"))
        elif func in spec["fault_sites"]:
            obs.attribution_evidence = (
                f"{func} is the declared function but the fault is at {path or '?'}:{line}, not in "
                f"{site_file}: a same-named routine elsewhere, or no line table for it")
        elif path.endswith(site_file):
            obs.attribution_evidence = (
                f"{path}:{line} is ASan's file for this case but the fault is in {func or '?'}, not "
                f"the declared {spec['fault_sites']}: another routine of the same file, which "
                f"the old any-line-in-the-file rule would have admitted")
        elif not path and leaf.get("outside") and func and func == leaf.get("func"):
            # A libc leaf: the fault is in the routine ASan's own frame #0 names (strlen under
            # xbuf_format_converter, CRASH-068). That is consistent with the defect and proves
            # nothing: strlen is on many paths, and whether THIS call came from the declared
            # caller needs the return address, which the launcher does not report
            # (runtime/virtual/wire.h cv_step carries cause, pc and address only). Unresolved,
            # said so, and not promoted.
            obs.attribution_evidence = (
                f"the fault is in {func}, the libc routine ASan's own frame #0 names for this "
                f"case, with no line table; whether this call came from the declared "
                f"{spec['fault_sites']} is not observable without a return address -- unresolved")
        else:
            obs.attribution_evidence = (f"faulted in {func or '?'} at {path or '?'}:{line} but the "
                                        f"declared site is {spec['fault_sites']} in {site_file}")

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
    # The second route: a fault in THIS run that the rule above attributed -- the declared
    # function, in its file, with a protective cause -- cannot have happened without the
    # trigger running. It licenses THIS observation only. Until 2026-10-27 a site hit on ANY
    # arm licensed reach on every arm of the case, which let a silence elsewhere read as
    # MISSED on the strength of another configuration's fault; that transfer is gone.
    obs.reached = at_site or (bool(stock_reached) and bool(precondition_met))
    if at_site:
        obs.reach_evidence = (f"this run faulted in {spec['fault_sites'][0]}, the case's own "
                              f"declared site, which cannot be reached without the trigger "
                              f"having run")
    elif obs.reached:
        obs.reach_evidence = ("the stock arm ran this case to its normal end, the corpus's "
                              "pristine record says it completes, and revocation-sanity "
                              "completed on this arm")
    else:
        obs.reach_evidence = ""
    if stock_reached and not precondition_met:
        # A case with a declared EXECUTION precondition that nothing in this measurement
        # confirmed was met. CRASH-092 is the shape of the problem: it creates its own loopback
        # listener and then echoes "done" WHETHER OR NOT the listener came up, so a silence here
        # cannot be told apart from a platform that caught nothing. Refusing it is the only
        # honest option; a MISSED would be a claim about revocation drawn from a socket.
        note(obs, f"declared precondition {spec['requires']!r} was not confirmed met in this "
                  f"run, so a silence here is not a reading")
    if fault is not None:
        return obs

    # normalise_paths, for the SAME reason the stock-arm comparison below applies it: PHP prints
    # the script's absolute path inside its own warnings and fatal errors, and the corpus's host
    # path and the guest's /mnt/vm path are both correct. Comparing them literally reports a
    # divergence on every such case -- "2 lines vs 2", differing only in the path.
    if said.strip() and not same_as_oracle(normalise_paths(said, case, spec.get("id")),
                                           normalise_paths(spec["pristine_stdout"], case, spec.get("id"))):
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
        src = args.corpus / case / "trigger.php"
        if not src.is_file():
            sys.exit(f"no case file at {src}")
        (args.scratch / f"{case}.php").write_bytes(src.read_bytes())
    (args.scratch / SANITY_SCRIPT).write_text(SANITY_TEXT)


# The sanity workload must allocate and free heavily through emalloc/efree and contain NO
# defect, so that "revocation is sound on a clean workload" is what it actually measures. A
# module constant, so a smoke run outside this runner can execute the very same script.
SANITY_TEXT = (
        "<?php\n"
        "// revocation-sanity: a benign workload. Every arm must run this to its normal end;\n"
        "// an arm that cannot is not a column, and a silence on it is not a MISSED.\n"
        "// It sorts, three ways, with a comparator that touches nothing: PHP's own\n"
        "// _zend_qsort_swap moved Bucket* elements through int and char stores, which\n"
        "// strips their tags, so every sort faulted at its first comparison (patch 0002).\n"
        "// Before this control existed, CRASH-022 and CRASH-078 read that fault as a catch.\n"
        "function phpb_cmp($x, $y) { return ($x < $y) ? -1 : 1; }\n"
        "$k = array(3 => 'c', 1 => 'a', 2 => 'b'); ksort($k);\n"
        "$w = array(5, 3, 8, 1, 9, 2, 7, 4, 6, 0); sort($w); usort($w, 'phpb_cmp');\n"
        "$a = array();\n"
        "for ($i = 0; $i < 400; $i++) { $a[] = str_repeat('x', ($i % 23) + 1); }\n"
        "for ($i = 0; $i < 400; $i += 2) { unset($a[$i]); }\n"
        "$s = '';\n"
        "foreach ($a as $v) { $s .= substr($v, 0, 3); }\n"
        "$u = unserialize(serialize(array('k' => $a, 'n' => 17)));\n"
        "echo 'sanity ', strlen($s), ' ', count($u['k']), ' ', $u['n'], ' ',\n"
        "     implode('', $k), ' ', implode(',', $w), \"\\n\";\n"
)


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
    # BY ENCODING, NOT BY MNEMONIC: objdump prints CDERIVE and CREVOKE as <unknown>, so a
    # mnemonic count reads zero on a CORRECT image. BOTH are required -- an image that can issue
    # but not revoke is exactly the shape that would report a clean sweep with nothing revoked.
    derives_in_image, revokes_in_image = cap_insn_counter()(io.StringIO(dis))
    if revokes_in_image == 0 or derives_in_image == 0:
        problems.append(f"the image cannot issue or cannot revoke (derive={derives_in_image} "
                        f"revoke={revokes_in_image}), so no arm could revoke anything and all "
                        "three temporal columns would read alike")
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
    return problems, derives_in_image, revokes_in_image


def sha256_file(path):
    """sha256 of a file on this host, for the platform record. A bundle that names its inputs
    by path says nothing about what those files contained when it ran."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def platform_record(args):
    """Every platform binary by sha256, with its path for the reader. A file that is absent is
    said to be absent -- an omitted key would read as an input that did not matter."""
    def binary(path):
        path = Path(path)
        return f"{sha256_file(path)}  {path}" if path.exists() else f"MISSING:{path}"
    virtual = run_virtual_module()
    return {
        "compiler": subprocess.run([str(args.llvm_bin / "clang"), "--version"],
                                   capture_output=True, text=True).stdout.splitlines()[0],
        "qemu": binary(args.qemu),
        "firmware": binary(args.images / "fw_jump.elf"),
        "kernel": binary(args.images / "Image"),
        "rootfs": binary(args.images / "rootfs.ext2"),
        "launcher": binary(args.adapter / "capstone-vexec"),
        "module": binary(args.adapter / "module" / "capstone_vm.ko"),
        # The two platform PROPERTIES, not just the binaries. runtime/virtual/run.py records
        # paths and hashes but deliberately not argv, and these properties decide whether the
        # platform runs at all: without x-capstone-test-csrs the paged module's one read of CSR
        # 0xcc0 is an illegal instruction, and at the default supervisor-slots=1 every thread
        # and shared-heap check fails. Two runs differing only in these are NOT equivalent.
        "qemu_cpu": virtual.QEMU_CPU,
        "module_args": f"node_initial_pages={virtual.NODE_PAGES} {virtual.MODULE_ARGS}",
        "runner": "ports/php/app/run-realphp.py",
    }


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
    p.add_argument("--score-oracle-crashes", action="store_true",
                   help="also score the cases whose pristine record is a SIGNAL -- native PHP "
                        "dies on them too, so a fault is NOT evidence the protection found "
                        "something plain execution missed; read the fault SITE, and report this "
                        "scoring BESIDE the default one, never instead of it")
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
    # NAME THE POPULATION THAT IS ACTUALLY RUNNING. This said "ASAN-blind" whichever set was
    # selected, so a temporal-asan-visible run announced itself as its own complement -- and a
    # reader checking which population a log belongs to got the wrong answer from the one line
    # that looks authoritative.
    print(f"population: {len(cases)} {args.population} cases, "
          "derived from the corpus's own definition")

    image_sha = subprocess.run(["sha256sum", str(args.image)], capture_output=True,
                               text=True).stdout.split()[0]
    # The bound probe is a DIFFERENT image on purpose -- probe-seam.dom, built from the seam's
    # sources without PHP -- so its transcript reports its own sha256 and must be checked
    # against its own file. Exempting it instead would make the guard unable to notice a stale
    # probe transcript, which is the one run in the session whose image can change on its own.
    probe_sha = subprocess.run(["sha256sum", str(args.probe)], capture_output=True,
                               text=True).stdout.split()[0]
    # THE PLATFORM RECORD IS TAKEN BEFORE THE FIRST BOOT, and by content. It used to be
    # written afterwards and as PATHS, which named files on this host and said nothing about
    # what they held: the kernel under build-qemu/images was built from a buildroot checkout
    # that is not the pin the superproject records, so no bundle written that way can say
    # which kernel it ran on. rootfs.ext2 in particular is WRITTEN BY THE GUEST, so a hash
    # taken after the arms have run is not the filesystem that was booted.
    # RE-JUDGING KEEPS THE RUN'S PROVENANCE. A --from-record pass judges transcripts a boot
    # produced earlier; hashing this host's binaries at re-judge time would stamp the bundle
    # with whatever is on disk NOW, which the 2026-10-27 sensitivity re-scoring did. The
    # record of the platform is written beside the transcripts by the pass that booted them,
    # and a re-judge reads that or refuses.
    platform_file = args.out / "platform.json"
    if args.from_record:
        if not platform_file.is_file():
            sys.exit(f"--from-record: no {platform_file} beside the stored runs; the platform "
                     f"that produced them is not known and cannot be hashed from this host")
        platform = json.loads(platform_file.read_text())
    else:
        platform = platform_record(args)
        args.out.mkdir(parents=True, exist_ok=True)
        platform_file.write_text(json.dumps(platform, indent=2, sort_keys=True) + "\n")
        stage_inputs(args, cases)
    for arm in ARMS:
        if not arms[arm].get("protective_causes"):
            sys.exit(f"{arm}: tools/arms.json declares no protective_causes for this "
                     f"configuration; the judge cannot tell a revocation from a corrupted pointer")

    per_arm, texts, raws = {}, {}, {}
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
        # THE ARM IS NAMED, NEVER DEFAULTED. The common application build bakes an arm into its
        # image (ZEND_CAP_DEFAULT_MODE) for the generic runners; such an image says
        # ZEND_CAP_MODE_DEFAULTED when it takes that default. A measurement run that shows that
        # line in ANY transcript did not name its arm and is refused, stored records included.
        # And every run that reports a mode must report the arm's: a transcript at another mode
        # is not this arm's reading, whatever the sanity run said.
        for name, lines in rec["output"].items():
            joined = "\n".join(lines)
            if "ZEND_CAP_MODE_DEFAULTED" in joined:
                sys.exit(f"{arm}/{name}: the image took a DEFAULT arm (ZEND_CAP_MODE_DEFAULTED); "
                         f"a measurement names its arm, so this image cannot be measured")
            # The bound probe is its own image and sets its own modes (rounded, then exact, each
            # reading index 9), so its mode lines are the probe's, never this arm's.
            seen = MODE_LINE.search(joined)
            if name != BOUND_PROBE and seen and int(seen.group(1)) != mode:
                sys.exit(f"{arm}/{name}: the transcript reports PHP_CAP_MODE {seen.group(1)}, "
                         f"not this arm's {mode}")
        texts[arm] = {n: app_output(l) for n, l in rec["output"].items()}
        raws[arm] = {n: "\n".join(l) for n, l in rec["output"].items()}
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

    problems, derives_in_image, revokes_in_image = cross_arm_gates(args, per_arm, args.llvm_bin)
    print(f"\ncross-arm gates: image capability instructions derive={derives_in_image} revoke={revokes_in_image}")
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
        elif ran and not same_as_oracle(normalise_paths(said, case, spec.get("id")),
                                        normalise_paths(spec["pristine_stdout"], case, spec.get("id"))):
            output_diverged[case] = (said, spec["pristine_stdout"])
            print(f"    {case}: ran, but its output differs from the oracle's "
                  f"({len(said.splitlines())} lines vs "
                  f"{len(spec['pristine_stdout'].splitlines())})")

    # DECLARED REQUIREMENTS, by kind (oracle.json requires_kind, classified by the corpus
    # builder from the declaration's text). A DETECTOR OPTION -- CRASH-052's "ASAN_OPTIONS
    # detect_stack_use_after_return=1" -- is what ASan needs to report the defect, not what
    # the trigger needs to run here; until 2026-10-27 it was treated as an unmet precondition
    # and the case refused on every arm, which is unsound: Capstone not having enabled an
    # ASan option says nothing about reach. It is recorded as not applicable, not as "met".
    # An EXECUTION requirement (a writable /tmp, a working loopback) is confirmed by nothing
    # this run can see: CRASH-092 prints its marker whether or not its listener came up, so
    # the stock arm completing with the oracle's output is not evidence the requirement held.
    # A fault on some arm at the case's site used to confirm it for every arm of the case;
    # that cross-arm transfer is gone (see observe). An arm's own attributed fault licenses
    # that arm, and a silence on any arm of such a case is not a reading.
    precondition_met = {}
    for case, spec in cases.items():
        kind = spec["requires_kind"]
        if not kind:
            precondition_met[case] = True
        elif kind == "detector-option":
            precondition_met[case] = True
            print(f"    {case}: requirement {spec['requires']!r} is a detector option, not a "
                  f"precondition on this platform -- not applicable")
        else:
            precondition_met[case] = False
            print(f"    {case}: precondition {spec['requires']!r} cannot be confirmed by this "
                  f"run -- a silence on this case is not a reading; an arm's own fault in the "
                  f"declared function licenses that arm alone")

    rows, table = {arm: [] for arm in ARMS}, {}
    for arm in ARMS:
        controls = arm_controls(arm, per_arm[arm]["sanity_ok"], per_arm[arm]["probe_ok"], arms)
        for case, spec in cases.items():
            text = texts[arm].get(case, "")
            obs = observe(case, spec, arm, image_sha, text,
                          per_arm[arm]["rec"]["exits"].get(case),
                          controls, stock_reached[case], args.llvm_bin, args.image,
                          precondition_met[case], args.score_oracle_crashes,
                          tuple(arms[arm]["protective_causes"]), raws[arm].get(case))
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

    # One bundle per arm, named by the FRAMEWORK arm (what corpus.json's verdict_bundles and
    # derive-verdicts.py key on) and recording the arms.json configuration it measured. The
    # stock arm is the judge's control, not a case arm; its bundle keeps its own name.
    for arm in ARMS:
        verdicts.write_bundle(args.out / f"bundle-{FRAMEWORK_ARM[arm]}", "php/engine-repros",
                              FRAMEWORK_ARM[arm], rows[arm],
                              {"platform": platform, "configuration": arm,
                               "build": {"application": "php", "profile": "virtual",
                                         "image_sha256": image_sha, "allocator": "seam",
                                         "mode": arms[arm]["mode"]},
                               "pre_registration": args.prereg,
                               "population": args.population,
                               # How the rows were scored, so a reader of the bundle alone can
                               # tell a default run from a sensitivity one, and which
                               # attribution rule judged it.
                               "scoring": {
                                   "score_oracle_crashes": bool(args.score_oracle_crashes),
                                   "attribution": ("declared function (case.json fault_sites) in "
                                                   "its file, with a protective cause "
                                                   "(arms.json protective_causes); own fault "
                                                   "licenses own reach; 2026-10-27"),
                                   "protective_causes": arms[arm]["protective_causes"]}})
    print(f"\nbundles written under {args.out}")


if __name__ == "__main__":
    main()
