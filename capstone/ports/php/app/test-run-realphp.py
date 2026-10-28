#!/usr/bin/env python3
"""Offline tests for run-realphp.py. No guest, no toolchain, no corpus.

WHY THIS EXISTS. The previous session left two NameErrors in a runner's own code; the first cost
a 15-minute boot to find. Every boot here is four boots and ~40 minutes, so main()'s wiring is
driven end to end against synthetic transcripts before anything is launched.

AND WHY EACH GATE IS SHOWN FIRING. A gate that has never failed is not known to work. The
cross-arm checks below are each driven to FAIL on purpose and asserted to refuse the run.
"""
import importlib.util
import io, json, os, re, subprocess, sys, tempfile, types
from pathlib import Path

# Case directories as bug-corpora/php/engine-repros names them: NN_<fix>_<id>. The runner keys
# cases by that name (it is what derive-verdicts.py keys case.json arms by); the suite keeps the
# corpus id in its own vocabulary and maps through K().
NUM = {"CRASH-057": 1, "CRASH-062": 2, "CRASH-086": 3, "CRASH-092": 4, "CRASH-110": 5, "CRASH-999": 6,
       "CRASH-052": 7}
def K(cid):
    return f"{NUM.get(cid, 99):02d}_fix_{cid.lower()}"

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("runreal", HERE / "run-realphp.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
sys.path.insert(0, str(HERE.parents[3] / "capstone/bug-corpora/tools"))
import verdicts

FAILURES = []


def check(name, got, want):
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {name}")


FAULT = ("capstone-exec: domain fault cause=25 pc=0x3f904e7f90 address=0x3f909c8074 "
         "entry=0x3f904009a0 code=0x3f90400000-0x3f90800000 last=0x39 preparing=0x39 "
         "sha256=%s image=php-real-nested.dom")
# The same fault line with the RESOURCE cause: the node table ran out, which lands at an
# arbitrary pc and must never be read as a catch.
FAULT30 = FAULT.replace("cause=25", "cause=30")
# ...and with the TAG cause: an untagged value used as a capability. That is the program
# dereferencing its own corrupted pointer, not the mechanism catching anything, and arms.json
# does not list it among the PHP configurations' protective causes.
FAULT24 = FAULT.replace("cause=25", "cause=24")
# ...and with the return address and frame pointer the launcher prints since 2026-10-27, as the
# escape left them in the frame's x1 and x8 slots. The pc here is musl's strlen in the real
# image; the ra is inside xbuf_format_converter.
FAULT_RA = FAULT.replace("pc=0x3f904e7f90 address=0x3f909c8074",
                         "pc=0x3f90411888 address=0x3f90a0ba20 ra=0x3f90508d7c fp=0x3f80a01f00")
# ...and with the cause the ADDRESS POLICY raises (QEMU pin 49a225e408): a control transfer to
# a refused address. Not a protective cause either.
FAULT29 = FAULT.replace("cause=25", "cause=29")
# The bound probe's own fault: the EXACT read past a 9-byte request, the BOUNDS cause.
PROBE_FAULT = FAULT.replace("cause=25", "cause=28")
PROBE_ROUNDED = "S7 rounded arm read index 9 = 0x00 -- the slack is reachable, as stock PHP"
PROBE_EXACT = "S7 exact arm now reading index 9 (expect a fault)"
PROBE_NO_FAULT = ("S7 RESULT: NO FAULT -- read 0x00 past a 9-byte request on the EXACT arm. "
                  "The bound is not being applied.")
# The probe's own report of the exact capability, printed before the exact read, as the real
# transcript carries it (run5: `exact    cursor=0x3faa5fe080 bounds=[0x3faa5fe080,0x3faa5fe089) tag=1`
# before a fault at address=0x3faa5fe089). The fixture fault's address, 0x3f909c8074, is this end.
PROBE_EXACT_REPORT = "exact    cursor=0x3f909c806b bounds=[0x3f909c806b,0x3f909c8074) tag=1"


def probe_transcript(shape, probe_sha=None):
    """The bound probe's transcript and exit in each shape the control must tell apart.

    Only "declared" is the bound firing: both reads announced, then cause 28. Every other
    shape is a probe that did something else -- read past the bound, faulted with another
    cause on the free/re-allocate/report path between the reads, or faulted before the exact
    read was announced -- and must read as the control NOT met."""
    probe_sha = probe_sha or PROBE_SHA
    f = PROBE_FAULT % probe_sha
    # The probe reports the exact capability before it reads: the fixture fault's address,
    # 0x3f909c8074, is that capability's end, cursor + 9.
    R, X = PROBE_ROUNDED, PROBE_EXACT
    E = PROBE_EXACT_REPORT
    shapes = {
        "declared": ([R, E, X, f], 139),
        "no-fault": ([R, E, X, PROBE_NO_FAULT], 1),
        "cause-24": ([R, E, X, FAULT24 % probe_sha], 139),
        "cause-25": ([R, E, X, FAULT % probe_sha], 139),
        "cause-29": ([R, E, X, FAULT29 % probe_sha], 139),
        "no-exact-line": ([R, E, f], 139),
        "no-rounded-line": ([E, X, f], 139),
        "fault-before-exact": ([R, E, f, X], 139),
        "rounded-after-fault": ([E, X, f, R], 139),
        # cause 28, both lines in order, but the pc is not the exact read in main(): in
        # report() (the probe's own bounds printer, static 0xf7e10 under the test's symbol
        # table), and outside the image altogether.
        "fault-in-report": ([R, E, X, f.replace("pc=0x3f904e7f90", "pc=0x3f904e7e10")], 139),
        "pc-outside-image": ([R, E, X, f.replace("pc=0x3f904e7f90", "pc=0x1234")], 139),
        # cause 28 in main() after both lines, but not the byte the exact capability ends at:
        # another load in main() faulting on another address, under the same symbol.
        "wrong-address": ([R, E, X, f.replace("address=0x3f909c8074", "address=0x3f909c8070")], 139),
        # the probe never reported the exact capability, so there is no end to check against
        "no-exact-report": ([R, X, f], 139),
        # the declared shape followed by a second fault line: a transcript that contradicts
        # itself is not the bound firing once
        "second-fault": ([R, E, X, f, FAULT29 % probe_sha], 139),
    }
    return shapes[shape]
SHA = "a" * 64
# The bound probe is a DIFFERENT image by design, so its transcript carries its own
# sha and the guard has to know that without being blinded to a stale probe.
PROBE_SHA = "b" * 64
STATS = ("CAPSTONE_VM_STATS arenas=16 pages=1778 peak=1994 nodes=%d steps=1943 faults=664 "
         "collections=0 reclaimed=0 rounds=54 bytes_in=544 bytes_out=21229 node_bytes=225280 "
         "node_capacity=%d node_growths=3 launch_ns=1 elapsed_ns=2")

SITE = "user_tick_function_call"
SPEC = {"cwe": "CWE-416", "requires": "", "requires_kind": "", "asan_func": SITE,
        "asan_file": "ext/standard/basic_functions.c", "asan_line": 2135,
        "asan_stack": [{"index": 0, "func": SITE, "file": "ext/standard/basic_functions.c", "line": 2135},
                       {"index": 1, "func": "execute", "file": "Zend/zend_execute.c", "line": 1391}],
        "fault_sites": [SITE], "site_files": {SITE: "ext/standard/basic_functions.c"},
        "pristine_stdout": "done\n", "pristine_outcome": "OK"}
# A case whose ASan leaf is a libc routine (CRASH-068's shape: strlen under
# xbuf_format_converter). The declared site is the PHP caller; the leaf is named in the stack
# and deliberately not declared.
LEAF_SPEC = dict(SPEC, asan_func="xbuf_format_converter", asan_file="main/spprintf.c", asan_line=518,
                 asan_stack=[{"index": 0, "func": "strlen", "outside": "sanitizer_common_interceptors.inc:425"},
                             {"index": 1, "func": "xbuf_format_converter", "file": "main/spprintf.c", "line": 518}],
                 fault_sites=["xbuf_format_converter"], site_files={"xbuf_format_converter": "main/spprintf.c"})


def fake_symbolize(path, line, func=SITE):
    """What the symbolizer would say: (function, file, line). The default function is the
    fixture case's declared site, so a test that only moves the FILE tests the file rule."""
    return lambda llvm_bin, image, fault, text: (func, path, line)


def controls(sanity="complete", probe="fault"):
    """The declared controls. probe=None for a temporal arm, which declares no bound probe --
    the judge RAISES on a control an arm does not declare, which is the behaviour that keeps a
    control from being borrowed from an arm that has one."""
    out = [verdicts.Control(name=R.SANITY, observed=sanity)]
    if probe is not None:
        out.append(verdicts.Control(name=R.BOUND_PROBE, observed=probe))
    return out


def observation_cases():
    print("\n-- observe() and the judge --")
    arms = verdicts.load_arms()
    exact = arms["php-real-exact"]
    pools = arms["php-real-object"]
    orig = R.symbolize

    # A fault at ASAN's own file AND line, with the trigger shown live: CAUGHT.
    R.symbolize = fake_symbolize("ext/standard/basic_functions.c", 2135)
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA + "\n" + STATS % (100, 9216),
                    139, controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("caught at ASAN's own line", (v, reason), (verdicts.CAUGHT, None))
    check("the evidence says 'ASan's own line'", "ASan's own line" in ev, True)

    # A neighbouring line in the same file is still the catch, and says so.
    R.symbolize = fake_symbolize("ext/standard/basic_functions.c", 2137)
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    v, _, ev = verdicts.judge(obs, exact)
    check("caught at a neighbouring line", v, verdicts.CAUGHT)
    check("the evidence names ASan's line", "ASan: line 2135" in ev, True)

    # A fault in a DIFFERENT file is not this case's defect.
    R.symbolize = fake_symbolize("Zend/zend_hash.c", 400)
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    check("wrong file is unattributed", verdicts.judge(obs, exact)[1], "unattributed")

    # A fault in the nested layer's OWN source is a port artefact, never a catch.
    R.symbolize = fake_symbolize("capstone/ports/php/zend-alloc/sublet/zend-cap-mode.c", 120)
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("a fault in the nested layer is unattributed", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and says it is a port artefact", "port artefact" in ev, True)

    # A case whose ASAN site IS the allocator cannot be attributed by site.
    alloc_spec = dict(SPEC, asan_file="Zend/zend_alloc.c", asan_line=403)
    R.symbolize = fake_symbolize("Zend/zend_alloc.c", 403)
    obs = R.observe("CRASH-073", alloc_spec, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("allocator-site case is unattributed", reason, "unattributed")
    check("and asks for the audit arm", "audit arm" in ev, True)

    # Faulted somewhere else, and the trigger was never shown to run: setup-fault, NOT a
    # catch and not `unattributed` either -- there is no reading to attribute. (A fault IN
    # the declared function is its own proof of reach; that case is below.)
    R.symbolize = fake_symbolize("Zend/zend_hash.c", 400, func="zend_hash_find")
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), False, "/x", "/y")
    check("fault elsewhere without reach is a setup-fault", verdicts.judge(obs, exact)[1], "setup-fault")
    R.symbolize = fake_symbolize("ext/standard/basic_functions.c", 2135)

    # Silent, trigger live, controls held: MISSED is a real reading.
    obs = R.observe("CRASH-086", SPEC, "php-real-stock", SHA, "done\n" + STATS % (100, 9216),
                    0, [verdicts.Control(name=R.SANITY, observed="complete")], True, "/x", "/y")
    check("silence with controls is MISSED", verdicts.judge(obs, arms["php-real-stock"])[0],
          verdicts.MISSED)

    # Silent, but the arm's own control did NOT hold: not a reading at all.
    obs = R.observe("CRASH-086", SPEC, "php-real-stock", SHA, "done\n", 0,
                    [verdicts.Control(name=R.SANITY, observed="did-not-complete")], True, "/x", "/y")
    check("silence on a broken arm is control-failed",
          verdicts.judge(obs, arms["php-real-stock"])[1], "control-failed")

    # The node table ran out. The fault SAYS so, on the line the pc comes from, so this is
    # keyed on the cause and not on a threshold over a counter -- see CAUSE_RESOURCE on why
    # the counter the pre-registration named is neither printed nor comparable to capacity.
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA,
                    FAULT30 % SHA + "\n" + STATS % (9000, 9216), 139, controls(), True, "/x", "/y")
    v, reason, _ = verdicts.judge(obs, exact)
    check("a cause-30 resource fault is refused", (v, reason), (verdicts.NO_READING, "infra"))

    # AND THE COUNTERS DO NOT REFUSE ANYTHING ON THEIR OWN. nodes= is cumulative, so it exceeds
    # the capacity on healthy runs; the old threshold fired on about twenty good runs of this
    # measurement, catches included.
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA,
                    FAULT % SHA + "\n" + STATS % (13480, 13312), 139, controls(), True, "/x", "/y")
    check("nodes over capacity is reported, not gated",
          verdicts.judge(obs, exact)[0], verdicts.CAUGHT)
    check("and the counters are in the notes", "cumulative, not a high-water" in obs.notes, True)

    # The nested layer itself failed: an instrument failure, never a verdict.
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA,
                    "ZEND_CAP_FAIL code=2 the heap could not grow its node table\n", 139, controls(), True, "/x", "/y")
    check("a nested layer failure is a setup-fault", verdicts.judge(obs, exact)[1], "setup-fault")
    # The nested layer refused the pointer efree was given (code 3: in no block; code 4: not at its
    # start): the program's invalid free, named, but nothing ties it to the declared site.
    for code in (3, 4):
        obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA,
                        f"ZEND_CAP_FAIL code={code} free of a pointer this heap did not issue a=0x40 b=0x0\n",
                        134, controls(), True, "/x", "/y")
        check(f"a named refusal (code {code}) is unattributed, not infra",
              verdicts.judge(obs, exact)[:2], (verdicts.NO_READING, "unattributed"))

    # A declared PLATFORM precondition (CRASH-084: putenv keeps an empty-name string in environ).
    # The probe's reading decides, never the declaration alone.
    present = ("PUTENV_PROBE control retained=1 rc=0 errno=0 (ok)\nPUTENV_PROBE empty-name retained=1 rc=0 errno=0 (ok)\n"
               "PUTENV_PROBE empty-name-value retained=1 rc=0 errno=0 (ok)\nPUTENV_PROBE DONE failures=0\n")
    absent = ("PUTENV_PROBE control retained=1 rc=0 errno=0 (ok)\nPUTENV_PROBE empty-name retained=0 rc=-1 errno=22 (Invalid argument)\n"
              "PUTENV_PROBE empty-name-value retained=0 rc=-1 errno=22 (Invalid argument)\nPUTENV_PROBE DONE failures=0\n")
    broken = ("PUTENV_PROBE control retained=0 rc=0 errno=0 (ok)\nPUTENV_PROBE FAIL the control variable was not retained\n"
              "PUTENV_PROBE empty-name retained=0 rc=-1 errno=22 (Invalid argument)\nPUTENV_PROBE DONE failures=1\n")
    check("probe: retained reads present", R.putenv_probe_outcome(present), True)
    check("probe: not retained with the control retained reads absent", R.putenv_probe_outcome(absent), False)
    check("probe: a failed control reads as no result", R.putenv_probe_outcome(broken), None)
    check("probe: no output reads as no result", R.putenv_probe_outcome(""), None)
    # ABSENT is a whole pattern, never "anything but retained": an unexpected failure of the
    # call (EFAULT, a retained=0 with rc 0, a missing second call) is a probe that did not
    # behave as declared and reads as no result (review 2026-10-28: every other result read
    # as absence, so a broken probe could take a case out of the denominator).
    efault = absent.replace("rc=-1 errno=22 (Invalid argument)", "rc=-1 errno=14 (Bad address)")
    check("probe: an unexpected errno is no result, not absence", R.putenv_probe_outcome(efault), None)
    rc0 = absent.replace("empty-name retained=0 rc=-1 errno=22", "empty-name retained=0 rc=0 errno=0")
    check("probe: not retained with rc 0 is no result", R.putenv_probe_outcome(rc0), None)
    one_call = "\n".join(l for l in absent.splitlines() if "empty-name-value" not in l) + "\n"
    check("probe: a missing second call is no result", R.putenv_probe_outcome(one_call), None)
    mixed = absent.replace("empty-name-value retained=0 rc=-1 errno=22 (Invalid argument)",
                           "empty-name-value retained=1 rc=0 errno=0 (ok)")
    check("probe: the two calls disagreeing is no result", R.putenv_probe_outcome(mixed), None)
    check("the runner knows the one probe by name (timeval retracted 2026-10-28)", sorted(R.PLATFORM_PROBES), ["putenv"])
    pp_spec = dict(SPEC, platform_precondition={"needs": "putenv keeps an empty-name string referenced from environ",
                                                "probe": "putenv", "present": "x", "absent": "y"})
    obs = R.observe("CRASH-084", pp_spec, "php-real-exact", SHA, "", 0, controls(), True, "/x", "/y",
                    platform_present=False)
    check("precondition proven absent: out of the denominator",
          verdicts.judge(obs, exact)[:2], (verdicts.NO_READING, "out-of-denominator"))
    check("and the note says the probe found it absent", "probe found it absent" in obs.notes, True)
    obs = R.observe("CRASH-084", pp_spec, "php-real-exact", SHA, "", 0, controls(), True, "/x", "/y",
                    platform_present=True)
    check("precondition present: judged normally (a completed run is MISSED)", verdicts.judge(obs, exact)[0], verdicts.MISSED)
    obs = R.observe("CRASH-084", pp_spec, "php-real-exact", SHA, "", 0, controls(), True, "/x", "/y",
                    platform_present=None)
    check("precondition unchecked: judged normally, with a note",
          (verdicts.judge(obs, exact)[0], "not checked" in obs.notes), (verdicts.MISSED, True))
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, "", 0, controls(), True, "/x", "/y",
                    platform_present=False)
    check("a case without the declaration ignores the probe", verdicts.judge(obs, exact)[0], verdicts.MISSED)
    # the precondition decides BEFORE the run's outcome: a watchdog kill (the hang the absent
    # behaviour causes, CRASH-092) is out of the denominator, not infra
    obs = R.observe("CRASH-092", pp_spec, "php-real-object", SHA, "VM_EXIT_TIMEOUT:30_x_crash-092\nKilled\n", 137,
                    controls(probe=None), False, "/x", "/y", platform_present=False)
    check("a watchdog kill with the precondition proven absent is out of the denominator",
          verdicts.judge(obs, pools)[:2], (verdicts.NO_READING, "out-of-denominator"))

    # Caught on the STOCK arm, so no control arm completed and stock_reached is False. The
    # fault in the declared function is itself the proof the trigger ran: without that route
    # this is `setup-fault` -- the instrument blamed for the platform's clearest result --
    # which is what CRASH-022 and CRASH-102 actually did.
    R.symbolize = fake_symbolize("ext/standard/basic_functions.c", 2135)
    obs = R.observe("CRASH-022", SPEC, "php-real-stock", SHA, FAULT % SHA, 139,
                    controls(probe=None), False, "/x", "/y")
    check("a fault in the declared function licenses its own reach",
          verdicts.judge(obs, arms["php-real-stock"])[0], verdicts.CAUGHT)
    check("and says which route licensed it",
          "own declared site" in obs.reach_evidence, True)

    # THE LICENCE IS THE OBSERVATION'S OWN, NEVER ANOTHER ARM'S. Until 2026-10-27 a site hit
    # on SOME arm licensed reach on every arm of the case, so a fault elsewhere on a second
    # arm read `unattributed` and a silence there read MISSED. Now a fault somewhere else,
    # with no completing control, is what it is: a fault before the trigger was shown to run.
    R.symbolize = fake_symbolize("ext/standard/url.c", 132, func="php_url_parse_ex")
    obs = R.observe("CRASH-022", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), False, "/x", "/y")
    check("another arm's catch does not license a fault elsewhere",
          verdicts.judge(obs, exact)[1], "setup-fault")
    obs = R.observe("CRASH-022", SPEC, "php-real-exact", SHA, "done\n", 0,
                    controls(), False, "/x", "/y")
    check("nor a silence", verdicts.judge(obs, exact)[1], "not-reached")
    R.symbolize = fake_symbolize("ext/standard/basic_functions.c", 2135)

    # QUALIFIED ATTRIBUTION: the declared FUNCTION in its file, not any line of the file.
    # Each of these is a shape the old rule admitted or the review asked to see refused.
    R.symbolize = fake_symbolize("ext/standard/basic_functions.c", 900, func="php_getenv")
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("another function in ASan's file is unattributed", (v, reason),
          (verdicts.NO_READING, "unattributed"))
    check("and the evidence says the old rule would have admitted it",
          "any-line-in-the-file" in ev, True)
    R.symbolize = fake_symbolize("Zend/zend_execute.c", 1391, func="execute")
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    check("a frame of ASan's stack that is not the declared site is unattributed",
          verdicts.judge(obs, exact)[1], "unattributed")
    R.symbolize = fake_symbolize("ext/standard/other.c", 2135)
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("the declared function in another file is unattributed", reason, "unattributed")
    check("and says so", "same-named routine elsewhere" in ev, True)
    # The tag cause: the program's own corrupted pointer, dereferenced. Site and function
    # match; it is still not the mechanism catching anything.
    R.symbolize = fake_symbolize("ext/standard/basic_functions.c", 2135)
    obs = R.observe("CRASH-064", SPEC, "php-real-exact", SHA, FAULT24 % SHA, 139,
                    controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("a cause-24 tag fault is unattributed even at the declared site", reason, "unattributed")
    check("and the evidence says it did land there", "did land in the declared function" in ev, True)
    check("the fault carries its symbol into the bundle", obs.fault.symbol, SITE)
    # ...and with no completing control it is still `unattributed`, not `setup-fault`: the
    # fault IS at the declared site, so the trigger ran; it is the cause that disqualifies it.
    obs = R.observe("CRASH-064", SPEC, "php-real-exact", SHA, FAULT24 % SHA, 139,
                    controls(), False, "/x", "/y")
    check("a cause-24 fault at the site reaches but does not attribute",
          verdicts.judge(obs, exact)[1], "unattributed")
    # The libc leaf: a fault in strlen, the routine ASan's frame #0 names for CRASH-068, with
    # no line table. Consistent with the defect, and not attributable without the caller.
    R.symbolize = fake_symbolize("", 0, func="strlen")
    obs = R.observe("CRASH-068", LEAF_SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("a fault in the libc leaf ASan names is unattributed", reason, "unattributed")
    check("and says it is unresolved for want of a return address",
          "unresolved" in ev and "return address" in ev, True)
    # WITH A RETURN ADDRESS the leaf is read through its caller, one hop, and only that.
    orig_at = R.symbolize_at
    def by_pc(table):
        return lambda llvm_bin, image, pc, text: table.get(pc, ("", "", 0))
    STRLEN_PC, RA = 0x3f90411888, 0x3f90508d7c
    R.symbolize = fake_symbolize("", 0, func="strlen")
    R.symbolize_at = by_pc({RA - 2: ("xbuf_format_converter", "/b/main/spprintf.c", 518)})
    obs = R.observe("CRASH-068", LEAF_SPEC, "php-real-exact", SHA, FAULT_RA % SHA, 139,
                    controls(), False, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("a libc leaf called from the declared function is CAUGHT", v, verdicts.CAUGHT)
    check("and the evidence names the caller and the ra",
          "called from xbuf_format_converter" in ev and "ra=0x3f90508d7c" in ev, True)
    check("and it licensed its own reach (no completing control needed)", obs.reached, True)
    check("the fault carries ra and fp into the bundle", (obs.fault.ra, obs.fault.fp),
          (0x3f90508d7c, 0x3f80a01f00))
    R.symbolize_at = by_pc({RA - 2: ("php_error_cb", "/b/main/main.c", 624)})
    obs = R.observe("CRASH-068", LEAF_SPEC, "php-real-exact", SHA, FAULT_RA % SHA, 139,
                    controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("a libc leaf called from ANOTHER function is unattributed", reason, "unattributed")
    check("and says where it was called from", "called from php_error_cb" in ev, True)
    R.symbolize_at = by_pc({RA - 2: ("xbuf_format_converter", "/b/main/other.c", 5)})
    obs = R.observe("CRASH-068", LEAF_SPEC, "php-real-exact", SHA, FAULT_RA % SHA, 139,
                    controls(), True, "/x", "/y")
    check("the declared caller in another file is unattributed",
          verdicts.judge(obs, exact)[1], "unattributed")
    # The caller is read ONE hop only: a strlen called from the declared function's own callee
    # is not seen as the declared function.
    R.symbolize_at = by_pc({RA - 2: ("vspprintf", "/b/main/spprintf.c", 725)})
    obs = R.observe("CRASH-068", LEAF_SPEC, "php-real-exact", SHA, FAULT_RA % SHA, 139,
                    controls(), True, "/x", "/y")
    check("a caller in the declared file but not the declared function is unattributed",
          verdicts.judge(obs, exact)[1], "unattributed")

    # THE CHAIN RULE (2026-10-28). The fault is outside PHP's tree -- the runtime's delegate copying
    # a write(2) buffer (CRASH-057's shape: the reused freed temporary printed by var_dump) -- and
    # the launcher printed the frame chain. The access belongs to the first frame of PHP's tree
    # above the output plumbing; a declared site there is a catch, anything else is not.
    FAULT_CHAIN = (FAULT_RA.replace("ra=0x3f90508d7c", "ra=0xa")
                   + " image=php.dom chain=0x3f90a00010,0x3f90a00020,0x3f90a00030,0x3f90a00040")
    # the real chain's names (run13): musl's write with its line table, the CLI's ub_write, and
    # PHP 5.0.0's php_body_write (output.c:119) -- every one on the forwarding list by name
    PLUMB = {0x3f90a00010 - 2: ("write", "/tmp/k/tmp/musl-src/musl-1.2.5/src/unistd/write.c", 6),
             0x3f90a00020 - 2: ("sapi_cli_ub_write", "/b/sapi/cli/php_cli.c", 205),
             0x3f90a00030 - 2: ("php_body_write", "/b/main/output.c", 119)}
    VD_SPEC = dict(SPEC, asan_func="_estrndup", asan_file="Zend/zend_alloc.c", asan_line=403,
                   asan_stack=[{"index": 0, "func": "memcpy", "outside": "memintrinsics.inc:117"},
                               {"index": 1, "func": "_estrndup", "file": "Zend/zend_alloc.c", "line": 403}],
                   fault_sites=["_estrndup", "php_var_dump"],
                   site_files={"_estrndup": "Zend/zend_alloc.c", "php_var_dump": "ext/standard/var.c"})
    R.symbolize = fake_symbolize("", 0, func="dl_call_once")         # the delegate's call, as the real record names it: a symbol, no line
    R.symbolize_at = by_pc({**PLUMB, 0x3f90a00040 - 2: ("php_var_dump", "/b/ext/standard/var.c", 126)})
    obs = R.observe("CRASH-057", VD_SPEC, "php-real-object", SHA, FAULT_CHAIN % SHA, 139,
                    controls(probe=None), False, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, pools)
    check("chain through the output plumbing to a declared site is CAUGHT", v, verdicts.CAUGHT)
    check("and the evidence names the skipped plumbing and the owner",
          "php_var_dump at var.c:126" in ev and "sapi_cli_ub_write" in ev, True)
    check("and it licensed its own reach", obs.reached, True)
    check("the chain is in the bundle", obs.fault.chain, (0x3f90a00010, 0x3f90a00020, 0x3f90a00030, 0x3f90a00040))
    # the first non-plumbing frame is NOT declared: no catch, whatever lies deeper
    R.symbolize_at = by_pc({**PLUMB, 0x3f90a00040 - 2: ("zend_assign_to_variable", "/b/Zend/zend_execute.c", 654)})
    obs = R.observe("CRASH-057", VD_SPEC, "php-real-object", SHA, FAULT_CHAIN % SHA, 139,
                    controls(probe=None), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, pools)
    check("chain reaching an undeclared engine frame first is unattributed", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and names that frame", "zend_assign_to_variable" in ev and "not a declared site" in ev, True)
    # a declared site BEYOND an undeclared engine frame does not count: the walk stops there
    FAULT_CHAIN5 = FAULT_CHAIN + ",0x3f90a00050"
    R.symbolize_at = by_pc({**PLUMB, 0x3f90a00040 - 2: ("zend_assign_to_variable", "/b/Zend/zend_execute.c", 654),
                                          0x3f90a00050 - 2: ("php_var_dump", "/b/ext/standard/var.c", 126)})
    obs = R.observe("CRASH-057", VD_SPEC, "php-real-object", SHA, FAULT_CHAIN5 % SHA, 139,
                    controls(probe=None), True, "/x", "/y")
    check("a declared site behind an undeclared frame is not reached", verdicts.judge(obs, pools)[0], verdicts.NO_READING)
    # a chain that never enters PHP's tree, every frame of it resolved (an unresolved one would
    # be a barrier, below)
    R.symbolize_at = by_pc({**PLUMB, 0x3f90a00040 - 2: ("fwrite", "/tmp/k/tmp/musl-src/musl-1.2.5/src/stdio/fwrite.c", 20)})
    obs = R.observe("CRASH-057", VD_SPEC, "php-real-object", SHA, FAULT_CHAIN % SHA, 139,
                    controls(probe=None), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, pools)
    check("a chain that reaches no PHP frame is unattributed", reason, "unattributed")
    check("and says so", "reaches no frame of PHP's tree" in ev, True)
    # BARRIERS (review 2026-10-28, which built a CAUGHT out of a memcpy fault whose chain ran
    # through zend_cap_reblock): a frame in the nested layer's own source, in PHP's allocator, or one
    # that resolves to nothing stops the walk, and no declared site beyond it counts. Each is
    # driven here with the declared site php_var_dump right behind the barrier.
    def chain_with(third, fourth=("php_var_dump", "/b/ext/standard/var.c", 126), spec=VD_SPEC):
        R.symbolize_at = by_pc({**PLUMB, 0x3f90a00030 - 2: third, 0x3f90a00040 - 2: fourth})
        # stock reached its marker, so a fault the chain cannot attribute is `unattributed`,
        # not a fault before the defect
        o = R.observe("CRASH-057", spec, "php-real-object", SHA, FAULT_CHAIN % SHA, 139,
                      controls(probe=None), True, "/x", "/y")
        return verdicts.judge(o, pools)
    v, reason, ev = chain_with(("zend_cap_reblock", "/b/Zend/sublet/zend-cap-mode.c", 88))
    check("a nested layer frame in the chain is a barrier: unattributed", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and the evidence names it", "the nested layer's own source (zend_cap_reblock@zend-cap-mode.c:88)" in ev, True)
    v, reason, ev = chain_with(("_estrndup", "/b/Zend/zend_alloc.c", 437))
    check("an allocator frame in the chain is a barrier, declared site or not (CRASH-073's memcpy <- _estrndup)",
          (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and the evidence names the allocator", "PHP's allocator (_estrndup@zend_alloc.c:437)" in ev, True)
    v, reason, ev = chain_with(("", "", 0))
    check("an unresolved chain frame is a barrier", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and the evidence names its address", "an unresolved frame (0x3f90a00030)" in ev, True)
    v, reason, ev = chain_with(("php_body_write", "/b/main/output.c", 119),
                               ("php_var_dump", "/b/ext/standard/other.c", 126))
    check("a same-named function in another file is not the site", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and the evidence says which file was declared", "not in its declared file ext/standard/var.c" in ev, True)
    v, reason, ev = chain_with(("php_body_write", "/b/main/output.c", 119),
                               spec=dict(VD_SPEC, site_files={"_estrndup": "Zend/zend_alloc.c"}))
    check("a site with no declared file cannot be matched by the chain", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and says so", "(none declared)" in ev, True)
    # the ra register resolving to nothing is NOT a barrier: a leaf may use it as scratch
    v, reason, ev = chain_with(("php_body_write", "/b/main/output.c", 119))
    check("with ra=0xa unresolved the chain still decides (CRASH-057)", v, verdicts.CAUGHT)
    check("and the evidence says the ra was skipped as a leaf's scratch", "ra=0xa (unresolved, a leaf's scratch register)" in ev, True)
    # THE SECOND REVIEW'S COUNTEREXAMPLES (2026-10-28): a frame with a SYMBOL and no line table
    # used to pass every barrier (they keyed on the path), and a directory used to pass as
    # plumbing. Now a frame passes by NAME, from the list of routines known to forward the
    # buffer, and everything else is a barrier or the owner.
    v, reason, ev = chain_with(("zend_cap_reblock", "", 0))
    check("a nested layer frame with a symbol and no line table is a barrier", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and is named as the nested layer", "the nested layer's own source (zend_cap_reblock@?:0)" in ev, True)
    v, reason, ev = chain_with(("_estrndup", "", 0))
    check("an allocator frame with a symbol and no line table is a barrier", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and is named as the allocator", "PHP's allocator (_estrndup@?:0)" in ev, True)
    v, reason, ev = chain_with(("zend_mm_alloc", "", 0))
    check("the allocator's internal layer too", "PHP's allocator (zend_mm_alloc@?:0)" in ev, True)
    v, reason, ev = chain_with(("capstone_delegate_pack", "", 0))
    check("a runtime routine not on the forwarding list is a barrier", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and says it is not known to forward", "a routine not known to forward the buffer (capstone_delegate_pack@?:0)" in ev, True)
    v, reason, ev = chain_with(("php_start_ob_buffer", "/b/main/output.c", 200))
    check("a routine of output.c that is not on the forwarding list is the owner, and undeclared", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and the evidence names it", "php_start_ob_buffer at output.c:200 first" in ev, True)
    v, reason, ev = chain_with(("php_body_write", "/b/main/output.c", 119),
                               ("php_var_dump", "/b/ext/other/notext/standard/var.c", 126))
    check("a declared file matched by suffix across a directory boundary is not the file", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("at_file: the whole path or after a slash only",
          (R.at_file("/b/ext/standard/var.c", "ext/standard/var.c"), R.at_file("ext/standard/var.c", "ext/standard/var.c"),
           R.at_file("/b/ext/other/notext/standard/var.c", "ext/standard/var.c"), R.at_file("", "ext/standard/var.c")),
          (True, True, False, False))
    check("frame kinds", tuple(R.frame_kind(f, p) for f, p in (("strcat", "/m/strcat.c"), ("zend_cap_issue", ""),
                                                                  ("_efree", ""), ("", ""), ("dl_transfer", ""),
                                                                  ("php_var_dump", "/b/ext/standard/var.c"),
                                                                  ("capstone_delegate_pack", ""), ("foo", "/b/main/output.c"))),
          ("forwarding", "nested", "allocator", "unresolved", "forwarding", "php", "other", "php"))
    # the probe transcript must NAME its image: the launcher's exit line or fault record
    check("transcript_image: the exit line's digest", R.transcript_image("x\ncapstone-exec: exit image=probe-putenv.dom sha256=" + "a" * 64 + "\n"), "a" * 64)
    check("transcript_image: no digest is None", R.transcript_image("PUTENV_PROBE DONE failures=0\n"), None)
    check("transcript_image: two digests are nobody's", R.transcript_image("sha256=" + "a" * 64 + "\nsha256=" + "b" * 64), None)
    absent2 = ("PUTENV_PROBE control retained=1 rc=0 errno=0 (ok)\nPUTENV_PROBE empty-name retained=0 rc=-1 errno=22 (Invalid argument)\n"
               "PUTENV_PROBE empty-name-value retained=0 rc=-1 errno=22 (Invalid argument)\nPUTENV_PROBE DONE failures=0\n")
    check("probe: a duplicate call line is no result, not the last line's reading",
          R.putenv_probe_outcome("PUTENV_PROBE empty-name retained=1 rc=0 errno=0 (ok)\n" + absent2), None)
    check("probe: the same transcript without the duplicate reads absent", R.putenv_probe_outcome(absent2), False)
    # THE THIRD REVIEW (2026-10-28): the FAULTING frame classified like a chain frame; the
    # digest read only off the launcher's own lines; exactly one completion record; the exit
    # line out of the application's output.
    R.symbolize_at = by_pc({**PLUMB, 0x3f90a00040 - 2: ("php_var_dump", "/b/ext/standard/var.c", 126)})
    for sym, what in (("zend_cap_reblock", "the nested layer's own source"), ("_emalloc", "PHP's own allocator"),
                      ("capstone_delegate_pack", "a routine not known to forward a caller's buffer")):
        R.symbolize = fake_symbolize("", 0, func=sym)
        obs = R.observe("CRASH-057", VD_SPEC, "php-real-object", SHA, FAULT_CHAIN % SHA, 139,
                        controls(probe=None), True, "/x", "/y")
        v, reason, ev = verdicts.judge(obs, pools)
        check(f"a fault IN {sym} (symbol, no line table) is not walked to its callers", (v, reason, obs.attribution),
              (verdicts.NO_READING, "unattributed", None))
        check(f"and the evidence says why", what in ev, True)
    R.symbolize = fake_symbolize("", 0, func="")
    obs = R.observe("CRASH-057", VD_SPEC, "php-real-object", SHA, FAULT_CHAIN % SHA, 139,
                    controls(probe=None), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, pools)
    check("a fault at an address nobody can name is not walked either", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and says so", "an address nobody can name" in ev, True)
    R.symbolize = fake_symbolize("", 0, func="dl_call_once")
    obs = R.observe("CRASH-057", VD_SPEC, "php-real-object", SHA, FAULT_CHAIN % SHA, 139,
                    controls(probe=None), False, "/x", "/y")
    check("a fault in a forwarding routine (dl_call_once, CRASH-057's) is walked and CAUGHT", verdicts.judge(obs, pools)[0], verdicts.CAUGHT)
    check("transcript_image: a digest on a line that is not the launcher's is nobody's",
          R.transcript_image("expected sha256=" + "a" * 64 + "\nPUTENV_PROBE DONE failures=0\n"), None)
    check("transcript_image: the fault record's digest counts",
          R.transcript_image(FAULT % ("b" * 64)), "b" * 64)
    check("probe: two DONE lines are no result", R.putenv_probe_outcome(absent2 + "PUTENV_PROBE DONE failures=1\n"), None)
    check("app_output drops the launcher's exit line and keeps its fault record",
          R.app_output(["done", "capstone-exec: exit image=x.dom sha256=" + "a" * 64, FAULT % SHA]).splitlines(),
          ["done", FAULT % SHA])
    # THE FOURTH REVIEW (2026-10-28): a catch through the call path is its own attribution
    # class, `chain`; the one-hop leaf rule applies the barriers to the caller; the direct rule
    # uses each declared site's own file; the probe reader takes whole records only.
    R.symbolize = fake_symbolize("", 0, func="dl_call_once")
    R.symbolize_at = by_pc({**PLUMB, 0x3f90a00040 - 2: ("php_var_dump", "/b/ext/standard/var.c", 126)})
    obs = R.observe("CRASH-057", VD_SPEC, "php-real-object", SHA, FAULT_CHAIN % SHA, 139,
                    controls(probe=None), False, "/x", "/y")
    check("a catch through the frame chain is attributed as `chain`, and says so",
          (verdicts.judge(obs, pools)[0], obs.attribution, "by the call path (chain)" in obs.attribution_evidence),
          (verdicts.CAUGHT, "chain", True))
    check("and its reach evidence says the FAULT is in the forwarding routine and the site is reached by the call path",
          ("faulted in dl_call_once, whose call path reaches php_var_dump" in obs.reach_evidence,
           "faulted in php_var_dump" in obs.reach_evidence), (True, False))
    # the leaf rule: memcpy with no chain, ra into the allocator (CRASH-073's shape) is refused
    SPEC73 = dict(VD_SPEC, fault_sites=["_estrndup"], site_files={"_estrndup": "Zend/zend_alloc.c"},
                  asan_stack=[{"index": 0, "func": "memcpy", "outside": "memintrinsics.inc:117"},
                              {"index": 1, "func": "_estrndup", "file": "Zend/zend_alloc.c", "line": 403}])
    R.symbolize = fake_symbolize("", 0, func="memcpy")
    R.symbolize_at = by_pc({RA - 2: ("_estrndup", "/b/Zend/zend_alloc.c", 403)})
    obs = R.observe("CRASH-073", SPEC73, "php-real-exact", SHA, FAULT_RA.replace("cause=25", "cause=28") % SHA, 139,
                    controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("a leaf fault whose ra caller is the allocator is refused like a chain barrier", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and says so", "its caller" in ev and "PHP's allocator" in ev, True)
    # the leaf rule's catch is a chain attribution too (one hop by ra)
    R.symbolize = fake_symbolize("", 0, func="strlen")
    R.symbolize_at = by_pc({RA - 2: ("xbuf_format_converter", "/b/main/spprintf.c", 518)})
    obs = R.observe("CRASH-068", LEAF_SPEC, "php-real-exact", SHA, FAULT_RA % SHA, 139, controls(), False, "/x", "/y")
    check("a leaf catch through ra is attributed as `chain`", (verdicts.judge(obs, exact)[0], obs.attribution), (verdicts.CAUGHT, "chain"))
    # the direct rule with a second declared site in its own file
    R.symbolize = fake_symbolize("/b/ext/standard/var.c", 126, func="php_var_dump")
    obs = R.observe("CRASH-057", VD_SPEC, "php-real-object", SHA, FAULT % SHA, 139, controls(probe=None), False, "/x", "/y")
    check("a fault directly in a second declared site, in that site's own file, is CAUGHT by function",
          (verdicts.judge(obs, pools)[0], obs.attribution), (verdicts.CAUGHT, "function"))
    R.symbolize = fake_symbolize("/b/Zend/zend_alloc.c", 126, func="php_var_dump")
    obs = R.observe("CRASH-057", VD_SPEC, "php-real-object", SHA, FAULT % SHA, 139, controls(probe=None), True, "/x", "/y")
    check("the same function in the primary site's file is not that site", verdicts.judge(obs, pools)[1], "unattributed")
    # the state the tests after the chain section expect (a fault inside PHP's tree)
    R.symbolize = fake_symbolize("/b/Zend/zend_execute.c", 700, func="zend_do_fcall_common_helper")
    check("probe: retained=2 is no result", R.putenv_probe_outcome(absent2.replace("empty-name retained=0", "empty-name retained=2")), None)
    check("probe: a DONE line with trailing garbage is no result", R.putenv_probe_outcome(absent2.replace("failures=0", "failures=0garbage")), None)
    check("probe: the clean absent transcript still reads absent", R.putenv_probe_outcome(absent2), False)
    # the fifth review: a malformed or unexpected PUTENV_PROBE line ANYWHERE voids the transcript,
    # even beside a complete well-formed pattern
    check("probe: an extra malformed DONE line voids the transcript",
          R.putenv_probe_outcome(absent2 + "PUTENV_PROBE DONE failures=1garbage\n"), None)
    check("probe: an extra malformed call line voids the transcript",
          R.putenv_probe_outcome(absent2 + "PUTENV_PROBE empty-name retained=2 rc=-1 errno=22 (Invalid argument)\n"), None)
    check("probe: the probe's own FAIL line voids the transcript",
          R.putenv_probe_outcome("PUTENV_PROBE FAIL the control variable was not retained\n" + absent2), None)
    check("probe: an unexpected fourth call voids the transcript",
          R.putenv_probe_outcome(absent2 + "PUTENV_PROBE other retained=0 rc=-1 errno=22 (Invalid argument)\n"), None)
    # A LEAF under line tables (CRASH-068's shape once musl carries them): strlen has a path now,
    # so the old `not path` branch no longer applies; and a leaf keeps no frame, so the chain's
    # first entry is the caller's CALLER (vspprintf) -- the caller itself is only in ra. The walk
    # starts at ra.
    FAULT_LEAF = FAULT_RA + " image=php.dom chain=0x3f90a00050,0x3f90a00060"
    R.symbolize = fake_symbolize("/tmp/k/musl-src/musl-1.2.5/src/string/strlen.c", 7, func="strlen")
    R.symbolize_at = by_pc({RA - 2: ("xbuf_format_converter", "/b/main/spprintf.c", 518),
                            0x3f90a00050 - 2: ("vspprintf", "/b/main/spprintf.c", 725),
                            0x3f90a00060 - 2: ("php_error_cb", "/b/main/main.c", 624)})
    obs = R.observe("CRASH-068", LEAF_SPEC, "php-real-exact", SHA, FAULT_LEAF % SHA, 139,
                    controls(), False, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("a leaf with a line table is attributed through ra first, then the chain", v, verdicts.CAUGHT)
    check("and the evidence names the caller from ra", "xbuf_format_converter at spprintf.c:518" in ev, True)
    # the same with ra clobbered: the chain alone reaches vspprintf, undeclared -> no catch
    FAULT_LEAF_BADRA = FAULT_LEAF.replace("ra=0x3f90508d7c", "ra=0xa")
    obs = R.observe("CRASH-068", LEAF_SPEC, "php-real-exact", SHA, FAULT_LEAF_BADRA % SHA, 139,
                    controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("with ra clobbered the chain's first PHP frame decides, and it is undeclared", (v, reason), (verdicts.NO_READING, "unattributed"))

    # the OBSERVED fault inside the allocator is what needs the audit arm, not the declared site
    R.symbolize = fake_symbolize("/b/Zend/zend_alloc.c", 403, func="_estrndup")
    obs = R.observe("CRASH-057", VD_SPEC, "php-real-object", SHA, FAULT_CHAIN % SHA, 139,
                    controls(probe=None), False, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, pools)
    check("a fault observed inside the allocator still needs the audit arm", (reason, "audit arm" in ev), ("unattributed", True))
    # the chain rule does not apply to a fault INSIDE PHP's tree: the site rules decide
    R.symbolize = fake_symbolize("/b/Zend/zend_execute.c", 700, func="zend_do_fcall_common_helper")
    R.symbolize_at = by_pc({**PLUMB, 0x3f90a00040 - 2: ("php_var_dump", "/b/ext/standard/var.c", 126)})
    obs = R.observe("CRASH-057", VD_SPEC, "php-real-object", SHA, FAULT_CHAIN % SHA, 139,
                    controls(probe=None), False, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, pools)
    check("a fault inside PHP's tree is judged by its own site, not by the chain",
          (v, "declared site is" in obs.attribution_evidence, "frame chain" in obs.attribution_evidence),
          (verdicts.NO_READING, True, False))
    R.symbolize_at = orig_at

    # An unrelated strlen -- a case whose ASan leaf is NOT strlen -- gets no such sentence.
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("a strlen fault on a case whose leaf is not strlen is plainly unattributed",
          (reason, "unresolved" in ev), ("unattributed", False))
    # And the leaf case caught in its declared caller IS a catch.
    R.symbolize = fake_symbolize("main/spprintf.c", 518, func="xbuf_format_converter")
    obs = R.observe("CRASH-068", LEAF_SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    check("the leaf case faulting in its declared caller is CAUGHT",
          verdicts.judge(obs, exact)[0], verdicts.CAUGHT)
    R.symbolize = fake_symbolize("ext/standard/basic_functions.c", 2135)

    # A `requires` case still needs the precondition: an execution requirement is confirmed
    # by nothing a silent run can show, so a silence is refused exactly as before.
    req = dict(SPEC, requires="network: working loopback", requires_kind="execution")
    obs = R.observe("CRASH-092", req, "php-real-stock", SHA, "done\n" + STATS % (100, 9216), 0,
                    [verdicts.Control(name=R.SANITY, observed="complete")], True, "/x", "/y",
                    False, False)
    check("an unconfirmed precondition is still not a reading",
          verdicts.judge(obs, arms["php-real-stock"])[1], "not-reached")

    # The oracle says native PHP does not complete this case: out of the denominator, and said
    # so, rather than decaying into a claim about our instrument or our run.
    nopremise = dict(SPEC, pristine_outcome="SIGNAL:11")
    obs = R.observe("CRASH-057", nopremise, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), False, "/x", "/y", True, False)
    v, reason, _ = verdicts.judge(obs, exact)
    check("a case the oracle crashes on is out of the denominator",
          (v, reason), (verdicts.NO_READING, "out-of-denominator"))
    obs = R.observe("CRASH-100", nopremise, "php-real-stock", SHA, "", 0,
                    controls(probe=None), False, "/x", "/y", True, False)
    check("and so is a silent one", verdicts.judge(obs, arms["php-real-stock"])[1],
          "out-of-denominator")
    # ...but a case PHP ENDS ITSELF, with a diagnostic and exit 255, is not that. It has an
    # oracle, so it is scoreable, and calling it out-of-denominator would discard a reading.
    fatal = dict(SPEC, pristine_outcome="EXIT:255", pristine_rc=255)
    check("a PHP fatal error is not a crash, for denominator purposes",
          R.oracle_crashes(fatal), False)
    check("a signal is", R.oracle_crashes(dict(SPEC, pristine_outcome="SIGNAL:11")), True)
    obs = R.observe("CRASH-062", fatal, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y", True, False)
    v, reason, _ = verdicts.judge(obs, exact)
    check("so it is judged on its merits, not thrown out",
          reason != "out-of-denominator", True)

    # AND IT MUST BE ABLE TO READ AS A PLAIN MISS. `completed` once hardcoded exit 0, so a case
    # whose oracle is EXIT:255 could not be completed on ANY arm and every cell decayed to
    # `inconclusive` -- including the control's, which takes the case's whole column with it.
    # This is the regression test for that: ending exactly as the oracle ended is completing.
    done = "ZEND_CAP_MODE 0 exact=0 revoke=0\ndone\n"
    obs = R.observe("CRASH-062", fatal, "php-real-stock", SHA, done, 255,
                    controls(probe=None), True, "/x", "/y")
    check("ending as the oracle ended counts as completing", obs.completed, True)
    check("so a fatal-error case can read as a plain MISSED",
          verdicts.judge(obs, arms["php-real-stock"])[0], verdicts.MISSED)
    # The converse: exit 0 where the oracle said 255 is NOT completing.
    obs = R.observe("CRASH-062", fatal, "php-real-stock", SHA, done, 0,
                    controls(probe=None), True, "/x", "/y")
    check("but exiting 0 where the oracle exited 255 is not", obs.completed, False)

    # THE SCRIPT PATH IS NOT A DIVERGENCE. PHP prints the script's absolute path inside its own
    # warnings and fatal errors; the corpus recorded a host path and the guest runs /mnt/vm. A
    # literal comparison reports a divergence on every such case, differing only in that path.
    pathy = dict(SPEC, pristine_stdout="Fatal error: nested in /corpus/CRASH-062.php on line 3")
    obs = R.observe("CRASH-062", pathy, "php-real-stock", SHA,
                    "ZEND_CAP_MODE 0 exact=0 revoke=0\n"
                    "Fatal error: nested in /mnt/vm/CRASH-062.php on line 3\n",
                    0, controls(probe=None), True, "/x", "/y")
    check("a differing script path is not an output divergence",
          "differs" in (obs.notes or ""), False)
    # ...but a real difference in what PHP said still is.
    obs = R.observe("CRASH-062", pathy, "php-real-stock", SHA,
                    "ZEND_CAP_MODE 0 exact=0 revoke=0\n"
                    "Fatal error: something else in /mnt/vm/CRASH-062.php on line 3\n",
                    0, controls(probe=None), True, "/x", "/y")
    check("a real difference in PHP's own words still is",
          "differs" in (obs.notes or ""), True)

    # The per-run watchdog fired. A killed run is the harness giving up, never a statement
    # about the program -- and in particular never the `not-reached` it would decay to if the
    # marker were ignored, because a killed run is silent and un-reached by construction.
    obs = R.observe("CRASH-092", SPEC, "php-real-stock", SHA,
                    "ZEND_CAP_MODE 0 exact=0 revoke=0\nVM_EXIT_TIMEOUT:CRASH-092\nKilled\n",
                    137, controls(), True, "/x", "/y")
    v, reason, _ = verdicts.judge(obs, arms["php-real-stock"])
    check("a watchdog kill is infra, not a reading", (v, reason), (verdicts.NO_READING, "infra"))

    # No exit status at all: the guest session ended before this run reported.
    obs = R.observe("CRASH-092", SPEC, "php-real-stock", SHA, "", None,
                    controls(), True, "/x", "/y")
    v, reason, _ = verdicts.judge(obs, arms["php-real-stock"])
    check("a run with no exit status is infra", (v, reason), (verdicts.NO_READING, "infra"))

    # Completed, but not with the oracle's output: recorded as a note, still MISSED.
    obs = R.observe("CRASH-086", SPEC, "php-real-stock", SHA, "something else\n", 0,
                    [verdicts.Control(name=R.SANITY, observed="complete")], True, "/x", "/y")
    check("divergent output is noted", "differs from the corpus's pristine record" in obs.notes, True)
    R.symbolize = orig


def helper_cases():
    print("\n-- the derivations --")
    # symbolize_at with the symbolizer's answer for a pc that has no line table. The function
    # comes from nm; the location is "??:0:0", which is NO file -- the leaf rule in observe()
    # keys on an empty file, and "??" reached it as a file name on the first run with ra.
    real_run = subprocess.run
    def stub(cmd, *a, **k):
        prog = str(cmd[0])
        if prog.endswith("llvm-nm"):
            return types.SimpleNamespace(stdout="0000000000011880 0000000000000040 T strlen\n", stderr="", returncode=0)
        if prog.endswith("llvm-readelf"):
            return types.SimpleNamespace(stdout="  LOAD  0x000000 0x0000000000010000 0x0000000000010000\n", stderr="", returncode=0)
        if "symbolizer" in prog:
            return types.SimpleNamespace(stdout="??:0:0\n", stderr="", returncode=0)
        return real_run(cmd, *a, **k)
    subprocess.run = stub
    R._SYMBOLS.clear()
    try:
        got = R.symbolize_at("/llvm", "/img-nolines", 0x3fb9001888,
                             "code=0x3fb9000000-0x3fb9400000")
    finally:
        subprocess.run = real_run
        R._SYMBOLS.clear()
    check("a pc with a symbol and no line table is (function, no file)", got, ("strlen", "", 0))
    check("app_output drops kernel lines and stats",
          R.app_output(["[    1.0] boot", "hello", "CAPSTONE_VM_STATS nodes=1", "world"]),
          "hello\nworld")
    # Taken from a REAL transcript: the launcher prints its stats with no preceding newline, so
    # when PHP's output does not end in one they share a line. A startswith() filter left the
    # whole thing in place and the oracle comparison then failed for CRASH-008 alone.
    check("stats concatenated onto PHP's last line are still stripped",
          R.app_output(["999999999999999CAPSTONE_VM_STATS arenas=39 nodes=1 node_capacity=2"]),
          "999999999999999")
    check("a stats line standing alone leaves no blank line",
          R.app_output(["done", "CAPSTONE_VM_STATS arenas=1 nodes=1 node_capacity=2"]), "done")
    # The launcher's grant diagnostics share the console with PHP; thousands of them on a
    # large case. They reached the oracle comparison and reported two matching outputs as
    # diverging (CRASH-084, CRASH-139, 2026-10-27).
    check("the launcher's grant lines are not PHP's output",
          R.app_output(["CAPSTONE_VM_ADD_OK slot=0 base=0x3faa000000 bytes=4194304 reg=3 perms=7 linear=0",
                        "v", "CAPSTONE_VM_ADD_OK slot=1 base=0x3faa500000 bytes=1048576 reg=2 perms=6 linear=0",
                        "end"]), "v\nend")
    # And the indentation var_dump emits must survive, or every indented case reads not-reached.
    check("leading whitespace is data, not noise",
          R.app_output(["array(2) {", '  ["scheme"]=>', "}"]),
          'array(2) {\n  ["scheme"]=>\n}')
    check("strip_nested_noise keeps only PHP's own output",
          R.strip_nested_noise("ZEND_CAP_MODE 6 exact=0 revoke=1\ndone\nPHP_VIRTUAL NESTED blocks=1 "
                             "frees=1 issues=1 retires=1 revokes=1"), "done")
    check("same_as_oracle ignores a trailing newline", R.same_as_oracle("done", "done\n"), True)
    check("same_as_oracle is otherwise exact", R.same_as_oracle("done!", "done"), False)
    check("node_pressure takes the high-water",
          R.node_pressure(R.STATS_LINE.pattern and (STATS % (10, 99) + "\n" + STATS % (40, 99))),
          (40, 99))
    check("nested_counters reads the mode and the revokes",
          R.nested_counters("ZEND_CAP_MODE 6 exact=0 revoke=1\nPHP_VIRTUAL NESTED blocks=3 frees=2 "
                          "issues=4 retires=2 revokes=77"), (6, 77))


def fake_corpus(root, ids=("CRASH-086", "CRASH-110")):
    """A corpus of case directories as tools/build-cases.py writes them: trigger.php and
    oracle.json per case, the population decided there and recorded."""
    root.mkdir(parents=True, exist_ok=True)
    def case(cid, pop, *, pristine="OK", rc=0, stdout="done\n", requires="", stock="OK",
             requires_kind=None):
        d = root / K(cid); d.mkdir(exist_ok=True)
        (d / "trigger.php").write_text("<?php echo \"done\\n\";\n")
        (d / "oracle.json").write_text(json.dumps({
            "id": cid, "population": pop, "cwe": "CWE-416", "requires": requires,
            "requires_kind": requires_kind if requires_kind is not None else ("execution" if requires else ""),
            "asan": {"class": "ASAN:heap-use-after-free", "func": SITE,
                     "file": "ext/standard/basic_functions.c", "line": 2135,
                     "stack": SPEC["asan_stack"]},
            "stockalloc": stock,
            "pristine": {"outcome": pristine, "rc": rc, "stdout": stdout},
            "crashes_on_pristine_build": pristine.startswith("SIGNAL:")}))
        # The framework's record, where the declared site lives (SCHEMA.md fault_sites).
        (d / "case.json").write_text(json.dumps({
            "case": NUM.get(cid, 99), "trigger": "trigger.php",
            "fault_sites": [SITE],
            "fault_sites_why": "the fixture's ASan frame #0, written by the builder from the native run"}))
    for cid in ids:
        case(cid, "asan-blind")
    # A case with a DECLARED PRECONDITION. CRASH-092's shape: it echoes its marker whether or
    # not its own listener came up, so a silence cannot be told from a platform that caught
    # nothing -- unless some arm faults at ASAN's site and thereby proves the precondition held.
    case("CRASH-092", "asan-blind", requires="network: working loopback")
    # A case whose requirement is a DETECTOR OPTION: what ASan needs to report it, not what
    # the trigger needs to run. CRASH-052's shape. It must not be refused for Capstone
    # lacking an ASan knob.
    case("CRASH-052", "asan-blind", requires="ASAN_OPTIONS detect_stack_use_after_return=1",
         requires_kind="detector-option")
    # A case the PRISTINE build already crashes on: there is no silence to interpret, so it must
    # read not-reached however the arms behave. This is CRASH-057's situation.
    case("CRASH-057", "asan-blind", pristine="SIGNAL:11", rc=-11)
    # A case that is NOT ASAN-blind (ASAN catches it with the stock allocator too) is the
    # complement population's member, never the ASAN-blind one.
    case("CRASH-999", "temporal-asan-visible", stock="ASAN:heap-use-after-free", stdout="")
    # A case PHP ends with its OWN fatal error (exit 255) completes in PHP's terms.
    case("CRASH-062", "asan-blind", pristine="EXIT:255", rc=255,
         stdout="\nFatal error: Class declarations may not be nested\n")


def population_cases():
    print("\n-- population() --")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus-root"
        fake_corpus(root)
        pop = R.population(root)
        check("only the ASAN-blind cases are in the denominator",
              sorted(pop), [K("CRASH-057"), K("CRASH-062"), K("CRASH-086"), K("CRASH-092"), K("CRASH-110"),
                            K("CRASH-052")])
        check("the expected site comes from ASAN's own frame",
              (pop[K("CRASH-086")]["asan_file"], pop[K("CRASH-086")]["asan_line"]),
              ("ext/standard/basic_functions.c", 2135))
        check("the pristine oracle is carried", pop[K("CRASH-086")]["pristine_stdout"], "done\n")
        check("a case PHP ends with its own fatal error stays in the denominator",
              K("CRASH-062") in pop, True)
        # The complement population: CWE-416 that ASAN finds unaided. CRASH-999 is the fixture's
        # not-blind case, so it is the one and only member here -- and must NOT appear above.
        tpop = R.population(root, "temporal-asan-visible")
        check("the temporal-asan-visible population is the complement",
              sorted(tpop), [K("CRASH-999")])
        check("and it does not overlap the ASAN-blind one",
              sorted(set(tpop) & set(pop)), [])
        # THE GATE MUST BE ABLE TO FIRE. Make the case's own flag disagree with its own run.
        op = root / K("CRASH-086") / "oracle.json"
        o = json.loads(op.read_text())
        o["crashes_on_pristine_build"] = True                 # but its run says OK
        op.write_text(json.dumps(o))
        try:
            R.population(root)
            check("a corpus whose crash flag contradicts its own run is refused", "ran", "refused")
        except SystemExit as e:
            check("a corpus whose crash flag contradicts its own run is refused",
                  "crashes_on_pristine_build" in str(e), True)


def synth_record(cases, mode, revokes, *, case_faults=(), sanity_ok=True, probe=True,
                 stdout="done", sha=SHA, probe_sha=PROBE_SHA):
    nested = f"ZEND_CAP_MODE {mode} exact={1 if mode & 1 else 0} revoke={1 if mode & 2 else 0}"
    stats = STATS % (100, 9216)
    out, exits = {}, {}
    out[R.SANITY] = [nested, f"PHP_VIRTUAL NESTED blocks=9 frees=9 issues=9 retires=9 revokes={revokes}",
                     "sanity 600 200 17" if sanity_ok else "", stats]
    exits[R.SANITY] = 0 if sanity_ok else 139
    if probe is not None:
        # probe: True = the bound firing as declared, False = the probe read past the bound,
        # or a shape name from probe_transcript().
        shape = {True: "declared", False: "no-fault"}.get(probe, probe)
        out[R.BOUND_PROBE], exits[R.BOUND_PROBE] = probe_transcript(shape, probe_sha)
    for c in cases:
        if c in case_faults:
            out[c] = [nested, FAULT % sha, stats]
            exits[c] = 139
        else:
            out[c] = [nested, stdout, stats]
            exits[c] = 0
    return {"booted": True, "reached_script": True, "completed": True, "panic": False,
            "exits": exits, "output": out, "serial_log": "x"}


def drive(tmp, *, revokes=(0, 0, 77, 77), faults=None, extra_argv=(), modes=None,
          defaulted=False, probe_mode_line=False, probe_shape=True):
    """Run main() end to end over stored records, with the toolchain stubbed out.

    defaulted: the first arm's sanity transcript carries ZEND_CAP_MODE_DEFAULTED, as an image
    built with a baked-in arm prints when no PHP_CAP_MODE named one.
    probe_shape: the bound probe's transcript on the arms that declare it (probe_transcript)."""
    root = tmp / "corpus-root"
    fake_corpus(root)
    out = tmp / "out"
    out.mkdir(exist_ok=True)
    cases = [K("CRASH-057"), K("CRASH-086"), K("CRASH-092"), K("CRASH-110"), K("CRASH-052")]
    faults = faults or {}
    # The platform record the booting pass writes beside its transcripts; a --from-record
    # judge reads it rather than hashing this host.
    (out / "platform.json").write_text(json.dumps(
        {k: "unrecorded (offline suite)" for k in verdicts.PLATFORM_KEYS}))
    declared = [verdicts.load_arms()[a]["mode"] for a in R.ARMS]
    modes = modes or declared
    for arm, mode, rev in zip(R.ARMS, modes, revokes):
        probe = probe_shape if R.BOUND_PROBE in verdicts.load_arms()[arm]["controls"] else None
        rec = synth_record(cases, mode, rev, case_faults=faults.get(arm, ()), probe=probe)
        if defaulted and arm == R.ARMS[0]:
            rec["output"][R.SANITY].insert(0, f"ZEND_CAP_MODE_DEFAULTED {mode}")
        if probe_mode_line and R.BOUND_PROBE in rec["output"]:
            rec["output"][R.BOUND_PROBE].insert(0, "ZEND_CAP_MODE 7 exact=1 revoke=1")
        (out / f"{arm}-run.json").write_text(json.dumps(rec))
    img = tmp / "php-real-nested.dom"
    img.write_bytes(b"\x7fELF" + b"\0" * 64)

    real_run, real_call = subprocess.run, subprocess.call
    def fake_run(cmd, *a, **k):
        prog = str(cmd[0])
        text = ""
        if "sha256sum" in prog:
            text = (PROBE_SHA if str(cmd[1]).endswith("p.dom") else SHA) + "  x\n"
        elif "objdump" in prog:
            # Real CDERIVE/CREVOKE encodings, because the gate reads instruction WORDS now:
            # llvm-objdump prints both as <unknown>, so a stub emitting the word "revoke" would
            # test a check that no longer exists. funct7 0x51/0x52, funct3 1, opcode 0x5b.
            text = ("   10000: 5b 10 00 a2  \t<unknown>\n"
                    "   10004: 5b 10 00 a4  \t<unknown>\n")
        elif "clang" in prog:
            text = "clang version test\n"
        elif "symbolizer" in prog:
            # --functions=none: the location only. The function comes from nm below.
            text = "/b/ext/standard/basic_functions.c:2135:9\n"
        elif "readelf" in prog:
            text = "  LOAD           0x000000 0x0000000000010000 0x0000000000010000\n"
        elif prog.endswith("llvm-nm"):
            # Sized symbols, one covering the fixture fault's static pc (FAULT's
            # pc - code + base = 0xf7f90), so the function is read from the symbol table as
            # it is on the real image, where the symbolizer's function line is a local label.
            # The PROBE image (p.dom) is its own program: its stage-7 reads are in main(),
            # which is what covers that pc there; report() sits below it.
            if str(cmd[-1]).endswith("p.dom"):
                text = ("00000000000f7e00 0000000000000100 T report\n"
                        "00000000000f7f00 0000000000000100 T main\n")
            else:
                text = ("0000000000010000 0000000000000010 T main\n"
                        f"00000000000f7f00 0000000000000100 T {SITE}\n")
        return types.SimpleNamespace(stdout=text, stderr="", returncode=0)
    subprocess.run, subprocess.call = fake_run, (lambda *a, **k: 0)
    argv = sys.argv
    sys.argv = ["run-realphp.py", "--qemu", "/q", "--images", str(tmp), "--adapter", str(tmp),
                "--image", str(img), "--probe", str(tmp / "p.dom"), "--corpus", str(root),
                "--out", str(out), "--llvm-bin", str(tmp), "--from-record",
                "--prereg", "deadbeef", *extra_argv]
    try:
        R.main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    finally:
        sys.argv = argv
        subprocess.run, subprocess.call = real_run, real_call


def main_path_cases():
    print("\n-- main() end to end, and every gate shown FIRING --")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rc = drive(tmp, faults={"php-real-exact": (K("CRASH-086"), K("CRASH-110"), K("CRASH-057"),
                                                   K("CRASH-092")),
                                "php-real-object": (K("CRASH-086"),)})
        check("a clean measurement completes", rc, 0)
        b = tmp / "out" / f"bundle-{R.FRAMEWORK_ARM['php-real-exact']}" / "verdicts.jsonl"
        check("the bundle is written", b.is_file(), True)
        rows = [json.loads(l) for l in b.read_text().splitlines()]
        check("every case is judged", len(rows), 6)
        by_case = {r["observation"]["case"]: r for r in rows}
        # A detector-option requirement is not a precondition here: the silent stock arm
        # reads the case as a plain MISSED instead of refusing it for an ASan knob.
        stock_rows = {json.loads(l)["observation"]["case"]: json.loads(l) for l in
                      (tmp / "out" / "bundle-php-real-stock" / "verdicts.jsonl").read_text().splitlines()}
        check("an ASAN_OPTIONS requirement does not refuse the case",
              (stock_rows[K("CRASH-052")]["verdict"], stock_rows[K("CRASH-052")]["reason"]),
              (verdicts.MISSED, None))
        check("the bundle says how it was scored",
              json.loads((tmp / "out" / "bundle-php-real-stock" / "inputs.json").read_text())
              ["scoring"]["score_oracle_crashes"], False)
        check("the fault arm caught the two with a premise",
              sorted(by_case[c]["verdict"] for c in (K("CRASH-086"), K("CRASH-110"))),
              [verdicts.CAUGHT, verdicts.CAUGHT])
        # The oracle says the pristine build crashes on CRASH-057, so even a fault at ASAN's own
        # site is not a reading: there is no silence on the control arm to compare it against,
        # and the reason names the oracle rather than blaming our setup for a native crash.
        check("a case the pristine build crashes on is not a reading",
              (by_case[K("CRASH-057")]["verdict"], by_case[K("CRASH-057")]["reason"]),
              (verdicts.NO_READING, "out-of-denominator"))
        # CRASH-092 faults at ASAN's site on the exact arm here, so its precondition is proved
        # and that licenses the catch.
        check("a precondition case that faults at ASAN's site is CAUGHT",
              by_case[K("CRASH-092")]["verdict"], verdicts.CAUGHT)
        inputs = json.loads((tmp / "out" / f"bundle-{R.FRAMEWORK_ARM['php-real-exact']}" / "inputs.json").read_text())
        check("the pre-registration is recorded", inputs["pre_registration"], "deadbeef")
        check("the bundle names the arms.json configuration it measured", inputs["configuration"], "php-real-exact")
        check("and the framework arm derive-verdicts.py keys on", inputs["arm"], "virtual-nested-exact")
        check("and the corpus it belongs to", inputs["corpus"], "php/engine-repros")

    # THE POSITIVE CONTROL FOR --score-oracle-crashes, which it did not have on 2026-10-26 and
    # which is why the flag shipped doing nothing. It lifts the out-of-denominator refusal above
    # so the 19 oracle-crashing cases can be read by their fault SITE; the first run of it
    # returned `setup-fault` on all 19, because `site_licensed` was ANDed with
    # `not oracle_crashes(spec)` in main()'s pass one while only observe()'s early return was
    # lifted. `reached` was then false by construction, and the judge tests `reached` BEFORE
    # attribution -- so every one of those faults, already attributed to the case's own
    # ASAN-recorded file by a revoked-capability load, was reported as OUR bug.
    #
    # A flag that moves 19 cells to one NO-READING reason is indistinguishable from a flag that
    # does nothing, which is what makes this a must-fire control and not a nicety. It is driven
    # end to end rather than against observe(), deliberately: the defect was in main(), so a
    # unit test passing site_licensed=True by hand would have stayed green throughout.
    with tempfile.TemporaryDirectory() as d3:
        tmp3 = Path(d3)
        rc = drive(tmp3, faults={"php-real-exact": (K("CRASH-086"), K("CRASH-110"), K("CRASH-057"),
                                                    K("CRASH-092")),
                                 "php-real-object": (K("CRASH-086"),)},
                   extra_argv=("--score-oracle-crashes",))
        check("scoring the oracle-crashing cases completes", rc, 0)
        rows3 = [json.loads(l) for l in
                 (tmp3 / "out" / f"bundle-{R.FRAMEWORK_ARM['php-real-exact']}" / "verdicts.jsonl").read_text().splitlines()]
        got3 = {r["observation"]["case"]: (r["verdict"], r["reason"]) for r in rows3}
        check("--score-oracle-crashes reads the fault SITE, not setup-fault",
              got3[K("CRASH-057")], (verdicts.CAUGHT, None))
        obs3 = {r["observation"]["case"]: r["observation"] for r in rows3}
        check("and the licence is the trigger having run, not the oracle's premise",
              obs3[K("CRASH-057")]["reached"], True)
        check("the cell still carries the warning that native PHP dies on it too",
              "NOT evidence" in (obs3[K("CRASH-057")]["notes"] or ""), True)
        # The flag must not leak into the default scoring: same records, no flag, still refused.
        check("without the flag the same record is still out-of-denominator",
              by_case[K("CRASH-057")]["reason"], "out-of-denominator")

    # ... and when NOTHING faults for it, its silence must be refused rather than read as MISSED.
    with tempfile.TemporaryDirectory() as d2:
        tmp2 = Path(d2)
        drive(tmp2, faults={"php-real-exact": (K("CRASH-086"),)})
        rows2 = [json.loads(l) for l in
                 (tmp2 / "out" / f"bundle-{R.FRAMEWORK_ARM['php-real-stock']}" / "verdicts.jsonl").read_text().splitlines()]
        got = {r["observation"]["case"]: (r["verdict"], r["reason"]) for r in rows2}
        check("an unconfirmed precondition refuses the silence",
              got[K("CRASH-092")], (verdicts.NO_READING, "not-reached"))
        check("a case with no precondition still reads MISSED",
              got[K("CRASH-086")][0], verdicts.MISSED)

    with tempfile.TemporaryDirectory() as d:
        rc = drive(Path(d), revokes=(0, 0, 0, 0))
        check("GATE: object revoking nothing is refused", rc != 0, True)
    with tempfile.TemporaryDirectory() as d:
        rc = drive(Path(d), revokes=(5, 0, 77, 77))
        check("GATE: stock revoking is refused", rc != 0, True)
    with tempfile.TemporaryDirectory() as d:
        rc = drive(Path(d), modes=[6, 6, 6, 6])
        check("GATE: arms reporting one mode word are refused", rc != 0, True)
    with tempfile.TemporaryDirectory() as d:
        rc = drive(Path(d), modes=[0, 4, 6, 6])
        check("GATE: a mode word disagreeing with arms.json is refused", rc != 0, True)


def probe_predicate_cases():
    print("\n-- the bound probe counts only when the EXACT read faults 28 --")
    # The predicate on every shape, the declared one positive and each broken one negative:
    # a control that only ever saw the declared shape has never been shown to fire. The
    # probe image's symbol table, as drive()'s nm stub serves it for p.dom: main() covers the
    # fixture's exact-read pc, report() sits below it.
    class ProbeSymbols:
        base = 0x10000
        funcs = [(0xf7e00, 0x100, "report"), (0xf7f00, 0x100, "main")]
        lookup = verdicts.Symbols.lookup
    symbols = ProbeSymbols()
    check("the test's symbol table puts the fixture's exact read in main()",
          symbols.lookup(0x3f904e7f90, 0x3f90400000), ("main", 0x90))
    for shape in ("declared", "no-fault", "cause-24", "cause-25", "cause-29",
                  "no-exact-line", "no-rounded-line", "fault-before-exact", "rounded-after-fault",
                  "fault-in-report", "pc-outside-image", "wrong-address", "no-exact-report",
                  "second-fault"):
        lines, _ = probe_transcript(shape)
        check(f"probe transcript '{shape}' reads {'as declared' if shape == 'declared' else 'NOT met'}",
              R.bound_probe_faulted_as_declared("\n".join(lines), symbols), shape == "declared")
    # End to end: a probe that faulted with the policy's cause voids the spatial arms' rows as
    # control-failed, while a temporal arm, which declares no probe, judges as before.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rc = drive(tmp, faults={"php-real-exact": (K("CRASH-086"),), "php-real-object": (K("CRASH-086"),)},
                   probe_shape="cause-29")
        check("a broken probe does not stop the run", rc, 0)
        exact = [json.loads(l) for l in (tmp / "out" / f"bundle-{R.FRAMEWORK_ARM['php-real-exact']}"
                                         / "verdicts.jsonl").read_text().splitlines()]
        by = {r["observation"]["case"]: (r["verdict"], r["reason"]) for r in exact}
        # The case that faulted at its site would have been CAUGHT; the broken probe voids it.
        # Rows that are infra or out-of-denominator keep those reasons (they rank first), so
        # the arm-wide claim is that no row survives as a verdict.
        check("the exact arm's catch is voided by a cause-29 probe fault",
              by[K("CRASH-086")], (verdicts.NO_READING, "control-failed"))
        check("no row of the exact arm survives as a verdict",
              sorted({v for v, _ in by.values()}), [verdicts.NO_READING])
        block = [json.loads(l) for l in (tmp / "out" / f"bundle-{R.FRAMEWORK_ARM['php-real-block']}"
                                         / "verdicts.jsonl").read_text().splitlines()]
        check("the block arm, which declares no probe, is not voided",
              "control-failed" in {r["reason"] for r in block}, False)


def policy_cases():
    print("\n-- the address policy is named in QEMU_CPU, and only sv39 or off --")
    # run-virtual.py reads PHP_VIRTUAL_ADDR_POLICY once at import, so the launch and the
    # platform record (which imports QEMU_CPU) cannot disagree. Loaded in a child so the env
    # is the only input.
    load = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('rv', sys.argv[1]); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.QEMU_CPU)")
    rv = str(Path(R.__file__).with_name("run-virtual.py"))
    for policy, want_rc, want_text in (("sv39", 0, "x-capstone-addr=sv39"), ("off", 0, "x-capstone-addr=off"),
                                        ("bogus", 1, "PHP_VIRTUAL_ADDR_POLICY='bogus'")):
        env = dict(os.environ, PHP_VIRTUAL_ADDR_POLICY=policy)
        p = subprocess.run([sys.executable, "-c", load, rv], env=env, capture_output=True, text=True)
        check(f"PHP_VIRTUAL_ADDR_POLICY={policy}: rc {want_rc} and the policy named",
              (p.returncode, want_text in p.stdout + p.stderr), (want_rc, True))
    env = dict(os.environ); env.pop("PHP_VIRTUAL_ADDR_POLICY", None)
    p = subprocess.run([sys.executable, "-c", load, rv], env=env, capture_output=True, text=True)
    check("unset: the measurement policy sv39", p.stdout.strip().endswith("x-capstone-addr=sv39"), True)


def stale_guard(label, *, sha=SHA, probe_sha=PROBE_SHA, want_refused=True):
    # Asserted on the REASON, not on the exit status: these records trip other gates too (the
    # synthetic arms all report the same revoke count), so "it exited non-zero" would pass
    # whether or not the sha guard was the thing that fired.
    """Drive the stale-transcript guard with the shas the stored runs claim.

    Parameterised rather than copied because the guard now has two sides and a bug on either
    is silent: the CASE runs must match the PHP image and the bound probe must match the
    PROBE image, which is a different file on purpose. The measurement found this the
    expensive way -- the guard refused the probe's own honest transcript and killed the run
    after three arms, because it compared every run against the PHP image."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = tmp / "corpus-root"; fake_corpus(root)
        out = tmp / "out"; out.mkdir()
        (out / "platform.json").write_text(json.dumps({k: "unrecorded (offline suite)" for k in verdicts.PLATFORM_KEYS}))
        for arm in R.ARMS:
            probe = True if R.BOUND_PROBE in verdicts.load_arms()[arm]["controls"] else None
            rec = synth_record([K("CRASH-086"), K("CRASH-110")], verdicts.load_arms()[arm]["mode"], 77,
                               case_faults=(K("CRASH-086"),), probe=probe, sha=sha,
                               probe_sha=probe_sha)
            (out / f"{arm}-run.json").write_text(json.dumps(rec))
        img = tmp / "php-real-nested.dom"; img.write_bytes(b"\x7fELF")
        real_run, real_call = subprocess.run, subprocess.call
        def stub(cmd, *a, **k):
            prog = str(cmd[0])
            if "sha256sum" in prog:
                text = (PROBE_SHA if str(cmd[1]).endswith("p.dom") else SHA) + "  x\n"
            elif prog.endswith("llvm-nm"):
                # the bound-probe control reads the probe image's symbol table
                text = "00000000000f7f00 0000000000000100 T main\n"
            elif "readelf" in prog:
                text = "  LOAD           0x000000 0x0000000000010000 0x0000000000010000\n"
            else:
                text = "revoke\n"
            return types.SimpleNamespace(stdout=text, stderr="", returncode=0)
        subprocess.run = stub
        subprocess.call = lambda *a, **k: 0
        argv = sys.argv
        sys.argv = ["x", "--qemu", "/q", "--images", str(tmp), "--adapter", str(tmp),
                    "--image", str(img), "--probe", str(tmp / "p.dom"), "--corpus", str(root),
                    "--out", str(out), "--llvm-bin", str(tmp), "--from-record"]
        why = ""
        try:
            R.main()
        except SystemExit as e:
            why = "" if isinstance(e.code, int) else str(e.code or "")
        finally:
            sys.argv = argv
            subprocess.run, subprocess.call = real_run, real_call
        check(label, "the stored transcript is from image" in why, want_refused)


def stale_image_case():
    print("\n-- the stale-transcript guard --")
    # A case transcript from another PHP image.
    stale_guard("a transcript from another image is refused", sha="c" * 64)
    # The probe's OWN transcript, carrying the probe image's sha, is what it should be.
    stale_guard("the probe's own sha is not treated as stale", want_refused=False)
    # ...and the exemption is not a hole: a probe transcript from a THIRD image is refused.
    stale_guard("a stale PROBE transcript is still refused", probe_sha="c" * 64)


def node_table_cases():
    """THE NODE TABLE IS PART OF THE INSTRUMENT, so its size is pinned by a test.

    CRASH-046 and CRASH-075 produced `setup-fault` on all four arms because the module's
    default 4-page node table could not grow fast enough for a workload holding ~12,600
    live blocks, and heap.c's ensure_ids(256) then handed back NULL. Nothing about that
    failure looks like a size: it surfaces as "the heap would not lend a block". If the
    insmod line silently loses its parameter the two cases simply go back to being
    unreadable, so the line itself is asserted here rather than trusted.
    """
    print("\n-- the node table --")
    spec = importlib.util.spec_from_file_location("rv", HERE / "run-virtual.py")
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    runs = [(K("CRASH-001"), "img 1")]
    text = rv.guest_script(runs)
    check("the guest insmods the module with a pre-grown node table",
          "insmod capstone_vm.ko node_initial_pages=1024 supervisor_slots=32 || exit 1" in text,
          True)
    # Both of these decide whether the platform runs AT ALL, so they are asserted rather than
    # assumed. supervisor_slots defaults to 1 (what silicon holds) and every thread and
    # shared-heap check in runtime/virtual/run.py fails there; x-capstone-test-csrs keeps CSR
    # 0xcc0, which the PAGED module reads once per context creation and which upstream made
    # illegal by default on 2026-10-18. Missing either one, the guest wedges with `cause = 24`
    # inside the monitor and names neither cause.
    check("the module gets more than silicon's single supervisor slot",
          "supervisor_slots=32" in text, True)
    check("and the emulator keeps the CSR the paged module reads",
          "x-capstone-test-csrs=true" in rv.QEMU_CPU, True)
    check("with matching supervisor slots on the emulator side",
          "x-capstone-supervisor-slots=32" in rv.QEMU_CPU, True)
    check("the exact-bounds profile is still selected",
          "x-capstone-exact-bounds=true" in rv.QEMU_CPU, True)
    check("1024 pages is 262144 nodes, at 256 per page", rv.NODE_PAGES * 256, 262144)
    check("and the size is overridable for a directed experiment",
          "node_initial_pages=7 " in rv.guest_script(runs, node_pages=7), True)


def cap_insn_cases():
    """The image gate reads instruction WORDS, not a mnemonic.

    This gate decides whether the whole measurement is believed: the nested layer is a runtime mode, so
    an image that cannot revoke would make all three temporal columns read alike. It was a
    `grep -c '\\brevoke\\b'` over objdump until 2026-10-26, and llvm-objdump prints both CDERIVE
    and CREVOKE as <unknown> -- so after the nested layer moved onto them the gate read zero on a
    CORRECT image. It existed in two copies, which drifted: the build's was fixed, the runner's
    was not, and the runner refused a whole 29-case measurement after all four arms had run.
    These pin the encoding so a third copy cannot drift silently."""
    print("\n-- the image capability-instruction gate --")
    spec = importlib.util.spec_from_file_location("cci", HERE / "count-cap-insns.py")
    cci = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cci)
    # funct7 0x51 CDERIVE / 0x52 CREVOKE, funct3 1, opcode 0x5b -- little-endian in the dump.
    derive = "   10000: 5b 10 00 a2  \t<unknown>\n"
    revoke = "   10004: 5b 10 00 a4  \t<unknown>\n"
    check("both instructions are counted from their encodings",
          cci.count(io.StringIO(derive + revoke)), (1, 1))
    check("the word 'revoke' in the text counts for NOTHING",
          cci.count(io.StringIO("revoke\nrevoke\nrevoke\n")), (0, 0))
    check("a near miss on funct7 is not counted",
          cci.count(io.StringIO("   10000: 5b 10 00 a6  \t<unknown>\n")), (0, 0))
    check("a near miss on the opcode is not counted",
          cci.count(io.StringIO("   10000: 5a 10 00 a4  \t<unknown>\n")), (0, 0))
    check("an image with derives but no revokes is not enough",
          cci.count(io.StringIO(derive * 9)), (9, 0))


def provenance_cases():
    """A bundle must identify its platform BY CONTENT. These were recorded as bare paths, so a
    bundle named files on one host and said nothing about what they held -- and the kernel here
    was built from a buildroot checkout that is not the pin the superproject records."""
    print("\n-- the platform record --")
    import hashlib
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "Image").write_bytes(b"kernel bytes")
        want = hashlib.sha256(b"kernel bytes").hexdigest()
        check("a platform binary is hashed, not just named",
              R.sha256_file(d / "Image"), want)

        images, adapter = d, d / "adapter"
        (d / "fw_jump.elf").write_bytes(b"fw")
        (d / "rootfs.ext2").write_bytes(b"rootfs")
        (adapter / "module").mkdir(parents=True)
        (adapter / "capstone-vexec").write_bytes(b"launcher")
        (adapter / "module" / "capstone_vm.ko").write_bytes(b"module")
        args = types.SimpleNamespace(llvm_bin=Path("/nonexistent"), qemu=d / "qemu",
                                     images=images, adapter=adapter)
        try:
            rec = R.platform_record(args)
        except (FileNotFoundError, IndexError):
            # clang is absent in this sandbox; the compiler line is not what is under test
            R.subprocess = types.SimpleNamespace(
                run=lambda *a, **k: types.SimpleNamespace(stdout="clang stub\n"))
            rec = R.platform_record(args)
            R.subprocess = subprocess

        check("the kernel is recorded by digest", rec["kernel"].split()[0], want)
        check("the launcher and the module are recorded too",
              all(len(rec[k].split()[0]) == 64 for k in ("launcher", "module")), True)
        check("the guest-written rootfs is recorded as well", "rootfs" in rec, True)
        check("an ABSENT binary says so rather than being omitted",
              rec["qemu"].startswith("MISSING:"), True)
        check("and a missing input is never mistaken for a 64-hex digest",
              len(rec["qemu"].split()[0]) == 64, False)
        # The platform is not only its binaries. These two properties decide whether the
        # platform runs at all, run.py records paths and hashes but deliberately not argv, and
        # two runs differing only in them are NOT equivalent -- so the bundle has to say.
        check("the record names the emulator's capability properties, not just its hash",
              "x-capstone-test-csrs=true" in rec.get("qemu_cpu", ""), True)
        check("and the arguments the module was loaded with",
              "supervisor_slots=32" in rec.get("module_args", ""), True)
        check("the node table size is recorded with them",
              "node_initial_pages=" in rec.get("module_args", ""), True)
        check("the judge's required keys are all present",
              [k for k in verdicts.PLATFORM_KEYS if k not in rec], [])


def defaulted_arm_cases():
    """An image that took a baked-in arm is not a measurement; a transcript at another arm's
    mode is not this arm's reading. Both refused, stored records included."""
    print("\n-- a defaulted or mismatched arm --")
    with tempfile.TemporaryDirectory() as d:
        check("a run whose sanity transcript says ZEND_CAP_MODE_DEFAULTED is refused",
              drive(Path(d), defaulted=True) != 0, True)
    with tempfile.TemporaryDirectory() as d:
        declared = [verdicts.load_arms()[a]["mode"] for a in R.ARMS]
        wrong = list(declared); wrong[-1] = declared[-2]          # the exact arm's records at the object arm's mode
        check("a run whose transcripts report another arm's mode is refused",
              drive(Path(d), modes=wrong) != 0, True)
    # The bound probe sets its own modes (stage 7 reads index 9 rounded, then exact), so its
    # transcript says PHP_CAP_MODE 7 on every arm that runs it; that is the probe's mode, not
    # the arm's, and must not be read as a mismatch. The first corpus run from
    # bug-corpora/php/engine-repros was refused exactly so (2026-10-27).
    with tempfile.TemporaryDirectory() as d:
        check("the bound probe's own mode line is not held against the arm",
              drive(Path(d), probe_mode_line=True) == 0, True)


def sanity_text_cases():
    """The sanity control's script is what every arm must complete; pin what it must contain."""
    print("\n-- the sanity script --")
    text = R.SANITY_TEXT
    check("the sanity script sorts three ways (ksort, sort, usort)",
          [f in text for f in ("ksort(", "sort(", "usort(")], [True, True, True])
    check("its comparator touches nothing but its arguments",
          "function phpb_cmp($x, $y) { return ($x < $y) ? -1 : 1; }" in text, True)
    check("the sorted array is not clobbered by the later foreach",
          "foreach ($a as $w)" in text, False)
    check("its one output line names every quantity it computes",
          "echo 'sanity ', strlen($s), ' ', count($u['k']), ' ', $u['n'], ' '," in text
          and "implode('', $k), ' ', implode(',', $w)" in text, True)


def lint():
    print("\n-- lint --")
    for mod in ("pyflakes", "flake8"):
        p = subprocess.run([sys.executable, "-m", mod, str(HERE / "run-realphp.py"),
                            str(Path(__file__))], capture_output=True, text=True)
        # `python -m X` with X not installed exits 1 and says so on stderr; that is "not
        # available", not "clean". Only exit 0 with nothing to report is clean: an empty stdout
        # with a non-zero status used to read as "ok pyflakes clean" when pyflakes was absent.
        if "No module named" in p.stderr:
            continue
        if p.returncode == 0 and not p.stdout.strip():
            print(f"  ok   {mod} clean")
        else:
            print((p.stdout + p.stderr).strip())
            FAILURES.append(f"{mod} reported findings (exit {p.returncode})")
        return
    print("  NOTE: neither pyflakes nor flake8 is installed, so NO linter ran. This is reported "
          "rather than skipped silently: an absent check reads exactly like a passing one.")


if __name__ == "__main__":
    helper_cases()
    population_cases()
    observation_cases()
    main_path_cases()
    probe_predicate_cases()
    policy_cases()
    stale_image_case()
    node_table_cases()
    cap_insn_cases()
    provenance_cases()
    defaulted_arm_cases()
    sanity_text_cases()
    lint()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES")
        for f in FAILURES:
            print(f"  {f}")
        sys.exit(1)
    print("all checks pass")
