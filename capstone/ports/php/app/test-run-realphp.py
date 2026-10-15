#!/usr/bin/env python3
"""Offline tests for run-realphp.py. No guest, no toolchain, no corpus.

WHY THIS EXISTS. The previous session left two NameErrors in a runner's own code; the first cost
a 15-minute boot to find. Every boot here is four boots and ~40 minutes, so main()'s wiring is
driven end to end against synthetic transcripts before anything is launched.

AND WHY EACH GATE IS SHOWN FIRING. A gate that has never failed is not known to work. The
cross-arm checks below are each driven to FAIL on purpose and asserted to refuse the run.
"""
import importlib.util, json, re, subprocess, sys, tempfile, types
from pathlib import Path

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


FAULT = ("capstone-exec: domain fault cause=24 pc=0x3f904e7f90 address=0x3f909c8074 "
         "entry=0x3f904009a0 code=0x3f90400000-0x3f90800000 last=0x39 preparing=0x39 "
         "sha256=%s image=php-real-seam.dom")
# The same fault line with the RESOURCE cause: the node table ran out, which lands at an
# arbitrary pc and must never be read as a catch.
FAULT30 = FAULT.replace("cause=24", "cause=30")
SHA = "a" * 64
# The bound probe is a DIFFERENT image by design, so its transcript carries its own
# sha and the guard has to know that without being blinded to a stale probe.
PROBE_SHA = "b" * 64
STATS = ("CAPSTONE_VM_STATS arenas=16 pages=1778 peak=1994 nodes=%d steps=1943 faults=664 "
         "collections=0 reclaimed=0 rounds=54 bytes_in=544 bytes_out=21229 node_bytes=225280 "
         "node_capacity=%d node_growths=3 launch_ns=1 elapsed_ns=2")

SPEC = {"cwe": "CWE-416", "requires": "", "asan_func": "user_tick_function_call",
        "asan_file": "ext/standard/basic_functions.c", "asan_line": 2135,
        "pristine_stdout": "done\n", "pristine_outcome": "OK"}


def fake_symbolize(path, line):
    return lambda llvm_bin, image, fault, text: (path, line)


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
    orig = R.symbolize

    # A fault at ASAN's own file AND line, with the trigger shown live: CAUGHT.
    R.symbolize = fake_symbolize("ext/standard/basic_functions.c", 2135)
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA + "\n" + STATS % (100, 9216),
                    139, controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("caught at ASAN's own line", (v, reason), (verdicts.CAUGHT, None))
    check("the evidence says 'its own line'", "its own line" in ev, True)

    # A neighbouring line in the same file is still the catch, and says so.
    R.symbolize = fake_symbolize("ext/standard/basic_functions.c", 2137)
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    v, _, ev = verdicts.judge(obs, exact)
    check("caught at a neighbouring line", v, verdicts.CAUGHT)
    check("the evidence names ASAN's line", "ASAN: line 2135" in ev, True)

    # A fault in a DIFFERENT file is not this case's defect.
    R.symbolize = fake_symbolize("Zend/zend_hash.c", 400)
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    check("wrong file is unattributed", verdicts.judge(obs, exact)[1], "unattributed")

    # A fault in the seam's OWN source is a port artefact, never a catch.
    R.symbolize = fake_symbolize("capstone/ports/php/zend-alloc/sublet/zend-cap-blocks.c", 120)
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("a fault in the seam is unattributed", (v, reason), (verdicts.NO_READING, "unattributed"))
    check("and says it is a port artefact", "port artefact" in ev, True)

    # A case whose ASAN site IS the allocator cannot be attributed by site.
    alloc_spec = dict(SPEC, asan_file="Zend/zend_alloc.c", asan_line=403)
    R.symbolize = fake_symbolize("Zend/zend_alloc.c", 403)
    obs = R.observe("CRASH-073", alloc_spec, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), True, "/x", "/y")
    v, reason, ev = verdicts.judge(obs, exact)
    check("allocator-site case is unattributed", reason, "unattributed")
    check("and asks for the audit arm", "audit arm" in ev, True)

    # Faulted, but the trigger was never shown to run: setup-fault, NOT a catch.
    R.symbolize = fake_symbolize("ext/standard/basic_functions.c", 2135)
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), False, "/x", "/y")
    check("fault without reach is a setup-fault", verdicts.judge(obs, exact)[1], "setup-fault")

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

    # The seam itself failed: an instrument failure, never a verdict.
    obs = R.observe("CRASH-086", SPEC, "php-real-exact", SHA,
                    "ZEND_CAP_FAIL code=3 unknown key 0x40\n", 139, controls(), True, "/x", "/y")
    check("a seam failure is a setup-fault", verdicts.judge(obs, exact)[1], "setup-fault")

    # Caught on the STOCK arm, so no control arm completed and stock_reached is False. Without
    # the site licence this is `setup-fault` -- the instrument blamed for the platform's
    # clearest result -- which is what CRASH-022 and CRASH-102 actually did.
    R.symbolize = fake_symbolize("ext/standard/basic_functions.c", 2135)
    obs = R.observe("CRASH-022", SPEC, "php-real-stock", SHA, FAULT % SHA, 139,
                    controls(probe=None), False, "/x", "/y", True, False)
    check("no licence: a stock-arm catch reads as setup-fault",
          verdicts.judge(obs, arms["php-real-stock"])[1], "setup-fault")
    obs = R.observe("CRASH-022", SPEC, "php-real-stock", SHA, FAULT % SHA, 139,
                    controls(probe=None), False, "/x", "/y", True, True)
    check("a fault at ASAN's own site licenses reach",
          verdicts.judge(obs, arms["php-real-stock"])[0], verdicts.CAUGHT)
    check("and says which route licensed it",
          "own ASAN-recorded site" in obs.reach_evidence, True)

    # THE LICENCE MUST NOT RESCUE A FAULT SOMEWHERE ELSE. It is granted by a fault at the
    # case's own site on SOME arm; an arm that faulted in the seam, or at an unrelated line,
    # still reports unattributed on its own account.
    R.symbolize = fake_symbolize("ext/standard/url.c", 132)
    obs = R.observe("CRASH-022", SPEC, "php-real-exact", SHA, FAULT % SHA, 139,
                    controls(), False, "/x", "/y", True, True)
    check("the licence does not attribute a fault elsewhere",
          verdicts.judge(obs, exact)[1], "unattributed")
    R.symbolize = fake_symbolize("ext/standard/basic_functions.c", 2135)

    # AND THE LICENCE DOES NOT CREATE A PREMISE. A case the pristine build itself crashes on
    # has no pair to read; where the trigger got to is beside the point. main() withholds the
    # licence for those, and the offline suite drives that path in main_path_cases().

    # A `requires` case still needs the precondition: the licence and the precondition are the
    # same fact, so a case with neither is refused exactly as before.
    req = dict(SPEC, requires="network: working loopback")
    obs = R.observe("CRASH-092", req, "php-real-stock", SHA, "done\n" + STATS % (100, 9216), 0,
                    [verdicts.Control(name=R.SANITY, observed="complete")], True, "/x", "/y",
                    False, False)
    check("an unconfirmed precondition is still not a reading",
          verdicts.judge(obs, arms["php-real-stock"])[1], "not-reached")

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
    # And the indentation var_dump emits must survive, or every indented case reads not-reached.
    check("leading whitespace is data, not noise",
          R.app_output(["array(2) {", '  ["scheme"]=>', "}"]),
          'array(2) {\n  ["scheme"]=>\n}')
    check("strip_seam_noise keeps only PHP's own output",
          R.strip_seam_noise("ZEND_CAP_MODE 6 exact=0 revoke=1\ndone\nPHP_VIRTUAL SEAM blocks=1 "
                             "frees=1 issues=1 retires=1 revokes=1"), "done")
    check("same_as_oracle ignores a trailing newline", R.same_as_oracle("done", "done\n"), True)
    check("same_as_oracle is otherwise exact", R.same_as_oracle("done!", "done"), False)
    check("node_pressure takes the high-water",
          R.node_pressure(R.STATS_LINE.pattern and (STATS % (10, 99) + "\n" + STATS % (40, 99))),
          (40, 99))
    check("seam_counters reads the mode and the revokes",
          R.seam_counters("ZEND_CAP_MODE 6 exact=0 revoke=1\nPHP_VIRTUAL SEAM blocks=3 frees=2 "
                          "issues=4 retires=2 revokes=77"), (6, 77))


def fake_corpus(root, ids=("CRASH-086", "CRASH-110")):
    """A corpus with the two files the population is defined by, plus the pristine oracle."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "corpus").mkdir(exist_ok=True)
    stderr = ("==1==ERROR: AddressSanitizer: heap-use-after-free\n"
              "    #0 0x55 in user_tick_function_call /b/ext/standard/basic_functions.c:2135\n")
    default, stock, pristine = [], [], []
    for i, cid in enumerate(ids):
        (root / "corpus" / f"{cid}.php").write_text("<?php echo \"done\\n\";\n")
        default.append({"id": cid, "cwe": "CWE-416", "requires": "",
                        "outcome": "ASAN:heap-use-after-free", "stderr": stderr})
        stock.append({"id": cid, "outcome": "OK"})
        pristine.append({"id": cid, "outcome": "OK", "stdout": "done\n"})
    # A case with a DECLARED PRECONDITION. CRASH-092's shape: it echoes its marker whether or
    # not its own listener came up, so a silence cannot be told from a platform that caught
    # nothing -- unless some arm faults at ASAN's site and thereby proves the precondition held.
    (root / "corpus" / "CRASH-092.php").write_text("<?php echo \"done\\n\";\n")
    default.append({"id": "CRASH-092", "cwe": "CWE-416",
                    "requires": "network: working loopback",
                    "outcome": "ASAN:heap-use-after-free", "stderr": stderr})
    stock.append({"id": "CRASH-092", "outcome": "OK"})
    pristine.append({"id": "CRASH-092", "outcome": "OK", "stdout": "done\n"})
    # A case the PRISTINE build already crashes on: there is no silence to interpret, so it must
    # read not-reached however the arms behave. This is CRASH-057's situation.
    (root / "corpus" / "CRASH-057.php").write_text("<?php echo \"done\\n\";\n")
    default.append({"id": "CRASH-057", "cwe": "CWE-416", "requires": "",
                    "outcome": "ASAN:heap-use-after-free", "stderr": stderr})
    stock.append({"id": "CRASH-057", "outcome": "OK"})
    pristine.append({"id": "CRASH-057", "outcome": "SIGNAL:11", "stdout": "done\n"})
    # A case that is NOT ASAN-blind (ASAN catches it with the stock allocator too) must be
    # excluded, or the denominator silently grows.
    default.append({"id": "CRASH-999", "cwe": "CWE-416", "requires": "",
                    "outcome": "ASAN:heap-use-after-free", "stderr": stderr})
    stock.append({"id": "CRASH-999", "outcome": "ASAN:heap-use-after-free"})
    pristine.append({"id": "CRASH-999", "outcome": "OK", "stdout": ""})
    (root / "results-asan-default.json").write_text(json.dumps({"results": default}))
    (root / "results-asan-stockalloc.json").write_text(json.dumps({"results": stock}))
    (root / "results-pristine-default.json").write_text(json.dumps({"results": pristine}))


def population_cases():
    print("\n-- population() --")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus-root"
        fake_corpus(root)
        pop = R.population(root)
        check("only the ASAN-blind cases are in the denominator",
              sorted(pop), ["CRASH-057", "CRASH-086", "CRASH-092", "CRASH-110"])
        check("the expected site comes from ASAN's own frame",
              (pop["CRASH-086"]["asan_file"], pop["CRASH-086"]["asan_line"]),
              ("ext/standard/basic_functions.c", 2135))
        check("the pristine oracle is carried", pop["CRASH-086"]["pristine_stdout"], "done\n")


def synth_record(cases, mode, revokes, *, case_faults=(), sanity_ok=True, probe=True,
                 stdout="done", sha=SHA, probe_sha=PROBE_SHA):
    seam = f"ZEND_CAP_MODE {mode} exact={1 if mode & 1 else 0} revoke={1 if mode & 2 else 0}"
    stats = STATS % (100, 9216)
    out, exits = {}, {}
    out[R.SANITY] = [seam, f"PHP_VIRTUAL SEAM blocks=9 frees=9 issues=9 retires=9 revokes={revokes}",
                     "sanity 600 200 17" if sanity_ok else "", stats]
    exits[R.SANITY] = 0 if sanity_ok else 139
    if probe is not None:
        out[R.BOUND_PROBE] = ["S7 rounded arm read index 9 = 0x00 -- the slack is reachable",
                              FAULT % probe_sha] if probe else ["S7 rounded arm read index 9 = 0x00"]
        exits[R.BOUND_PROBE] = 139 if probe else 0
    for c in cases:
        if c in case_faults:
            out[c] = [seam, FAULT % sha, stats]
            exits[c] = 139
        else:
            out[c] = [seam, stdout, stats]
            exits[c] = 0
    return {"booted": True, "reached_script": True, "completed": True, "panic": False,
            "exits": exits, "output": out, "serial_log": "x"}


def drive(tmp, *, revokes=(0, 0, 77, 77), faults=None, extra_argv=(), modes=None):
    """Run main() end to end over stored records, with the toolchain stubbed out."""
    root = tmp / "corpus-root"
    fake_corpus(root)
    out = tmp / "out"
    out.mkdir(exist_ok=True)
    cases = ["CRASH-057", "CRASH-086", "CRASH-092", "CRASH-110"]
    faults = faults or {}
    declared = [verdicts.load_arms()[a]["mode"] for a in R.ARMS]
    modes = modes or declared
    for arm, mode, rev in zip(R.ARMS, modes, revokes):
        probe = True if R.BOUND_PROBE in verdicts.load_arms()[arm]["controls"] else None
        rec = synth_record(cases, mode, rev, case_faults=faults.get(arm, ()), probe=probe)
        (out / f"{arm}-run.json").write_text(json.dumps(rec))
    img = tmp / "php-real-seam.dom"
    img.write_bytes(b"\x7fELF" + b"\0" * 64)

    real_run, real_call = subprocess.run, subprocess.call
    def fake_run(cmd, *a, **k):
        prog = str(cmd[0])
        text = ""
        if "sha256sum" in prog:
            text = (PROBE_SHA if str(cmd[1]).endswith("p.dom") else SHA) + "  x\n"
        elif "objdump" in prog:
            text = "revoke\nrevoke\n"
        elif "clang" in prog:
            text = "clang version test\n"
        elif "symbolizer" in prog:
            text = "??\n/b/ext/standard/basic_functions.c:2135:9\n"
        elif "readelf" in prog:
            text = "  LOAD           0x000000 0x0000000000010000 0x0000000000010000\n"
        elif prog.endswith("llvm-nm"):
            text = "0000000000010000 00000010 T main\n"
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
        rc = drive(tmp, faults={"php-real-exact": ("CRASH-086", "CRASH-110", "CRASH-057",
                                                   "CRASH-092"),
                                "php-real-object": ("CRASH-086",)})
        check("a clean measurement completes", rc, 0)
        b = tmp / "out" / "bundle-php-real-exact" / "verdicts.jsonl"
        check("the bundle is written", b.is_file(), True)
        rows = [json.loads(l) for l in b.read_text().splitlines()]
        check("every case is judged", len(rows), 4)
        by_case = {r["observation"]["case"]: r for r in rows}
        check("the fault arm caught the two with a premise",
              sorted(by_case[c]["verdict"] for c in ("CRASH-086", "CRASH-110")),
              [verdicts.CAUGHT, verdicts.CAUGHT])
        # The oracle says the pristine build crashes on CRASH-057, so even a fault at ASAN's own
        # site is not a reading: there is no silence on the control arm to compare it against.
        check("a case the pristine build crashes on is not a reading",
              (by_case["CRASH-057"]["verdict"], by_case["CRASH-057"]["reason"]),
              (verdicts.NO_READING, "setup-fault"))
        # CRASH-092 faults at ASAN's site on the exact arm here, so its precondition is proved
        # and that licenses the catch.
        check("a precondition case that faults at ASAN's site is CAUGHT",
              by_case["CRASH-092"]["verdict"], verdicts.CAUGHT)
        inputs = json.loads((tmp / "out" / "bundle-php-real-exact" / "inputs.json").read_text())
        check("the pre-registration is recorded", inputs["pre_registration"], "deadbeef")

    # ... and when NOTHING faults for it, its silence must be refused rather than read as MISSED.
    with tempfile.TemporaryDirectory() as d2:
        tmp2 = Path(d2)
        drive(tmp2, faults={"php-real-exact": ("CRASH-086",)})
        rows2 = [json.loads(l) for l in
                 (tmp2 / "out" / "bundle-php-real-stock" / "verdicts.jsonl").read_text().splitlines()]
        got = {r["observation"]["case"]: (r["verdict"], r["reason"]) for r in rows2}
        check("an unconfirmed precondition refuses the silence",
              got["CRASH-092"], (verdicts.NO_READING, "not-reached"))
        check("a case with no precondition still reads MISSED",
              got["CRASH-086"][0], verdicts.MISSED)

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
        for arm in R.ARMS:
            probe = True if R.BOUND_PROBE in verdicts.load_arms()[arm]["controls"] else None
            rec = synth_record(["CRASH-086", "CRASH-110"], verdicts.load_arms()[arm]["mode"], 77,
                               case_faults=("CRASH-086",), probe=probe, sha=sha,
                               probe_sha=probe_sha)
            (out / f"{arm}-run.json").write_text(json.dumps(rec))
        img = tmp / "php-real-seam.dom"; img.write_bytes(b"\x7fELF")
        real_run, real_call = subprocess.run, subprocess.call
        subprocess.run = lambda cmd, *a, **k: types.SimpleNamespace(
            stdout=((PROBE_SHA if str(cmd[1]).endswith("p.dom") else SHA) + "  x\n")
            if "sha256sum" in str(cmd[0]) else "revoke\n", stderr="", returncode=0)
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


def lint():
    print("\n-- lint --")
    for mod in ("pyflakes", "flake8"):
        p = subprocess.run([sys.executable, "-m", mod, str(HERE / "run-realphp.py"),
                            str(Path(__file__))], capture_output=True, text=True)
        if p.returncode != 127 and (p.stdout or p.stderr or p.returncode == 0):
            if p.stdout.strip():
                print(p.stdout.strip())
                FAILURES.append(f"{mod} reported findings")
            else:
                print(f"  ok   {mod} clean")
            return
    print("  NOTE: neither pyflakes nor flake8 is installed, so NO linter ran. This is reported "
          "rather than skipped silently: an absent check reads exactly like a passing one.")


if __name__ == "__main__":
    helper_cases()
    population_cases()
    observation_cases()
    main_path_cases()
    stale_image_case()
    lint()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES")
        for f in FAILURES:
            print(f"  {f}")
        sys.exit(1)
    print("all checks pass")
