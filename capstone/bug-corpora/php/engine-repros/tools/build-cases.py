#!/usr/bin/env python3
"""Build the case directories of php/engine-repros from the PHP 5.0.0 bug corpus and our own
native runs of it.

    build-cases.py --corpus <unpacked php-5.0.0-bug-corpus> --oracles <dir> [--write]

The corpus is the collaborator's: 185 cases, each a .php trigger with a metadata record
(CWE, the defective file:line in the 5.0.0 source, the php-src fix commit, how it was
collected). It is not in this repository; `--corpus` is its unpacked zip, identified by
sha256 in every PROVENANCE.md. What IS in this repository is derived from it by this script,
so a reader can rebuild the cases from the zip and the three result files and diff.

The ORACLE is ours, not the corpus's: `--oracles` holds results-ours-{asan-default,
asan-stockalloc,pristine-default}.json, written by the corpus's own run-corpus.py against
three builds we made of the pinned tarball on our host (the corpus README's recipe: its asan
variant, that variant without the two zend_alloc.c detector patches = `stockalloc`, and the
plain build). The corpus ships its own asan/pristine runs; it does not ship stockalloc, and
the two populations below are DEFINED by differencing against stockalloc, so the one file the
populations rest on had to be measured here. Where our run and the corpus's disagree on a
case, PROVENANCE.md says so, and ours is what oracle.json carries.

Populations (the port's two measured sets, recomputed here, never retyped):

  asan-blind              ASan reports the defect with PHP's allocator PATCHED (cache off, no
                          rounding) and reads OK with it AS SHIPPED: emalloc's cache or its
                          8-byte rounding hides the defect from any detector at free().
  temporal-asan-visible   CWE-416 that ASan reports unaided (as shipped) and classifies as a
                          use-after-free (patched build): efree reached free(), so a detector
                          at the libc boundary sees it. The control population. A case no
                          build reports as a use-after-free is neither (CRASH-064).

Each case directory holds trigger.php (the corpus script, byte for byte), oracle.json (what
run-realphp.py needs: population, ASan's first PHP frame and its whole ACCESS stack, the kind
of requirement the case declares, the pristine run's outcome and stdout), case.json (the
framework's script-trigger record, with `fault_sites` naming the one function a fault may land
in and be this defect's) and PROVENANCE.md.

--write creates or rewrites the directories; without it the script only reports what it
would do and whether existing directories differ. Idempotent: the same inputs give the same
bytes.
"""
import argparse, hashlib, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE.parent                      # bug-corpora/php/engine-repros
REPO = CORPUS_DIR.parents[3]
POPULATIONS = ("asan-blind", "temporal-asan-visible")
# The same frame rule as ports/php/app/run-realphp.py: ASan's first frame inside PHP's tree.
ASAN_FRAME = re.compile(r"#\d+ 0x[0-9a-f]+ in (\S+) [^ ]*?((?:Zend|main|ext|sapi|TSRM|regex)/[\w./]+):(\d+)")
# One frame of any ASan stack: index, function, the rest of the line (a source location or a
# `(binary+0x..)` for a frame without line tables).
ANY_FRAME = re.compile(r"^\s*#(\d+) 0x[0-9a-f]+ in (\S+) (.*)$")
PHP_PATH = re.compile(r"((?:Zend|main|ext|sapi|TSRM|regex)/[\w./]+):(\d+)")
# A requirement the corpus declares per case is one of two kinds, and the judge treats them
# differently: an EXECUTION requirement (a writable /tmp, a working loopback) is a precondition
# the trigger needs met on OUR platform to run at all, so a silence on an arm is a reading only
# once something confirmed it was met. A DETECTOR OPTION (an ASAN_OPTIONS knob) is a
# requirement on ASan's own instrument -- what it needs to REPORT the defect -- and is not a
# precondition for the trigger running here at all; refusing a case because Capstone has not
# enabled an ASan option would be unsound. Classified here, from the declaration's text, never
# from a run.
DETECTOR_OPTION = re.compile(r"^\s*ASAN_OPTIONS\b")


# A PLATFORM PRECONDITION a case declares beyond its execution requirement: a libc behaviour the
# defect needs. CRASH-084's use-after-free exists only where putenv keeps an empty-name string in
# environ (glibc); musl's putenv does not, so on this platform the dangling reference never comes
# to be. The declaration is checked by a probe every run (ports/php/app/probe-putenv.c), never
# asserted: the runner excludes the case only when the probe PROVES the behaviour absent, and
# judges it normally when the probe is missing or fails. Kept here, with the generator, so that
# regenerating the corpus keeps it; the schema is bug-corpora/SCHEMA.md "platform_precondition".
# A SECOND DECLARED SITE a case carries beyond ASan's first PHP frame: the access path this
# platform takes to the same dead object, read off the source and the trigger's own account.
# CRASH-057: natively the copy at assignment reads the freed temporary (_estrndup); here the
# assignment keeps the temporary's buffer and var_dump prints it (php_var_dump, var.c:126).
EXTRA_FAULT_SITES = {
    "CRASH-057": ({"php_var_dump": "ext/standard/var.c"}, " On this platform the access is not the copy: the assignment keeps the temporary's buffer as the result's (the trigger's own account: the freed IS_TMP_VAR is reused as the assignment's result value), and var_dump($x) prints it -- php_var_dump, ext/standard/var.c:126, the PHPWRITE of Z_STRVAL -- through main/output.c and the CLI's write, which pass the buffer through. php_var_dump is therefore a declared site: the print of the reused temporary that the trigger documents, read off the source, with the nested arm's frame chain (residues/057-chain, 2026-10-28) as the confirming reading, not the admitting one."),
}

# RETRACTED 2026-10-28, same day it was declared: a precondition for CRASH-092, "select() rejects
# a timeval whose tv_usec is out of range (EINVAL) rather than sleeping on it". PHP 5.0.0's
# liveness check (main/streams/xp_socket.c, PHP_STREAM_OPTION_CHECK_LIVENESS) hands select() a
# timeval with only tv_sec set, and musl 1.2.5 folds an out-of-range tv_usec into the seconds
# (`s += us/1000000`), so the script can hang before its defect. The declaration said Linux
# rejects such a timeval with EINVAL. It does not: fs/select.c (v6.12, kern_select) computes
# `tv.tv_sec + (tv.tv_usec / USEC_PER_SEC)` and rejects only a NEGATIVE value, so glibc's select
# folds a positive overflow exactly as musl does. The hang is therefore not a libc behaviour the
# defect needs but whatever uninitialised stack contents the run happens to pass, on either
# libc; a probe with a fixed value cannot read that. CRASH-092 carries no precondition: a run
# that hangs is `infra`, one that completes without reaching the defect is `not-reached`, and
# the case stays in the denominator. The probe (probe-timeval.c) was removed with the declaration.
PLATFORM_PRECONDITIONS = {
    "CRASH-084": {
        "platform_precondition": {
            "needs": "putenv keeps an empty-name string referenced from environ",
            "probe": "putenv",
            "present": "PUTENV_PROBE empty-name retained=1",
            "absent": "PUTENV_PROBE empty-name retained=0"
        },
        "platform_precondition_why": "musl 1.2.5's putenv (src/env/putenv.c) takes the unsetenv branch for an empty name -- `if (!l || !s[l]) return unsetenv(s);` -- and unsetenv refuses a name containing '=' with EINVAL, so the string is never stored in environ. The use-after-free this case records needs that stored reference: the second empty-name putenv frees the string and the following environ scan reads it (the corpus's native oracle, glibc). Without the reference there is nothing dangling, so a run that completes on this libc is not a miss of the protection; it is a defect whose precondition this platform does not meet. Checked, not asserted: ports/php/app/probe-putenv.c runs on every arm and says what this libc retains, with a control that proves it can see retention at all.",
    },
}


def requires_kind(text):
    if not text.strip():
        return ""
    return "detector-option" if DETECTOR_OPTION.match(text) else "execution"


def access_stack(stderr):
    """ASan's ACCESS stack -- the first stack of the report, the one under the READ/WRITE line --
    as a list of frames, in order. A frame inside PHP's tree carries file and line; any other
    (a libc routine, a sanitizer interceptor) carries only the function and where it came
    from, so the judge can see that the leaf is not PHP without mistaking it for a site.

    The allocation and free stacks that follow in the report are NOT taken: a frame from
    those names where the object was made or released, which is not where the defect's access
    is, and admitting one would turn a fault at any free() into this case's catch.
    """
    frames, started = [], False
    for line in (stderr or "").splitlines():
        m = ANY_FRAME.match(line)
        if not m:
            if started:
                break
            continue
        started = True
        index, func, rest = int(m.group(1)), m.group(2), m.group(3)
        php = PHP_PATH.search(rest)
        if php:
            frames.append({"index": index, "func": func, "file": php.group(1), "line": int(php.group(2))})
        else:
            frames.append({"index": index, "func": func, "outside": rest.split("(")[0].strip() or rest.strip()})
    return frames
TEMPORAL = {"CWE-416": "temporal", "CWE-415": "temporal", "CWE-590": "temporal"}
SPATIAL = {"CWE-125": "spatial", "CWE-787": "spatial", "CWE-190": "spatial"}
ARMS = {
    "virtual-malloc": "the case on the virtual profile with the seam present but returning every freed block's loan, so only musl mallocng revokes, per BLOCK, when PHP's cache hands a block back to free(): a fault at the ASan site, or the trigger ends and this arm did not see it (tools/arms.json php-real-block)",
    "virtual-nested-pools": "the same image with the allocator seam retiring the payload at EVERY _efree, cache-absorbed or not, and re-deriving it under a fresh node when the cache hands the block to its next tenant: a fault at the ASan site, or the trigger ends and this arm did not see it (php-real-object)",
    "virtual-nested-exact": "virtual-nested-pools with each payload bounded to the caller's request instead of REAL_SIZE(n): the arm that can see a defect inside PHP's own rounding slack (php-real-exact)",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def populations(asan, stock, meta):
    """{id: population} over the cases the default build runs, from the three result files."""
    out = {}
    for cid, e in asan.items():
        blind = e["outcome"] != "OK" and stock.get(cid, {}).get("outcome") == "OK"
        if blind:
            out[cid] = "asan-blind"
        elif (meta[cid]["cwe"] == "CWE-416" and stock.get(cid, {}).get("outcome", "").startswith("ASAN:")
              and e["outcome"] == "ASAN:heap-use-after-free"):
            # Visible means ASan REPORTS it with the allocator as shipped -- the stockalloc run,
            # not the patched one. A case the as-shipped build ends with PHP's own fatal error
            # (EXIT:255, no report) is visible to nobody and belongs to neither population; the
            # first build of this corpus tested the patched run here and admitted two such.
            # And the defect has to BE a use-after-free as ASan classifies one: with the
            # allocator as shipped 18 of these die as ASan:SEGV (the stale access lands on
            # reused memory), so the class is read from the patched run, which names them
            # heap-use-after-free. A case NEITHER build reports as a use-after-free is not one
            # a detector at free() can see: CRASH-064 dereferences an integer read from a
            # reused object-store bucket (the free-list link written over the object pointer
            # in the bucket's union) -- a bad pointer, not a freed one, on both builds -- and
            # does not belong to the population that exists as the control for such detectors.
            out[cid] = "temporal-asan-visible"
    return out


def dir_name(number, fix, cid):
    return f"{number:02d}_{fix}_{cid.lower()}"


def fix_id(commit):
    commit = commit.strip()
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise SystemExit(f"fix commit {commit!r} is not a hex id")
    return commit[:12]


def collaborator_populations():
    """The populations the collaborator's instrument measured (ports/php/app/results/2026-10-26),
    read from its verdict bundles: id -> population. Recorded per case so the two instruments'
    disagreement is visible case by case, never summed away."""
    out = {}
    root = REPO / "capstone/ports/php/app/results/2026-10-26"
    for pop in POPULATIONS:
        path = root / pop / "php-real-exact" / "verdicts.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                out[json.loads(line)["observation"]["case"]] = pop
    return out


def one_case(number, cid, pop, meta, asan, stock, pris, theirs, corpus_sha, oracle_shas, hosts):
    m, a, p = meta[cid], asan[cid], pris[cid]
    frame = ASAN_FRAME.search(a.get("stderr", ""))
    if not frame:
        raise SystemExit(f"{cid}: our ASan stderr has no PHP source frame; no site for the judge")
    func, path, line = frame.group(1), frame.group(2), int(frame.group(3))
    stack = access_stack(a.get("stderr", ""))
    site = next((f for f in stack if "file" in f), None)
    if site is None or (site["func"], site["file"], site["line"]) != (func, path, line):
        raise SystemExit(f"{cid}: the access stack's first PHP frame is not the frame rule's "
                         f"({site} vs {func} {path}:{line}); the report has more than one stack "
                         f"before its PHP frame, or the parser disagrees with itself")
    leaf = stack[0] if stack and "outside" in stack[0] else None
    crashed = p["outcome"].startswith("SIGNAL:")
    fix = fix_id(m["upstream_fix_commit"])
    name = dir_name(number, fix, cid)
    cls = TEMPORAL.get(m["cwe"]) or SPATIAL.get(m["cwe"]) or {"CWE-562": "stack-address-escape"}.get(m["cwe"], m["cwe"].lower())
    consumer = m["c_file_line"].split(" ")[0].rstrip(";,")
    oracle = {
        "id": cid, "population": pop, "cwe": m["cwe"], "category": m["category"],
        "requires": m["requires"],
        "requires_kind": requires_kind(m["requires"]),
        "asan": {"class": a["outcome"], "func": func, "file": path, "line": line, "stack": stack},
        "stockalloc": stock[cid]["outcome"],
        "pristine": {"outcome": p["outcome"], "rc": p["rc"], "stdout": p.get("stdout", "")},
        "crashes_on_pristine_build": crashed,
        "collaborator_population": theirs["pops"].get(cid, "neither"),
        "sources": {"results-ours-asan-default.json": oracle_shas["asan"],
                    "results-ours-asan-stockalloc.json": oracle_shas["stock"],
                    "results-ours-pristine-default.json": oracle_shas["pristine"]},
    }
    why = ("ASan reports it only with PHP's allocator patched (cache off, no 8-byte rounding); with "
           "zend_alloc.c as shipped the run reads OK. That is the selection, by measurement; which "
           "mechanism hides it, and which arm sees it, is what the arms record" if pop == "asan-blind" else
           "a CWE-416 that ASan reports with zend_alloc.c as shipped (the stockalloc run): efree "
           "reached free(), so a detector at the libc boundary sees it; the control population")
    case = {
        "case": number,
        "upstream_fix": fix,
        "title": f"{cid}: {m['root_cause_id'].replace('-', ' ')} ({m['category']})",
        "consumer": consumer,
        "class": cls,
        "trigger": "trigger.php",
        "live_in_pin": True,
        "live_proof": (f"measured, not asserted: our native ASan build of the pinned 5.0.0 tarball "
                       f"({hosts['asan']}) reports {a['outcome']} at {path}:{line} in {func} when "
                       f"trigger.php runs; the corpus records the fix as php-src {m['upstream_fix_commit'].strip()}, "
                       f"which postdates the release, and names the defect at {m['c_file_line']}"),
        "fidelity": ("the corpus's own trigger script, run through the real interpreter (PHP's cli SAPI, "
                     "ext/standard only, the port's 145 translation units) rather than reduced to a C "
                     "model. What it does NOT establish is which arm ought to catch it. "
                     + ("The pristine build crashes on it natively, so a silence on an arm is not a MISSED: "
                        "under the judge's default scoring the case reads out-of-denominator on every arm; the "
                        "sensitivity scoring (run-realphp.py --score-oracle-crashes) reads it by fault site instead. "
                        if crashed else "The pristine build completes it natively, so an arm that completes it saw nothing. ")
                     + (f"Needs: {m['requires']}. " if m["requires"] else "")),
        # The one function a fault may land in and be this defect's: the defective access as
        # ASan's own access stack names it, from our native run -- an independent measurement,
        # never the Capstone fault it will admit. ONE function, not the stack: `execute`,
        # `main` and the libc routine under the access are on every script's stack and would
        # make any fault this case's. A libc leaf (strlen under xbuf_format_converter) is
        # named in the reason and deliberately NOT declared: a fault in strlen is attributable
        # only through its caller, which the launcher does not report.
        "fault_sites": [func],
        "fault_sites_why": (f"ASan's access stack in our native run (results-ours-asan-default.json, sha256 "
                            f"{oracle_shas['asan'][:12]}) has {func} at {path}:{line} as its first frame inside "
                            f"PHP's tree (frame #{site['index']}"
                            + (f", under the libc leaf {leaf['func']} at frame #0, which is not declared: a fault "
                               f"there can be tied to this case only through its caller, and the launcher reports "
                               f"no return address" if leaf else "")
                            + "); the stack's remaining frames are the interpreter's dispatch and are not sites"),
        "arms": {arm: {"oracle": text} for arm, text in ARMS.items()},
        "status": "built 2026-10-27 from the corpus and our native oracle; what has run is in this record's arms (verdict, verdict_from) and under results/",
        "note": f"population: {pop} -- {why}. Collected by the corpus as {m['how_collected']['origin_stream']} ({m['how_collected']['history_status']}, confidence {m['how_collected']['confidence']}).",
    }
    case.update(PLATFORM_PRECONDITIONS.get(cid, {}))
    # The file each declared site must be seen in: ASan's own frame for the main site, the table
    # above for a site declared beyond ASan's stack. A judge matches a site by function AND file,
    # so a same-named routine in another file is never the site (review 2026-10-28).
    case["fault_site_files"] = {func: path}
    if cid in EXTRA_FAULT_SITES:
        # `extra_why`, not `why`: the population's reason is still needed by PROVENANCE.md below,
        # and reusing its name here wrote the second site's reason into CRASH-057's population
        # line (caught on the 2026-10-28 regeneration).
        extra, extra_why = EXTRA_FAULT_SITES[cid]
        case["fault_sites"] = case["fault_sites"] + [s for s in extra if s not in case["fault_sites"]]
        case["fault_site_files"].update(extra)
        case["fault_sites_why"] = case["fault_sites_why"] + extra_why
    disagreements = []
    t = theirs
    if t["asan"].get(cid, {}).get("outcome") != a["outcome"]:
        disagreements.append(f"ASan class: corpus recorded {t['asan'].get(cid, {}).get('outcome')!r}, ours {a['outcome']!r}")
    if t["pristine"].get(cid, {}).get("outcome") != p["outcome"]:
        disagreements.append(f"pristine outcome: corpus recorded {t['pristine'].get(cid, {}).get('outcome')!r}, ours {p['outcome']!r}")
    their_pop = theirs["pops"].get(cid, "neither")
    if their_pop != pop:
        disagreements.append(f"population: the collaborator's instrument (gcc 13 ASan, 2026-10-26) placed it in {their_pop!r}, ours (gcc 15 ASan) in {pop!r} -- the stockalloc run decides, and that run's detector differs")
    flag = str(m["crashes_on_pristine_build"])
    if flag.startswith(("True", "False")) and (flag.startswith("True") != crashed):
        disagreements.append(f"crashes_on_pristine_build: corpus says {flag}, our pristine run read {p['outcome']}")
    prov = f"""# {cid}: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `{corpus_sha}`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{json.dumps(m, indent=1)}
```

**Trigger.** `trigger.php` is `corpus/{cid}.php` from that zip, byte for byte (sha256
`{sha256(Path(hosts['corpus_root']) / 'corpus' / f'{cid}.php')}`).

**Oracle, ours.** Measured on {hosts['asan']} with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `{a['outcome']}` at `{path}:{line}` in
`{func}`, stockalloc `{stock[cid]['outcome']}`, pristine `{p['outcome']}` (rc {p['rc']}).
Hence **{pop}**: {why}.

**Where our run and the corpus's record disagree:** {('; '.join(disagreements)) if disagreements else 'nowhere -- ASan class, pristine outcome and the crash flag all agree'}.

**Upstream fix.** php-src `{m['upstream_fix_commit'].strip()}`, the directory prefix `{fix}`.
"""
    return name, oracle, case, prov


def self_test():
    """The population rule and the frame rule on synthetic entries; the derivations the offline
    suite of run-realphp.py no longer exercises since its fixture became case directories."""
    meta = {k: {"cwe": v} for k, v in {"A": "CWE-416", "B": "CWE-416", "C": "CWE-416", "D": "CWE-125", "E": "CWE-416", "F": "CWE-416", "G": "CWE-416", "H": "CWE-416"}.items()}
    asan = {"A": {"outcome": "ASAN:heap-use-after-free"}, "B": {"outcome": "ASAN:heap-use-after-free"},
            "C": {"outcome": "ASAN:heap-use-after-free"}, "D": {"outcome": "ASAN:heap-buffer-overflow"},
            "E": {"outcome": "OK"}, "F": {"outcome": "ASAN:heap-use-after-free"}, "G": {"outcome": "ASAN:SEGV"},
            "H": {"outcome": "ASAN:heap-use-after-free"}}
    stock = {"A": {"outcome": "OK"},                              # blind: patched reports, as shipped OK
             "B": {"outcome": "ASAN:heap-use-after-free"},        # visible: as shipped reports
             "C": {"outcome": "EXIT:255"},                        # PHP's own fatal error as shipped: neither
             "D": {"outcome": "OK"},                              # blind, spatial
             "E": {"outcome": "OK"},                              # nothing anywhere: neither
             "F": {"outcome": "SIGNAL:11"},                       # crashes as shipped, no report: neither
             "G": {"outcome": "ASAN:SEGV"},                       # SEGV on both builds (CRASH-064): neither
             "H": {"outcome": "ASAN:SEGV"}}                       # SEGV as shipped, UAF patched: visible
    got = populations(asan, stock, meta)
    want = {"A": "asan-blind", "B": "temporal-asan-visible", "D": "asan-blind", "H": "temporal-asan-visible"}
    checks = [("the population rule", got, want)]
    frame = ASAN_FRAME.search("    #0 0x55 in user_tick_function_call /x/y/ext/standard/basic_functions.c:2135\n")
    checks.append(("the frame rule takes the first PHP frame", (frame.group(1), frame.group(2), frame.group(3)),
                   ("user_tick_function_call", "ext/standard/basic_functions.c", "2135")))
    checks.append(("a frame outside PHP's tree is not a site", ASAN_FRAME.search("    #0 0x55 in free /lib/libasan.so\n"), None))
    # The access stack: the FIRST stack only, PHP frames with their site, a libc leaf named as
    # outside PHP. The free and allocation stacks that follow are not taken -- a frame from
    # those is where the object was released, which a fault at any free() would match.
    report = ("==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1 at pc 0x2\n"
              "READ of size 3 at 0x1 thread T0\n"
              "    #0 0x3 in strlen ../../src/libsanitizer/sanitizer_common_interceptors.inc:425\n"
              "    #1 0x4 in xbuf_format_converter /b/php-5.0.0/main/spprintf.c:518:7\n"
              "    #2 0x5 in execute /b/php-5.0.0/Zend/zend_execute.c:1391:3\n"
              "\n"
              "0x1 is located 0 bytes inside of 2-byte region\n"
              "freed by thread T0 here:\n"
              "    #0 0x6 in free ../../src/libsanitizer/asan_malloc_linux.cpp:52\n"
              "    #1 0x7 in zend_register_constant /b/php-5.0.0/Zend/zend_constants.c:320:4\n")
    checks.append(("the access stack is the first stack, in order, leaf marked as outside PHP",
                   access_stack(report),
                   [{"index": 0, "func": "strlen", "outside": "../../src/libsanitizer/sanitizer_common_interceptors.inc:425"},
                    {"index": 1, "func": "xbuf_format_converter", "file": "main/spprintf.c", "line": 518},
                    {"index": 2, "func": "execute", "file": "Zend/zend_execute.c", "line": 1391}]))
    checks.append(("the free stack is not taken", any(f["func"] == "zend_register_constant" for f in access_stack(report)), False))
    checks.append(("no stack, no frames", access_stack("nothing here\n"), []))
    checks.append(("an ASAN_OPTIONS requirement is a detector option",
                   requires_kind("ASAN_OPTIONS detect_stack_use_after_return=1"), "detector-option"))
    checks.append(("a loopback requirement is an execution requirement",
                   requires_kind("network: working loopback"), "execution"))
    checks.append(("no requirement, no kind", requires_kind(""), ""))
    checks.append(("CRASH-084 declares its libc precondition with a probe and its two readings",
                   sorted(PLATFORM_PRECONDITIONS["CRASH-084"]["platform_precondition"]), ["absent", "needs", "present", "probe"]))
    checks.append(("CRASH-057 declares var_dump's print of the reused temporary as a second site",
                   list(EXTRA_FAULT_SITES["CRASH-057"][0]), ["php_var_dump"]))
    checks.append(("with the file the site must be seen in",
                   EXTRA_FAULT_SITES["CRASH-057"][0]["php_var_dump"], "ext/standard/var.c"))
    checks.append(("and says why, naming the source line",
                   "var.c:126" in EXTRA_FAULT_SITES["CRASH-057"][1], True))
    checks.append(("CRASH-092 declares no precondition (retracted 2026-10-28: Linux folds a positive tv_usec too)",
                   "CRASH-092" in PLATFORM_PRECONDITIONS, False))
    checks.append(("only CRASH-084 declares one", sorted(PLATFORM_PRECONDITIONS), ["CRASH-084"]))
    checks.append(("and says why, naming the musl branch",
                   "unsetenv" in PLATFORM_PRECONDITIONS["CRASH-084"]["platform_precondition_why"], True))
    checks.append(("the fix id is the first 12 hex digits", fix_id("4f161fe28997f5d1f818fc6ae95a8131066cf1a2"), "4f161fe28997"))
    checks.append(("an 11-digit fix id stays as given", fix_id("b599e434add"), "b599e434add"))
    bad = [(n, g, w) for n, g, w in checks if g != w]
    for n, g, w in bad:
        print(f"  FAIL {n}: got {g!r}, want {w!r}")
    print(f"self-test: {len(checks) - len(bad)} of {len(checks)} pass")
    return 0 if not bad else 1


def main():
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True, help="the unpacked php-5.0.0-bug-corpus directory")
    ap.add_argument("--corpus-zip", type=Path, required=True, help="the zip it was unpacked from, for its sha256")
    ap.add_argument("--oracles", type=Path, required=True, help="dir with results-ours-*.json")
    ap.add_argument("--host", default="p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27",
                    help="where and with what the oracle runs were made")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    meta = {e["id"]: e for e in load(a.corpus / "metadata" / "corpus.json")}
    asan = {e["id"]: e for e in load(a.oracles / "results-ours-asan-default.json")["results"]}
    stock = {e["id"]: e for e in load(a.oracles / "results-ours-asan-stockalloc.json")["results"]}
    pris = {e["id"]: e for e in load(a.oracles / "results-ours-pristine-default.json")["results"]}
    theirs = {"asan": {e["id"]: e for e in load(a.corpus / "results-asan-default.json")["results"]},
              "pristine": {e["id"]: e for e in load(a.corpus / "results-pristine-default.json")["results"]},
              "pops": collaborator_populations()}
    # Only the cases the default (ext/standard-only) build runs are candidates: the corpus's own
    # asan-default run is that build, so its id set is the denominator of the whole corpus here.
    default_ids = set(theirs["asan"])
    pops = {cid: p for cid, p in populations({i: asan[i] for i in default_ids}, stock, meta).items()}
    order = sorted(pops, key=lambda c: (POPULATIONS.index(pops[c]), c))
    counts = {p: sum(1 for c in pops.values() if c == p) for p in POPULATIONS}
    print(f"populations from our oracle: {counts} over {len(default_ids)} default-build cases")
    tp = theirs["pops"]
    moved = sorted((c, tp.get(c, "neither"), pops[c]) for c in pops if tp.get(c, "neither") != pops[c])
    gone = sorted(c for c in tp if c not in pops)
    print(f"against the collaborator's populations: {len(moved)} placed differently, {len(gone)} of theirs absent here: {gone}")
    corpus_sha = sha256(a.corpus_zip)
    oracle_shas = {"asan": sha256(a.oracles / "results-ours-asan-default.json"),
                   "stock": sha256(a.oracles / "results-ours-asan-stockalloc.json"),
                   "pristine": sha256(a.oracles / "results-ours-pristine-default.json")}
    hosts = {"asan": a.host, "corpus_root": a.corpus}
    seen, differ = set(), []
    for n, cid in enumerate(order, 1):
        name, oracle, case, prov = one_case(n, cid, pops[cid], meta, asan, stock, pris, theirs, corpus_sha, oracle_shas, hosts)
        d = CORPUS_DIR / name
        seen.add(name)
        files = {"trigger.php": (a.corpus / "corpus" / f"{cid}.php").read_bytes(),
                 "oracle.json": (json.dumps(oracle, indent=1, sort_keys=True) + "\n").encode(),
                 "case.json": (json.dumps(case, indent=1) + "\n").encode(),
                 "PROVENANCE.md": prov.encode()}
        for fname, data in files.items():
            path = d / fname
            if not path.exists() or path.read_bytes() != data:
                differ.append(str(path.relative_to(REPO)))
                if a.write:
                    d.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
    stale = [p.name for p in CORPUS_DIR.glob("[0-9][0-9]_*") if p.name not in seen]
    if stale:
        print(f"directories not produced by this input (remove by hand): {stale}")
    print(f"{len(order)} cases; {len(differ)} files {'written' if a.write else 'would change'}")
    return 0 if (a.write or not differ) else 1


if __name__ == "__main__":
    sys.exit(main())
