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
  temporal-asan-visible   CWE-416 that ASan reports unaided: efree reached free(), so a
                          detector at the libc boundary sees it. The control population.

Each case directory holds trigger.php (the corpus script, byte for byte), oracle.json (what
run-realphp.py needs: population, ASan's first PHP frame, the pristine run's outcome and
stdout), case.json (the framework's script-trigger record) and PROVENANCE.md.

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
        elif meta[cid]["cwe"] == "CWE-416" and e["outcome"].startswith("ASAN:"):
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
    crashed = p["outcome"].startswith("SIGNAL:")
    fix = fix_id(m["upstream_fix_commit"])
    name = dir_name(number, fix, cid)
    cls = TEMPORAL.get(m["cwe"]) or SPATIAL.get(m["cwe"]) or {"CWE-562": "stack-address-escape"}.get(m["cwe"], m["cwe"].lower())
    consumer = m["c_file_line"].split(" ")[0].rstrip(";,")
    oracle = {
        "id": cid, "population": pop, "cwe": m["cwe"], "category": m["category"],
        "requires": m["requires"],
        "asan": {"class": a["outcome"], "func": func, "file": path, "line": line},
        "stockalloc": stock[cid]["outcome"],
        "pristine": {"outcome": p["outcome"], "rc": p["rc"], "stdout": p.get("stdout", "")},
        "crashes_on_pristine_build": crashed,
        "collaborator_population": theirs["pops"].get(cid, "neither"),
        "sources": {"results-ours-asan-default.json": oracle_shas["asan"],
                    "results-ours-asan-stockalloc.json": oracle_shas["stock"],
                    "results-ours-pristine-default.json": oracle_shas["pristine"]},
    }
    why = ("ASan reports it only with PHP's allocator patched (cache off, no 8-byte rounding); with "
           "zend_alloc.c as shipped the run reads OK, so emalloc's cache or rounding hides it from "
           "any detector at free()" if pop == "asan-blind" else
           "a CWE-416 that ASan reports with zend_alloc.c as shipped: efree reached free(), so a "
           "detector at the libc boundary sees it; the control population")
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
                        "the judge reads such a case out of the denominator unless an arm faults at the ASan site. "
                        if crashed else "The pristine build completes it natively, so an arm that completes it saw nothing. ")
                     + (f"Needs: {m['requires']}. " if m["requires"] else "")),
        "arms": {arm: {"oracle": text} for arm, text in ARMS.items()},
        "status": "built 2026-10-27 from the corpus and our native oracle; not yet run on the virtual profile from this corpus",
        "note": f"population: {pop} -- {why}. Collected by the corpus as {m['how_collected']['origin_stream']} ({m['how_collected']['history_status']}, confidence {m['how_collected']['confidence']}).",
    }
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


def main():
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
