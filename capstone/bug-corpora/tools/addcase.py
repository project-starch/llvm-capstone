#!/usr/bin/env python3
"""Add a case to a corpus: write the three files, recompute the two counters, extend the tables.

Only the reduction (case.c) is hand-written. Everything the checker validates -- the schema, the
arm set, the arm keys -- is generated from the corpus's own sibling case, or from its
`required_arms` when the corpus is new, so a case cannot drift from what check-corpus.py expects.

Both counters are RECOMPUTED from the tree rather than incremented, and both README tables are
upserted by case number, so re-running a batch to correct one case is idempotent. That matters:
correcting a case by re-running its batch is the normal way this is used, and an incrementing
counter made `expect_live_in_pin` drift away from `cases` on the second run.

Used to build the 64 cases added on 2026-10-08. The per-case content lives in the committed
case.c / case.json / PROVENANCE.md, so the one-shot batch scripts that drove this are not kept;
this module and mkharness.py are what a later pass needs.
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent   # capstone/bug-corpora


def add(s):
    corpus = ROOT / s["corpus"]
    d = corpus / f"{s['case']:02d}_{s['fix']}_{s['slug']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "case.c").write_text(s["case_c"])

    # --- case.json, structurally cloned from a sibling -------------------------------------
    # A brand-new corpus has no sibling to clone, so the arm set is taken from the corpus's own
    # declaration instead. Doing it from `required_arms` rather than from a hand-written list is
    # the point: the checker validates cases against that same field, so the two cannot drift.
    sibs = sorted(p for p in corpus.glob("[0-9][0-9]_*/case.json"))
    if sibs:
        j = json.loads(sibs[0].read_text())
    else:
        decl = json.loads((corpus / "corpus.json").read_text())
        j = {"arms": {name: {} for name in decl.get("required_arms", [])}}
    for k in ("distinguishing", "sibling_issue", "size_class", "size_note", "layer_note",
              "note", "advisory", "capstone_column", "comparison", "taxonomy_class",
              "allocator_consumed", "channel", "harness_limit", "oracle_is_recording",
              "citation_constraint", "live_note"):
        j.pop(k, None)
    j.update({k: s[k] for k in ("case", "title", "consumer", "object", "lifetime_ender",
                                "shape", "allocator_layer", "fidelity", "status")})
    j["upstream_fix"] = s["fix"]
    j["live_in_pin"] = s["live_in_pin"]
    j["live_proof"] = s["live_proof"]
    if "nested" in s:
        j["nested"] = s["nested"]
        j["nested_why"] = s["nested_why"]
    if s.get("distinguishing"):
        j["distinguishing"] = s["distinguishing"]
    if s.get("citation_constraint"):
        j["citation_constraint"] = s["citation_constraint"]

    for name, arm in list(j.get("arms", {}).items()):
        if not isinstance(arm, dict):
            continue
        keep = {k: arm[k] for k in ("mode", "signal", "si_code", "cause", "target",
                                    "guest_default") if k in arm}
        new = dict(keep)
        if name in s.get("arms", {}):
            new.update(s["arms"][name])
        elif name in ("native-detect", "native-fix-differential", "backing"):
            new["oracle"] = s["native_oracle"]
            new["status"] = "measured"
        elif name.startswith("poisoncap") and not keep:
            # No PoisonCap run exists for this corpus yet, and these arms' schema requires
            # run-specific keys -- mode, signal, si_code. Those are properties of the runner and
            # of the actual fault, not things to predict, so the arm declares itself unwritten
            # rather than carrying a value that would read as a measurement.
            new = {"status": "not written"}
        else:
            new["oracle"] = s["predicted"]
            new["status"] = "predicted"
        j["arms"][name] = new
    (d / "case.json").write_text(json.dumps(j, indent=2, ensure_ascii=False) + "\n")
    (d / "PROVENANCE.md").write_text(s["provenance"])

    # --- the two counters ------------------------------------------------------------------
    # Both counters are RECOMPUTED from the tree, never incremented: an increment makes a re-run
    # of the same batch drift `expect_live_in_pin` away from `cases`, and re-running a batch to
    # correct one case is the normal way this is used.
    cj = corpus / "corpus.json"
    c = json.loads(cj.read_text())
    c["cases"] = len(list(corpus.glob("[0-9][0-9]_*")))
    eli = {}
    for p in sorted(corpus.glob("[0-9][0-9]_*/case.json")):
        k = {True: "true", False: "false", None: "not_asserted"}[
            json.loads(p.read_text()).get("live_in_pin")]
        eli[k] = eli.get(k, 0) + 1
    c["expect_live_in_pin"] = {k: eli[k] for k in sorted(eli)}   # stable key order, no diff noise
    cj.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n")

    # --- the README tables ------------------------------------------------------------------
    # Two tables, both optional in shape: a "| shape | cases |" summary whose last column is the
    # case number, and a richer "| case | ... |" detail table whose FIRST column is the case
    # number in bold. Both appends are idempotent, keyed on the case number -- the fix hash is
    # not usable as the key because the summary table is only two columns wide and drops it.
    rp = corpus / "README.md"
    t = rp.read_text()

    def upsert(header_re, cells, key_col):
        nonlocal t
        m = re.search(header_re + r".*?\n\n", t, re.S)
        if not m:
            return False
        block = m.group(0)
        rows = [l for l in block.splitlines() if l.startswith("|")]
        ncols = rows[0].count("|") - 1
        c = list(cells[:ncols]) + ["" for _ in range(max(0, ncols - len(cells)))]
        newrow = "| " + " | ".join(c) + " |"
        key = c[key_col]
        dup = [l for l in rows[2:]
               if len(l.split("|")) > key_col + 1 and l.split("|")[key_col + 1].strip() == key]
        if dup:
            for extra in dup[1:]:            # clean up any rows an earlier non-idempotent run left
                t = t.replace("\n" + extra, "", 1)
            t = t.replace(dup[0], newrow, 1)
        else:
            t = t.replace(block, block.rstrip("\n") + "\n" + newrow + "\n\n", 1)
        return True

    if not upsert(r"\| shape \| cases \|", [s["shape_row"], str(s["case"])], key_col=1):
        raise SystemExit(f"{s['corpus']}: shape table not found")
    upsert(r"\| case \| upstream \|",
           [f"**{s['case']}**"] + list(s.get("shape_cells", [])), key_col=0)
    rp.write_text(t)
    return d
