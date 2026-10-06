#!/usr/bin/env python3
"""The reference count's traffic and reclamation, per 1000 instructions, against the run
without it.

    summarize_rc.py [--md] [--before DIR] name=<reports prefix> [...]

<prefix>.npl4.json is Capstone's node cache (16-byte records, four per line) over the
whole trace, reference-count updates included (capstone-qemu cap_refcount.h, sites
rc_*); <prefix>.npl4-nosweep.json leaves out the supervisor's sweep at the domain's end,
<prefix>.npl4-rcelide.json the same-node pairs an implementation can skip,
<prefix>.npl4-norc.json every count update. <prefix>.refcount.txt is the emulator's own
statistics line, <prefix>.alias.json aliasstat's report (its free-time alias count must
be 0 under the count), <prefix>.clover.json Clover's stream on the same trace. --before
DIR names the reports of the run without the count (fresh ids, no reuse); its
<name>.npl4.json gives the baseline misses.

The denominator is 1000 instructions the domain retired in C-mode blocks (the trace's
INSN record, format 04), the paper's; the table also gives the checks and the memory
operations per 1000 instructions so the earlier per-check figures convert. --md prints
Markdown tables.
"""
import json
import os
import re
import sys

SIZES = [64, 1024, 16384]
ARROW = " → "


def fmt(x):
    return f"{x:.3f}" if x < 10 else f"{x:.1f}" if x < 1000 else f"{x:,.0f}"


def full_cache(report, s):
    return [c for c in report["caches"] if c["entries"] == s and c["ways"] == "full"][0]


def misses(report, k):
    return [full_cache(report, s)["misses"] / k for s in SIZES]


def sibling(directory, prefix, suffix):
    """the report of the same program in another results directory, by either naming"""
    base = os.path.basename(prefix)
    for name in (base, base.replace("ptr-seq-", "").replace("mix-seq-", "")):
        f = os.path.join(directory, name + suffix)
        if os.path.exists(f):
            return f
    return None


def load(prefix, before, models=None):
    full = json.load(open(prefix + ".npl4.json"))
    acc = full["accesses"]
    r = {"insns": full["domain_instructions"]}
    if not r["insns"]:
        raise SystemExit(f"{prefix}: no instruction count in the trace (format 04 needed)")
    k = r["insns"] / 1000                                   # per 1000 instructions
    r["k"] = k
    r["checks"] = (acc["read"]["ldst"] + acc["read"]["ldc"]) / k
    r["memops"] = acc["read"]["ldst"] / k
    w = acc["write"]
    r["reg"] = (w["rc_reg_inc"] + w["rc_reg_dec"]) / k        # register writes, nothing to hide behind
    r["ld"] = (w["rc_ld_inc"] + w["rc_ld_dec"]) / k           # capability loads
    r["mem"] = (w["rc_mem_inc"] + w["rc_mem_dec"]) / k        # capability and data stores
    r["same"] = w["rc_same"] / k                              # records, two per cancelling pair: a slot rewritten with its node
    r["move"] = w.get("rc_move", 0) / k                       # a copy moved between slots (a linear move)
    r["call"] = w.get("rc_call", 0) / k                       # the return register rewritten at a call
    r["free"] = (w["rc_free"] + acc["free"]["rc_free"]) / k   # unlink writes and the free-list write
    r["sweep"] = (w["rc_sweep_dec"] + w["rc_sweep_free"] + acc["free"]["rc_sweep_free"]) / k
    r["tree"] = sum(acc["read"][s] + acc["write"][s] for s in ("mrev", "split", "revoke", "delin", "drop")) / k
    r["elided"] = r["reg"] + r["ld"] + r["mem"] + r["free"]
    r["pairs"] = r["same"] + r["move"] + r["call"]
    r["literal"] = r["elided"] + r["pairs"]
    ids = full["ids"]
    r["allocs"], r["footprint"], r["live"] = ids["allocations"], ids["footprint"], ids["peak_live"]
    r["first_reuse"] = ids["first_reuse_allocation"]
    r["m_literal"] = misses(full, k)
    # the paper's node cache, 8 KB 2-way: misses per 1000 instructions, miss rate, write-backs
    paper = [c for c in full["caches"] if c["entries"] == 128 and c["ways"] == 2]
    r["paper"] = None
    if paper:
        c = paper[0]
        r["paper"] = {"m": c["misses"] / k, "rate": 100.0 * c["misses"] / max(c["hits"] + c["misses"], 1),
                      "wb": c.get("writebacks", 0) / k}
    r["wb"] = [full_cache(full, s).get("writebacks", 0) / k for s in SIZES]
    # the run without its teardown: everything from the last allocation on, counted apart
    after = full.get("after_last_allocation")
    r["m_run"] = None
    if after:
        r["m_run"] = [(full_cache(full, s)["misses"] - [c for c in after["full_caches"] if c["entries"] == s][0]["misses"]) / k for s in SIZES]
        r["teardown_records"] = after["logical_records"]
    r["m_reg_miss"] = [(full_cache(full, s)["by_site"]["rc_reg_inc"][1] + full_cache(full, s)["by_site"]["rc_reg_dec"][1]) / k
                       for s in SIZES]
    for tag in ("nosweep", "rcelide", "norc"):
        f = f"{prefix}.npl4-{tag}.json"
        r["m_" + tag] = misses(json.load(open(f)), k) if os.path.exists(f) else None
        r["wb_" + tag] = None
        if os.path.exists(f):
            rep_ = json.load(open(f))
            r["wb_" + tag] = [full_cache(rep_, s).get("writebacks", 0) / k for s in SIZES]
            a = rep_.get("after_last_allocation")
            if a:
                r["m_" + tag + "_run"] = [(full_cache(rep_, s)["misses"] - [c for c in a["full_caches"] if c["entries"] == s][0]["misses"]) / k for s in SIZES]
    r["m_before"] = None
    if before:
        f = sibling(before, prefix, ".npl4.json")
        if f:
            b = json.load(open(f))
            # the earlier run has no instruction count: scale by its checks, which match within 0.2 %
            kb = (b["accesses"]["read"]["ldst"] + b["accesses"]["read"]["ldc"]) / r["checks"]
            r["m_before"] = misses(b, kb)
    r["stats"] = open(prefix + ".refcount.txt").read().strip() if os.path.exists(prefix + ".refcount.txt") else ""
    m = re.search(r"releases (\d+) \(valid (\d+) unlinked, invalid (\d+), by the sweep (\d+)\), max count (\d+), unsynced (\d+), stale (\d+)", r["stats"])
    r["rel"] = tuple(int(x) for x in m.groups()) if m else None
    r["free_alias_max"] = r["alias_errors"] = None
    f = prefix + ".alias.json"
    if os.path.exists(f):
        al = json.load(open(f))
        r["free_alias_max"] = al["aliases"]["at_free_memory"]["max"]
        r["alias_errors"] = sum(al["errors"].values())
    r["clover"] = None
    r["models_from"] = "this run"
    f = prefix + ".clover.json"
    if not os.path.exists(f) and models:
        f = sibling(models, prefix, ".clover.json") or f
        r["models_from"] = "the run without the count"
    if os.path.exists(f):
        cl = json.load(open(f))
        # that run has no instruction count: per check there, times this run's checks per instruction
        r["clover"] = {"acc": cl["clover"]["accesses"] / (cl["lifetime_checks"] / 1000) * r["checks"] / 1000,
                       "m": [cl["clover"]["misses"][cl["sizes_lines"].index(s)] / (cl["lifetime_checks"] / 1000) * r["checks"] / 1000 for s in SIZES]}
    return r


def table(md, title, hdr, rows, width=16):
    if md:
        print(f"\n{title}\n")
        print("| " + " | ".join(hdr) + " |")
        print("|---|" + "---:|" * (len(hdr) - 1))
        for cells in rows:
            print("| " + " | ".join(cells) + " |")
    else:
        print(f"\n{title}")
        print("".join(f"{h:>{width}s}" for h in hdr))
        for cells in rows:
            print("".join(f"{c:>{width}s}" for c in cells))


def main():
    md = "--md" in sys.argv
    before = models = None
    names = []
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--md":
            continue
        if a == "--before":
            before = next(it)
            continue
        if a == "--models":
            models = next(it)
            continue
        names.append(a)
    rows = []
    for a in names:
        name, prefix = a.split("=", 1)
        r = load(prefix, before, models)
        r["name"] = name
        rows.append(r)

    # 1. the streams per 1000 instructions
    table(md, "metadata accesses per 1000 instructions",
          ["", "instructions", "checks", "memory ops", "tree ops", "count: register", "count: load", "count: store", "frees", "pairs ×2: rewrite", "move", "call", "count, literal", "count, pairs elided", "sweep at exit"],
          [[r["name"], f"{r['insns']:,}", fmt(r["checks"]), fmt(r["memops"]), fmt(r["tree"]), fmt(r["reg"]), fmt(r["ld"]), fmt(r["mem"]),
            fmt(r["free"]), fmt(r["same"]), fmt(r["move"]), fmt(r["call"]), fmt(r["literal"]), fmt(r["elided"]), fmt(r["sweep"])] for r in rows], 14)

    # 2. the overhead's components: occupancy by origin, traffic, latency
    table(md, "the count's overhead per 1000 instructions (exposed: register writes, nothing to hide behind; hideable: at a load or store)",
          ["", "exposed", "hideable", "elidable pairs", "extra misses in the run @64 / 1024 / 16384", "write-backs @1024: no count → with", "exposed misses @64 / 1024 / 16384", "paper's 8 KB 2-way: misses, miss rate, write-backs"],
          [[r["name"], fmt(r["reg"]), fmt(r["ld"] + r["mem"]), fmt(r["pairs"]),
            " / ".join(fmt(r["m_nosweep_run"][i] - r["m_norc_run"][i]) for i in range(3)) if r.get("m_nosweep_run") and r.get("m_norc_run") else
            (" / ".join(fmt(r["m_nosweep"][i] - r["m_norc"][i]) for i in range(3)) + " (teardown in)" if r["m_nosweep"] and r["m_norc"] else "?"),
            f"{r['wb_norc'][1]:.3f} → {r['wb_nosweep'][1]:.3f}" if r.get("wb_norc") and r.get("wb_nosweep") else "?",
            " / ".join(fmt(x) for x in r["m_reg_miss"]),
            f"{r['paper']['m']:.3f}, {r['paper']['rate']:.3f} %, {r['paper']['wb']:.3f}" if r["paper"] else "?"] for r in rows], 22)

    # 3. reclamation
    table(md, "reclamation",
          ["", "allocations", "ids used", "peak live (by revoke)", "first reuse at", "frees by the count", "of them valid (unlinked)", "frees in the exit sweep", "max count", "aliases at free (max)", "unsynced / stale"],
          [[r["name"], f"{r['allocs']:,}", f"{r['footprint']:,}", f"{r['live']:,}", f"{r['first_reuse']:,}" if r["first_reuse"] else "never",
            f"{r['rel'][0]:,}" if r["rel"] else "?", f"{r['rel'][1]:,}" if r["rel"] else "?", f"{r['rel'][3]:,}" if r["rel"] else "?",
            f"{r['rel'][4]:,}" if r["rel"] else "?", str(r["free_alias_max"]), f"{r['rel'][5]} / {r['rel'][6]}" if r["rel"] else "?"] for r in rows], 18)

    # 4. misses per 1000 instructions
    table(md, "misses per 1000 instructions: without the count (fresh ids) → the count's reuse alone → count, pairs elided → count, literal (sweep left out) → Clover naive",
          [""] + [f"{s} lines" for s in SIZES],
          [[r["name"]] + [ARROW.join("?" if x is None else fmt(x) for x in
                                    [r["m_before"][i] if r["m_before"] else None, r["m_norc"][i] if r["m_norc"] else None,
                                     r["m_rcelide"][i] if r["m_rcelide"] else None, r["m_nosweep"][i] if r["m_nosweep"] else None,
                                     r["clover"]["m"][i] if r["clover"] else None])
                          for i in range(3)] for r in rows], 40)

    # 5. against Clover's stream on the same trace
    table(md, "accesses per 1000 instructions: Capstone without the count → with it, pairs elided → literal → Clover naive",
          ["", "accesses", "misses @1024"],
          [[r["name"], ARROW.join(fmt(x) for x in [r["checks"] + r["tree"], r["checks"] + r["tree"] + r["elided"], r["checks"] + r["tree"] + r["literal"]] + ([r["clover"]["acc"]] if r["clover"] else [])),
            ARROW.join("?" if x is None else fmt(x) for x in [r["m_before"][1] if r["m_before"] else None, r["m_rcelide"][1] if r["m_rcelide"] else None,
                                                             r["m_nosweep"][1] if r["m_nosweep"] else None, r["clover"]["m"][1] if r["clover"] else None])] for r in rows], 44)

    borrowed = [r["name"] for r in rows if r["models_from"] != "this run"]
    if borrowed:
        print("\nClover's stream for " + ", ".join(borrowed) + " is from the run without the count, scaled to this run's instructions.")

    bad = [(r["name"], r["free_alias_max"], r["alias_errors"]) for r in rows
           if (r["free_alias_max"] or 0) != 0 or (r["alias_errors"] or 0) != 0]
    if bad:
        print("\nALIAS CHECK FAILED (copies left at a free, or aliasstat errors):", bad)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
