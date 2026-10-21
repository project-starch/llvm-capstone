#!/usr/bin/env python3
"""Judge one set's observations and write its result tables.

    judge.py --manifest M --juliet <dir>/C --build B --run R [--seeded-build SB --seeded-run SR]
             [--twin-build TB --twin-run TR] [--probe P] [--asan A] --llvm-bin <llvm bin dir> --out <results dir>

A BATCH COUNTS only if its boot completed, its steps are exactly the sequence run.py must produce (derived here from
the build, not taken from the run: ok, both controls, [randcheck], the images, both controls, ok), `ok` completed at its
start and at its end, and both controls read as they must at the start and at the end (bounds-malloc prints its mark, faults with
cause 28 in control_write_probe, exits on SIGSEGV and never RETURNs; uaf-malloc the same with cause 25 in
control_read_probe). A batch with randcheck also needs it to exit 0 with the four seed patterns of common.SEEDS.
Every image of a batch that does not count is NO-READING:batch-void. Each observation must name an image the build holds,
with the build's sha256 and the build's case, class, arm, seed and twin family; every built image must be observed
exactly once. All runs, the probe and the build must share one platform and one compiler, and the Juliet sources must
still hash as the manifest says.

PER IMAGE:
  good  CORRECT          "Finished good()", exit 0, no fault, no timeout
        FALSE-POSITIVE?  a domain fault (each one is to be examined)
  bad   CAUGHT           "Calling bad()" printed BEFORE a domain fault with the class's cause (common.EXPECTED_CAUSE),
                         no "Finished bad()", killed by SIGSEGV, the strict check below passes, AND the case's good
                         image (the same flow without the defect) is CORRECT
        MISSED           "Finished bad()" and exit 0, no fault
  anything else is NO-READING:<reason>, including NO-READING:unattributed(<strict reason>) for a fault with the right
  cause that the strict check does not tie to the defective object.

STRICT, from QEMU's own line about the faulting access, against common.declared_sizes(the case's bad section). This
attributes by SIZE: the faulting capability has the size of an object the bad section declares or allocates (good
objects and copy sources excluded). Where the section declares exactly one such object the size names it (`:unique`);
otherwise the row says how many candidates there were (`:ambiguous-<k>`), and a fault on another candidate of a
matching size would pass too.
  temporal  the capability is reported REVOKED, its length is an object the bad section allocates, the cursor is inside
  spatial   the access is outside [base, end) in the class's direction (past the end for 121/122/126, below the start
            for 124/127), on a capability whose length is a declared buffer (STRICT), or a declared STACK buffer
            rounded up to 16 bytes (STRICT-ROUNDED16: the bytes between the end and the rounded bound were not
            protected)

A rand_flow case takes its verdict from the seeded run: CAUGHT if any seed's image, in a batch that counts, is CAUGHT;
MISSED only if exactly one image per seed pattern was read, in batches that count, and all four are MISSED; otherwise
NO-READING.

A MISSED case is relabelled only on evidence that holds for the measured execution:
  NOT-TRIGGERED:wide-snprintf  the case's own twin printed wcslen=1 after the copy: swprintf(.., L"%s", wide) wrote one
                               character. What swprintf writes does not depend on the run, so the twin's reading holds
                               for the measured image.
  NOT-TRIGGERED:sizeof-ptr     probe.py read sizeof(T*) >= sizeof(T) for double, int64_t and the struct in the guest,
                               so malloc(sizeof(pointer)) holds the one element the case stores
For a rand_flow twin, every seed's twin must have completed, at least one must have printed, and every printed value
must be the safe one. A CWE-170 twin's dest99 reading is only reported (column `twin`): it is an uninitialised stack
word of the twin's execution, which the measured execution need not share. The native ASan oracle (asan.py) is also
only reported (column `asan`).
Intra-object cases are reported apart: there are no sub-object bounds on this profile.
Exit 0 when every batch counted, 75 otherwise.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import common as C
import verdicts  # noqa: E402

REVOKED = re.compile(r"Cap mem access on revoked capability: .*?cursor = (\w+), bounds = \((\w+), (\w+)\)")
OOB = re.compile(r"Cap mem access OOB: .*?addr = (\w+), size = (\d+), bounds = \((\w+), (\w+)\)")
SAFE = {"wide-snprintf": ("wcslen", "1")}          # families whose twin reading may relabel
REPORTED = {"cwe170": "dest99"}                     # families whose twin reading is only reported
SUPPORT_HEAD = ["ok-0", "ctl-bounds-0", "ctl-uaf-0"]
SUPPORT_TAIL = ["ctl-bounds-1", "ctl-uaf-1", "ok-1"]


class Run:
    """One run's observations, bound to its build."""

    def __init__(self, run_dir, build_dir, llvm_bin):
        self.obs = C.load(run_dir / "observations.json")
        self.build = C.load(build_dir / "build.json")
        self.img = build_dir / "img"
        self._sym, self.llvm_bin = {}, llvm_bin
        self.records = {i["image"]: i for i in self.build["images"] if i["ok"]}
        known = {i: r["sha256"] for i, r in self.records.items()}
        known.update(self.build["support_images"])
        for i, h in known.items():
            if not h or C.sha256(self.img / f"{i}.dom") != h:
                sys.exit(f"{self.img / i}.dom differs from {build_dir}/build.json")
        seen, logical = set(), set()
        for s in self.obs["steps"]:
            if s["image"] not in known or s["sha256"] != known[s["image"]]:
                sys.exit(f"{run_dir}: step {s['batch']}/{s['step']} ran an image the build does not hold as recorded")
            if s["step"].startswith("s") and s["step"][1:].isdigit():
                r = self.records.get(s["image"])
                if r is None or [s.get(k) for k in ("case", "cwe", "arm", "seed_pattern", "twin_family")] != \
                        [r["case"], r["cwe"], r["arm"], r["seed_pattern"], r.get("twin")]:
                    sys.exit(f"{run_dir}: {s['image']} observed with metadata that differs from the build")
                key = (s["case"], s["arm"], s["seed_pattern"], s["twin_family"])
                if s["image"] in seen or key in logical:
                    sys.exit(f"{run_dir}: {s['image']} observed twice")
                seen.add(s["image"]); logical.add(key)
        if seen != set(self.records):
            sys.exit(f"{run_dir}: {len(set(self.records) - seen)} built images were never observed")
        self.valid = self.batch_validity()

    def symbols(self, image):
        p = self.img / f"{image}.dom"
        return self._sym.setdefault(p, verdicts.Symbols(self.llvm_bin, p))

    def fault(self, s):
        return verdicts.domain_fault(s["fault_line"], self.symbols(s["image"])) if s["fault_line"] else None

    def batch_validity(self):
        by, order = defaultdict(dict), defaultdict(list)
        for s in self.obs["steps"]:
            if s["step"] in by[s["batch"]]:
                sys.exit(f"step {s['batch']}/{s['step']} observed twice")
            by[s["batch"]][s["step"]] = s
            order[s["batch"]].append(s["step"])
        rand = ["randcheck"] if "randcheck" in self.build["support_images"] else []
        valid = {}
        for b in self.obs["batches"]:
            st = by[b["batch"]]
            n = sum(1 for k in order[b["batch"]] if k.startswith("s") and k[1:].isdigit())
            want = SUPPORT_HEAD + rand + [f"s{k:03d}" for k in range(n)] + SUPPORT_TAIL
            ok = b["completed"] and order[b["batch"]] == want == b["plan"] == b["serial_sequence"][:-1] \
                and b["serial_sequence"][-1:] == ["stats"]
            for k in ("0", "1"):
                o = st.get(f"ok-{k}", {})
                ok &= ("OK-PROGRAM done" in o.get("events", []) and o.get("exit") == ["exit", 0]
                       and not o.get("timeout") and not o.get("fault_line"))
                for name, cause, sym, mark in ((f"ctl-bounds-{k}", 28, "control_write_probe", "CONTROL bounds-malloc mark"),
                                               (f"ctl-uaf-{k}", 25, "control_read_probe", "CONTROL uaf-malloc mark")):
                    s = st.get(name)
                    f = self.fault(s) if s else None
                    ok &= bool(s and mark in s["events"] and "RETURNED" not in s["events"] and f is not None
                               and f.cause == cause and f.symbol == sym and s["exit"] == ["signal", 11] and not s["timeout"])
            if rand:
                r = st.get("randcheck", {})
                ok &= (r.get("exit") == ["exit", 0] and not r.get("timeout") and not r.get("fault_line")
                       and r.get("randcheck") == {str(v): k for k, v in C.SEEDS.items()})
            valid[b["batch"]] = bool(ok)
        return valid

    def images(self):
        return [s for s in self.obs["steps"] if "case" in s]


def strict(cwe, qemu_line, declared):
    """STRICT / STRICT-ROUNDED16, suffixed :unique when the bad section declares exactly one candidate object (the size
    then names it) or :ambiguous-<k> when it declares k; or STRICT-FAIL:<reason>."""
    sizes, k = declared
    st = _strict(cwe, qemu_line, sizes)
    return st if st.startswith("STRICT-FAIL") else f"{st}:{'unique' if k == 1 else f'ambiguous-{k}'}"


def _strict(cwe, qemu_line, sizes):
    if cwe in (415, 416):
        m = REVOKED.search(qemu_line or "")
        if not m: return "STRICT-FAIL:no-revoked-line"
        cur, lo, hi = (int(x, 16) for x in m.groups())
        if not lo <= cur < hi: return "STRICT-FAIL:outside"
        return "STRICT" if hi - lo in sizes else f"STRICT-FAIL:size-{hi - lo}"
    m = OOB.search(qemu_line or "")
    if not m: return "STRICT-FAIL:no-oob-line"
    addr, size, lo, hi = int(m.group(1), 16), int(m.group(2)), int(m.group(3), 16), int(m.group(4), 16)
    if lo <= addr and addr + size <= hi: return "STRICT-FAIL:inside"
    if not ((addr + size > hi) if cwe in C.OVER else (addr < lo)): return "STRICT-FAIL:direction"
    if hi - lo in sizes: return "STRICT"
    if any(hi - lo == (n + 15) // 16 * 16 and "stack" in kinds for n, kinds in sizes.items()):
        return "STRICT-ROUNDED16"
    return f"STRICT-FAIL:size-{hi - lo}"


def per_image(run, s, sizes):
    """(verdict, fault, strict) for one observed image of a batch that counts."""
    if s["timeout"]: return "NO-READING:timeout", None, ""
    if s["exit"][0] == "none": return "NO-READING:infra", None, ""
    f = run.fault(s)
    ev = s["events"]
    if s["arm"] == "good":
        if f: return "FALSE-POSITIVE?", f, ""
        if "Finished good()" in ev and s["exit"] == ["exit", 0]: return "CORRECT", None, ""
        return "NO-READING:other", None, ""
    if "Calling bad()" not in ev: return "NO-READING:not-reached", f, ""
    if f:
        ordered = ev.index("Calling bad()") < ev.index("domain fault cause=") and "Finished bad()" not in ev
        if not (ordered and s["exit"] == ["signal", 11]): return "NO-READING:fault-out-of-order", f, ""
        if f.cause != C.EXPECTED_CAUSE[s["cwe"]]: return "NO-READING:unexpected-cause", f, ""
        st = strict(s["cwe"], s["qemu_line"], sizes)
        return ("CAUGHT" if st.startswith("STRICT") and not st.startswith("STRICT-FAIL") else f"NO-READING:unattributed({st})"), f, st
    if "Finished bad()" in ev and s["exit"] == ["exit", 0]: return "MISSED", None, ""
    return "NO-READING:other", None, ""


def judge_images(run, sizes_of):
    out = []
    for s in run.images():
        if run.valid[s["batch"]]:
            v, f, st = per_image(run, s, sizes_of(s["case"]))
        else:
            v, f, st = "NO-READING:batch-void", None, ""
        out.append({"case": s["case"], "cwe": s["cwe"], "arm": s["arm"], "seed_pattern": s["seed_pattern"],
                    "verdict": v, "cause": f.cause if f else None,
                    "site": f"{f.symbol}+{f.offset:#x}" if f and f.symbol else None, "strict": st,
                    "twin": s.get("twin", {}), "twin_family": s.get("twin_family"), "batch_valid": run.valid[s["batch"]]})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--juliet", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--seeded-build", type=Path)
    p.add_argument("--seeded-run", type=Path)
    p.add_argument("--twin-build", type=Path)
    p.add_argument("--twin-run", type=Path)
    p.add_argument("--probe", type=Path, help="probe.py's probe.json")
    p.add_argument("--asan", type=Path, help="asan.py's asan.json")
    p.add_argument("--llvm-bin", type=verdicts.llvm_bin_dir, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    man = C.load(a.manifest)
    cases = {c["case"]: c for c in man["cases"]}
    for c in man["cases"]:
        for f, h in c["sha256"].items():
            if C.sha256(a.juliet / "testcases" / f) != h:
                sys.exit(f"{f} differs from the manifest")
    size_cache = {}
    sizes_of = lambda k: size_cache.setdefault(k, C.declared_sizes(C.bad_sections(a.juliet, cases[k])))

    runs = {"census": Run(a.run, a.build, a.llvm_bin)}
    if a.seeded_run: runs["seeded"] = Run(a.seeded_run, a.seeded_build, a.llvm_bin)
    if a.twin_run: runs["twins"] = Run(a.twin_run, a.twin_build, a.llvm_bin)
    ref = runs["census"]
    man_sha = C.sha256(a.manifest)
    for k, r in runs.items():
        if r.obs["platform"] != ref.obs["platform"] or r.build["sdk_fingerprint"] != ref.build["sdk_fingerprint"] \
                or r.build["set"] != man["set"] or r.build["manifest_sha256"] != man_sha:
            sys.exit(f"the {k} run is not from the census run's platform, SDK and manifest")
    judged = {k: judge_images(r, sizes_of) for k, r in runs.items()}
    census = {(r["case"], r["arm"]): r for r in judged["census"]}
    seeded, twins = defaultdict(list), defaultdict(list)
    for r in judged.get("seeded", []): seeded[r["case"]].append(r)
    for r in judged.get("twins", []): twins[r["case"]].append(r)

    probe = C.load(a.probe) if a.probe else None
    if probe and (probe.get("platform") != ref.obs["platform"] or probe.get("sdk_fingerprint") != ref.build["sdk_fingerprint"]):
        sys.exit("the probe is not from the census run's platform and SDK")
    sizeof_ok = False
    if probe and probe.get("completed") and probe.get("exit") == ["exit", 0]:
        sz = {k: int(v) for k, v in probe["readings"].get("sizeof-ptr", {}).items()}
        sizeof_ok = bool(sz) and all(sz[f"{t}*"] >= sz[t] for t in ("double", "int64_t", "struct"))
    asan = C.load(a.asan) if a.asan else None
    if asan and asan.get("manifest_sha256") != man_sha:
        sys.exit("the ASan oracle was run for another manifest")
    asan_cases = (asan or {}).get("cases", {}) if (asan or {}).get("control", {}).get("reported") else {}

    def twin_reading(k):
        fam = C.family(k)
        key = SAFE.get(fam, (REPORTED.get(fam),))[0]
        rs = twins.get(k, [])
        vals = [f"{r['seed_pattern'] or '-'}:{r['twin'][key]}" for r in rs
                if key and key in r["twin"] and r["batch_valid"] and r["verdict"] == "MISSED"]
        bad = sum(1 for r in rs if not (r["batch_valid"] and r["verdict"] == "MISSED"))
        return (f"{key}=" + ",".join(vals) if vals else "") + (f" ({bad} twin runs did not count)" if bad else "")

    def twin_verdict(k):
        fam = C.family(k)
        rs = twins.get(k, [])
        if fam not in SAFE or not rs or not all(r["batch_valid"] for r in rs): return None
        if cases[k]["rand_flow"] and sorted(r["seed_pattern"] for r in rs) != sorted(C.SEEDS): return None
        if not all(r["verdict"] == "MISSED" for r in rs): return None
        key, safe = SAFE[fam]
        vals = [r["twin"].get(key) for r in rs if key in r["twin"]]
        return f"NOT-TRIGGERED:{fam}" if vals and all(v == safe for v in vals) else None

    rows, caught_now = [], set()
    for k, c in cases.items():
        b, g = census.get((k, "bad")), census.get((k, "good"))
        v, detail = (b["verdict"], b) if b else ("NO-READING:not-built", None)
        if c["rand_flow"]:
            rs = seeded.get(k, [])
            hit = [r for r in rs if r["batch_valid"] and r["verdict"] == "CAUGHT"]
            complete = sorted(r["seed_pattern"] for r in rs) == sorted(C.SEEDS) and all(r["batch_valid"] for r in rs)
            if hit:
                v, detail = "CAUGHT", hit[0]
            elif complete and all(r["verdict"] == "MISSED" for r in rs):
                v, detail = "MISSED", rs[0]
            else:
                v, detail = ("NO-READING:seeded(" + ",".join(sorted({r["verdict"] for r in rs})) + ")" if complete
                             else "NO-READING:seeded-incomplete"), None
        if v == "CAUGHT" and (not g or g["verdict"] != "CORRECT"):
            v = "NO-READING:good-image-not-correct"      # the paired control: the same flow without the defect must run
        if v == "MISSED":
            v = twin_verdict(k) or ("NOT-TRIGGERED:sizeof-ptr" if C.family(k) == "sizeof-ptr" and sizeof_ok else "MISSED")
        if c["intra_object"]:
            v = f"INTRA-OBJECT:{v}"
        if asan and v == "CAUGHT":
            caught_now.add(k)
        rows.append({"case": k, "cwe": c["cwe"], "wchar_t": c["wchar_t"], "intra_object": c["intra_object"],
                     "rand_flow": c["rand_flow"], "verdict": v, "good": g["verdict"] if g else "NO-READING:not-built",
                     "cause": detail["cause"] if detail else None, "site": detail["site"] if detail else None,
                     "strict": detail["strict"] if detail and v == "CAUGHT" else "",
                     "asan": asan_cases.get(k, "") if v == "MISSED" or v.startswith("NOT-TRIGGERED") else "",
                     "twin": twin_reading(k) if v == "MISSED" or v.startswith("NOT-TRIGGERED") else ""})

    a.out.mkdir(parents=True, exist_ok=True)
    cols = ["case", "cwe", "wchar_t", "intra_object", "rand_flow", "verdict", "good", "cause", "site", "strict", "asan", "twin"]
    with open(a.out / f"{man['set']}-verdicts.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    summary = {"set": man["set"], "suite": man["suite"], "language": "C", "cases": len(rows), "left_out": man["left_out"],
               "judge_sha256": C.sha256(Path(__file__)), "common_sha256": C.sha256(C.HERE / "common.py"),
               "runs": {k: {"batches": len(r.valid), "batches_counted": sum(r.valid.values()), "wall_s": r.obs["wall_s"],
                            "runner_sha256": r.obs["runner_sha256"], "helper_sha256": r.obs["helper_sha256"],
                            "sdk_fingerprint": r.build["sdk_fingerprint"]} for k, r in runs.items()},
               "manifest_sha256": man_sha,
               "platform": runs["census"].obs["platform"], "sdk": runs["census"].build["sdk"], "per_cwe": {}}
    if probe:
        summary["mechanism_probe"] = {k: probe[k] for k in ("completed", "exit", "image_sha256", "source_sha256", "readings")}
    if asan and not set(asan["control"]["cases"]) <= caught_now:
        sys.exit("the ASan control sample is not drawn from this judgement's CAUGHT cases")
    if asan:
        summary["asan_oracle"] = {"compiler": asan["compiler"], "flags": asan["flags"],
                                  "control": {k: v for k, v in asan["control"].items() if k != "cases"}}
    for cwe in sorted({r["cwe"] for r in rows}):
        rs = [r for r in rows if r["cwe"] == cwe]
        caught = sum(r["verdict"] == "CAUGHT" for r in rs)
        missed = sum(r["verdict"] == "MISSED" for r in rs)
        summary["per_cwe"][str(cwe)] = {
            "bad": dict(Counter(r["verdict"] for r in rs)), "good": dict(Counter(r["good"] for r in rs)),
            "strict": dict(Counter(r["strict"] for r in rs if r["verdict"] == "CAUGHT")),
            "caught": caught, "missed": missed}
    (a.out / f"{man['set']}-summary.json").write_text(json.dumps(summary, indent=1))
    for cwe, s in summary["per_cwe"].items():
        print(f"CWE-{cwe}: caught {s['caught']} missed {s['missed']}; bad {s['bad']}; good {s['good']}; strict {s['strict']}")
    print("runs:", {k: (v["batches_counted"], v["batches"]) for k, v in summary["runs"].items()})
    sys.exit(0 if all(all(r.valid.values()) for r in runs.values()) else 75)


if __name__ == "__main__":
    main()
