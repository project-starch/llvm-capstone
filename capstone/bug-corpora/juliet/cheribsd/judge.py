#!/usr/bin/env python3
"""Judge one set's CheriBSD observations and write its result tables.

    judge.py --manifest M --juliet <dir>/C --build B --run R [--seeded-build SB --seeded-run SR]
             [--twin-build TB --twin-run TR] [--off-run OR] [--every-free-run ER]
             --llvm-bin <SDK bin dir> [--rootfs <purecap rootfs>] --out <results dir>

A BATCH COUNTS only if its boot completed without a panic, every sysctl read back as its configuration requires, its
steps are exactly the sequence run.py must produce (derived here from the build), no step timed out, `ok` completed
at both ends, the ABI probe read pointer_bytes=16 with the configuration's runtime_revocation at both ends, and the
controls read as they must at both ends:
  ctl-bounds   (abi-probe oob) prints CHERI_BOUNDARY_READY, dies on SIGPROT PROT_CHERI_BOUNDS -- every configuration
  ctl-revoked  (sicode-selftest revoked: free, forced sweep, read) dies on SIGPROT PROT_CHERI_TAG at the default and
               every-free; completes with revocation off
  ctl-uaf      (free, no forced sweep, read) dies on SIGPROT PROT_CHERI_TAG at every-free; completes with revocation
               off; at the default it is RECORDED, not required (what the quarantine does with one free is a reading)
and the seed finder found all four patterns, the same in every batch of every run. Every image of a batch that does
not count is NO-READING:batch-void. Each observation must name an image the build holds, with the build's sha256 and
metadata; every built image must be observed exactly once. All runs must share one build and one platform.

PER IMAGE, from supervise's lines (the program's own prints are evidence only when it completed: a fault leaves
them in a stdio buffer):
  good  CORRECT          "Finished good()", exit 0, no fault, no timeout
        FALSE-POSITIVE?  any fault (each one is to be examined)
  bad   CAUGHT           killed by SIGPROT with the class's si_code (PROT_CHERI_TAG for 415/416, PROT_CHERI_BOUNDS for
                         121-127; a spatial case whose pointer left its bounds far enough to lose its tag faults
                         PROT_CHERI_TAG instead, and counts, with the code reported), no "Finished bad()", AND the
                         case's good image (the same flow without the defect) is CORRECT
        MISSED           "Finished bad()" and exit 0, no fault
        STOPPED:<what>   ended by something that is not the class's capability fault: SIGABRT (on this libc, mrs's
                         free() validation aborts on a second free of a block it already quarantined or revoked --
                         an allocator consistency check, never scored as a capability catch), another signal, or
                         SIGPROT with another code. Reported apart from CAUGHT and MISSED.
  anything else is NO-READING:<reason>.
ATTRIBUTION is by SITE: the function the fault pc lies in (in the image or, with --rootfs, in the library supervise
named) and the function the return address lies in (the caller only when the site is a leaf). It is weaker than the
virtual harness's strict check, which reads the faulting capability's bounds from QEMU; CheriBSD's kernel reports no
bounds, so a size-based attribution is not available here, and the column says so.

A rand_flow case takes its verdict from the seeded run, as in ../judge.py: CAUGHT if any seed's image, in a batch
that counts, is CAUGHT; MISSED only if exactly one image per pattern was read and all four are MISSED. A MISSED case
is relabelled NOT-TRIGGERED only on evidence from the measured platform: wide-snprintf by its own twin's wcslen=1
reading (what this libc's swprintf writes is deterministic), sizeof-ptr by the probe's sizeof(T*) >= sizeof(T) read
in the guest. CWE-170 twin readings are reported only. Intra-object cases are reported apart.
The off and every-free runs add a column each: the census bad image's per-image reading in that configuration.
Exit 0 when every batch of every run counted, 75 otherwise.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import shared as S
import verdicts  # noqa: E402  (bug-corpora/tools, put on sys.path by shared)

SAFE = {"wide-snprintf": ("wcslen", "1")}
REPORTED = {"cwe170": "dest99"}
PATTERNS = ("11", "10", "01", "00")
SIGNAL_NAME = {6: "SIGABRT", 10: "SIGBUS", 11: "SIGSEGV", 4: "SIGILL", 8: "SIGFPE", 9: "SIGKILL", 15: "SIGTERM"}


def killed_by(s, signal, code=None):
    f = s["fault"]
    return bool(f and f["signal"] == signal and (code is None or f["code"] == code)
                and s["supervise_exit"] == ["signalled", signal] and not s["timeout"])


def completed(s, event):
    return event in s["events"] and s["supervise_exit"] == ["status", 0] and not s["fault"] and not s["timeout"]


class Run:
    """One run's observations, bound to its build."""

    def __init__(self, run_dir, build_dir):
        self.obs = S.load(run_dir / "observations.json")
        self.build = S.load(build_dir / "build.json")
        self.img = build_dir / "img"
        self.records = {i["image"]: i for i in self.build["images"] if i["ok"]}
        known = {i: r["sha256"] for i, r in self.records.items()}
        known.update(self.build["support_images"])
        for i, h in known.items():
            if not h or S.sha256(self.img / i) != h:
                sys.exit(f"{self.img / i} differs from {build_dir}/build.json")
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
        self.revocation = self.obs["revocation"]
        self.valid, self.why = self.batch_validity()
        self.seeds = next((s["randseed"] for s in self.obs["steps"] if s["step"] == "randseed" and s["randseed"]), None)

    def batch_validity(self):
        by, order = defaultdict(dict), defaultdict(list)
        for s in self.obs["steps"]:
            if s["step"] in by[s["batch"]]:
                sys.exit(f"step {s['batch']}/{s['step']} observed twice")
            by[s["batch"]][s["step"]] = s
            order[s["batch"]].append(s["step"])
        _, expected_knobs, abi_rev = S.REVOCATION[self.revocation]
        valid, why = {}, {}
        for b in self.obs["batches"]:
            st = by[b["batch"]]
            n = sum(1 for k in order[b["batch"]] if k.startswith("s") and k[1:].isdigit())
            want = S.SUPPORT_HEAD + [f"s{k:03d}" for k in range(n)] + S.SUPPORT_TAIL
            fails = []
            if not (b["booted"] and b["mounted"] and b["completed"] and not b["panic"]):
                fails.append("boot")
            if b.get("sysctl") != expected_knobs:
                fails.append("sysctl")
            if not (order[b["batch"]] == want == b["plan"] == b["serial_sequence"]):
                fails.append("sequence")
            if any(s["timeout"] for s in st.values()):
                fails.append("timeout")
            for k in ("0", "1"):
                s = st.get(f"ok-{k}")
                if not (s and completed(s, "OK-PROGRAM done")):
                    fails.append(f"ok-{k}")
                abi = st.get(f"abi-{k}", {})
                if not (abi.get("abi") == {"pointer_bytes": 16, "runtime_revocation": abi_rev}
                        and abi.get("supervise_exit") == ["status", 0]):
                    fails.append(f"abi-{k}")
                s = st.get(f"ctl-bounds-{k}")
                if not (s and "CHERI_BOUNDARY_READY" in s["events"] and killed_by(s, S.SIGPROT, 1)):
                    fails.append(f"ctl-bounds-{k}")
                s = st.get(f"ctl-revoked-{k}")
                want_fault = self.revocation != "off"
                if not (s and (killed_by(s, S.SIGPROT, 2) if want_fault else completed(s, "SELFTEST no fault"))):
                    fails.append(f"ctl-revoked-{k}")
                s = st.get(f"ctl-uaf-{k}")
                if self.revocation == "every-free" and not (s and killed_by(s, S.SIGPROT, 2)):
                    fails.append(f"ctl-uaf-{k}")
                if self.revocation == "off" and not (s and completed(s, "CONTROL uaf-free done")):
                    fails.append(f"ctl-uaf-{k}")
            r = st.get("randseed", {})
            if not (r.get("supervise_exit") == ["status", 0] and sorted(r.get("randseed", {})) == sorted(PATTERNS)):
                fails.append("randseed")
            pr = st.get("probe", {})
            if not (pr.get("supervise_exit") == ["status", 0] and "sizeof-ptr" in pr.get("probe", {})):
                fails.append("probe")
            valid[b["batch"]] = not fails
            why[b["batch"]] = fails
        seeds = {s["batch"]: s["randseed"] for s in self.obs["steps"] if s["step"] == "randseed"}
        if len({json.dumps(v, sort_keys=True) for v in seeds.values()}) > 1:
            for k in valid:
                valid[k] = False; why[k].append("randseed-differs")
        return valid, why

    def images(self):
        return [s for s in self.obs["steps"] if "case" in s]

    def uaf_control(self):
        """What ctl-uaf read in each batch that counts: the quarantine's answer to one free, no forced sweep."""
        out = Counter()
        for s in self.obs["steps"]:
            if s["step"].startswith("ctl-uaf-") and self.valid[s["batch"]]:
                v, code = reading(s, "CONTROL uaf-free done", 416)
                out["completed" if v == "MISSED" else f"faulted:{S.SICODE_NAME.get(code, code)}" if v == "CAUGHT" else v] += 1
        return dict(out)


class Sites:
    """pc -> function, from the image or the rootfs library supervise named."""

    def __init__(self, llvm_bin, img_dir, rootfs):
        self.llvm_bin, self.img_dir, self.rootfs, self.cache = llvm_bin, img_dir, rootfs, {}

    def elf(self, path, image):
        if path == f"{S.GUEST_IMAGES}/{image}":
            local = self.img_dir / image
        elif self.rootfs and (self.rootfs / path.lstrip("/")).is_file():
            local = self.rootfs / path.lstrip("/")
        else:
            return None
        if local not in self.cache:
            self.cache[local] = S.Symbols(self.llvm_bin, local)
        return self.cache[local]

    def name(self, obj, image):
        if not obj:
            return None
        syms = self.elf(obj["path"], image)
        fn, off = syms.lookup(obj["addr"], obj["base"]) if syms else (None, None)
        return f"{fn}+{off:#x}" if fn else f"{Path(obj['path']).name}+{obj['addr'] - obj['base']:#x}"


def reading(s, finished, cwe):
    """(verdict, si_code) for one bad image or the uaf control; `finished` is the marker a completion prints."""
    if s["timeout"]:
        return "NO-READING:timeout", None
    if s["exit"] is None or s["supervise_exit"] is None:
        return "NO-READING:infra", None
    f = s["fault"]
    if f:
        if f["signal"] == S.SIGPROT and f["code"] == S.EXPECTED_SICODE[cwe] and finished not in s["events"] \
                and s["supervise_exit"] == ["signalled", S.SIGPROT]:
            return "CAUGHT", f["code"]
        if f["signal"] == S.SIGPROT and f["code"] == 2 and cwe in S.C.OVER | S.C.UNDER and finished not in s["events"] \
                and s["supervise_exit"] == ["signalled", S.SIGPROT]:
            return "CAUGHT", f["code"]     # a spatial pointer moved out of representable bounds: tag cleared
        if s["supervise_exit"][0] != "signalled":
            return "NO-READING:fault-then-exit", f["code"]
        if f["signal"] == S.SIGPROT:
            return f"STOPPED:SIGPROT-{S.SICODE_NAME.get(f['code'], f['code'])}", f["code"]
        return f"STOPPED:{SIGNAL_NAME.get(f['signal'], 'signal-%d' % f['signal'])}", None
    if finished in s["events"] and s["supervise_exit"] == ["status", 0]:
        return "MISSED", None
    return "NO-READING:other", None


def per_image(s):
    """(verdict, si_code) for one observed image of a batch that counts."""
    if s["arm"] == "good":
        if s["timeout"]:
            return "NO-READING:timeout", None
        if s["fault"]:
            return "FALSE-POSITIVE?", s["fault"]["code"]
        if completed(s, "Finished good()"):
            return "CORRECT", None
        return "NO-READING:other", None
    return reading(s, "Finished bad()", s["cwe"])


def judge_images(run, sites):
    out = []
    for s in run.images():
        v, code = per_image(s) if run.valid[s["batch"]] else ("NO-READING:batch-void", None)
        out.append({"case": s["case"], "cwe": s["cwe"], "arm": s["arm"], "seed_pattern": s["seed_pattern"],
                    "verdict": v, "sicode": code,
                    "site": sites.name(s["fault_object"], s["image"]) if s["fault"] else None,
                    "ra_site": sites.name(s["fault_ra"], s["image"]) if s["fault"] else None,
                    "twin": s.get("twin", {}), "twin_family": s.get("twin_family"), "batch_valid": run.valid[s["batch"]]})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--juliet", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True, help="the census run at the platform default")
    p.add_argument("--seeded-build", type=Path)
    p.add_argument("--seeded-run", type=Path)
    p.add_argument("--twin-build", type=Path)
    p.add_argument("--twin-run", type=Path)
    p.add_argument("--off-run", type=Path, help="the census build run with revocation off")
    p.add_argument("--every-free-run", type=Path, help="the census build run with every-free revocation")
    p.add_argument("--llvm-bin", type=verdicts.llvm_bin_dir, required=True)
    p.add_argument("--rootfs", type=Path, help="resolve fault sites inside the guest's libraries")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    man = S.load(a.manifest)
    cases = {c["case"]: c for c in man["cases"]}
    for c in man["cases"]:
        for f, h in c["sha256"].items():
            if S.sha256(a.juliet / "testcases" / f) != h:
                sys.exit(f"{f} differs from the manifest")

    runs = {"census": Run(a.run, a.build)}
    if a.seeded_run: runs["seeded"] = Run(a.seeded_run, a.seeded_build)
    if a.twin_run: runs["twins"] = Run(a.twin_run, a.twin_build)
    if a.off_run: runs["off"] = Run(a.off_run, a.build)
    if a.every_free_run: runs["every-free"] = Run(a.every_free_run, a.build)
    ref = runs["census"]
    man_sha = S.sha256(a.manifest)
    for k, r in runs.items():
        if r.obs["platform"] != ref.obs["platform"] or r.build["toolchain"] != ref.build["toolchain"] \
                or r.build["set"] != man["set"] or r.build["manifest_sha256"] != man_sha:
            sys.exit(f"the {k} run is not from the census run's platform, toolchain and manifest")
        want = {"census": "default", "seeded": "default", "twins": "default", "off": "off", "every-free": "every-free"}[k]
        if r.revocation != want:
            sys.exit(f"the {k} run was made with revocation {r.revocation}, not {want}")
        if k in ("seeded", "twins") and r.build["seeds"] and r.build["seeds"] != ref.seeds:
            sys.exit(f"the {k} build's seeds are not the census run's")
    sites = {k: Sites(a.llvm_bin, r.img, a.rootfs) for k, r in runs.items()}
    judged = {k: judge_images(r, sites[k]) for k, r in runs.items()}
    census = {(r["case"], r["arm"]): r for r in judged["census"]}
    other = {k: {r["case"]: r for r in judged[k] if r["arm"] == "bad"} for k in ("off", "every-free") if k in judged}
    seeded, twins = defaultdict(list), defaultdict(list)
    for r in judged.get("seeded", []): seeded[r["case"]].append(r)
    for r in judged.get("twins", []): twins[r["case"]].append(r)

    probe = next((s["probe"]["sizeof-ptr"] for s in ref.obs["steps"]
                  if s["step"] == "probe" and ref.valid[s["batch"]] and "sizeof-ptr" in s["probe"]), None)
    sizeof_ok = bool(probe) and all(int(probe[f"{t}*"]) >= int(probe[t]) for t in ("double", "int64_t", "struct"))

    def twin_reading(k):
        fam = S.C.family(k)
        key = SAFE.get(fam, (REPORTED.get(fam),))[0]
        rs = twins.get(k, [])
        vals = [f"{r['seed_pattern'] or '-'}:{r['twin'][key]}" for r in rs
                if key and key in r["twin"] and r["batch_valid"] and r["verdict"] == "MISSED"]
        bad = sum(1 for r in rs if not (r["batch_valid"] and r["verdict"] == "MISSED"))
        return (f"{key}=" + ",".join(vals) if vals else "") + (f" ({bad} twin runs did not count)" if bad else "")

    def twin_verdict(k):
        fam = S.C.family(k)
        rs = twins.get(k, [])
        if fam not in SAFE or not rs or not all(r["batch_valid"] for r in rs): return None
        if cases[k]["rand_flow"] and sorted(r["seed_pattern"] for r in rs) != sorted(PATTERNS): return None
        if not all(r["verdict"] == "MISSED" for r in rs): return None
        key, safe = SAFE[fam]
        vals = [r["twin"].get(key) for r in rs if key in r["twin"]]
        return f"NOT-TRIGGERED:{fam}" if vals and all(v == safe for v in vals) else None

    rows = []
    for k, c in cases.items():
        b, g = census.get((k, "bad")), census.get((k, "good"))
        v, detail = (b["verdict"], b) if b else ("NO-READING:not-built", None)
        if c["rand_flow"] and "seeded" in runs:
            rs = seeded.get(k, [])
            hit = [r for r in rs if r["batch_valid"] and r["verdict"] == "CAUGHT"]
            complete = sorted(r["seed_pattern"] for r in rs) == sorted(PATTERNS) and all(r["batch_valid"] for r in rs)
            if hit:
                v, detail = "CAUGHT", hit[0]
            elif complete and all(r["verdict"] == "MISSED" for r in rs):
                v, detail = "MISSED", rs[0]
            else:
                v, detail = ("NO-READING:seeded(" + ",".join(sorted({r["verdict"] for r in rs})) + ")" if complete
                             else "NO-READING:seeded-incomplete"), None
        elif c["rand_flow"]:
            v = f"NO-READING:rand-flow-unseeded({v})"
        if v == "CAUGHT" and (not g or g["verdict"] != "CORRECT"):
            v = "NO-READING:good-image-not-correct"
        if v == "MISSED":
            v = twin_verdict(k) or ("NOT-TRIGGERED:sizeof-ptr" if S.C.family(k) == "sizeof-ptr" and sizeof_ok else "MISSED")
        if c["intra_object"]:
            v = f"INTRA-OBJECT:{v}"
        rows.append({"case": k, "cwe": c["cwe"], "wchar_t": c["wchar_t"], "intra_object": c["intra_object"],
                     "rand_flow": c["rand_flow"], "verdict": v, "good": g["verdict"] if g else "NO-READING:not-built",
                     "sicode": detail["sicode"] if detail else None, "site": detail["site"] if detail else None,
                     "ra_site": detail["ra_site"] if detail else None,
                     "off": other["off"][k]["verdict"] if "off" in other and k in other["off"] else "",
                     "every_free": other["every-free"][k]["verdict"] if "every-free" in other and k in other["every-free"] else "",
                     "twin": twin_reading(k) if v == "MISSED" or v.startswith("NOT-TRIGGERED") else ""})

    a.out.mkdir(parents=True, exist_ok=True)
    cols = ["case", "cwe", "wchar_t", "intra_object", "rand_flow", "verdict", "good", "sicode", "site", "ra_site",
            "off", "every_free", "twin"]
    with open(a.out / f"{man['set']}-verdicts.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    capbounds = next((s["capbounds"] for s in ref.obs["steps"] if s["step"] == "capbounds" and ref.valid[s["batch"]]), None)
    summary = {"set": man["set"], "suite": man["suite"], "language": "C", "cases": len(rows), "left_out": man["left_out"],
               "judge_sha256": S.sha256(Path(__file__)), "shared_sha256": S.sha256(S.HERE / "shared.py"),
               "manifest_sha256": man_sha, "attribution": "site: the function holding the fault pc, and the return address's; no capability bounds are reported by CheriBSD's kernel, so no size-based strict check",
               "runs": {k: {"revocation": r.revocation, "batches": len(r.valid), "batches_counted": sum(r.valid.values()),
                            "batches_void": {b: w for b, w in r.why.items() if w}, "wall_s": r.obs["wall_s"],
                            "guests": r.obs.get("guests"), "runner_sha256": r.obs["runner_sha256"],
                            "sysctl_expected": r.obs["sysctl_expected"], "uaf_control": r.uaf_control()}
                        for k, r in runs.items()},
               "seeds": ref.seeds, "mechanism_probe": {"sizeof-ptr": probe, "holds_one_element": sizeof_ok},
               "malloc_bounds": capbounds, "platform": ref.obs["platform"], "toolchain": ref.build["toolchain"],
               "support_sources": ref.build["support_sources"], "per_cwe": {}}
    for cwe in sorted({r["cwe"] for r in rows}):
        rs = [r for r in rows if r["cwe"] == cwe]
        summary["per_cwe"][str(cwe)] = {
            "bad": dict(Counter(r["verdict"] for r in rs)), "good": dict(Counter(r["good"] for r in rs)),
            "sicode": dict(Counter(S.SICODE_NAME.get(r["sicode"], r["sicode"]) for r in rs if r["verdict"] == "CAUGHT")),
            "caught": sum(r["verdict"] == "CAUGHT" for r in rs), "missed": sum(r["verdict"] == "MISSED" for r in rs),
            "stopped": sum(r["verdict"].startswith("STOPPED") for r in rs),
            **({"off": dict(Counter(r["off"] for r in rs))} if "off" in other else {}),
            **({"every_free": dict(Counter(r["every_free"] for r in rs))} if "every-free" in other else {})}
    (a.out / f"{man['set']}-summary.json").write_text(json.dumps(summary, indent=1))
    for cwe, s in summary["per_cwe"].items():
        print(f"CWE-{cwe}: caught {s['caught']} missed {s['missed']} stopped {s['stopped']}; bad {s['bad']}; "
              f"good {s['good']}; sicode {s['sicode']}"
              + (f"; off {s['off']}" if "off" in s else "") + (f"; every-free {s['every_free']}" if "every_free" in s else ""))
    print("runs:", {k: (v["batches_counted"], v["batches"]) for k, v in summary["runs"].items()})
    sys.exit(0 if all(all(r.valid.values()) for r in runs.values()) else 75)


if __name__ == "__main__":
    main()
