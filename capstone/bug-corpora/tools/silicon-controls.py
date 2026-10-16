#!/usr/bin/env python3
"""Read the nested-lifetime controls' board run (docs/plans/2026-10-11-nested-lifetime-controls.md) beside their
virtual readings.

  silicon-controls.py --bundle silicon-runs/<set> --transcript board-corpus.txt --stage ARM=DIR... --llvm-bin DIR
                      --prereg-commit SHA --out-name NAME [--platform KEY=VALUE...] [--dry-run]

The controls are not corpus cases: they show that a port's lifetime patch leaves every permitted access alone
(parent, sibling, successor) and rejects only the stale one. So they get no case.json verdict. Each corpus gets one
bundle, <corpus>/results/<NAME>/ (results.json and README.md), with the silicon reading of every control run beside
the virtual reading of the same control.

Per run, from the board transcript (`CORPUS-RUN <arm>:<row> BEGIN ... END kind=K value=V`):
- the image is found in the arm's stage by the matrix row's sha256, so a row whose image is missing is an error;
- a capstone-exec fault line, resolved with the image's own symbols, reads `CAUGHT <cause> at <function>`;
- otherwise exit 0 with the family's completion mark reads `completes`;
- anything else is an error, not a reading.
The reading must equal the board matrix's outcome and the pre-registered predicted_board (qemu-and-predictions.json,
which must be unchanged since --prereg-commit). Otherwise nothing is written.

The virtual reading comes from each family's recorded results.json (VIRTUAL below). Every repetition must agree;
the reading is the one form they share.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
CORPORA = TOOLS.parent
REPO = TOOLS.parents[2]
sys.path.insert(0, str(TOOLS))
import verdicts as v  # noqa: E402

BLOCK = re.compile(r"^CORPUS-RUN (\S+) BEGIN$(.*?)^CORPUS-RUN \1 END kind=(\w+) value=(-?\d+)$", re.M | re.S)

# Board arm -> (corpus, which side of the virtual record it mirrors).
ARMS = {
    "silicon-pool-stock": ("ffmpeg/pool-repros", "stock"),
    "silicon-pool-sublet": ("ffmpeg/pool-repros", "protected"),
    "silicon-mc-stock": ("memcached/allocator-repros", "stock"),
    "silicon-mc-sublet": ("memcached/allocator-repros", "protected"),
    "silicon-wm-stock": ("wireshark/wmem-repros", "stock"),
    "silicon-wm-sublet": ("wireshark/wmem-repros", "protected"),
}
FAMILY = re.compile(r"^(ffmpeg-pool-controls-nested-lifetime|memcached-allocator-controls-lifetime-cache|"
                    r"memcached-allocator-controls-lifetime|wireshark-wmem-controls-lifetime-block-fast|"
                    r"wireshark-wmem-controls-lifetime)-(\d\d)-(\S+)$")


def completion(family, number, variant):
    if family.startswith("ffmpeg"):
        return "CTL-RESULT completed"
    if family.startswith("memcached"):
        return f"CTL-COMPLETED control={number} variant={variant}"
    return f"WM_DEFECT case={int(number)} mode=0 completed"


def norm_fault(cause, symbol):
    return f"CAUGHT {cause} at {symbol}"


# ---- the virtual records, one adapter per family's results.json ------------------------------------------------

WM_MODE = re.compile(r"^fault cause=(\d+) .*?\(link \S+, (\w+)\+")
MC_MODE = re.compile(r"^fault cause=(\d+) pc=\S+(?: \((\w+)\+)?")


def one(forms, what):
    forms = set(forms)
    if len(forms) != 1:
        raise SystemExit(f"virtual {what}: repetitions disagree or are missing: {sorted(forms)}")
    return forms.pop()


def virtual_ffmpeg(rel, number, variant, side):
    d = json.loads((CORPORA / rel / "results/2026-10-11-controls/results.json").read_text())
    arm = {"stock": "release", "protected": "protected"}[side]
    forms = []
    for boot in d["boots"]:
        if boot["arm"] != arm:
            continue
        run = boot["runs"][f"{number}-{variant}"]
        forms.append(norm_fault(run["fault"]["cause"], run["fault"]["symbol"]) if run["outcome"] == "fault"
                     else "completes" if run["outcome"] == "completed" else f"? {run['outcome']}")
    return one(forms, f"ffmpeg {number}-{variant} {arm}"), "ffmpeg/pool-repros/results/2026-10-11-controls"


def virtual_memcached(rel, number, variant, side, family):
    stamp = "2026-10-12-controls-ext" if family.endswith("cache") else "2026-10-11-controls"
    d = json.loads((CORPORA / rel / f"results/{stamp}/results.json").read_text())
    arm = {"stock": "virtual-malloc", "protected": "virtual-nested-pools"}[side]
    forms = []
    for s in d["summary"]:
        if s["arm"] == arm and s["control"] == number and s["variant"] == variant:
            if s["outcome"] == "completed":
                forms.append("completes")
            else:
                m = MC_MODE.match(s["outcome"])
                if not m:
                    raise SystemExit(f"virtual memcached {number}-{variant}: unreadable outcome {s['outcome']!r}")
                sym = m.group(2) or (s.get("fault") or {}).get("symbol") or "?"
                forms.append(norm_fault(m.group(1), sym))
    return one(forms, f"memcached {number}-{variant} {arm}"), f"{rel}/results/{stamp}"


def virtual_wmem(rel, number, variant, side, family):
    stamp = "2026-10-12-controls-ext" if family.endswith("block-fast") else "2026-10-11-controls"
    d = json.loads((CORPORA / rel / f"results/{stamp}/results.json").read_text())
    arm = {"stock": "release", "protected": "protected"}[side]
    keys = [k for k in d["controls"] if k.startswith(f"{number}-")]
    if len(keys) != 1:
        raise SystemExit(f"virtual wmem {family} {number}: controls {keys}")
    forms = []
    for boot in d["controls"][keys[0]][arm]["boots"]:
        o = boot["outcome"]
        if o == "completed":
            forms.append("completes")
        else:
            m = WM_MODE.match(o)
            if not m:
                raise SystemExit(f"virtual wmem {keys[0]}: unreadable outcome {o!r}")
            forms.append(norm_fault(m.group(1), m.group(2)))
    return one(forms, f"wmem {keys[0]} {arm}"), f"{rel}/results/{stamp}"


def virtual(rel, family, number, variant, side):
    if family.startswith("ffmpeg"):
        return virtual_ffmpeg(rel, number, variant, side)
    if family.startswith("memcached"):
        return virtual_memcached(rel, number, variant, side, family)
    return virtual_wmem(rel, number, variant, side, family)


# ---- the silicon run -------------------------------------------------------------------------------------------

def matrix_form(row):
    if row["outcome"] == "COMPLETED":
        return "completes"
    m = re.search(r"cause=(\d+) .*? in (\w+)\+", row["detail"])
    if row["outcome"] != "CAUGHT" or not m:
        raise SystemExit(f"{row['name']}: matrix outcome {row['outcome']} / {row['detail']!r} is neither")
    return norm_fault(m.group(1), m.group(2))


def prereg_form(p):
    if p["outcome"] == "COMPLETED":
        return "completes"
    if p["outcome"] == "CAUGHT":
        return norm_fault(p["cause"], p["site"])
    raise SystemExit(f"pre-registered outcome {p} is neither")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True, help="the board bundle, relative to bug-corpora/")
    p.add_argument("--transcript", type=Path, required=True)
    p.add_argument("--stage", action="append", required=True, help="ARM=DIR")
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--prereg-commit", required=True)
    p.add_argument("--out-name", required=True)
    p.add_argument("--platform", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    text = a.transcript.read_text(errors="replace").replace("\r", "")
    runs = {m.group(1): (m.group(2), m.group(3), int(m.group(4))) for m in BLOCK.finditer(text)}
    if not runs:
        raise SystemExit(f"{a.transcript}: no CORPUS-RUN blocks")
    stages = dict(kv.split("=", 1) for kv in a.stage)
    bundle = CORPORA / a.bundle
    per_corpus, tool_controls = {}, {}
    for arm, (rel, side) in ARMS.items():
        mpath = bundle / arm / "board-matrix.json"
        matrix = json.loads(mpath.read_text())
        q = json.loads((bundle / arm / "qemu-and-predictions.json").read_text())
        rel_q = (bundle / arm / "qemu-and-predictions.json").relative_to(REPO)
        pre = subprocess.run(["git", "-C", str(REPO), "show", f"{a.prereg_commit}:{rel_q}"],
                             capture_output=True, text=True, check=True).stdout
        if json.loads(pre) != q:
            raise SystemExit(f"{rel_q} changed since {a.prereg_commit}")
        prereg = {r["name"]: prereg_form(r["predicted_board"]) for r in q["rows"]}
        stage = Path(stages[arm])
        by_hash = {v.sha256(f): f for f in stage.glob("*.dom")}
        tool_controls[arm] = {k: c["outcome"] for k, c in matrix["controls"].items()}
        bad = [k for k, c in matrix["controls"].items() if not c["ok"]]
        if bad:
            raise SystemExit(f"{arm}: tool controls {bad} did not behave as required")
        for row in matrix["rows"]:
            name = row["name"]
            fam = FAMILY.match(name)
            if not fam:
                raise SystemExit(f"{arm}: row {name} is in no known control family")
            family, number, variant = fam.groups()
            image = by_hash.get(row["image_sha256"])
            if image is None:
                raise SystemExit(f"{arm}:{name}: no image with the matrix's sha256 in {stage}")
            block = runs.get(f"{arm}:{name}")
            if block is None:
                raise SystemExit(f"{arm}:{name}: no run in the transcript")
            body, kind, value = block
            fault = v.domain_fault(body, v.Symbols(a.llvm_bin, image))
            mark = completion(family, number, variant)
            if fault:
                silicon = norm_fault(fault.cause, fault.symbol)
                evidence = f"cause={fault.cause} pc={fault.pc:#x} in {fault.symbol}+{fault.offset:#x}"
            elif kind == "exit" and value == 0 and mark in body:
                silicon, evidence = "completes", f"`{mark}`, exit 0"
            else:
                raise SystemExit(f"{arm}:{name}: neither a fault nor a completion (end kind={kind} value={value})")
            for what, form in (("board matrix", matrix_form(row)), ("pre-registration", prereg.get(name))):
                if form != silicon:
                    raise SystemExit(f"{arm}:{name}: silicon reads {silicon!r}, the {what} {form!r}")
            vform, vsource = virtual(rel, family, number, variant, side)
            per_corpus.setdefault(rel, []).append({
                "family": family.split("-controls-")[1], "control": number, "run": variant, "arm": arm,
                "side": side, "image": image.name, "image_sha256": row["image_sha256"], "silicon": silicon,
                "silicon_evidence": evidence, "virtual": vform, "virtual_record": vsource,
                "same_as_virtual": vform == silicon})
    platform = dict(kv.split("=", 1) for kv in a.platform)
    for rel, rows in sorted(per_corpus.items()):
        out = CORPORA / rel / "results" / a.out_name
        differ = [r for r in rows if not r["same_as_virtual"]]
        print(f"{rel}: {len(rows)} runs, {len(differ)} differ from virtual -> {out.relative_to(REPO)}")
        for r in differ:
            print(f"  {r['arm']} {r['family']} {r['control']}-{r['run']}: virtual {r['virtual']}, silicon {r['silicon']}")
        if a.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        arms = sorted({r["arm"] for r in rows})
        doc = {"what": "the nested-lifetime controls on silicon, beside their virtual readings (tools/silicon-controls.py)",
               "pre_registration": ["capstone/docs/plans/2026-10-11-nested-lifetime-controls.md",
                                    f"the board lane's {a.bundle}/README.md and qemu-and-predictions.json, lane "
                                    f"{a.prereg_commit} (before the boot)"],
               "source_bundle": f"capstone/bug-corpora/{a.bundle}", "platform": platform,
               "tool_controls": {k: tool_controls[k] for k in arms}, "rows": rows}
        (out / "results.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        lines = [f"# {rel.split('/')[0]}'s nested-lifetime controls on silicon, beside their virtual readings",
                 "", "Generated by `tools/silicon-controls.py` from the board transcript of "
                 f"`{a.bundle}` (pre-registered at lane `{a.prereg_commit}`, before the boot). These are the plan's",
                 "controls, not defects, so no `case.json` carries them. Each silicon reading was re-read from the "
                 "transcript", "and equals the board matrix and the pre-registration. The virtual column is the "
                 "recorded virtual run of the", "same control (`virtual_record` in `results.json`).", "",
                 "Stock is the port without its lifetime patch (the virtual record's release / virtual-malloc arm, the "
                 "board's `*-stock` arm); protected is the port with it (protected / virtual-nested-pools, `*-sublet`).",
                 "",
                 "| family | control | run | stock: virtual | stock: silicon | protected: virtual | protected: silicon |",
                 "|---|---|---|---|---|---|---|"]
        keys = []
        for r in rows:
            k = (r["family"], r["control"], r["run"])
            if k not in keys:
                keys.append(k)
        for k in sorted(keys, key=lambda k: (k[0], k[1])):  # stable: a control's runs stay in board order
            cell = {}
            for r in rows:
                if (r["family"], r["control"], r["run"]) == k:
                    cell[r["side"]] = r
            def c(side, which):
                return cell[side][which] if side in cell else "(no such run on this arm)"
            lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {c('stock', 'virtual')} | {c('stock', 'silicon')} | "
                         f"{c('protected', 'virtual')} | {c('protected', 'silicon')} |")
        lines += ["", f"{len(rows)} silicon runs; {len(differ)} differ from the virtual reading of the same control."]
        if differ:
            lines += [f"- {r['arm']} {r['family']} {r['control']}-{r['run']}: virtual {r['virtual']}, silicon "
                      f"{r['silicon']}" for r in differ]
        lines += ["", "Platform: " + "; ".join(f"{k} {v}" for k, v in platform.items()) + ".", ""]
        (out / "README.md").write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
