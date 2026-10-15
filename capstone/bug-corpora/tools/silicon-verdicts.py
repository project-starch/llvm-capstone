#!/usr/bin/env python3
"""Turn a board run of the corpora (tools/run-silicon-cases.py, b0run.sh `corpus-<arm>`) into verdicts-v1 bundles.

  silicon-verdicts.py --arm ARM --configuration CONFIG --matrix board-matrix.json --transcript board-corpus.txt
                      --images DIR --llvm-bin DIR --stamp STAMP --platform KEY=VALUE... [--extra KEY=VALUE...]
                      [--prefix ARM:] [--row-control ROW=CONTROL...]

--prefix is for a boot that ran several arms: its blocks are named `<arm>:<run>`. --row-control turns a matrix row
that is a control rather than a case (FFmpeg pool-repros' controls/virtual/90, for example) into the configuration's
named control: its buggy run's fault means `fault`, and a clean exit means `complete`.

The board lane's matrix is a REPORT. This tool re-reads the run itself from the board transcript, and judges every
case with tools/verdicts.py exactly as the QEMU arms are judged. It writes one bundle per corpus, at
<corpus>/results/<STAMP>/<ARM>/, for tools/derive-verdicts.py.
- The transcript is the run's verbatim log: one `CORPUS-RUN <name> BEGIN ... END kind=K value=V` block per run.
- Each case is two runs: <name>-fixed and <name>-buggy, both of the same image <images>/<name>.dom.
- Each control is one run, control-<which>.

Per case:
- The image's sha256 must equal the matrix row's.
- The fixed run must print `VERDICT FIXED` and exit 0. Otherwise the case is NO-READING control-failed.
- The buggy run's `case=N arm=buggy` line is the reach marker.
- Its capstone-exec fault line is resolved with the image's own symbols (verdicts.Symbols), and attributed to the
  labelled probe (the corpus PROBE rule) or to a function the case declares (`fault_sites`).
- A run that printed `VERDICT DEFECT-REPRODUCED` and exited 0 COMPLETED.

The judge's verdict must agree with the matrix's outcome (CAUGHT, or DEFECT-REPRODUCED = MISSED) on every row.
Otherwise the tool writes nothing and says which rows disagree. A transcript with no blocks, or a row with no run,
is an error, not a NO-READING.
"""
import argparse
import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[2]
sys.path.insert(0, str(TOOLS))
import verdicts as v  # noqa: E402

PROBE = re.compile(r"(^|_)(read|write)_probe(_u8|_u32)?$")
BLOCK = re.compile(r"^CORPUS-RUN (\S+) BEGIN$(.*?)^CORPUS-RUN \1 END kind=(\w+) value=(-?\d+)$", re.M | re.S)
MATRIX_VERDICT = {"CAUGHT": v.CAUGHT, "DEFECT-REPRODUCED": v.MISSED}


def blocks(transcript):
    text = Path(transcript).read_text(errors="replace").replace("\r", "")
    out = {m.group(1): (m.group(2), m.group(3), int(m.group(4))) for m in BLOCK.finditer(text)}
    if not out:
        raise SystemExit(f"{transcript}: no CORPUS-RUN blocks")
    return out


def control_observed(block, symbols):
    if block is None:
        return "none", "no run"
    body, kind, value = block
    fault = v.domain_fault(body, symbols)
    if fault:
        return "fault", f"cause={fault.cause} pc={fault.pc:#x} in {fault.symbol}+{fault.offset:#x}"
    if kind == "exit" and value == 0:
        line = next((l for l in body.splitlines() if l.startswith("CONTROL ")), "exit 0")
        return "complete", line.strip()
    return "none", f"end kind={kind} value={value}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True)
    p.add_argument("--configuration", required=True)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--transcript", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--stamp", required=True)
    p.add_argument("--platform", action="append", default=[])
    p.add_argument("--extra", action="append", default=[])
    p.add_argument("--prefix", default="")
    p.add_argument("--row-control", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    arms = v.load_arms()
    if a.configuration not in arms:
        raise SystemExit(f"{a.configuration} is not a configuration in tools/arms.json")
    spec = arms[a.configuration]
    matrix = json.loads(a.matrix.read_text())
    runs = blocks(a.transcript)

    pre = a.prefix
    row_controls = dict(kv.split("=", 1) for kv in a.row_control)
    ctl_symbols = v.Symbols(a.llvm_bin, a.images / "controls.dom")
    controls = []
    for name in spec["controls"]:
        if name in row_controls.values():
            continue
        observed, evidence = control_observed(runs.get(f"{pre}control-{name}"), ctl_symbols)
        controls.append(v.Control(name=name, observed=observed, evidence=evidence))
    for row in matrix["rows"]:
        if row["name"] in row_controls:
            image = a.images / f"{row['name']}.dom"
            if v.sha256(image) != row["image_sha256"]:
                raise SystemExit(f"{row['name']}: image sha256 is not the matrix's")
            observed, evidence = control_observed(runs.get(f"{pre}{row['name']}-buggy"), v.Symbols(a.llvm_bin, image))
            controls.append(v.Control(name=row_controls[row["name"]], observed=observed, evidence=evidence))
    missing = sorted(set(spec["controls"]) - {c.name for c in controls})
    if missing:
        raise SystemExit(f"controls {missing} of {a.configuration} have no run in this transcript")

    per_corpus, disagree = {}, []
    for row in matrix["rows"]:
        if row["name"] in row_controls:
            continue
        name, case = row["name"], row["case"]
        corpus = REPO / row["corpus"]
        image = a.images / f"{name}.dom"
        if not image.is_file():
            raise SystemExit(f"{name}: no image {image}")
        digest = v.sha256(image)
        if digest != row["image_sha256"]:
            raise SystemExit(f"{name}: image sha256 {digest[:16]} is not the matrix's {row['image_sha256'][:16]}")
        fixed, buggy = runs.get(f"{pre}{name}-fixed"), runs.get(f"{pre}{name}-buggy")
        if fixed is None or buggy is None:
            raise SystemExit(f"{name}: the transcript has no {'fixed' if fixed is None else 'buggy'} run")
        obs = v.Observation(case=case, arm=a.arm, controls=controls, image_sha256=digest)
        fbody, fkind, fval = fixed
        if not ("VERDICT FIXED" in fbody and fkind == "exit" and fval == 0):
            obs.infra = "control-failed"
            obs.notes = f"the fixed run did not print VERDICT FIXED and exit 0 (end kind={fkind} value={fval})"
        else:
            body, kind, value = buggy
            marker = re.search(r"^case=(\d+) arm=buggy$", body, re.M)
            obs.reached = bool(marker)
            obs.reach_evidence = marker.group(0) if marker else ""
            obs.fault = v.domain_fault(body, v.Symbols(a.llvm_bin, image))
            if obs.fault:
                sites = json.loads((corpus / case / "case.json").read_text()).get("fault_sites") or ()
                if obs.fault.symbol and PROBE.search(obs.fault.symbol):
                    obs.attribution = "probe"
                    obs.attribution_evidence = f"the fault is at the labelled probe {obs.fault.symbol}"
                elif obs.fault.symbol in sites:
                    obs.attribution = "function"
                    obs.attribution_evidence = f"{obs.fault.symbol} is a fault site declared in case.json"
            elif kind == "exit" and value == 0 and "VERDICT DEFECT-REPRODUCED" in body:
                obs.completed = True
            obs.notes = f"silicon; buggy run end kind={kind} value={value}"
        verdict = v.judge(obs, spec)
        want = MATRIX_VERDICT.get(row["outcome"])
        if verdict[0] != want:
            disagree.append(f"{name}: judged {verdict[0]} ({verdict[1]}: {verdict[2]}), matrix {row['outcome']}")
        per_corpus.setdefault(row["corpus"], []).append((obs, verdict))

    if disagree:
        raise SystemExit("the judge and the board matrix disagree:\n  " + "\n  ".join(disagree))
    platform = dict(kv.split("=", 1) for kv in a.platform)
    extra = dict(kv.split("=", 1) for kv in a.extra)
    for rel, rows in sorted(per_corpus.items()):
        out = REPO / rel / "results" / a.stamp / a.arm
        tally = {}
        for _, (verdict, reason, _) in rows:
            tally[verdict] = tally.get(verdict, 0) + 1
        print(f"{rel}: {len(rows)} cases {dict(sorted(tally.items()))} -> {out.relative_to(REPO)}")
        if not a.dry_run:
            v.write_bundle(out, rel, a.arm, rows, dict(extra, configuration=a.configuration, platform=platform))
    print(f"controls: {[(c.name, c.observed) for c in controls]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
