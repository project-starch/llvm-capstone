#!/usr/bin/env python3
"""Turn a board run of the corpora (tools/run-silicon-cases.py, b0run.sh `corpus-<arm>`) into verdicts-v1 bundles.

  silicon-verdicts.py --arm ARM --configuration CONFIG --matrix board-matrix.json --transcript board-corpus.txt
                      --images DIR --llvm-bin DIR --stamp STAMP --platform KEY=VALUE... [--extra KEY=VALUE...]
                      [--prefix ARM:] [--row-control ROW[:fixed|:buggy]=CONTROL...] [--skip-row ROW...]

--prefix is for a boot that ran several arms: its blocks are named `<arm>:<run>`. --row-control turns a matrix row
that is a control rather than a case (FFmpeg pool-repros' controls/virtual/90, for example) into the configuration's
named control. By default it reads the row's buggy run; `ROW:fixed=CONTROL` reads its fixed run instead, so one row
can give two controls (the carved corpus's re-carve control 98: live-carve from its fixed run, uaf-carve from its buggy
run). A fault means `fault`, and a clean exit means `complete`. --skip-row leaves out a row that belongs to another
corpus's later run (a BUILD-FAILED plane row in the carved boot); every other row must be a case or a control.

The board lane's matrix is a REPORT. This tool re-reads the run itself from the board transcript, and judges every
case with tools/verdicts.py exactly as the QEMU arms are judged. It writes one bundle per corpus, at
<corpus>/results/<STAMP>/<ARM>/, for tools/derive-verdicts.py.
- The transcript is the run's verbatim log: one `CORPUS-RUN <name> BEGIN ... END kind=K value=V` block per run.
- Each case is two runs: <name>-fixed and <name>-buggy, both of the same image <images>/<name>.dom.
- Each control is one run, control-<which>.

Per case:
- The image's sha256 must equal the matrix row's.
- The fixed run must print `VERDICT FIXED` and exit 0. Otherwise the case is NO-READING control-failed, and the note
  names the fixed run's fault if it had one.
- A buggy run that exits 75 refused its own setup (the case's CHECK printed CONTROL-FAILED): NO-READING control-failed,
  as tools/run-virtual-cases.py reads the same exit.
- The buggy run's `case=N arm=buggy` line (optionally followed by more fields), or wmem's `WM_DEFECT case=N ready`,
  is the reach marker.
- Its capstone-exec fault line is resolved with the image's own symbols (verdicts.Symbols), and attributed to the
  labelled probe (the corpus PROBE rule) or to a function the case declares (`fault_sites`).
- A run that printed `VERDICT DEFECT-REPRODUCED` and exited 0 COMPLETED.

Two-image corpora (PostgreSQL's c-repros and mmgr-repros) build each case twice: the case, and a negative control
(-DPGCLIENT_NEGATIVE_CONTROL / -DPG_NEGATIVE_CONTROL) that moves one value to the safe side and must complete. The
matrix carries them as `<name>-case` and `<name>-ctl-control` rows; the control row gates its case as a fixed run
does. The reach mark, the completion line and the probe are the family's (TWO_IMAGE): c-repros names the function
it expects to fault in (`expect_fault_in=<fn>`), mmgr-repros faults in pg_probe / pg_write_probe. A corpus control
row whose run key is its own row name (`--row-control ROW=CONTROL`) is read under that key.

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

# The corpus PROBE rule, plus wmem's two probe functions (wireshark/wmem-repros/shared/run-defects.py PROBES).
PROBE = re.compile(r"(^|_)(read|write)_probe(_u8|_u32)?$|^wm_probe$")
# A buggy run's reach mark: the corpus drivers' `case=N arm=buggy` (the carved driver appends carve=<mode>), or
# wmem's hosted driver's `WM_DEFECT case=N ready`.
REACH = re.compile(r"^(?:case=(\d+) arm=buggy(?: .*)?|WM_DEFECT case=(\d+) ready)$", re.M)
BLOCK = re.compile(r"^CORPUS-RUN (\S+) BEGIN$(.*?)^CORPUS-RUN \1 END kind=(\w+) value=(-?\d+)$", re.M | re.S)
# FIXED-ARM-FAILED (the fixed run did not read FIXED) and CONTROL-FAILED (the buggy run refused its own setup, exit
# 75) are the board's names for runs that are not readings.
MATRIX_VERDICT = {"CAUGHT": v.CAUGHT, "DEFECT-REPRODUCED": v.MISSED, "COMPLETED": v.MISSED,
                  "FIXED-ARM-FAILED": v.NO_READING, "CONTROL-FAILED": v.NO_READING}
# Two-image families, by matrix row kind: the reach mark, the completion line and the probe rule.
TWO_IMAGE = {
    "pg-c": dict(reach=r"^PG_DEFECT case={n} mark$", done="case {n} RETURNED",
                 probe_from=r"PG_DEFECT case=\d+ expect_fault_in=([A-Za-z_][\w.$]*)@"),
    "pg-mmgr": dict(reach=r"^PG_DEFECT case={n} ready$", done="PG_DEFECT case={n} mode=0 completed",
                    probe=r"^pg_(write_)?probe$"),
}


class Images:
    """The stage's images, found by name and checked by sha256, or found by sha256 alone."""
    def __init__(self, root):
        self.root, self._by_hash = root, None

    def get(self, name, sha):
        p = self.root / f"{name}.dom"
        if p.is_file() and v.sha256(p) == sha:
            return p
        if self._by_hash is None:
            self._by_hash = {v.sha256(q): q for q in sorted(self.root.glob("*.dom"))}
        if sha in self._by_hash:
            return self._by_hash[sha]
        raise SystemExit(f"{name}: no image with the matrix's sha256 {sha[:16]} in {self.root}")


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
        # The control's result line (`CONTROL clean RETURNED 1`), not its opening `CONTROL clean BEGIN`.
        lines = [l for l in body.splitlines() if l.startswith("CONTROL ")]
        line = next((l for l in lines if "RETURNED" in l), lines[-1] if lines else "exit 0")
        return "complete", line.strip()
    return "none", f"end kind={kind} value={value}"


def two_image(row, by_name, runs, pre, images, a, controls, spec):
    """One case of a two-image corpus: its negative control must complete; then the case is judged."""
    fam = TWO_IMAGE[row["kind"]]
    name, case = row["name"], row["case"]
    if not name.endswith("-case"):
        raise SystemExit(f"{name}: a {row['kind']} row should be named <case>-case")
    base = name[: -len("-case")]
    ctl_row = by_name.get(f"{base}-ctl-control")
    if ctl_row is None:
        raise SystemExit(f"{name}: no negative-control row {base}-ctl-control in the matrix")
    image = images.get(base, row["image_sha256"])
    ctl_image = images.get(f"{base}-ctl", ctl_row["image_sha256"])
    blk, ctl_blk = runs.get(f"{pre}{name}"), runs.get(f"{pre}{ctl_row['name']}")
    if blk is None or ctl_blk is None:
        raise SystemExit(f"{name}: the transcript has no {'case' if blk is None else 'control'} run")
    n = int(case[:2])
    done = fam["done"].format(n=n)
    obs = v.Observation(case=case, arm=a.arm, controls=controls, image_sha256=row["image_sha256"])
    cbody, ckind, cval = ctl_blk
    body, kind, value = blk
    if not (ckind == "exit" and cval == 0 and done in cbody):
        f = v.domain_fault(cbody, v.Symbols(a.llvm_bin, ctl_image))
        obs.infra = "control-failed"
        obs.notes = (f"the negative-control image did not print `{done}` and exit 0 (end kind={ckind} value={cval})"
                     + (f"; it faulted cause={f.cause} pc={f.pc:#x} in {f.symbol}+{f.offset:#x}" if f else ""))
    elif kind == "exit" and value == 75:
        obs.infra = "control-failed"
        obs.notes = next((l.strip() for l in body.splitlines() if "CONTROL-FAILED" in l),
                         "exit 75: the case refused its own setup (a CHECK)") + "; case run end kind=exit value=75"
    else:
        marker = re.search(fam["reach"].format(n=n), body, re.M)
        obs.reached = bool(marker)
        obs.reach_evidence = marker.group(0) if marker else ""
        obs.fault = v.domain_fault(body, v.Symbols(a.llvm_bin, image))
        if obs.fault:
            if "probe_from" in fam:
                m = re.search(fam["probe_from"], body)
                probe, how = (re.compile(rf"^{re.escape(m.group(1))}$") if m else None), "the function the case named before its access"
            else:
                probe, how = re.compile(fam["probe"]), "the labelled probe"
            sites = json.loads((REPO / row["corpus"] / case / "case.json").read_text()).get("fault_sites") or ()
            if probe and obs.fault.symbol and probe.search(obs.fault.symbol):
                obs.attribution = "probe"
                obs.attribution_evidence = f"the fault is in {obs.fault.symbol}, {how}"
            elif obs.fault.symbol in sites:
                obs.attribution = "function"
                obs.attribution_evidence = f"{obs.fault.symbol} is a fault site declared in case.json"
        elif kind == "exit" and value == 0 and done in body:
            obs.completed = True
        obs.notes = f"silicon; case run end kind={kind} value={value}; negative control completed"
    return obs, v.judge(obs, spec)


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
    p.add_argument("--skip-row", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    arms = v.load_arms()
    if a.configuration not in arms:
        raise SystemExit(f"{a.configuration} is not a configuration in tools/arms.json")
    spec = arms[a.configuration]
    matrix = json.loads(a.matrix.read_text())
    runs = blocks(a.transcript)

    pre = a.prefix
    row_controls = []
    for kv in a.row_control:
        key, control = kv.split("=", 1)
        row, _, run = key.partition(":")
        row_controls.append((row, run or "buggy", control))
    not_cases = {r for r, _, _ in row_controls} | set(a.skip_row)
    ctl_symbols = v.Symbols(a.llvm_bin, a.images / "controls.dom")
    controls = []
    for name in spec["controls"]:
        if name in {c for _, _, c in row_controls}:
            continue
        observed, evidence = control_observed(runs.get(f"{pre}control-{name}"), ctl_symbols)
        controls.append(v.Control(name=name, observed=observed, evidence=evidence))
    by_name = {row["name"]: row for row in matrix["rows"]}
    images = Images(a.images)
    for rname, run, control in row_controls:
        row = by_name.get(rname)
        if row is None:
            raise SystemExit(f"--row-control {rname}: no such matrix row")
        image = images.get(rname, row["image_sha256"])
        block = runs.get(f"{pre}{rname}-{run}") or runs.get(f"{pre}{rname}")
        observed, evidence = control_observed(block, v.Symbols(a.llvm_bin, image))
        controls.append(v.Control(name=control, observed=observed, evidence=evidence))
    missing = sorted(set(spec["controls"]) - {c.name for c in controls})
    if missing:
        raise SystemExit(f"controls {missing} of {a.configuration} have no run in this transcript")

    per_corpus, disagree = {}, []
    for row in matrix["rows"]:
        if row["name"] in not_cases:
            continue
        kind = row.get("kind", "")
        if kind.endswith("-control") and kind[:-len("-control")] in TWO_IMAGE:
            continue        # a negative control: read with its case
        if kind in TWO_IMAGE:
            obs, verdict = two_image(row, by_name, runs, pre, images, a, controls, spec)
            want = MATRIX_VERDICT.get(row["outcome"])
            if verdict[0] != want:
                disagree.append(f"{row['name']}: judged {verdict[0]} ({verdict[1]}: {verdict[2]}), matrix {row['outcome']}")
            per_corpus.setdefault(row["corpus"], []).append((obs, verdict))
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
            ffault = v.domain_fault(fbody, v.Symbols(a.llvm_bin, image))
            obs.notes = (f"the fixed run did not print VERDICT FIXED and exit 0 (end kind={fkind} value={fval})"
                         + (f"; it faulted cause={ffault.cause} pc={ffault.pc:#x} in {ffault.symbol}+{ffault.offset:#x}"
                            if ffault else ""))
        elif buggy[1] == "exit" and buggy[2] == 75:
            obs.infra = "control-failed"
            obs.notes = next((l.strip() for l in buggy[0].splitlines() if "CONTROL-FAILED" in l),
                             "exit 75: the case refused its own setup (a CHECK)") + "; buggy run end kind=exit value=75"
        else:
            body, kind, value = buggy
            marker = REACH.search(body)
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
