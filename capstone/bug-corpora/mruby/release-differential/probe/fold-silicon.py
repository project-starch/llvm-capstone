#!/usr/bin/env python3
"""Read the board's silicon bounds-only arm of this corpus (tools/run-silicon-cases.py kind `mruby-rb`) into the
corpus's own records: results/<stamp>/<arm>/matrix.tsv and each case.json's arms[<arm>].

    fold-silicon.py --set-dir <silicon-runs/.../silicon-mruby-bounds> --transcript board-corpus.txt --stage <dir>
                    --llvm-bin <dir> [--arm silicon-mruby-bounds] [--stamp 2026-10-16-silicon]
                    [--skipped NN=reason ...] [--write]

The corpus is not on verdicts-v1: its arms are matrix.tsv rows and hand-held case.json verdicts. This tool writes
both from the board transcript, re-read here, and refuses when its reading disagrees with the board's matrix.

What one run is: B5's mruby.dom, the same image for every case, run on the case's own `trigger.rb`, staged as
NN-trigger.rb (byte-equal to the corpus's file, checked). The reading:
  fault        the capstone-exec domain fault line, its pc resolved with mruby.dom's symbols
  completed    exit 0 with the harness footer's `["PASS"]`; or exit 0 with the harness's failure lines (a wrong
               answer), which go to the harness column as the corpus's other matrices record them
  none         anything else (a non-zero exit, a signal without a fault line, a missing run)

The arm revokes nothing, so a fault is a detection only where per-object bounds are the mechanism. By the case's
class (case.json `class`), as pre-registered by the board lane and agreed before its results:
  spatial            a capability fault is CAUGHT; a cause other than 28 is flagged
  temporal           MISSED as a detection whatever it reads; the reading is recorded in the note
  both               CAUGHT only with cause 28 (its spatial half); otherwise as temporal
  any other class    the corpus's own rule (a capability fault is CAUGHT, a completion MISSED), outside the
                     temporal/spatial scope and counted in neither total
The gate: the port's smoke.rb prints SMOKE_DONE and exits 0, and the tool's four controls read as the board matrix
requires. If either fails, every case is NO-READING.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[1]
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import verdicts as v  # noqa: E402

BLOCK = re.compile(r"^CORPUS-RUN (\S+) BEGIN$(.*?)^CORPUS-RUN \1 END kind=(\w+) value=(-?\d+)$", re.M | re.S)
NEED = {"RETURNED": "complete", "CAUGHT": "fault"}
ORACLE = {
    "spatial": "silicon, bounds only: B5's mruby.dom over silicon-alloc.c, every object narrowed to its request, nothing "
               "revoked. A capability fault is CAUGHT (per-object bounds; cause 28 is the RTL's out-of-bounds trap), a "
               "run that completes is MISSED",
    "temporal": "silicon, bounds only: B5's mruby.dom over silicon-alloc.c, every object narrowed to its request, "
                "nothing revoked. This arm has no lifetimes, so a temporal row is MISSED as a detection whatever it "
                "reads; the note records how the defect shows (a fault on an untagged value, a wrong answer, or PASS)",
    "both": "silicon, bounds only: B5's mruby.dom over silicon-alloc.c, nothing revoked. CAUGHT only with cause 28, the "
            "spatial half; any other reading is MISSED as a detection, as for a temporal row",
    "other": "silicon, bounds only: B5's mruby.dom over silicon-alloc.c, nothing revoked. Outside the temporal/spatial "
             "scope: recorded under this corpus's own rule (a capability fault is CAUGHT, a completion MISSED) and "
             "counted in neither total",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def reading(block, symbols):
    """(outcome, cause, function, harness, end) for one run."""
    if block is None:
        return "none", "", "", "", "no run"
    body, kind, value = block
    fault = v.domain_fault(body, symbols)
    end = f"end kind={kind} value={value}"
    if fault:
        return "fault", str(fault.cause), fault.symbol or "", "", end
    lines = [l.strip() for l in body.splitlines() if l.strip().startswith('["')]
    if kind == "exit" and value == 0 and lines:
        return "completed", "", "", " ".join(lines), end
    return "none", "", "", "", end


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-dir", type=Path, required=True)
    p.add_argument("--transcript", type=Path, required=True)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--arm", default="silicon-mruby-bounds")
    p.add_argument("--stamp", default="2026-10-16-silicon")
    p.add_argument("--skipped", action="append", default=[], metavar="NN=reason")
    p.add_argument("--write", action="store_true")
    a = p.parse_args()

    matrix = json.loads((a.set_dir / "board-matrix.json").read_text())
    text = a.transcript.read_text(errors="replace").replace("\r", "")
    runs = {m.group(1): (m.group(2), m.group(3), int(m.group(4))) for m in BLOCK.finditer(text)}
    if not runs:
        raise SystemExit(f"{a.transcript}: no CORPUS-RUN blocks")
    pre = f"{a.arm}:"

    # The stage is what SHA256SUMS says it is.
    sums = {}
    for line in (a.set_dir / "SHA256SUMS").read_text().splitlines():
        h, name = line.split()
        sums[Path(name).name] = h
    for name, h in sums.items():
        f = a.stage / name
        if not f.is_file() or sha(f) != h:
            raise SystemExit(f"{f}: missing, or not SHA256SUMS's {h[:16]}")
    image = a.stage / "mruby.dom"
    symbols = v.Symbols(a.llvm_bin, image)
    ctl_symbols = v.Symbols(a.llvm_bin, a.stage / "controls.dom")

    # The gate.
    problems = []
    smoke = runs.get(f"{pre}mruby-smoke-control")
    if not (smoke and smoke[1] == "exit" and smoke[2] == 0 and "SMOKE_DONE" in smoke[0]):
        problems.append("smoke.rb did not print SMOKE_DONE and exit 0")
    for name, want in matrix["controls"].items():
        blk = runs.get(f"{pre}control-{name}")
        seen, _, fn, _, end = reading(blk, ctl_symbols)
        seen = {"fault": "fault", "completed": "complete"}.get(seen, "complete" if blk and blk[1] == "exit"
                                                              and blk[2] == 0 else "none")
        if seen != NEED.get(want["required"]):
            problems.append(f"control {name}: {seen} ({end}), the matrix requires {want['required']}")
    gate_ok = not problems

    rows, cells, disagree = [], {}, []
    for row in matrix["rows"]:
        if row["kind"] != "mruby-rb":
            continue
        name, case = row["name"], row["case"]
        n = case[:2]
        d = CORPUS / case
        claims = json.loads((d / "case.json").read_text())
        if row["image_sha256"] != sha(image):
            raise SystemExit(f"{name}: the matrix's image {row['image_sha256'][:16]} is not mruby.dom")
        staged = a.stage / f"{n}-trigger.rb"
        if staged.read_bytes() != (d / claims.get("trigger", "trigger.rb")).read_bytes():
            raise SystemExit(f"{staged}: not byte-equal to {case}/trigger.rb")
        outcome, cause, fn, harness, end = reading(runs.get(f"{pre}{name}"), symbols)
        board = {"fault": "CAUGHT", "completed": "COMPLETED" if harness == '["PASS"]' else "WRONG-ANSWER"}.get(outcome)
        if board != row["outcome"]:
            disagree.append(f"{name}: read {outcome} {cause} {fn} {harness[:60]!r} ({end}), matrix {row['outcome']}")
        cls = claims.get("class", "")
        key = cls if cls in ORACLE else "other"
        what = (f"capability fault, cause {cause}, in {fn}" if outcome == "fault" else
                "completed, the harness printed PASS" if harness == '["PASS"]' else
                "completed with a wrong answer: the harness printed its failures" if outcome == "completed" else
                f"no reading ({end})")
        if not gate_ok:
            verdict, note = "NO-READING", "the arm's gate failed: " + "; ".join(problems)
        elif outcome == "none":
            verdict, note = "NO-READING", what
        elif key == "temporal":
            verdict, note = "MISSED", f"{what}; not a detection: this arm revokes nothing"
        elif key == "both":
            verdict = "CAUGHT" if cause == "28" else "MISSED"
            note = what if cause == "28" else f"{what}; not a detection: neither the bounds check (28) nor revocation"
        else:
            verdict = "CAUGHT" if outcome == "fault" else "MISSED"
            note = what + ("" if key == "other" or outcome != "fault" or cause == "28"
                           else "; FLAG: a bounds row trapped with a cause other than 28")
            if key == "other":
                note += f"; class {cls}, outside the temporal/spatial scope"
        rows.append("\t".join([case, a.arm, outcome, cause, fn, harness]))
        cells[case] = {"oracle": ORACLE[key], "verdict": verdict, "verdict_note": note,
                       "verdict_from": f"results/{a.stamp}/{a.arm}/matrix.tsv {n}"}
        print(f"  {case[:40]:<40} {cls:<18} {outcome:<9} {cause:>3} {fn[:22]:<22} -> {verdict}")
    if disagree:
        raise SystemExit("the reading and the board matrix disagree:\n  " + "\n  ".join(disagree))
    skipped = dict(kv.split("=", 1) for kv in a.skipped)
    for d in sorted(CORPUS.glob("[0-9][0-9]_*")):
        if d.name not in cells:
            if d.name[:2] not in skipped:
                raise SystemExit(f"{d.name}: neither run nor given a --skipped reason")
            cells[d.name] = {"status": "not written", "note": f"not run on silicon: {skipped[d.name[:2]]}"}
    print(f"gate: {'ok' if gate_ok else problems}; "
          f"{sum(c.get('verdict') == 'CAUGHT' for c in cells.values())} CAUGHT, "
          f"{sum(c.get('verdict') == 'MISSED' for c in cells.values())} MISSED, "
          f"{sum(c.get('verdict') == 'NO-READING' for c in cells.values())} NO-READING, "
          f"{sum('status' in c for c in cells.values())} not written")
    if not a.write:
        return 0
    out = CORPUS / "results" / a.stamp / a.arm
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix.tsv").write_text("case\tarm\toutcome\tcause\tfunction\tharness\n" + "\n".join(rows) + "\n")
    for case, cell in cells.items():
        path = CORPUS / case / "case.json"
        raw = path.read_text()
        j = json.loads(raw)
        j["arms"][a.arm] = cell
        ind = 1 if raw.startswith('{\n "') else 2
        if json.dumps(json.loads(raw), indent=ind, ensure_ascii=False) + "\n" != raw:
            raise SystemExit(f"{path}: does not round-trip; refusing to rewrite it")
        path.write_text(json.dumps(j, indent=ind, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(REPO)}/matrix.tsv and {len(cells)} case.json entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
