#!/usr/bin/env python3
"""Read the engine corpus's silicon control arm (tests/rtl-smoke/repro322-silicon-*) into a verdicts-v1 bundle.

    fold-silicon.py OUT --folder <repro322-silicon folder> --boot <dir> [--boot <dir> ...] --images <dir>
                    --llvm-bin <dir> --arm silicon-sqlite-stock

This adapter REPORTS; tools/verdicts.py decides, by the same rules as shared/run-virtual.py where the harness
allows. The board ran each cell on the classic line (fpga_driver/run_sqlite_stages_fpga.py through
drivers/board-r322.sh): sqlite_host.user loads r322-<cell>.dom into a domain, enters it, and prints the domain's
output only after the domain returns. Each --boot is that boot's artefact folder: boot-raw.txt (the console
capture), driver.log (the runner, with the debug-mux readout taken at a wedge) and fw.sha.

The capture can open with the end of the previous boot, so only text after its one OpenSBI banner is read, and a
capture with no banner or two is refused. One Observation per case cell:

  reached     a returning cell: the case's own lines, printed after `SQ: H/return`. A wedged cell: `SQ: G/enter`,
              the host entering the domain, which is the nearest point to run-virtual.py's BEGIN line that this
              harness prints. Neither is a marker at the defective access
  completed   `### TEST k/n END ... rc=0`, after the harness tag's `<tag> NOTRAP done` (the case's last line)
  fault       a cell with no END line wedged the core. The runner's readout gives the trap log
              {seen, mcause[6:0]}, mepc and tval. The pc is resolved with the image's own symbols against the
              loader's `DBAS:` line, the domain's physical base
  attribution `function` when that symbol is a case.json fault_sites entry, as on the virtual arms

The configuration's controls (controls/control_*_mem5.c) are cells too: a control completes when it printed
`CONTROL <name> mark` and returned. Every image is checked against the folder's SHA256SUMS before it is used.
Cells that are not corpus cases (the base domain, the legacy blobclose, the review branch's 02) are reported and
left out.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[1]
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import verdicts as v  # noqa: E402

# The folder's cells that are corpus cases, by case number, and the configuration's controls.
CASE_CELLS = {"wschema": "24", "blobwrite": "04", "mem5design": "18", "blobclose03": "03", "jsonstatic": "17",
              "jsonroot": "16", "backupattach": "02", "detachtrig": "05"}
CONTROL_CELLS = {"ctluafmem5": "uaf-mem5", "ctlboundsmem5": "bounds-mem5"}
NOT_CASES = {"base322": "the 3.22.0 base domain", "blobclose": "the legacy standalone domain, not case 03's case.c",
             "backupattachrev": "the external review's case 02 (branch sqlite-review), not on dev"}
START = re.compile(r"^### TEST (\d+)/(\d+) START /test-domains/r322-(\S+)\.dom ###$", re.M)
END = r"^### TEST {k}/{n} END /test-domains/r322-{cell}\.dom rc=(-?\d+) ###$"
MUX = re.compile(r"\[wedge\] sw=(\d+) (.*?)\s+(0x[0-9a-f]{2}) [01]{8}$", re.M)


def boot_text(capture):
    text = Path(capture).read_text(errors="replace").replace("\r", "")
    banners = [m.start() for m in re.finditer(r"OpenSBI v\d", text)]
    if len(banners) != 1:
        raise SystemExit(f"{capture}: {len(banners)} OpenSBI banners; a capture must hold exactly one boot")
    return text[banners[0]:]


def cell_blocks(text):
    """{cell: (block text, END rc or None)}; a cell with no END line is the boot's wedge."""
    starts = list(START.finditer(text))
    out = {}
    for i, m in enumerate(starts):
        k, n, cell = m.group(1), m.group(2), m.group(3)
        stop = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        body = text[m.end():stop]
        e = re.search(END.format(k=k, n=n, cell=re.escape(cell)), body, re.M)
        if cell in out:
            raise SystemExit(f"{cell}: run twice in one boot")
        out[cell] = (body[:e.start()] if e else body, int(e.group(1)) if e else None)
    return out


def wedge_readout(driver_log):
    """mcause, mepc, tval from the runner's debug-mux readout at the wedge, or None."""
    text = Path(driver_log).read_text(errors="replace")
    i = text.find("[stages] WEDGED")
    if i < 0:
        return None
    mux = {}
    for m in MUX.finditer(text[i:]):
        mux.setdefault(m.group(2).strip(), int(m.group(3), 16))
    log = mux.get("TRAP LOG {seen,mcause[6:0]}")

    def word(name):
        parts = [mux.get(f"trap {name}[{hi}:{lo}]" + suffix) for lo, hi in ((b, b + 7) for b in range(0, 64, 8))
                 for suffix in ("   (LATCHED)", "  (LATCHED)", " (LATCHED)") if f"trap {name}[{hi}:{lo}]" + suffix in mux]
        if len(parts) != 8:
            raise SystemExit(f"{driver_log}: the {name} readout has {len(parts)} of 8 bytes")
        return sum(b << (8 * j) for j, b in enumerate(parts))
    if log is None or not log & 0x80:
        raise SystemExit(f"{driver_log}: wedged, but the trap log shows no trap ({log})")
    return log & 0x7F, word("mepc"), word("tval")


def disasm(llvm_bin, image, symbols, pc, code_start):
    vaddr = pc - code_start + symbols.base
    out = subprocess.run([str(Path(llvm_bin) / "llvm-objdump"), "-d", "--no-show-raw-insn",
                          f"--start-address={vaddr:#x}", f"--stop-address={vaddr + 4:#x}", str(image)],
                         capture_output=True, text=True, check=True).stdout
    m = re.search(rf"^\s*{vaddr:x}:\s+(.+)$", out, re.M)
    return " ".join(m.group(1).split()) if m else "?"


def harness_tag(source):
    m = re.search(r'REPRO322_MAIN\("([^"]+)"\)', Path(source).read_text())
    if not m:
        raise SystemExit(f"{source}: no REPRO322_MAIN tag")
    return m.group(1)


def own_lines(body, tag):
    after = body.split("SQ: H/return", 1)
    return [l.strip() for l in after[1].splitlines() if l.startswith(f"{tag} ")] if len(after) == 2 else []


def observe(cell, run, tag, sites, image, symbols, a):
    o = v.Observation(case="", arm="", image_sha256=v.sha256(image))
    if run is None:
        o.infra, o.notes = "infra", "no START line: the cell did not run"
        return o
    body, rc, wedge = run
    if rc is None:
        if "SQ: G/enter" not in body:
            o.infra, o.notes = "infra", "wedged before the host entered the domain"
            return o
        if wedge is None:
            raise SystemExit(f"{cell}: no END line, and the runner read no wedge")
        cause, mepc, tval = wedge
        base = re.search(r"^DBAS:([0-9A-Fa-f]+)$", body, re.M)
        if not base:
            raise SystemExit(f"{cell}: no DBAS line, so the pc cannot be placed in the image")
        code_start = int(base.group(1), 16)
        fn, off = symbols.lookup(mepc, code_start)
        o.reached = True
        o.reach_evidence = ("the host entered the domain (SQ: G/enter); this harness prints a case's lines only "
                            "after the domain returns, so a wedge leaves no marker inside the case")
        o.fault = v.Fault(cause=cause, pc=mepc, symbol=fn, offset=off)
        insn = disasm(a.llvm_bin, image, symbols, mepc, code_start)
        where = f"`{insn}` at {fn}+{off:#x}, trap tval {tval:#x}"
        if fn and fn in sites:
            o.attribution = "function"
            o.attribution_evidence = f"{fn} is where host ASan reported it; the trapping instruction is {where}"
        else:
            o.attribution_evidence = f"in {fn or 'no known function'}; ASan site {sorted(sites) or 'not recorded'}; {where}"
        return o
    lines = own_lines(body, tag)
    if not lines:
        o.notes = f"END rc={rc} with none of the case's own lines"
        return o
    o.reached = True
    o.reach_evidence = "the case ran and printed its own lines; no marker at the access on this harness: " + \
        " / ".join(lines[-2:])
    o.completed = rc == 0 and f"{tag} NOTRAP done" in lines
    if not o.completed:
        o.notes = f"END rc={rc}; last line {lines[-1]!r}"
    return o


def control_seen(name, run):
    if run is None:
        return "none", "did not run"
    body, rc, wedge = run
    after = body.split("SQ: H/return", 1)
    if rc == 0 and len(after) == 2 and f"CONTROL {name} mark" in after[1]:
        return "complete", f"CONTROL {name} mark, then returned rc=0"
    if rc is None:
        return "none", f"wedged; the readout is {wedge}"
    return "none", f"END rc={rc}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--boot", type=Path, action="append", required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--platform", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE")
    a = p.parse_args()

    config = json.loads((CORPUS / "corpus.json").read_text())["arm_configurations"][a.arm]
    spec = v.load_arms()[config]
    sums = {}
    for line in (a.folder / "SHA256SUMS").read_text().splitlines():
        h, name = line.split()
        sums[name[len("r322-"):-len(".dom")]] = h

    runs, boots = {}, []
    for d in a.boot:
        blocks = cell_blocks(boot_text(d / "boot-raw.txt"))
        wedged = [c for c, (_, rc) in blocks.items() if rc is None]
        if len(wedged) > 1:
            raise SystemExit(f"{d}: {len(wedged)} cells without an END line")
        readout = wedge_readout(d / "driver.log") if wedged else None
        for cell, (body, rc) in blocks.items():
            if cell in runs and cell not in NOT_CASES:     # the base domain runs in every boot, unread
                raise SystemExit(f"{cell}: run in two boots; name which reading counts")
            runs[cell] = (body, rc, readout if rc is None else None)
        boots.append({"dir": d.name, "firmware_sha256": (d / "fw.sha").read_text().split()[0],
                      "cells": sorted(blocks), "wedged": wedged})
    unknown = set(runs) - set(CASE_CELLS) - set(CONTROL_CELLS) - set(NOT_CASES)
    if unknown:
        raise SystemExit(f"cells this adapter does not know: {sorted(unknown)}")

    def image_of(cell):
        img = a.images / cell / f"r322-{cell}.dom"
        if not img.is_file() or v.sha256(img) != sums[cell]:
            raise SystemExit(f"{img}: missing, or not the folder's {sums[cell][:16]}")
        return img

    controls = []
    for cell, name in CONTROL_CELLS.items():
        image_of(cell)
        observed, evidence = control_seen(name, runs.get(cell))
        controls.append(v.Control(name, observed, evidence))
        print(f"  control {name:<12} {observed:<9} (expected {spec['controls'][name]})")

    rows = []
    for cell, number in sorted(CASE_CELLS.items(), key=lambda kv: kv[1]):
        d = next(CORPUS.glob(f"{number}_*"))
        claims = json.loads((d / "case.json").read_text())
        img = image_of(cell)
        o = observe(cell, runs.get(cell), harness_tag(d / "case.c"), set(claims.get("fault_sites", [])), img,
                    v.Symbols(a.llvm_bin, img), a)
        o.case, o.arm, o.controls = d.name, a.arm, list(controls)
        verdict = v.judge(o, spec)
        rows.append((o, verdict))
        print(f"  {d.name[:44]:<44} {verdict[0]:<10} {verdict[1] or ''}")
    for cell, why in sorted(NOT_CASES.items()):
        if cell in runs:
            print(f"  (left out) {cell}: {why}")

    inputs = {
        "configuration": config,
        "platform": dict(kv.split("=", 1) for kv in a.platform),
        "source_folder": str(a.folder.resolve().relative_to(REPO)) if a.folder.resolve().is_relative_to(REPO)
        else str(a.folder),
        "boots": boots,
        "cells_left_out": {c: NOT_CASES[c] for c in sorted(NOT_CASES) if c in runs},
        **dict(kv.split("=", 1) for kv in a.extra),
    }
    record = v.write_bundle(a.output, str(CORPUS.relative_to(REPO)), a.arm, rows, inputs)
    print(f"wrote {a.output}: {record['tally']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
