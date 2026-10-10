#!/usr/bin/env python3
"""Run one plain-heap, plain-temporal, subobject, plane or carved corpus on a Capstone application domain arm.

    run-capstone-domain.py --corpus <corpus dir> --arm spatial|sublet \\
        --sdk <application SDK> --state <capstone-vm state> --out <fresh dir> [--cc-arg ARG]...

--cc-arg adds a source, archive or flag to every case's compile, for the two corpora whose cases
link a library: subobject-repros (the buffer-pool port's pool core) and plane-repros (FFmpeg's
libavutil). Every --cc-arg is recorded in record.json.

Every case becomes one domain image, built by the SDK's capstone-cc at -O0 (the CheriBSD and
native arms build at -O0 too: an optimiser that folds or hoists the crossing moves the fault off
the labelled probe). Each image runs twice in the same boot, `fixed N` and then `buggy N`.

WHAT THE TWO ARMS ARE. The SDK decides the arm, not the label: the tool reads the SDK's
CAPSTONE_APPLICATION_HEAP and refuses a mismatch.
  spatial  level0 heap. malloc narrows the returned capability to the bytes requested
           (CAPSTONE_LEVEL0_OBJECT_BOUNDS, on by default since 2026-09-30); free only marks
           the block free, so a stale pointer keeps working.
  sublet   the Sublet system allocator. Per-object bounds as above, and free revokes every
           capability to the released object.

WHAT THIS REFUSES TO SCORE.
  * Any row, unless this boot's own controls behaved: a one-past-the-end write must fault on
    both arms; a use-after-free read must fault on sublet and must complete on spatial. A
    control that does not do that means the arm is not what it says, and the run exits 75
    with no matrix.
  * A buggy row whose FIXED run did not print `VERDICT FIXED` and exit 0. That run is the
    case's own control on this platform: the same image, the same allocations, minus the
    defect. It also proves argv reached the program, since only `fixed` can print FIXED.
  * A missing case. Every case directory gets a row: built and run, or build-failed with the
    compiler's first lines. A corpus with a directory and no row is an error.
  * A fault without attribution. The fault pc is mapped back to the ELF's symbols, and the row
    says which function it landed in and whether that is the corpus's labelled probe.

Exit: 0 every row as predicted, 1 at least one row differs (data, not failure), 75 a control
or infrastructure failure (not a reading).
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verdicts  # noqa: E402
REPO = HERE.parents[2]
RUN = REPO / "capstone/ports/common/application/run.py"
SDK_HEAP = {"spatial": "level0", "sublet": "sublet", "sublet-chunks": "sublet",
            "capstone-subobject": "level0", "capstone-carve-bounds": "level0", "sublet-carve": "sublet",
            "sublet-full": "sublet"}
REVOKING = ("sublet", "sublet-chunks", "sublet-carve", "sublet-full")
# sublet-full: a plain case in its program's whole Sublet configuration -- the Sublet heap plus the
# program's nested-allocator port, linked and brought up before main() by tools/full-config/<program>.c.
# Every run must print that constructor's `FULLCONFIG <program> ... live` line, or the image is not
# the configuration it claims and the run exits 75.
FULLCONFIG = re.compile(r"^FULLCONFIG \S+ .*\blive\b.*$", re.M)
# The carve switch each carve arm needs, per corpus. capstone-carve-bounds narrows a pointer into
# the malloc'd block; sublet-carve is the Sublet port of the carve: the block is lent LINEAR by the
# Sublet heap and split into one region per carve (carved corpus only).
ARM_CARVE = {"capstone-carve-bounds": {"carved-repros": "-DFFC_CARVE_BOUNDS",
                                       "plane-repros": "-DFFP_CARVE_BOUNDS"},
             "sublet-carve": {"carved-repros": "-DFFC_SUBLET_CARVE", "plane-repros": "-DFFP_SUBLET_CARVE"}}
# The Sublet carve's own controls, per corpus: one shows the port's BOUND (a crossing out of a region
# it issued), the other its REVOCATION (an alias used after its region was carved again, or after the
# frame was freed). Each runs fixed and buggy in the same boot; buggy must fault at the probe.
CARVE_CONTROLS = {"carved-repros": (("carve-control.c", 99), ("carve-recarve-control.c", 98)),
                  "plane-repros": (("plane-bound-control.c", 99), ("plane-free-control.c", 98))}

# The tool's own controls, compiled with the corpus's SDK into the same boot.
CONTROL_C = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
__attribute__((noinline, used)) void ctl_write_probe(volatile unsigned char *p, unsigned char v) { *p = v; }
__attribute__((noinline, used)) unsigned ctl_read_probe(const volatile unsigned char *p) { return *p; }
int main(int argc, char **argv) {
  const char *which = argc > 1 ? argv[1] : "";
  printf("CONTROL %s BEGIN\n", which);
  if (!strcmp(which, "clean")) {
    unsigned char *p = malloc(24);
    if (!p) return 75;
    ctl_write_probe(p + 23, 1);
    printf("CONTROL clean RETURNED %u\n", ctl_read_probe(p + 23));
    free(p);
    return 0;
  }
  if (!strcmp(which, "oob")) {          /* one past the end of a 24-byte object */
    unsigned char *p = malloc(24);
    if (!p) return 75;
    ctl_write_probe(p + 24, 0x5a);
    printf("CONTROL oob RETURNED\n");
    return 0;
  }
  if (!strcmp(which, "subobj")) {       /* one past an 8-byte FIELD, into its sibling field */
    struct pair { unsigned char a[8]; unsigned char b[8]; } *s = malloc(sizeof *s);
    if (!s) return 75;
    memset(s, 0, sizeof *s);
    volatile unsigned char *field = s->a; /* the field, materialised as a pointer */
    ctl_write_probe(field + 8, 0x5a);     /* lands on s->b[0]: inside the allocation */
    printf("CONTROL subobj RETURNED b0=0x%02x\n", s->b[0]);
    return 0;
  }
  if (!strcmp(which, "uaf")) {          /* free, same-size allocation, stale read */
    unsigned char *p = malloc(48);
    if (!p) return 75;
    memset(p, 0x11, 48);
    volatile unsigned char *stale = p;
    free(p);
    unsigned char *q = malloc(48);
    if (!q) return 75;
    memset(q, 0xaa, 48);
    unsigned v = ctl_read_probe(stale);
    printf("CONTROL uaf RETURNED observed=0x%02x reissued=%d\n", v, (void *)q == (void *)p);
    return 0;
  }
  return 75;
}
'''


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def corpus_kind(corpus):
    name = corpus.name
    if name == "plain-heap-repros":
        return "spatial"
    if name == "plain-temporal-repros":
        return "temporal"
    if name in ("subobject-repros", "plane-repros", "carved-repros"):
        return "interior"     # the crossing stays inside ONE allocation, by each case's own CHECKs
    sys.exit(f"CONTROL-FAILED {corpus}: this tool runs plain-heap, plain-temporal, subobject, plane and "
             f"carved corpora only")


def predicted(kind, arm):
    """The buggy run's predicted outcome, fixed before any run (see the module docstring)."""
    if arm in ("capstone-carve-bounds", "sublet-carve"):
        return "CAUGHT"                     # each carved region is narrowed to its own extent
    if kind == "spatial":
        return "CAUGHT"                     # both arms: the crossing leaves an exact bound
    if kind == "interior":
        return "DEFECT-REPRODUCED"          # both arms: an allocation-granular bound is in bounds for it
    return "CAUGHT" if arm in REVOKING else "DEFECT-REPRODUCED"


# pc attribution and the fault line are shared with every runner, in verdicts.py.
Symbols = verdicts.Symbols
FAULT = verdicts.FAULT_LINE


def run_image(state, image, argv, out_base):
    result_path = out_base.with_suffix(".json")
    log_path = out_base.with_suffix(".out")
    with log_path.open("w") as log:
        try:
            subprocess.run([sys.executable, str(RUN), "--state", str(state), "--cwd", "/tmp",
                            "--result", str(result_path), str(image), "--", *argv],
                           stdout=log, stderr=subprocess.STDOUT, timeout=300)
        except subprocess.TimeoutExpired:
            log.write("\n[runner] TIMEOUT\n")
    text = log_path.read_text(errors="replace")
    result = json.loads(result_path.read_text()) if result_path.is_file() and result_path.stat().st_size else {}
    return text, result


def classify(text, result, symbols, probe_re):
    """(outcome, detail). outcome: CAUGHT, DEFECT-REPRODUCED, NOT-REISSUED, FIXED,
    CONTROL-FAILED, INCONCLUSIVE, OTHER."""
    if "[runner] TIMEOUT" in text:
        return "OTHER", "runner timeout"
    fault = result.get("fault") or ""
    m = FAULT.search(fault) or FAULT.search(text)
    if m:
        cause, pc, addr, code = m.group(1), int(m.group(2), 16), m.group(3), int(m.group(4), 16)
        fn, off = symbols.lookup(pc, code)
        where = f"{fn}+{off:#x}" if fn else "an address outside every function symbol"
        at_probe = bool(fn and probe_re.search(fn))
        return "CAUGHT", (f"cause={cause} pc={pc:#x} address={addr} in {where}"
                          f"{' (the labelled probe)' if at_probe else ' (NOT the labelled probe)'}")
    if result.get("kind") == "signal":
        return "OTHER", f"signal {result.get('value')} with no domain fault line"
    if "CONTROL-FAILED" in text or (result.get("kind") == "exit" and result.get("value") == 75):
        line = next((l for l in text.splitlines() if "CONTROL-FAILED" in l), "exit 75")
        return "CONTROL-FAILED", line.strip()
    v = re.search(r"^VERDICT (\S+)", text, re.M)
    if not v:
        return "OTHER", f"no VERDICT line; result={result}"
    return v.group(1), next(l for l in text.splitlines() if l.startswith("VERDICT"))[:200]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--arm", choices=sorted(SDK_HEAP), required=True)
    ap.add_argument("--sdk", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cc-arg", action="append", default=[],
                    help="extra compile input for every case (source, archive or flag); repeatable")
    ap.add_argument("--predictions", type=Path,
                    help="JSON {case directory name: CAUGHT|DEFECT-REPRODUCED} overriding the per-kind "
                         "prediction -- for the capstone-subobject arm, where it depends on whether the "
                         "crossing leaves a FIELD; must be committed before the run")
    ap.add_argument("--llvm-bin", type=Path,
                    help="for llvm-nm/llvm-readelf; default: the SDK's recorded compiler")
    a = ap.parse_args()

    corpus = a.corpus.resolve()
    kind = corpus_kind(corpus)
    subobject = a.arm == "capstone-subobject"
    # The driver spelling -fcapstone-subobject-bounds is ACCEPTED AND IGNORED through capstone-cc
    # (measured 2026-10-09: the same shrink count as no flag); only the cc1 form narrows. Require it.
    SUBOBJ = ["-Xclang", "-fcapstone-subobject-bounds"]
    has_subobj = any(a.cc_arg[i:i + 2] == SUBOBJ for i in range(len(a.cc_arg)))
    if subobject != has_subobj:
        print("CONTROL-FAILED capstone-subobject needs exactly --cc-arg=-Xclang "
              "--cc-arg=-fcapstone-subobject-bounds, and no other arm may carry it", file=sys.stderr)
        return 75
    # capstone-carve-bounds is the carved corpus's source remedy: ffc_carve() narrows each region
    # to its own extent when FFC_CARVE_BOUNDS is defined. Like the field-bounds flag, the switch
    # decides the arm, so it is required exactly where the arm is named and refused elsewhere.
    # The plane corpus has the same remedy for av_frame_get_buffer's carve (FFP_CARVE_BOUNDS).
    every = sorted({f for per in ARM_CARVE.values() for f in per.values()})
    flags = [f for f in every if f in a.cc_arg]
    need = ARM_CARVE.get(a.arm, {}).get(corpus.name)
    if (a.arm in ARM_CARVE and flags != [need]) or (a.arm not in ARM_CARVE and flags):
        print(f"CONTROL-FAILED {a.arm} needs exactly its corpus's carve switch ({ARM_CARVE}), "
              "and no other arm may carry one", file=sys.stderr)
        return 75
    overrides = json.loads(a.predictions.read_text()) if a.predictions else {}
    cache = (a.sdk / "CMakeCache.txt").read_text()
    heap = re.search(r"^CAPSTONE_APPLICATION_HEAP:STRING=(\S+)", cache, re.M)
    if not heap or heap.group(1) != SDK_HEAP[a.arm]:
        print(f"CONTROL-FAILED arm {a.arm} needs an SDK built with HEAP={SDK_HEAP[a.arm]}; "
              f"{a.sdk} has {heap.group(1) if heap else 'none'}", file=sys.stderr)
        return 75
    llvm_dir = re.search(r"^CAPSTONE_LLVM_BUILD_DIR:\S*=(\S+)", cache, re.M)
    llvm_bin = a.llvm_bin or (Path(llvm_dir.group(1)) / "bin" if llvm_dir else None)
    if not llvm_bin or not (llvm_bin / "llvm-nm").exists():
        print("CONTROL-FAILED no llvm-nm; pass --llvm-bin", file=sys.stderr)
        return 75
    cc = a.sdk / "capstone-cc"
    if a.out.exists():
        print(f"CONTROL-FAILED {a.out} exists: use a fresh output directory", file=sys.stderr)
        return 75
    status = subprocess.run([sys.executable, "-m", "capstone_vm", "--state", str(a.state), "status"],
                            cwd=str(REPO / "capstone/runtime/host"), capture_output=True, text=True)
    if "running" not in status.stdout:
        print("CONTROL-FAILED the VM is not up; this runner does not boot it", file=sys.stderr)
        return 75
    bindir, rundir = a.out / "bin", a.out / "runs"
    bindir.mkdir(parents=True)
    rundir.mkdir()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # ---- build ------------------------------------------------------------------------
    shared = corpus / "shared"
    cases = sorted(d for d in corpus.glob("[0-9][0-9]_*") if d.is_dir())
    if not cases:
        print(f"CONTROL-FAILED no case directories in {corpus}", file=sys.stderr)
        return 75
    images, rows = {}, []
    for d in cases:
        img = bindir / f"{d.name}.dom"
        b = subprocess.run([str(cc), "-O0", f"-I{shared}", str(d / "case.c"), str(shared / "driver.c"),
                            *a.cc_arg, "-o", str(img)], capture_output=True, text=True)
        if b.returncode or not img.is_file():
            first = " | ".join((b.stderr or b.stdout).strip().splitlines()[:3])
            rows.append(dict(case=d.name, outcome="BUILD-FAILED", detail=first[:300]))
            print(f"  {d.name:<62} BUILD-FAILED", flush=True)
            continue
        images[d.name] = img
    ctl_src = a.out / "control.c"
    ctl_src.write_text(CONTROL_C)
    ctl_img = bindir / "control.dom"
    ctl_flags = SUBOBJ if subobject else []  # the arm's own codegen
    b = subprocess.run([str(cc), "-O0", *ctl_flags, str(ctl_src), "-o", str(ctl_img)], capture_output=True, text=True)
    if b.returncode:
        print(f"CONTROL-FAILED control build: {b.stderr[:300]}", file=sys.stderr)
        return 75

    # ---- controls: the arm must be what it says --------------------------------------
    ctl_syms = Symbols(llvm_bin, ctl_img)
    ctl_probe = re.compile(r"^ctl_(read|write)_probe$")
    want = {"clean": "RETURNED", "oob": "CAUGHT",
            "uaf": "CAUGHT" if a.arm in REVOKING else "RETURNED",
            # Field bounds are what the capstone-subobject arm adds; every other arm must NOT see a
            # crossing that stays inside the allocation, or it is not the arm it says it is.
            "subobj": "CAUGHT" if subobject else "RETURNED"}
    controls = {}
    for which in ("clean", "oob", "uaf", "subobj"):
        text, result = run_image(a.state, ctl_img, [which], rundir / f"control-{which}")
        outcome, detail = classify(text, result, ctl_syms, ctl_probe)
        if outcome == "OTHER" and f"CONTROL {which} RETURNED" in text and result.get("value") == 0:
            outcome, detail = "RETURNED", next(l for l in text.splitlines() if "RETURNED" in l)
        controls[which] = dict(outcome=outcome, detail=detail, required=want[which])
        ok = outcome == want[which] and (outcome != "CAUGHT" or "(the labelled probe)" in detail)
        print(f"  control {which:<6} {outcome:<10} required {want[which]:<9} {'ok' if ok else 'FAILED'}  {detail}",
              flush=True)
        if not ok:
            print(f"CONTROL-FAILED control {which}: {outcome} ({detail}); the arm is not what it says, "
                  f"so no row is a reading", file=sys.stderr)
            (a.out / "controls.json").write_text(json.dumps(controls, indent=2) + "\n")
            return 75

    # ---- the Sublet carve's own controls: the bound AND the revocation, in this boot ----
    # A prefixed probe (ffh_read_probe, wsh_write_probe_u8) or the subobject corpus's unprefixed one
    # (write_probe). The prefix used to be required, so subobject 05 and 08 -- at write_probe+0x58 --
    # printed "NOT the labelled probe".
    probe_re = re.compile(r"(^|_)(read|write)_probe(_u8|_u32)?$")
    if a.arm == "sublet-carve":
        for src, num in CARVE_CONTROLS[corpus.name]:
            img = bindir / f"control-{num}.dom"
            b = subprocess.run([str(cc), "-O0", f"-I{shared}", str(shared / src), str(shared / "driver.c"),
                                *a.cc_arg, "-o", str(img)], capture_output=True, text=True)
            if b.returncode:
                print(f"CONTROL-FAILED {src} build: {b.stderr[:300]}", file=sys.stderr)
                return 75
            syms = Symbols(llvm_bin, img)
            ftext, fres = run_image(a.state, img, ["fixed", str(num)], rundir / f"control-{num}-fixed")
            fout, fdetail = classify(ftext, fres, syms, probe_re)
            btext, bres = run_image(a.state, img, ["buggy", str(num)], rundir / f"control-{num}-buggy")
            bout, bdetail = classify(btext, bres, syms, probe_re)
            ok = (fout == "FIXED" and fres.get("value") == 0 and bout == "CAUGHT"
                  and "(the labelled probe)" in bdetail)
            controls[f"carve-{num}"] = dict(fixed=f"{fout} {fdetail}", buggy=f"{bout} {bdetail}",
                                            required="fixed FIXED, buggy CAUGHT at the probe")
            print(f"  control carve-{num} fixed={fout} buggy={bout} {'ok' if ok else 'FAILED'}  {bdetail[:100]}",
                  flush=True)
            if not ok:
                print(f"CONTROL-FAILED {src}: the Sublet carve is not what it says", file=sys.stderr)
                (a.out / "controls.json").write_text(json.dumps(controls, indent=2) + "\n")
                return 75

    # ---- the cases --------------------------------------------------------------------
    expect = predicted(kind, a.arm)
    for d in cases:
        if d.name not in images:
            continue
        n = str(int(d.name[:2]))
        img = images[d.name]
        syms = Symbols(llvm_bin, img)
        ftext, fres = run_image(a.state, img, ["fixed", n], rundir / f"{d.name}-fixed")
        fout, fdetail = classify(ftext, fres, syms, probe_re)
        fixed_ok = fout == "FIXED" and fres.get("kind") == "exit" and fres.get("value") == 0
        btext, bres = run_image(a.state, img, ["buggy", n], rundir / f"{d.name}-buggy")
        bout, bdetail = classify(btext, bres, syms, probe_re)
        if a.arm == "sublet-full":
            for which, text in (("fixed", ftext), ("buggy", btext)):
                if not FULLCONFIG.search(text) or "FULLCONFIG-FAILED" in text:
                    print(f"CONTROL-FAILED {d.name} {which}: no live FULLCONFIG line; the image is not the "
                          "full configuration", file=sys.stderr)
                    return 75
        if not fixed_ok:
            outcome = "FIXED-ARM-FAILED"
            detail = f"fixed run read {fout}: {fdetail}; buggy run ({bout}: {bdetail}) is not scored"
        else:
            outcome, detail = bout, bdetail
        site = outcome == "CAUGHT" and "(the labelled probe)" not in detail and at_declared_site(d, detail)
        if site:
            detail += " -- a fault site the case declared before the run (case.json fault_sites)"
        want_row = overrides.get(d.name, expect)
        row = dict(case=d.name, outcome=outcome, detail=detail, predicted=want_row,
                   as_predicted=outcome == want_row and (outcome != "CAUGHT" or "(the labelled probe)" in detail
                                                         or site),
                   image_sha256=sha256(img), fixed=f"{fout} exit={fres.get('value')}",
                   **({"full_config": FULLCONFIG.search(btext).group(0)} if a.arm == "sublet-full" else {}),
                   buggy_exit=f"{bres.get('kind')}={bres.get('value')}")
        rows.append(row)
        print(f"  {d.name:<62} {outcome:<18} {'ok ' if row['as_predicted'] else 'DIFF'} {detail[:110]}", flush=True)

    # ---- coverage, then the record ---------------------------------------------------------
    named = {r["case"] for r in rows}
    missing = [d.name for d in cases if d.name not in named]
    if missing:
        print(f"CONTROL-FAILED cases with no row: {missing}", file=sys.stderr)
        return 75
    vmcfg = json.loads((a.state / "config.json").read_text())
    record = dict(
        corpus=str(corpus.relative_to(REPO)), kind=kind, arm=a.arm, started=started,
        predicted_buggy_outcome=expect,
        sdk=dict(path=str(a.sdk), heap=heap.group(1),
                 heap_log=(re.search(r"^CAPSTONE_APPLICATION_HEAP_LOG:STRING=(\S+)", cache, re.M) or [None, None])[1],
                 runtime_sha256=sha256(a.sdk / "libapplication-runtime.a"),
                 sdk_json_sha256=sha256(a.sdk / "sdk.json") if (a.sdk / "sdk.json").exists() else None),
        compiler=subprocess.run([str(cc), "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        platform={k: v.get("sha256") for k, v in vmcfg["identity"]["files"].items()},
        platform_environment=vmcfg["identity"].get("environment"),
        tool_sha256=sha256(Path(__file__)), cc_args=a.cc_arg,
        controls=controls, rows=rows)
    (a.out / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    tally = {}
    for r in rows:
        tally[r["outcome"]] = tally.get(r["outcome"], 0) + 1
    diff = [r["case"] for r in rows if not r.get("as_predicted")]
    print(f"\n{corpus.parent.name}/{corpus.name} on {a.arm}: {len(rows)} cases; {tally}; "
          f"predicted {expect}; differing: {diff or 'none'}")
    lost = infra_rows(rows)
    if lost:
        # A row that printed no verdict (a dead VM, a signal with no domain fault line) differs from
        # its prediction for a reason that is not the mechanism. It used to make this run exit 1,
        # "data"; it is infrastructure, so the run is not a reading.
        print(f"INFRA: {len(lost)} row(s) produced no verdict ({', '.join(lost)}): not a reading",
              file=sys.stderr)
        return 75
    return 1 if diff else 0


def at_declared_site(case_dir, detail):
    """A fault outside the labelled probe counts only in a function the case declared BEFORE the run
    (case.json `fault_sites`, justified in `fault_sites_why` from the source). Until 2026-10-10 the
    capstone-subobject arm accepted a fault anywhere, which is how ff2_case_run+off and memcpy+0xdc
    became catches with nothing but the fixed run tying them to the defect."""
    sites = json.loads((case_dir / "case.json").read_text()).get("fault_sites") or []
    m = re.search(r" in ([A-Za-z_][\w.$]*)\+0x", detail)
    return bool(m and m.group(1) in sites)


def infra_rows(rows):
    """Cases whose run produced no verdict at all -- outcome OTHER -- which no prediction names."""
    return [str(r["case"]) for r in rows if r.get("outcome") == "OTHER"]


if __name__ == "__main__":
    sys.exit(main())
