#!/usr/bin/env python3
"""The plain corpora on the silicon ABI: build, run on the silicon-mimic QEMU, and judge a board transcript.

    run-silicon-cases.py build  --arm A --out OUT [--jobs N] CORPUS...
    run-silicon-cases.py qemu   --arm A --out OUT --state VMSTATE
    run-silicon-cases.py board  --arm A --out OUT --transcript FILE

docs/plans/2026-10-15-bug-corpus-on-fpga.md is the plan; silicon-predictions/2026-10-16-plain is the pre-registered
table. Each case is case.c + shared/driver.c, built through the B0 pipeline (runtime/silicon/build-b0-hello.sh:
gp-captable, full LTO, the pool carve, check-repr under the full scheme, no-fp-gate) at -O0, one image per case, run
as `fixed N` then `buggy N` in the same boot as the arm's controls (run-capstone-domain.py's CONTROL_C).

THE ARMS.
  silicon-level0     level0 as it ships: malloc narrows each object to its request (per-object bounds); free only
                     marks the block free. Temporal cases are MISSED by construction: this is the control arm.
  silicon-lifetimes  the same, built with CAPSTONE_LEVEL0_LIFETIMES: every object is a CDERIVE child of one group
                     over the arena at exactly its request size, and free CREVOKEs it. The bug corpus's silicon-malloc
                     arm (lead decision (b), 2026-10-16), measured as a pattern on silicon by B6.
  silicon-carve-sublet  FFmpeg's carved corpus with FFC_SUBLET_CARVE over the lifetimes heap: each carved region is
                     a CDERIVE child of a group over the block, and a re-carve CREVOKEs it (virtual-nested-pools'
                     mechanism).
  silicon-carve-bounds  the carved and plane corpora with their source remedy (FFC_CARVE_BOUNDS / FFP_CARVE_BOUNDS):
                     each region narrowed by SHRINK at the carving code, over level0.
  silicon-mc-sublet  memcached's nested allocators with the application's patch 0006 (slabs.c and cache.c hand out
                     CDERIVE children and revoke them at slabs_free / cache_free) over the lifetimes heap, driven by
                     the corpus's driver-virtual.c: virtual-nested-pools' configuration on the silicon ABI.
  silicon-mc-stock   the same allocators stock (patches 0001 + 0002) on the stock-malloc ledger (MCP_ADAPTER_BACKING):
                     virtual-malloc's configuration. The prepared sources and the ledger come with --extra-src.
  silicon-wm-sublet  wireshark's wmem with the port's patch 0001 (block and block_fast objects are CDERIVE children
                     of their block, revoked at free and at the pool reset) over the lifetimes heap: virtual-nested-
                     pools' configuration. The buggy run is the plain sequence (`0 N`).
  silicon-wm-stock   wmem as released over the lifetimes heap: virtual-malloc's. The buggy run is the fix
                     differential (`0 N buggy`), whose temporal cases assert the stale chunk was reoccupied.
wmem's fixed run is `0 N fixed` on both; its three controls run once as `0 N` (never the differential, whose verdict
line would judge a control by a case's observation) and gate the arm.
The nested-lifetime controls (docs/plans/2026-10-11-nested-lifetime-controls.md: parent, sibling and successor at
memcached's slabs and cache, FFmpeg's AVBufferPool and wmem's block and block_fast allocators) are their own kind: each
control program has a list of runs, every one of which must complete, except the successor's stale read on a Sublet
arm, which must fault at the family's probe. Pass the family's directory as the corpus.
postgres/c-repros: each case runs as `<image> N` against the real upstream functions (shared/upstream_*.c), announces
`PG_DEFECT case=N expect_fault_in=<function>` before its access, and prints `case N RETURNED` if nothing faulted. A
fault counts in that function or in a case.json fault site. Each case is also built with -DPGCLIENT_NEGATIVE_CONTROL,
which moves one value to the safe side of the boundary; that image must complete and gates the arm.
postgres/mmgr-repros: the same two images per case (-DPG_NEGATIVE_CONTROL), run as `0 N` against PostgreSQL's own
memory-context managers (the port's `spatial` variant on silicon-pg-stock, `sublet` with patch 0003 on
silicon-pg-sublet; sources via --extra-src). A fault counts in pg_probe or a case.json fault site; completion is
`PG_DEFECT case=N mode=0 completed`. Its corpus control 00 (controls/) completes on stock and faults on Sublet.
cpython/pymalloc-repros: each case is the port's hosted replay (ports/cpython/pymalloc/src/native/main.c) around the
case's PYC_CASE body, over the port's obmalloc.c and backing.c (--extra-src): the `ported` variant (patches 0001 + 0002)
on silicon-pym-stock, `protected` (+ patch 0003: every block a CDERIVE child of its arena, CREVOKEd on free) on
silicon-pym-sublet. A run is `<fixture> <report>`: the fixture is run-virtual.py's own one-event file naming the case,
staged beside the images (argv carries it as @DIR@, resolved per platform: /mnt/host on QEMU, --board-dir on the board).
A fault counts only in read_probe; completion is `PYM completed=1`. The corpus control 00 (controls/) completes on
stock and faults at read_probe on Sublet.
mruby/release-differential: each case is upstream's own regression test (trigger.rb) run by one interpreter image
(build --image: B5's mruby.dom, silicon-alloc.c's exact per-object bounds, no revocation) as `<image> <trigger.rb>`,
the script staged beside it. The image is staged once and shared. A run reads CAUGHT on any capability fault (the
corpus's bounds-arm oracle is the cause, not a probe), COMPLETED when the harness prints ["PASS"], and WRONG-ANSWER
when it prints its failures and exits 0. The port's scripts/smoke.rb is the arm's control (SMOKE_DONE). Cases the
image cannot run are skipped with the reason: a C-API case (its own build), a case that needs mruby-task (not in
the silicon build), and a case whose recorded reading is a watchdog (it does not terminate). smoke.rb writes its file
into /tmp, writable on both platforms (its M12 then reads false on both: smoke.rb is not in /tmp).
The carved corpus also runs its own controls as rows: 99 (one byte past a 16-byte carve, inside its block) on every
arm, and 98 (a re-carved region's old alias) on the Sublet carve. They gate the arm like the four tool controls.

The judge is classify() from run-capstone-domain.py and the rules of tools/verdicts.py: a fault is CAUGHT only at the
labelled probe or a fault site the case declared; a buggy row counts only when the same image's fixed run printed
VERDICT FIXED and exited 0; no row counts unless the boot's own controls behaved as the arm declares.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("rcd", HERE / "run-capstone-domain.py")
rcd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcd)

ARMS = {
    "silicon-level0": {"defs": "", "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "RETURNED",
                                               "subobj": "RETURNED"}},
    "silicon-lifetimes": {"defs": "-DCAPSTONE_LEVEL0_LIFETIMES=1",
                          "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "CAUGHT",
                                       "subobj": "RETURNED"}},
    # FFmpeg's AVBufferPool corpus over the lifetimes heap, its pools stock (the "traced" FFmpeg variant, patch 0001)
    # or as Sublet lifetimes (the "protected" variant, patch 0003): which FFmpeg tree is passed with --extra-src
    "silicon-pool-stock": {"defs": "-DCAPSTONE_LEVEL0_LIFETIMES=1",
                           "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "CAUGHT", "subobj": "RETURNED"}},
    "silicon-pool-sublet": {"defs": "-DCAPSTONE_LEVEL0_LIFETIMES=1",
                            "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "CAUGHT", "subobj": "RETURNED"}},
    # level0's bounds plus field bounds (the cc1 form; the driver spelling is accepted and ignored, run-capstone-domain.py)
    "silicon-subobject": {"defs": "", "app_cflags": "-Xclang -fcapstone-subobject-bounds",
                          "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "RETURNED",
                                       "subobj": "CAUGHT"}},
    # the carve switch is applied per corpus by `build`, never by hand, so an arm cannot carry the wrong one
    "silicon-carve-sublet": {"defs": "-DCAPSTONE_LEVEL0_LIFETIMES=1",
                             "carve": {"carved-repros": f"-DFFC_SUBLET_CARVE -I{REPO}/capstone/runtime/include"},
                             "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "CAUGHT", "subobj": "RETURNED"}},
    "silicon-carve-bounds": {"defs": "", "carve": {"carved-repros": "-DFFC_CARVE_BOUNDS",
                                                   "plane-repros": "-DFFP_CARVE_BOUNDS"},
                             "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "RETURNED",
                                          "subobj": "RETURNED"}},
    # memcached/allocator-repros and its control 90 (controls/virtual): the per-corpus switches select the arm
    "silicon-mc-sublet": {"defs": "-DCAPSTONE_LEVEL0_LIFETIMES=1", "driver": "driver-virtual.c",
                          "corpus_defs": {"allocator-repros": "-DMC_CARVE_BOUNDS -DMC_CAPSTONE_SUBLET"},
                          "virtual": "virtual-nested-pools",
                          "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "CAUGHT", "subobj": "RETURNED"}},
    "silicon-mc-stock": {"defs": "-DCAPSTONE_LEVEL0_LIFETIMES=1",
                         "corpus_defs": {"allocator-repros": "-DMCP_ADAPTER_BACKING"},
                         "virtual": "virtual-malloc",
                         "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "CAUGHT", "subobj": "RETURNED"}},
    # wireshark/wmem-repros and its controls: which wmem source the image links is the arm (--extra-src)
    "silicon-wm-sublet": {"defs": "-DCAPSTONE_LEVEL0_LIFETIMES=1", "virtual": "virtual-nested-pools",
                          "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "CAUGHT", "subobj": "RETURNED"}},
    # postgres/mmgr-repros: which memory-context variant the image links is the arm (--extra-src)
    "silicon-pg-stock": {"defs": "-DCAPSTONE_LEVEL0_LIFETIMES=1", "virtual": "virtual-malloc",
                         "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "CAUGHT", "subobj": "RETURNED"}},
    "silicon-pg-sublet": {"defs": "-DCAPSTONE_LEVEL0_LIFETIMES=1", "virtual": "virtual-pg-pools",
                          "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "CAUGHT", "subobj": "RETURNED"}},
    # cpython/pymalloc-repros: which obmalloc.c variant the image links is the arm (--extra-src)
    "silicon-pym-stock": {"defs": "-DCAPSTONE_LEVEL0_LIFETIMES=1", "virtual": "virtual-malloc",
                          "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "CAUGHT", "subobj": "RETURNED"}},
    "silicon-pym-sublet": {"defs": "-DCAPSTONE_LEVEL0_LIFETIMES=1", "virtual": "virtual-nested-pools",
                           "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "CAUGHT", "subobj": "RETURNED"}},
    # mruby/release-differential on one interpreter image (--image), the corpus's sysalloc-bounds configuration
    "silicon-mruby-bounds": {"defs": "", "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "RETURNED",
                                                     "subobj": "RETURNED"}},
    "silicon-wm-stock": {"defs": "-DCAPSTONE_LEVEL0_LIFETIMES=1", "virtual": "virtual-malloc", "wm_differential": True,
                         "controls": {"clean": "RETURNED", "oob": "CAUGHT", "uaf": "CAUGHT", "subobj": "RETURNED"}},
}
# wmem's controls, as controls/README.md declares them per arm: 00 uaf-chunk, 01 uaf-jumbo, 02 bounds-chunk
WM_CONTROLS = {"silicon-wm-sublet": {"00": "CAUGHT", "01": "CAUGHT", "02": "CAUGHT"},
               "silicon-wm-stock": {"00": "DEFECT-REPRODUCED", "01": "CAUGHT", "02": "DEFECT-REPRODUCED"}}
CARVE_SWITCHES = ("FFC_SUBLET_CARVE", "FFC_CARVE_BOUNDS", "FFP_CARVE_BOUNDS", "MC_CARVE_BOUNDS", "MC_CAPSTONE_SUBLET",
                  "MCP_ADAPTER_BACKING")
# The nested-lifetime control families, by their path under bug-corpora: the driver (relative to the family), extra
# include directories, each run as (control number, variant, argv), the completion marker, and the stale run's probe.
LIFETIME = {
    ("memcached", "allocator-repros", "controls", "lifetime"): dict(
        driver="driver-controls.c", inc=[], probe=r"^read_probe$", done="CTL-COMPLETED control={n} variant={v}",
        runs=lambda sublet: [(91, "page"), *([(91, "generation")] if sublet else []), (92, "sibling"),
                             (93, "through-c"), (93, "stale")],
        argv=lambda n, v: [str(n), v]),
    ("memcached", "allocator-repros", "controls", "lifetime-cache"): dict(
        driver="../lifetime/driver-controls.c", inc=["../lifetime", "../../shared"], probe=r"^read_probe$",
        done="CTL-COMPLETED control={n} variant={v}",
        runs=lambda sublet: [(94, "list-head"), (95, "sibling"), (96, "through-c"), (96, "stale")],
        argv=lambda n, v: [str(n), v]),
    ("ffmpeg", "pool-repros", "controls", "nested-lifetime"): dict(
        driver="shared/driver.c", inc=[], probe=r"^stale_read_probe$", done="CTL-RESULT completed",
        defs=lambda sublet: f"-DCTL_PROTECTED={1 if sublet else 0}",
        runs=lambda sublet: [(91, "run"), (92, "run"), (93, "run"), (93, "stale")],
        argv=lambda n, v: [v, str(n)]),
    ("wireshark", "wmem-repros", "controls", "lifetime"): dict(
        driver="shared/driver.c", inc=[], probe=r"^wm_probe$", done="WM_DEFECT case={n} mode=0 completed",
        runs=lambda sublet: [(0, "parent"), (1, "sibling"), (2, "successor"), (3, "stale")],
        argv=lambda n, v: ["0", str(n)]),
    ("wireshark", "wmem-repros", "controls", "lifetime-block-fast"): dict(
        driver="shared/driver.c", inc=[], probe=r"^wm_probe$", done="WM_DEFECT case={n} mode=0 completed",
        runs=lambda sublet: [(0, "parent"), (1, "successor"), (2, "stale")],
        argv=lambda n, v: ["0", str(n)]),
}


# Families whose rows are runs with their own completion marker. c-repros names its expected function in its output.
PGC = ("postgres", "c-repros")
PGM = ("postgres", "mmgr-repros")
PGM_CTL = ("postgres", "mmgr-repros", "controls")
RUNFAM = dict(LIFETIME)
RUNFAM[PGC] = dict(probe_from=r"PG_DEFECT case=\d+ expect_fault_in=([A-Za-z_][\w.$]*)@", done="case {n} RETURNED")
RUNFAM[PGM] = dict(probe=r"^pg_(write_)?probe$", done="PG_DEFECT case={n} mode=0 completed")
RUNFAM[PGM_CTL] = RUNFAM[PGM]
PYM = ("cpython", "pymalloc-repros")
PYM_CTL = ("cpython", "pymalloc-repros", "controls")
PYM_MAIN = REPO / "capstone/ports/cpython/pymalloc/src/native/main.c"
RUNFAM[PYM] = dict(probe=r"^read_probe$", done="PYM completed=1")
RUNFAM[PYM_CTL] = RUNFAM[PYM]
MRB = ("mruby", "release-differential")
MRB_SMOKE = REPO / "capstone/ports/mruby/app/scripts/smoke.rb"
RUNFAM[MRB] = dict(probe=r"(?!)", done='["PASS"]')


def mrb_expect(oracle):
    """The corpus's recorded bounds-arm reading (case.json arms.sysalloc-bounds.oracle) as this tool's outcome."""
    if oracle.startswith("capability fault"):
        return "CAUGHT"
    if "reports PASS" in oracle:
        return "COMPLETED"
    if "wrong answer" in oracle:
        return "WRONG-ANSWER"
    return None


def pym_fixture(n):
    """The virtual runner's own fixture for case n (runners/virtual/run-virtual.py), so both platforms read one file."""
    spec = importlib.util.spec_from_file_location(
        "pym_virtual", REPO / "capstone/bug-corpora/cpython/pymalloc-repros/runners/virtual/run-virtual.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fixture(n)
# Corpora built as two images per case: the case and its negative control, which must complete and gates the arm.
TWO_IMAGE = {
    PGC: dict(kind="pg-c", neg="-DPGCLIENT_NEGATIVE_CONTROL", argv=lambda n: [str(n)], extra_glob="upstream_*.c",
              virtual={"silicon-level0": "spatial", "silicon-lifetimes": "virtual-malloc"}),
    PGM: dict(kind="pg-mmgr", neg="-DPG_NEGATIVE_CONTROL", argv=lambda n: ["0", str(n)], extra_glob=None,
              virtual={"silicon-pg-stock": "virtual-malloc", "silicon-pg-sublet": "virtual-pg-pools"}),
}


def sublet_arm(arm):
    return arm.endswith("-sublet")


# corpus controls that gate their arm like the four tool controls (pool-control predates this and stays a plain row,
# as set 2 was pre-registered with it)
CONTROL_ROWS = ("carve-control", "mc-control", "wmem-control", "pg-c-control", "pg-mmgr-control",
                "pg-mmgr-corpus-control", "pym-control", "mruby-control")
PROBE = re.compile(r"(^|_)(read|write)_probe(_u8|_u32)?$")
CTL_PROBE = re.compile(r"^ctl_(read|write)_probe$")


def kind_of(corpus):
    """The corpus kinds this tool scores: run-capstone-domain.py's, plus the AVBufferPool corpus and its control 90."""
    rel = corpus.relative_to(REPO / "capstone/bug-corpora").parts
    if rel in LIFETIME:
        return "lifetime-control"
    if rel in TWO_IMAGE:
        return TWO_IMAGE[rel]["kind"]
    if rel == PGM_CTL:
        return "pg-mmgr-corpus-control"
    if rel in (PYM, PYM_CTL):
        return "pym" if rel == PYM else "pym-control"
    if rel == MRB:
        return "mruby-rb"
    if rel[1] == "pool-repros":
        return "pool-control" if "controls" in rel else "pool"
    if rel[1] == "wmem-repros":
        return "wmem-control" if "controls" in rel else "wmem"
    if rel[1] == "allocator-repros":
        return "mc-control" if "controls" in rel else "mc"
    if rel[1] in ("carved-repros", "plane-repros"):
        return rel[1].split("-")[0]          # carved / plane: interior to one block, caught only by a carve arm
    return rcd.corpus_kind(corpus)


def predicted(kind, arm, case_dir=None):
    if kind in ("carved", "plane", "carve-control"):
        # a carve arm bounds every region at the carving code; 98 (the re-carve) exists only on the Sublet carve.
        # Every region here is below 4 KiB, which the encoder bounds exactly at any cursor (bounds-model.py)
        return "CAUGHT" if ARMS[arm].get("carve", {}).get(f"{kind if kind != 'carve-control' else 'carved'}-repros") \
            else "DEFECT-REPRODUCED"
    if kind == "mc-control":
        # control 90: slabs_free revokes the chunk under patch 0006, and keeps it in its page otherwise
        return "CAUGHT" if arm == "silicon-mc-sublet" else "DEFECT-REPRODUCED"
    if kind == "wmem-control":
        return WM_CONTROLS[arm][case_dir.name[:2]]
    if kind in ("mc", "wmem"):
        # the virtual arm with the same mechanism, as recorded in case.json; NO-READING there was its fixed run failing
        v = json.loads((case_dir / "case.json").read_text())["arms"][ARMS[arm]["virtual"]]["verdict"]
        return {"CAUGHT": "CAUGHT", "MISSED": "DEFECT-REPRODUCED"}.get(v, "FIXED-ARM-FAILED")
    if kind in ("pool", "pool-control"):
        # stock pools reissue an entry with no revocation (virtual-malloc MISSED; control 90 completes); patch 0003
        # revokes the user's child on release (virtual-nested-pools CAUGHT; control 90 faults)
        return "CAUGHT" if arm == "silicon-pool-sublet" else "DEFECT-REPRODUCED"
    if arm == "silicon-subobject" and kind == "interior" and case_dir is not None:
        # field bounds catch what QEMU's capstone-subobject arm caught, on the same compiler option
        v = json.loads((case_dir / "case.json").read_text()).get("arms", {}).get("capstone-subobject", {}).get("verdict")
        return "CAUGHT" if v == "CAUGHT" else "DEFECT-REPRODUCED"
    if kind == "spatial":
        return "CAUGHT"
    if kind == "temporal":
        return "CAUGHT" if arm == "silicon-lifetimes" else "DEFECT-REPRODUCED"
    return "DEFECT-REPRODUCED"


def argv_for(arm, kind, n, which):
    """The program's arguments for one run, or None when the row has no such run (wmem's controls: no fixed run)."""
    if kind == "wmem":
        if which == "fixed":
            return ["0", n, "fixed"]
        return ["0", n] + (["buggy"] if ARMS[arm].get("wm_differential") else [])
    if kind == "wmem-control":
        return None if which == "fixed" else ["0", n]
    return [which, n]


WM_DONE = "WM_DEFECT case={n} mode=0 completed"


def wm_probe(case_dir):
    """run-virtual-cases.py's rule: the probe function the case's own case.c calls (wm_probe or wm_write_probe), or
    None for a case that calls neither and declares fault_sites (judged against those)."""
    text = (case_dir / "case.c").read_text()
    reads, writes = "WM_READ" in text or "wm_probe(" in text, "WM_WRITE" in text or "wm_write_probe(" in text
    if not reads and not writes and json.loads((case_dir / "case.json").read_text()).get("fault_sites"):
        return None
    if reads == writes:
        sys.exit(f"CONTROL-FAILED {case_dir.name} calls {'both' if reads else 'no'} labelled probe")
    return "wm_probe" if reads else "wm_write_probe"


def classify_run(c, case_dir, text, result, syms):
    """run-capstone-domain.py's classify, plus wmem's protocol: its probe is the case's own function, and a buggy run
    of the plain sequence prints no VERDICT -- `WM_DEFECT case=N mode=0 completed` with exit 0 is its completion."""
    if c["kind"] not in ("wmem", "wmem-control"):
        return rcd.classify(text, result, syms, PROBE)
    probe = wm_probe(case_dir)
    outcome, detail = rcd.classify(text, result, syms, re.compile(rf"^{probe}$") if probe else re.compile(r"(?!)"))
    if outcome == "OTHER" and result.get("kind") == "exit" and result.get("value") == 0 \
            and WM_DONE.format(n=c["n"]) in text and not re.search(r"^VERDICT", text, re.M):
        return "DEFECT-REPRODUCED", f"completed: `{WM_DONE.format(n=c['n'])}`, exit 0, no fault"
    return outcome, detail


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_one(job):
    name, srcs, cflags, defs, out, data, arena = job
    env = dict(os.environ, B0_APP=name, B0_APP_SRCS=" ".join(srcs), B0_APP_CFLAGS=cflags, B0_GLUE_CARVE="pool",
               B0_CFLAGS_EXTRA=defs, OUT_DIR=str(out / "build" / name))
    if data:
        env["B0_DATA"] = str(data)
    if arena:
        env["B0_ARENA"] = str(arena)
    log = out / "build" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        rc = subprocess.run(["nice", "-n", "10", "bash", str(REPO / "capstone/runtime/silicon/build-b0-hello.sh")],
                            env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    img = out / "build" / name / f"{name}.dom"
    pass1 = out / "build" / name / "pass1.dom"
    if pass1.exists():
        pass1.unlink()      # 8 MB each; the image is what runs
    return name, rc, img if rc == 0 and img.is_file() else None


def cmd_build(a):
    arm = ARMS[a.arm]
    out = a.out / a.arm
    if out.exists():
        sys.exit(f"CONTROL-FAILED {out} exists: use a fresh output directory")
    out.mkdir(parents=True)
    ctl = out / "controls.c"
    ctl.write_text(rcd.CONTROL_C)
    appf = arm.get("app_cflags", "")
    jobs, cases = [("controls", [str(ctl)], f"-O0 {appf}".strip(), arm["defs"], out, None, None)], {}
    skipped_all = {}     # cases a corpus cannot run on this arm, with the reason (recorded in the manifest)
    if any(f"-D{sw}" in a.extra_cflags for sw in CARVE_SWITCHES):
        sys.exit("CONTROL-FAILED a carve switch comes from the arm (ARMS[...]['carve']), never from --extra-cflags")
    for corpus in a.corpus:
        corpus = corpus.resolve()
        kind = kind_of(corpus)
        shared = corpus / "shared"
        rel = corpus.relative_to(REPO / "capstone/bug-corpora").parts   # memcached-heap, ffmpeg-subobject, ffmpeg-pool
        tag = f"{rel[0]}-{'-'.join(rel[1:]).replace('-repros', '').replace('plain-', '')}"
        carve = arm.get("carve", {}).get(rel[1], "")
        flags = f"{a.opt} -I{shared} {carve} {arm.get('corpus_defs', {}).get(rel[1], '')} {a.extra_cflags} {appf}".split()
        driver = shared / arm.get("driver", "driver.c")
        if rel in TWO_IMAGE:
            ti = TWO_IMAGE[rel]
            ups = sorted(str(f) for f in shared.glob(ti["extra_glob"])) if ti["extra_glob"] else []
            v = ti["virtual"].get(a.arm)
            if v is None:
                sys.exit(f"CONTROL-FAILED {'/'.join(rel)} has no recorded arm to predict {a.arm} from")
            for d in sorted(x for x in corpus.glob("[0-9][0-9]_*") if (x / "case.c").is_file()):
                name, n = f"{tag}-{d.name[:2]}", int(d.name[:2])
                verdict = json.loads((d / "case.json").read_text())["arms"][v]["verdict"]
                cases[name] = dict(corpus=str(corpus.relative_to(REPO)), case=d.name, kind=ti["kind"], n=str(n),
                                   runs=[dict(variant="case", argv=ti["argv"](n),
                                              expect="CAUGHT" if verdict == "CAUGHT" else "COMPLETED")])
                jobs.append((name, [str(d / "case.c"), str(driver), *ups, *a.extra_src], " ".join(flags), arm["defs"],
                             out, a.data, a.arena))
                cases[f"{name}-ctl"] = dict(corpus=str(corpus.relative_to(REPO)), case=d.name,
                                            kind=f"{ti['kind']}-control", n=str(n),
                                            runs=[dict(variant="control", argv=ti["argv"](n), expect="COMPLETED")])
                jobs.append((f"{name}-ctl", [str(d / "case.c"), str(driver), *ups, *a.extra_src],
                             " ".join(flags + [ti["neg"]]), arm["defs"], out, a.data, a.arena))
            continue
        if rel in (PYM, PYM_CTL):   # one image per case around the port's replay; the fixture is staged with it
            v = arm.get("virtual")
            if v is None:
                sys.exit(f"CONTROL-FAILED {'/'.join(rel)} has no recorded arm to predict {a.arm} from")
            files = out / "files"
            files.mkdir(exist_ok=True)
            for d in sorted(x for x in corpus.glob("[0-9][0-9]_*") if (x / "case.c").is_file()):
                name, n = f"{tag}-{d.name[:2]}", int(d.name[:2])
                fx = files / f"pym-fixture-{n:02d}.bin"
                fx.write_bytes(pym_fixture(n))
                if rel == PYM:
                    verdict = json.loads((d / "case.json").read_text())["arms"][v]["verdict"]
                    expect = "CAUGHT" if verdict == "CAUGHT" else "COMPLETED"
                else:
                    expect = "CAUGHT" if sublet_arm(a.arm) else "COMPLETED"
                cases[name] = dict(corpus=str(corpus.relative_to(REPO)), case=d.name, kind=kind_of(corpus), n=str(n),
                                   files=[str(fx)],
                                   runs=[dict(variant="case" if rel == PYM else "control",
                                              argv=[f"@DIR@/{fx.name}", "/tmp/pym-report.bin"], expect=expect)])
                jobs.append((name, [str(PYM_MAIN), str(d / "case.c"), *a.extra_src], " ".join(flags), arm["defs"],
                             out, a.data, a.arena))
            continue
        if rel == MRB:          # one interpreter image, a script per case; nothing is compiled
            if not a.image:
                sys.exit("CONTROL-FAILED mruby/release-differential needs --image (the interpreter .dom)")
            img = Path(a.image).resolve()
            files = out / "files"
            files.mkdir(exist_ok=True)
            (files / "smoke.rb").write_bytes(MRB_SMOKE.read_bytes())
            cases["mruby-smoke"] = dict(corpus=str(corpus.relative_to(REPO)), case="ports/mruby/app/scripts/smoke.rb",
                                        kind="mruby-control", n="0", files=[str(files / "smoke.rb")],
                                        stage_image=img.name, image=str(img), sha256=sha256(img),
                                        runs=[dict(variant="control", argv=["@DIR@/smoke.rb", "/tmp"], expect="COMPLETED")])
            skipped = {}
            for d in sorted(x for x in corpus.glob("[0-9][0-9]_*") if (x / "case.json").is_file()):
                cj = json.loads((d / "case.json").read_text())
                trig = d / cj.get("trigger", "")
                oracle = cj["arms"].get("sysalloc-bounds", {}).get("oracle", "")
                why = ("a C-API case, which needs its own build" if trig.suffix != ".rb" else
                       "needs mruby-task, which the silicon build does not carry" if re.search(r"\bTask\b", trig.read_text()) else
                       "its recorded reading is the watchdog: it does not terminate" if "watchdog" in oracle else
                       f"no outcome for the recorded oracle {oracle!r}" if mrb_expect(oracle) is None else None)
                if why:
                    skipped[d.name] = why
                    continue
                name, n = f"{tag}-{d.name[:2]}", int(d.name[:2])
                fx = files / f"{d.name[:2]}-trigger.rb"
                fx.write_bytes(trig.read_bytes())
                cases[name] = dict(corpus=str(corpus.relative_to(REPO)), case=d.name, kind="mruby-rb", n=str(n),
                                   files=[str(fx)], stage_image=img.name, image=str(img), sha256=sha256(img),
                                   runs=[dict(variant="case", argv=[f"@DIR@/{fx.name}"], expect=mrb_expect(oracle))])
            for k, why in skipped.items():
                print(f"  skipped {k}: {why}")
            skipped_all.update(skipped)
            continue
        if rel == PGM_CTL:      # the corpus's own control: completes on stock, faults at the probe on Sublet
            for d in sorted(x for x in corpus.glob("[0-9][0-9]_*") if (x / "case.c").is_file()):
                name, n = f"{tag}-{d.name[:2]}", int(d.name[:2])
                cases[name] = dict(corpus=str(corpus.relative_to(REPO)), case=d.name, kind="pg-mmgr-corpus-control",
                                   n=str(n), runs=[dict(variant="control", argv=["0", str(n)],
                                                        expect="CAUGHT" if sublet_arm(a.arm) else "COMPLETED")])
                jobs.append((name, [str(d / "case.c"), str(driver), *a.extra_src], " ".join(flags), arm["defs"],
                             out, a.data, a.arena))
            continue
        fam = LIFETIME.get(rel)
        if fam:
            driver = (corpus / fam["driver"]).resolve()
            flags += [f"-I{corpus}"] + [f"-I{(corpus / i).resolve()}" for i in fam["inc"]]
            flags += fam["defs"](sublet_arm(a.arm)).split() if "defs" in fam else []
        for d in sorted(x for x in corpus.glob("[0-9][0-9]_*") if (x / "case.c").is_file()):
            name = f"{tag}-{d.name[:2]}"
            cases[name] = dict(corpus=str(corpus.relative_to(REPO)), case=d.name, kind=kind, n=str(int(d.name[:2])))
            if fam:
                n = int(d.name[:2])
                cases[name]["runs"] = [dict(variant=v, argv=fam["argv"](n, v),
                                            expect="CAUGHT" if v == "stale" and sublet_arm(a.arm) else "COMPLETED")
                                       for m, v in fam["runs"](sublet_arm(a.arm)) if m == n]
            jobs.append((name, [str(d / "case.c"), str(driver), *a.extra_src],
                         " ".join(flags), arm["defs"], out, a.data, a.arena))
        if kind == "carved":   # the corpus's own carve controls, built and run like a case
            ctls = [("99", "carve-control.c")] + ([("98", "carve-recarve-control.c")] if "FFC_SUBLET_CARVE" in carve else [])
            for n, src in ctls:
                name = f"{tag}-c{n}"
                cases[name] = dict(corpus=str(corpus.relative_to(REPO)), case=f"shared/{src}", kind="carve-control", n=n)
                jobs.append((name, [str(shared / src), str(driver), *a.extra_src],
                             " ".join(flags), arm["defs"], out, a.data, a.arena))
    built = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for name, rc, img in ex.map(build_one, jobs):
            print(f"  {name:<28} {'ok ' + sha256(img)[:16] if img else 'BUILD-FAILED rc=' + str(rc)}", flush=True)
            if img:
                built[name] = dict(image=str(img), sha256=sha256(img))
    for n, c in cases.items():          # cases that run a prebuilt image (mruby's interpreter) need no build
        if c.get("image") and n not in built:
            built[n] = dict(image=c["image"], sha256=c["sha256"])
    manifest = dict(arm=a.arm, defs=arm["defs"], app_cflags=appf, carve=arm.get("carve", {}),
                    corpus_defs=arm.get("corpus_defs", {}), driver=arm.get("driver", "driver.c"), extra_src=a.extra_src,
                    extra_cflags=a.extra_cflags, skipped=skipped_all,
                    controls=built.get("controls"), cases={
        n: dict(cases[n], **built[n]) if n in built else dict(cases[n], image=None) for n in cases})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    bad = [n for n in manifest["cases"] if not manifest["cases"][n]["image"]]
    print(f"built {len(cases) - len(bad)} of {len(cases)} cases; controls {'ok' if built.get('controls') else 'FAILED'}")
    return 75 if bad or not built.get("controls") else 0


def judge_run(outcome, detail):
    return outcome == "CAUGHT" and "(the labelled probe)" in detail


def score(a, runs):
    """runs: name -> (text, result) for controls ('control-<which>') and cases ('<case>-fixed' / '-buggy')."""
    out = a.out / a.arm
    m = json.loads((out / "manifest.json").read_text())
    llvm = Path(os.environ.get("CAPSTONE_LLVM_BIN", REPO / "llvm/cmake-build-debug/bin"))
    ctl_syms = rcd.Symbols(llvm, m["controls"]["image"])
    controls, ok = {}, True
    for which, want in ARMS[a.arm]["controls"].items():
        text, result = runs.get(f"control-{which}", ("", {}))
        outcome, detail = rcd.classify(text, result, ctl_syms, CTL_PROBE)
        if outcome == "OTHER" and f"CONTROL {which} RETURNED" in text and result.get("value") == 0:
            outcome, detail = "RETURNED", next(l for l in text.splitlines() if "RETURNED" in l)
        good = outcome == want and (outcome != "CAUGHT" or "(the labelled probe)" in detail)
        ok &= good
        controls[which] = dict(outcome=outcome, detail=detail, required=want, ok=good)
        print(f"  control {which:<6} {outcome:<10} required {want:<9} {'ok' if good else 'FAILED'}  {detail[:110]}")
    rows = []
    for name, c in m["cases"].items():
        if not c["image"]:
            rows.append(dict(name=name, outcome="BUILD-FAILED", as_predicted=False))
            continue
        if c.get("runs") is not None:       # a run-based row: a nested-lifetime control, or a c-repros case/control
            fam = RUNFAM[tuple(Path(c["corpus"]).relative_to("capstone/bug-corpora").parts)]
            syms = rcd.Symbols(llvm, c["image"])
            for r in c["runs"]:
                text, result = runs.get(f"{name}-{r['variant']}", ("", {}))
                if "probe_from" in fam:
                    m_fn = re.search(fam["probe_from"], text)
                    probe_re = re.compile(rf"^{re.escape(m_fn.group(1))}$" if m_fn else r"(?!)")
                else:
                    probe_re = re.compile(fam["probe"])
                outcome, detail = rcd.classify(text, result, syms, probe_re)
                if outcome == "CAUGHT" and "(the labelled probe)" not in detail and c["kind"] in ("pg-c", "pg-mmgr") \
                        and rcd.at_declared_site(REPO / c["corpus"] / c["case"], detail):
                    detail += " (a declared fault site)"
                if outcome != "CAUGHT" and result.get("kind") == "exit" and result.get("value") == 0 \
                        and fam["done"].format(n=int(c["n"]), v=r["variant"]) in text:
                    outcome, detail = "COMPLETED", f"completed: `{fam['done'].format(n=int(c['n']), v=r['variant'])}`, exit 0"
                if c["kind"] == "mruby-control" and outcome != "CAUGHT":
                    ok_smoke = result.get("kind") == "exit" and result.get("value") == 0 and "SMOKE_DONE" in text
                    outcome, detail = ("COMPLETED", "SMOKE_DONE, exit 0") if ok_smoke else (outcome, detail)
                if c["kind"] == "mruby-rb" and outcome not in ("CAUGHT", "COMPLETED") and result.get("kind") == "exit" \
                        and result.get("value") == 0 and re.search(r'^\["', text, re.M):
                    fails = [l for l in text.splitlines() if l.startswith('["')]
                    outcome, detail = "WRONG-ANSWER", f"exit 0, the harness's failures: {'; '.join(fails)[:200]}"
                if c["kind"] == "mruby-rb" and outcome == "CAUGHT" and rcd.at_declared_site(REPO / c["corpus"] / c["case"], detail):
                    detail += " (a declared fault site)"
                rows.append(dict(name=f"{name}-{r['variant']}", case=c["case"], corpus=c["corpus"], kind=c["kind"],
                                 outcome=outcome, detail=detail, predicted=r["expect"], image_sha256=c["sha256"],
                                 as_predicted=outcome == r["expect"] and (outcome != "CAUGHT" or "(the labelled probe)" in detail
                                                                          or "(a declared fault site)" in detail
                                                                          or c["kind"] == "mruby-rb")))
            continue
        syms = rcd.Symbols(llvm, c["image"])
        case_dir = REPO / c["corpus"] / c["case"]
        bt, br = runs.get(f"{name}-buggy", ("", {}))
        bo, bd = classify_run(c, case_dir, bt, br, syms)
        if argv_for(a.arm, c["kind"], c["n"], "fixed") is None:
            fixed_ok = True     # a control with no fixed run
        else:
            ft, fr = runs.get(f"{name}-fixed", ("", {}))
            fo, fd = classify_run(c, case_dir, ft, fr, syms)
            fixed_ok = fo == "FIXED" and fr.get("kind") == "exit" and fr.get("value") == 0
        outcome, detail = (bo, bd) if fixed_ok else ("FIXED-ARM-FAILED", f"fixed read {fo}: {fd}")
        site = outcome == "CAUGHT" and "(the labelled probe)" not in detail and c["kind"] not in CONTROL_ROWS \
            and rcd.at_declared_site(REPO / c["corpus"] / c["case"], detail)     # a control has no declared sites
        want = predicted(c["kind"], a.arm, REPO / c["corpus"] / c["case"])
        rows.append(dict(name=name, case=c["case"], corpus=c["corpus"], kind=c["kind"], outcome=outcome, detail=detail,
                         predicted=want, image_sha256=c["sha256"],
                         as_predicted=outcome == want and (outcome != "CAUGHT" or "(the labelled probe)" in detail or bool(site))))
    for r in rows:
        print(f"  {r['name']:<24} {r['outcome']:<18} {'as predicted' if r['as_predicted'] else 'DIFFERS (predicted ' + r.get('predicted', '?') + ')'}"
              f"  {r.get('detail', '')[:100]}")
    for r in rows:
        if r.get("kind") in CONTROL_ROWS and not r["as_predicted"]:
            ok = False      # the corpus's own carve control: the arm is not what it says
            controls[r["name"]] = dict(outcome=r["outcome"], detail=r.get("detail", ""), required=r["predicted"], ok=False)
    rows_cases = [r for r in rows if r.get("kind") not in CONTROL_ROWS]
    n_ok = sum(r["as_predicted"] for r in rows_cases)
    print(f"controls {'ok' if ok else 'FAILED'}; {n_ok} of {len(rows_cases)} rows as predicted")
    return controls, rows, ok


def cmd_qemu(a):
    out = a.out / a.arm
    m = json.loads((out / "manifest.json").read_text())
    vm = os.environ.get("CAPSTONE_VM", "/tmp/capstone/b0/tools/bin/capstone-vm")
    share = Path(a.share)
    runs = {}
    def run(img, argv, key):
        dst = share / Path(img).name
        if not dst.exists() or sha256(dst) != sha256(img):
            dst.write_bytes(Path(img).read_bytes())
        res = out / "qemu" / f"{key}.json"
        res.parent.mkdir(parents=True, exist_ok=True)
        if res.exists():
            res.unlink()
        p = subprocess.run([vm, "--state", str(a.state), "run", "-e", "CAPSTONE_EXEC_DIAGNOSTICS=1", "--result", str(res),
                            f"/mnt/host/{dst.name}", *argv], capture_output=True, text=True, timeout=600)
        text = p.stdout + p.stderr
        (out / "qemu" / f"{key}.out").write_text(text)
        result = json.loads(res.read_text()) if res.exists() and res.stat().st_size else {}
        runs[key] = (text, result)
    for which in ARMS[a.arm]["controls"]:
        run(m["controls"]["image"], [which], f"control-{which}")
    for name, c in m["cases"].items():
        for f in c.get("files", []):
            (share / Path(f).name).write_bytes(Path(f).read_bytes())
        if c["image"] and c.get("runs") is not None:
            for r in c["runs"]:
                run(c["image"], [x.replace("@DIR@", "/mnt/host") for x in r["argv"]], f"{name}-{r['variant']}")
        elif c["image"]:
            for which in ("fixed", "buggy"):
                argv = argv_for(a.arm, c["kind"], c["n"], which)
                if argv is not None:
                    run(c["image"], argv, f"{name}-{which}")
    controls, rows, ok = score(a, runs)
    (out / "qemu-matrix.json").write_text(json.dumps(dict(controls=controls, rows=rows), indent=1) + "\n")
    return 0 if ok else 75


def cmd_board(a):
    """The board transcript: b0run.sh's corpus rung prints, per run, `CORPUS-RUN <key> BEGIN`, the program's stdout and
    stderr (capstone-exec's fault line included), and `CORPUS-RUN <key> END kind=<exit|signal> value=<n>`."""
    text = Path(a.transcript).read_text(errors="replace").replace("\r", "")
    runs = {}
    for mt in re.finditer(r"CORPUS-RUN (\S+) BEGIN\n(.*?)CORPUS-RUN \1 END kind=(\w+) value=(\d+)", text, re.S):
        key, body, kind, value = mt.group(1), mt.group(2), mt.group(3), int(mt.group(4))
        fault = rcd.FAULT.search(body)
        result = dict(kind=kind, value=value, fault=fault.group(0) if fault else "")
        runs[key] = (body, result)
    pre = f"{a.arm}:"
    if any(k.startswith(pre) for k in runs):          # a boot that carried several arms: this arm's keys only
        runs = {k[len(pre):]: v for k, v in runs.items() if k.startswith(pre)}
    if not runs:
        sys.exit(f"CONTROL-FAILED no CORPUS-RUN blocks in {a.transcript}")
    controls, rows, ok = score(a, runs)
    if a.prereg:
        # the pre-registered prediction, not the tool's rule, is what a board row is read against
        pre = {r["name"]: r for r in json.loads(Path(a.prereg).read_text())["rows"]}
        for r in rows:
            p = pre.get(r["name"], {}).get("predicted_board")
            if p is None:
                continue
            r["predicted_tool"], r["predicted"] = r.get("predicted"), p["outcome"]
            site = re.search(r" in ([A-Za-z_][\w.$]*)\+0x", r.get("detail", ""))
            r["as_predicted"] = r["outcome"] == p["outcome"] and (p["outcome"] != "CAUGHT" or
                                                                 (site and site.group(1)) == p.get("site"))
        n = [r for r in rows if r["outcome"] != "BUILD-FAILED"]
        print(f"against the pre-registration: {sum(r['as_predicted'] for r in n)} of {len(n)} rows by outcome and site")
    (a.out / a.arm / "board-matrix.json").write_text(json.dumps(dict(controls=controls, rows=rows), indent=1) + "\n")
    return 0 if ok else 75


def cmd_stage(a):
    """The board's staging for one arm: <out>/<arm>/stage/ holds every image under a short name and runs.txt, the list
    b0run.sh's corpus-<arm> rung reads (controls first, clean before the first expected fault, then the cases)."""
    out = a.out / a.arm
    m = json.loads((out / "manifest.json").read_text())
    st = out / "stage"
    if st.exists():
        sys.exit(f"CONTROL-FAILED {st} exists")
    st.mkdir()
    lines = []
    (st / "controls.dom").write_bytes(Path(m["controls"]["image"]).read_bytes())
    pre = f"{a.arm}:" if a.prefix_keys else ""   # several arms in one boot: keys must not collide
    for which in ARMS[a.arm]["controls"]:
        lines.append(f"{pre}control-{which} controls.dom {which}")
    for name, c in m["cases"].items():
        if not c["image"]:
            continue
        img_name = c.get("stage_image", f"{name}.dom")     # a shared image (mruby's interpreter) is staged once
        if not (st / img_name).exists():
            (st / img_name).write_bytes(Path(c["image"]).read_bytes())
        for f in c.get("files", []):
            if not a.board_dir:
                sys.exit(f"CONTROL-FAILED {name} stages files: pass --board-dir, the directory the stage is baked to")
            (st / Path(f).name).write_bytes(Path(f).read_bytes())
        if c.get("runs") is not None:
            for r in c["runs"]:
                argv = [x.replace("@DIR@", a.board_dir) if a.board_dir else x for x in r["argv"]]
                lines.append(f"{pre}{name}-{r['variant']} {img_name} {' '.join(argv)}")
            continue
        for which in ("fixed", "buggy"):
            argv = argv_for(a.arm, c["kind"], c["n"], which)
            if argv is not None:
                lines.append(f"{pre}{name}-{which} {name}.dom {' '.join(argv)}")
    (st / "runs.txt").write_text("\n".join(lines) + "\n")
    sums = "".join(f"{sha256(f)}  {f.name}\n" for f in sorted(p for p in st.iterdir() if p.suffix in (".dom", ".bin")))
    (st / "SHA256SUMS").write_text(sums)
    print(f"staged {len(list(st.glob('*.dom')))} images, {len(lines)} runs, {sum(f.stat().st_size for f in st.glob('*.dom'))} bytes")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--arm", choices=ARMS, required=True)
    b.add_argument("--out", type=Path, required=True); b.add_argument("--jobs", type=int, default=6)
    b.add_argument("--extra-src", action="append", default=[], help="a source every case links (subobject: the pool core)")
    b.add_argument("--extra-cflags", default="", help="flags every case's sources get")
    b.add_argument("--data", type=int, default=None, help="B0_DATA for the cases (the pool core's arenas need more)")
    b.add_argument("--arena", type=int, default=None, help="B0_ARENA (level0's heap) for the cases")
    b.add_argument("--opt", default="-O0", help="the cases' optimisation level (the pool corpus builds at -O1)")
    b.add_argument("--image", help="a prebuilt image every case runs (mruby/release-differential: the interpreter)")
    b.add_argument("corpus", type=Path, nargs="+")
    q = sub.add_parser("qemu"); q.add_argument("--arm", choices=ARMS, required=True)
    q.add_argument("--out", type=Path, required=True); q.add_argument("--state", type=Path, required=True)
    q.add_argument("--share", required=True)
    s = sub.add_parser("stage"); s.add_argument("--arm", choices=ARMS, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--prefix-keys", action="store_true", help="prefix every run key with the arm (several arms per boot)")
    s.add_argument("--board-dir", default="", help="where the stage is baked (make-bake-multi.py: "
                   "/test-domains/corpus/<set>-<label>); resolves @DIR@ in a run's argv")
    r = sub.add_parser("board"); r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--out", type=Path, required=True); r.add_argument("--transcript", required=True)
    r.add_argument("--prereg", help="the arm's committed qemu-and-predictions.json: rows are read against it")
    a = ap.parse_args()
    return {"build": cmd_build, "qemu": cmd_qemu, "stage": cmd_stage, "board": cmd_board}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
