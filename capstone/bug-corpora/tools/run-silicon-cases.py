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
}
CARVE_SWITCHES = ("FFC_SUBLET_CARVE", "FFC_CARVE_BOUNDS", "FFP_CARVE_BOUNDS", "MC_CARVE_BOUNDS", "MC_CAPSTONE_SUBLET",
                  "MCP_ADAPTER_BACKING")
# corpus controls that gate their arm like the four tool controls (pool-control predates this and stays a plain row,
# as set 2 was pre-registered with it)
CONTROL_ROWS = ("carve-control", "mc-control")
PROBE = re.compile(r"(^|_)(read|write)_probe(_u8|_u32)?$")
CTL_PROBE = re.compile(r"^ctl_(read|write)_probe$")


def kind_of(corpus):
    """The corpus kinds this tool scores: run-capstone-domain.py's, plus the AVBufferPool corpus and its control 90."""
    rel = corpus.relative_to(REPO / "capstone/bug-corpora").parts
    if rel[1] == "pool-repros":
        return "pool-control" if "controls" in rel else "pool"
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
    if kind == "mc":
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
        for d in sorted(x for x in corpus.glob("[0-9][0-9]_*") if (x / "case.c").is_file()):
            name = f"{tag}-{d.name[:2]}"
            cases[name] = dict(corpus=str(corpus.relative_to(REPO)), case=d.name, kind=kind, n=str(int(d.name[:2])))
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
    manifest = dict(arm=a.arm, defs=arm["defs"], app_cflags=appf, carve=arm.get("carve", {}),
                    corpus_defs=arm.get("corpus_defs", {}), driver=arm.get("driver", "driver.c"), extra_src=a.extra_src,
                    extra_cflags=a.extra_cflags,
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
        syms = rcd.Symbols(llvm, c["image"])
        ft, fr = runs.get(f"{name}-fixed", ("", {}))
        fo, fd = rcd.classify(ft, fr, syms, PROBE)
        bt, br = runs.get(f"{name}-buggy", ("", {}))
        bo, bd = rcd.classify(bt, br, syms, PROBE)
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
        if c["image"]:
            run(c["image"], ["fixed", c["n"]], f"{name}-fixed")
            run(c["image"], ["buggy", c["n"]], f"{name}-buggy")
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
        (st / f"{name}.dom").write_bytes(Path(c["image"]).read_bytes())
        lines.append(f"{pre}{name}-fixed {name}.dom fixed {c['n']}")
        lines.append(f"{pre}{name}-buggy {name}.dom buggy {c['n']}")
    (st / "runs.txt").write_text("\n".join(lines) + "\n")
    sums = "".join(f"{sha256(f)}  {f.name}\n" for f in sorted(st.glob("*.dom")))
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
    b.add_argument("corpus", type=Path, nargs="+")
    q = sub.add_parser("qemu"); q.add_argument("--arm", choices=ARMS, required=True)
    q.add_argument("--out", type=Path, required=True); q.add_argument("--state", type=Path, required=True)
    q.add_argument("--share", required=True)
    s = sub.add_parser("stage"); s.add_argument("--arm", choices=ARMS, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--prefix-keys", action="store_true", help="prefix every run key with the arm (several arms per boot)")
    r = sub.add_parser("board"); r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--out", type=Path, required=True); r.add_argument("--transcript", required=True)
    a = ap.parse_args()
    return {"build": cmd_build, "qemu": cmd_qemu, "stage": cmd_stage, "board": cmd_board}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
