#!/usr/bin/env python3
"""Run a make-stage.py stage in one boot, on the QEMU twin or on the board, and read it.

  run-stage.py run  --backend qemu|board --kit KIT --stage DIR --out OUT [--control DOM ARGS] [--step-s S]
  run-stage.py read --stage DIR --tree SPEC17 --text OUT/step.txt [--twin OTHER/result.json] [--out OUT]

`run` boots once through bug-corpora/tools/virtualvm.py (QEMU) or boardvm.py (the board's baked-rung recipe), with
the same stage and the same run.sh: an optional control step (a known-good image, which must exit 0), then the SPEC
step. It saves the SPEC step's text and reads it.

`read` takes the step's text and reports, per arm and command: rc, the uptime span, the launcher's CAPSTONE_VM_STATS
(peak pinned pages, nodes, collections) and the module's bench line (cycles, instret, entries inside the context).
It decodes every in-band output, runs runcpu's own host-side specdiff commands (plan.json) on them, compares every
output hash across the arms and, with --twin, against the twin's. No data is an error: a command without its END
line, an arm without its hashes, or a missing SPEC-DONE makes the exit status non-zero.
"""
import argparse
import base64
import gzip
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[2] / "bug-corpora" / "tools"


def run(a):
    sys.path.insert(0, str(TOOLS))
    import virtualvm  # noqa: E402
    import boardvm  # noqa: E402
    a.out.mkdir(parents=True, exist_ok=False)
    B = boardvm.BoardBatch if a.backend == "board" else virtualvm.Batch
    b = B(a.out / "batch")
    spec = b.put("spec", a.stage / "stage" if (a.stage / "stage").is_dir() else a.stage)
    root = spec.rsplit("/", 1)[0]
    if a.control:
        dom = Path(a.control[0])
        ctl = b.put("control/" + dom.name, dom)
        b.step("control", f"{root}/capstone-vexec {ctl} {' '.join(shlex.quote(x) for x in a.control[1:])}")
    inband = "" if a.backend == "board" else "SPEC_INBAND_MAX=67108864 "   # the twin carries every output
    b.step("spec", f"{inband}SPEC_VEXEC={root}/capstone-vexec sh {spec}/run.sh")
    b.bounds = {"spec": a.step_s / 20, "control": 60}
    runs, transcript, completed = b.execute(a.kit, a.out / "guest", timeout=a.step_s + 600)
    for name in ("control", "spec"):
        if name in runs:
            (a.out / f"{name}.txt").write_text(runs[name][0])
            (a.out / f"{name}.result.json").write_text(json.dumps(runs[name][1], sort_keys=True) + "\n")
    print(f"RUN backend {a.backend} kit {a.kit} completed={completed} transcript={transcript} "
          f"control={runs.get('control', (None, None))[1]} spec={runs.get('spec', (None, None))[1]}")
    if "spec" not in runs:
        print("RUN: NO SPEC STEP in the transcript")
        return 2
    rc = read(argparse.Namespace(stage=a.stage, tree=a.tree, text=a.out / "spec.txt", twin=a.twin, out=a.out))
    if a.control:
        # A boot whose known-good control fails is VOID (the board-run skill): it carries no verdict about the SPEC step.
        cres = runs.get("control", (None, {"kind": "none"}))[1]
        ok = cres.get("kind") == "exit" and cres.get("value") == 0 and "fault" not in cres
        print(f"RUN control {'PASSED' if ok else 'FAILED: the boot is VOID'} {json.dumps(cres, sort_keys=True)}")
        rc = rc if ok else 2
    return rc


def unsplice(t):
    """boardvm's heartbeat can land inside a line; every @@HB is removed with its newline (tier3-perfpilot.py)."""
    return re.sub(r"@@HB \d+ out=\S* tail=[A-Za-z0-9+/=]*\n", "", t)


def read(a):
    root = a.stage if (a.stage / "plan.json").exists() else a.stage.parent
    plan = json.loads((root / "plan.json").read_text())
    text = unsplice(Path(a.text).read_text(errors="replace"))
    out = Path(a.out or Path(a.text).parent)
    rc = 0
    res = dict(bench=plan["bench"], size=plan["size"], arms={}, done=bool(re.search(rf"^@@SPEC-DONE {re.escape(plan['bench'])} ", text, re.M)))
    begins = {m.group(1): m.end() for m in re.finditer(r"^@@SPEC-BEGIN (\d+) .*$", text, re.M)}
    ends = list(re.finditer(r"^@@SPEC-END (\d+) (\w+) (\S+) (\d+) rc=(\d+) up0=(\S+) up1=(\S+) nbench=(\d+) ?(.*?) :: (.*)$", text, re.M))
    prev_bench = None
    for m in ends:
        n, tag, arm, i, r, u0, u1, nb, stats, bench = m.groups()
        st = dict(re.findall(r"(\w+)=(\d+)", stats)) if "CAPSTONE_VM_STATS" in stats else {}
        bl = dict(re.findall(r"(\w+)=(-?\d+)", bench))
        # The bench line is the module's LAST in dmesg; a command that printed none would quote the previous one. The
        # count of bench lines cannot tell (the ring buffer drops old lines), so a quote equal to the previous is stale.
        fresh = tag == "gen" or (int(nb) > 0 and bench.strip() != prev_bench)
        if tag != "gen":
            prev_bench = bench.strip()
        e = dict(n=int(n), tag=tag, i=int(i), rc=int(r), seconds=round(float(u1) - float(u0), 2),
                 peak_pages=int(st["peak"]) if "peak" in st else None, nodes=int(st["nodes"]) if "nodes" in st else None,
                 collections=int(st["collections"]) if "collections" in st else None,
                 cycles=int(bl["cycles"]) if fresh and "cycles" in bl else None,
                 instret=int(bl["instret"]) if fresh and "instret" in bl else None,
                 entries=int(bl["entries"]) if fresh and "entries" in bl else None,
                 bench_line_fresh=fresh, body=text[begins.get(n, m.start()):m.start()].strip()[-400:])
        res["arms"].setdefault(arm, dict(commands=[], hashes={}, files={}))["commands"].append(e)
    for m in re.finditer(r"^@@SPEC-HASH-BEGIN (\S+)\n(.*?)^@@SPEC-HASH-END \1$", text, re.M | re.S):
        h = dict((p.split(None, 1)[1].strip().removeprefix("./"), p.split()[0]) for p in m.group(2).splitlines() if len(p.split()) == 2)
        res["arms"].setdefault(m.group(1), dict(commands=[], hashes={}, files={}))["hashes"] = h
    fdir = out / "files"
    for m in re.finditer(r"^@@SPEC-FILE (\S+) (\S+)\n(.*?)^@@SPEC-FILE-END$", text, re.M | re.S):
        arm, name, b64 = m.groups()
        data = gzip.decompress(base64.b64decode("".join(b64.split())))
        p = fdir / arm / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        res["arms"].setdefault(arm, dict(commands=[], hashes={}, files={}))["files"][name] = len(data)
    # Host-side validation: runcpu's specdiff commands, on each arm's decoded outputs.
    spec = Path(a.tree).resolve() if a.tree else None
    # specdiff is SPEC's perl and needs SPEC's environment (shrc sets SPEC, PATH and the perl library path).
    env = None
    if spec:
        e = subprocess.run(["bash", "-c", "set +u; source ./shrc > /dev/null && env -0"], cwd=spec, capture_output=True, check=True)
        env = dict(x.split("=", 1) for x in e.stdout.decode().split("\0") if "=" in x)
    for arm, d in res["arms"].items():
        d["validation"] = []
        for c in plan["host_validation"]:
            argv = [x.replace("ARM", arm) for x in c["argv"]]
            local = [x for x in argv if "/" not in x and (fdir / arm / x).exists()]
            if not spec:
                d["validation"].append(dict(cmd=argv[-1], verdict="NOT-RUN (no --tree)"))
                continue
            missing = [x for x in argv[2:] if "/" not in x and not x.startswith("-") and not re.match(r"^[0-9.e+-]+$", x)
                       and not (fdir / arm / x).exists()]
            if missing:
                d["validation"].append(dict(cmd=" ".join(argv[-2:]), verdict=f"NO-OUTPUT {missing}"))
                continue
            full = [str(spec / x[5:]) if x.startswith("SPEC/") else (str(fdir / arm / x) if x in local else x) for x in argv]
            p = subprocess.run(full, capture_output=True, text=True, env=env)
            ok = p.returncode == 0 and "specdiff run completed" in p.stdout
            d["validation"].append(dict(cmd=" ".join(argv[-2:]), verdict="PASS" if ok else "FAIL", rc=p.returncode,
                                        tail=(p.stdout + p.stderr)[-300:]))
    # Hashes: every arm against the first, and against the twin.
    arms = plan["arms"]
    outs = {x for x in plan["outputs"]}
    def outhash(arm):
        h = res["arms"].get(arm, {}).get("hashes", {})
        # .err files carry the launcher's stats line (launch and elapsed ns), which differs on every run.
        return {k.replace(arm, "ARM"): v for k, v in h.items()
                if (k.replace(arm, "ARM") in outs or k not in plan["inputs"]) and not k.endswith(".err")}
    base = outhash(arms[0])
    res["cross_arm"] = {arm: sorted(k for k in set(base) | set(outhash(arm)) if base.get(k) != outhash(arm).get(k)) for arm in arms[1:]}
    if a.twin:
        tw = json.loads(Path(a.twin).read_text())
        res["vs_twin"] = {}
        for arm in arms:
            th = {k.replace(arm, "ARM"): v for k, v in tw["arms"].get(arm, {}).get("hashes", {}).items()}
            mine = outhash(arm)
            th = {k: v for k, v in th.items() if not k.endswith(".err") and (k in outs or k not in plan["inputs"])}
            res["vs_twin"][arm] = sorted(k for k in set(mine) | set(th) if mine.get(k) != th.get(k))
    # Verdict lines.
    ncmd = len(plan["commands"]) + len(plan["target_validation"]) + len(plan["generation"])
    for arm in arms:
        d = res["arms"].get(arm)
        if not d:
            print(f"READ {plan['bench']} {plan['size']} {arm}: NO DATA"); rc = 2; continue
        cs = d["commands"]
        bad = [c for c in cs if c["rc"] != 0]
        vals = d.get("validation", [])
        print(f"READ {plan['bench']} {plan['size']} {arm}: {len(cs)}/{ncmd} commands ended, rc!=0: {len(bad)}, "
              f"hashes {len(d['hashes'])}, in-band {len(d['files'])}, specdiff "
              f"{sum(v['verdict'] == 'PASS' for v in vals)}/{len(vals)} PASS")
        for c in cs:
            print(f"   {c['tag']}{c['i']}: rc {c['rc']} {c['seconds']} s instret {c['instret']} cycles {c['cycles']} "
                  f"entries {c['entries']} peak_pages {c['peak_pages']} nodes {c['nodes']} collections {c['collections']}"
                  + ("" if c["bench_line_fresh"] else " (bench line NOT fresh)"))
        for v in vals:
            if v["verdict"] != "PASS":
                print(f"   specdiff {v['verdict']}: {v['cmd']} :: {v.get('tail', '')[-160:]!r}")
        if len(cs) != ncmd or bad or not d["hashes"] or any(v["verdict"] != "PASS" for v in vals):
            rc = max(rc, 1)
    for arm, diff in res.get("cross_arm", {}).items():
        print(f"READ outputs {arms[0]} vs {arm}: {'identical' if not diff else 'DIFFER ' + str(diff[:8])}")
        rc = max(rc, 1 if diff else 0)
    for arm, diff in res.get("vs_twin", {}).items():
        print(f"READ outputs {arm} vs twin: {'identical' if not diff else 'DIFFER ' + str(diff[:8])}")
    if not res["done"]:
        print("READ: SPEC-DONE ABSENT"); rc = 2
    if re.search(r"Oops|Unable to handle|BUG:|Out of memory|oom-kill", text):
        print("READ: KERNEL OOPS/OOM in the step's text"); rc = 2
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=["qemu", "board"], required=True)
    r.add_argument("--kit", type=Path, required=True)
    r.add_argument("--stage", type=Path, required=True, help="make-stage.py's --out")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--tree", type=Path, help="the SPEC17 directory, for specdiff")
    r.add_argument("--twin", type=Path, help="the twin's result.json")
    r.add_argument("--control", nargs="+", help="a known-good image and its arguments, run first")
    r.add_argument("--step-s", type=int, default=7200)
    d = sub.add_parser("read")
    d.add_argument("--stage", type=Path, required=True)
    d.add_argument("--tree", type=Path)
    d.add_argument("--text", type=Path, required=True)
    d.add_argument("--twin", type=Path)
    d.add_argument("--out", type=Path)
    a = ap.parse_args()
    sys.exit(run(a) if a.cmd == "run" else read(a))


if __name__ == "__main__":
    main()
