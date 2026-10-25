#!/usr/bin/env python3
"""Run xlang/repro and xlang/lua-cdp's C shims on the virtual Capstone profile, all in one boot.

    run-virtual.py --arm virtual-malloc --virtual-kit <kit> --sdk <virtual SDK> --llvm-bin <dir with llvm-nm>
                   (or --board-kit <board kit>: the same steps in one vp2 board boot, tools/boardvm.py)
                   [--out-repro DIR] [--out-lua-cdp DIR] [--raw DIR]

This runner REPORTS; capstone/bug-corpora/tools/verdicts.py decides. Its shape is
sqlite/capi-repros/shared/run-virtual.py's: the kit group, the C-81 gate, the configuration's malloc
controls and every row in one boot, one verdicts-v1 bundle per corpus.

THE PROJECT LEAD'S DECISIONS, relayed by the runtime lane on 2026-10-22:
  (a) attribution is the malloc controls plus a per-row declared fault site, with no test-only
      no-revoke runtime (the classic column's XLANG_NO_REVOKE control has no virtual analogue);
  (b) the mruby shim rows are dropped, because the real-interpreter arms cover mruby. They stay in
      the bundle as out-of-denominator rows, so the corpus's total stays visible.

THE BUILD: each shim unchanged, `capstone-cc -O0 -g`, with its mock (repro: cheri/mock-mruby/
mock_mruby.c; lua-cdp: lua-cdp/cheri/mock_report.c) and virtual-main.c through -Wl,--wrap=main.

ONE Observation per row:
  reached     `XLANG BEGIN`, printed by the main wrapper at main's ENTRY. A shim prints nothing
              before its stale access, so this is entry, not the access, and every reading says so
  completed   the process exited 0 (a shim prints `MOCK <row> use-after-free-survived` and returns 0
              when its stale access completed)
  fault       the launcher's domain fault line, the pc resolved from the row image's own symbols
  attribution `function` when that symbol is one of the row's fault_sites in sites.tsv (host ASan's
              frame #0, plus memcpy where ASan's line is an explicit memcpy call; none for the stack
              use-after-return row, which therefore can read MISSED or unattributed but never CAUGHT)

Exit: 0 judged, 75 nothing could be judged, 78 the C-81 SDK-builtins gate refused the SDK.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
XLANG = HERE.parents[1]
REPO = XLANG.parent
TOOLS = REPO / "capstone/bug-corpora/tools"
sys.path.insert(0, str(TOOLS))
import appvm  # noqa: E402
import boardvm  # noqa: E402
import virtualvm  # noqa: E402
import verdicts as v  # noqa: E402

CONTROLS_C = REPO / "capstone/bug-corpora/postgres/c-repros/shared/controls.c"
FLAGS = ["-O0", "-g"]
SETS = {
    "repro": {"table": XLANG / "capstone/rows.tsv", "key": "key", "shim": "row",
              "src": XLANG / "cheri/shims", "mock": XLANG / "cheri/mock-mruby/mock_mruby.c",
              "inc": [XLANG / "cheri/mock-mruby", XLANG / "cheri/shims"]},
    "lua-cdp": {"table": XLANG / "lua-cdp/capstone/rows.tsv", "key": "slug", "shim": "shim",
                "src": XLANG / "lua-cdp/shims", "mock": XLANG / "lua-cdp/cheri/mock_report.c",
                "inc": [XLANG / "lua-cdp/shims", XLANG / "lua-cdp/cheri"]},
}
DROPPED = "mruby_"   # decision (b): every repro row whose shim is an mruby shim
PREDICTED = {"virtual-malloc": "every kept temporal row CAUGHT at its declared site; the stack use-after-return row MISSED"}


def table(path):
    lines = [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    head = lines[0].split("\t")
    return [dict(zip(head, l.split("\t"))) for l in lines[1:]]


def sites():
    out = {}
    for r in table(HERE.with_name("sites.tsv")):
        out[(r["set"], r["row"])] = set(filter(None, r["fault_sites"].split(",")))
    return out


def rows():
    """[(set, case name, shim stem, dropped)] in table order."""
    out = []
    for st, d in SETS.items():
        for r in table(d["table"]):
            shim = r[d["shim"]]
            out.append((st, r[d["key"]], shim, st == "repro" and shim.startswith(DROPPED)))
    return out


def observe(name, sites_, text, result, symbols, rec):
    o = v.Observation(case=name, arm=rec)
    if "[runner] TIMEOUT" in text:
        o.infra, o.notes = "infra", "runner timeout"
        return o
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    o.reached = "XLANG BEGIN" in text
    o.reach_evidence = "main's entry (XLANG BEGIN, the --wrap=main shim); no marker at the access" if o.reached else ""
    o.fault = fault
    o.completed = result.get("kind") == "exit" and result.get("value") == 0 and not fault
    if not o.reached:
        o.notes = "no XLANG BEGIN: main never ran"
        if not fault:
            o.infra = "infra"
        return o
    if fault:
        if fault.symbol and fault.symbol in sites_:
            o.attribution = "function"
            o.attribution_evidence = f"{fault.symbol} is a declared site of the row (sites.tsv, from host ASan)"
        else:
            o.attribution_evidence = f"in {fault.symbol or 'no known function'}; declared {sorted(sites_) or 'none'}"
    elif not o.completed:
        o.notes = f"returned without completing: {result.get('kind')} {result.get('value')}"
    else:
        o.notes = next((l.strip() for l in text.splitlines() if l.startswith("MOCK ")), "no MOCK line")
    return o


def control_seen(name, text, result):
    if f"CONTROL {name} mark" not in text:
        return "none"
    if v.domain_fault(result.get("fault") or "") or v.domain_fault(text):
        return "fault"
    return "complete" if f"CONTROL {name} RETURNED" in text else "none"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=sorted(PREDICTED))
    p.add_argument("--arm-name", help="the arm the bundles record (default: --arm; with --board-kit, vp2-<arm>)")
    kits = p.add_mutually_exclusive_group(required=True)
    kits.add_argument("--virtual-kit", type=Path, help="the QEMU kit (virtualvm)")
    kits.add_argument("--board-kit", type=Path, help="a vp2 board kit (boardvm): the same steps in one board boot")
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--llvm-bin", type=v.llvm_bin_dir, required=True)
    p.add_argument("--out-repro", type=Path)
    p.add_argument("--out-lua-cdp", type=Path)
    p.add_argument("--raw", type=Path)
    a = p.parse_args()
    gate = subprocess.run([sys.executable, str(TOOLS / "check-sdk-builtins.py"), str(a.sdk)])
    if gate.returncode:
        print(f"CONTROL-FAILED {a.sdk}: SDK-BUILTINS gate exit {gate.returncode}", file=sys.stderr)
        return 78
    rec = a.arm_name or boardvm.record_arm(a.arm, a.board_kit)
    configs = {st: json.loads((XLANG / st / "corpus.json").read_text())["arm_configurations"][a.arm] for st in SETS}
    if len(set(configs.values())) != 1:
        print(f"CONTROL-FAILED the two corpora map {a.arm} to different configurations: {configs}", file=sys.stderr)
        return 75
    config = configs["repro"]
    arms = v.load_arms()
    if config not in arms or arms[config]["target"] != "capstone-virtual":
        print(f"CONTROL-FAILED arm {a.arm} maps to {config!r}; it must be a capstone-virtual configuration", file=sys.stderr)
        return 75
    spec = arms[config]
    heap, profile = appvm.sdk_identity(a.sdk)
    if profile != "virtual":
        print(f"CONTROL-FAILED {a.sdk} is a {profile} SDK (heap {heap}); this runner needs the virtual profile", file=sys.stderr)
        return 75
    if a.board_kit:
        kit, Batch, kit_files, kit_platform = a.board_kit, boardvm.BoardBatch, boardvm.FILES, boardvm.platform
    else:
        kit, Batch, kit_files, kit_platform = a.virtual_kit, virtualvm.Batch, virtualvm.FILES, virtualvm.platform
    missing = [f for f in kit_files if not (kit / f).is_file()]
    if missing:
        print(f"CONTROL-FAILED kit {kit} lacks {missing}", file=sys.stderr)
        return 75
    stamp = time.strftime("%Y%m%d-%H%M%S")
    sub = "vp2" if a.board_kit else "virtual"
    outs = {"repro": a.out_repro or XLANG / "repro/results" / f"{stamp}-{sub}" / rec,
            "lua-cdp": a.out_lua_cdp or XLANG / "lua-cdp/results" / f"{stamp}-{sub}" / rec}
    raw = a.raw or Path("/tmp/capstone/virtual-cases-raw") / f"{stamp}-xlang-{rec}"
    bindir = raw / "bin"
    bindir.mkdir(parents=True, exist_ok=False)
    cc = a.sdk / "capstone-cc"
    declared = sites()
    # Path-independent images (as sqlite/capi-repros): -g would record the build directory and the checkout.
    pm = [f"-ffile-prefix-map={bindir}=/xlang-build", f"-ffile-prefix-map={REPO}=/repo"]

    wrap = bindir / "virtual-main.o"
    b = subprocess.run([str(cc), "-c", *FLAGS, *pm, str(HERE.with_name("virtual-main.c")), "-o", str(wrap)], capture_output=True, text=True, cwd=bindir)
    if b.returncode:
        print(f"CONTROL-FAILED virtual-main.c: {b.stderr[-300:]}", file=sys.stderr)
        return 75
    images, errors = {}, {}
    for st, name, shim, dropped in rows():
        if dropped:
            continue
        d = SETS[st]
        img = bindir / f"{st}--{shim}.dom"
        b = subprocess.run([str(cc), *FLAGS, *pm, *[f"-I{i}" for i in d["inc"]], str(d["src"] / f"{shim}.c"), str(d["mock"]),
                            str(wrap), "-Wl,--wrap=main", "-o", str(img)], capture_output=True, text=True, cwd=bindir)
        if b.returncode:
            errors[(st, name)] = b.stderr[-300:]
        else:
            images[(st, name)] = img
    controls_img = bindir / "controls.dom"
    b = subprocess.run([str(cc), "-O0", *pm, str(CONTROLS_C), "-o", str(controls_img)], capture_output=True, text=True, cwd=bindir)
    if b.returncode:
        print(f"CONTROL-FAILED controls build: {b.stderr[:300]}", file=sys.stderr)
        return 75
    plan = [(f"control-{n}", controls_img, [n]) for n in spec["controls"]]
    plan += [(f"{st}--{name}", img, []) for (st, name), img in images.items()]

    # ---- one boot ----
    batch = Batch(raw / "stage")
    for step, image, argv in plan:
        batch.put(f"images/{image.name}", image)
        batch.step(step, f"./capstone-vexec ./images/{image.name} {' '.join(argv)}".rstrip())
    runs, serial, completed = batch.execute(kit, raw / "guest", timeout=300 + 60 * len(plan))
    print(f"  boot: {'completed' if completed else 'DID NOT COMPLETE'}; serial {serial}", flush=True)
    platform = kit_platform(kit, a.llvm_bin / "clang", HERE)
    if a.board_kit:
        boardvm.record_firmware(platform, raw / "guest")

    controls = []
    for n in spec["controls"]:
        if f"control-{n}" not in runs:
            controls.append(v.Control(n, "none", "the control did not run"))
            continue
        text, result = runs[f"control-{n}"]
        controls.append(v.Control(n, control_seen(n, text, result),
                                  (result.get("fault") or next(iter(text.strip().splitlines()[-1:]), ""))[:160]))
        print(f"  control {n:<14} {controls[-1].observed:<9} (expected {spec['controls'][n]})", flush=True)

    by_set = {st: [] for st in SETS}
    for st, name, shim, dropped in rows():
        if dropped:
            o = v.Observation(case=name, arm=rec, infra="out-of-denominator",
                              notes="an mruby shim, dropped by the project lead's decision (b) of 2026-10-22: the real-interpreter arms cover mruby")
        elif (st, name) in errors:
            o = v.Observation(case=name, arm=rec, infra="build-failed", notes=errors[(st, name)])
        elif f"{st}--{name}" not in runs:
            o = v.Observation(case=name, arm=rec, infra="infra", notes="the boot ended before this row ran")
        else:
            text, result = runs[f"{st}--{name}"]
            o = observe(name, declared.get((st, name), set()), text, result, v.Symbols(a.llvm_bin, images[(st, name)]), rec)
            o.image_sha256 = v.sha256(images[(st, name)])
        o.case, o.arm, o.controls = name, rec, list(controls)
        verdict = v.judge(o, spec)
        by_set[st].append((o, verdict))
        print(f"{st:<8} {name:<22} {shim[:34]:<34} {verdict[0]}{':' + verdict[1] if verdict[1] else ''}  {verdict[2][:80]}", flush=True)
    status = 0
    for st, out_rows in by_set.items():
        record = v.write_bundle(outs[st], f"xlang/{st}", rec, out_rows, {
            "configuration": config, "predicted": PREDICTED[a.arm],
            "decisions": "the project lead, relayed by the runtime lane, 2026-10-22: (a) malloc controls plus per-row "
                         "declared sites, no no-revoke runtime; (b) the mruby shim rows dropped",
            "build": {"flags": FLAGS, "prefix_maps": ["<bindir>=/xlang-build", "<repo>=/repo"], "mock": str(SETS[st]["mock"].relative_to(REPO)), "link": ["-Wl,--wrap=main", "xlang/virtual/virtual-main.c"],
                      "sites": "xlang/virtual/sites.tsv", "sites_sha256": v.sha256(HERE.with_name("sites.tsv")),
                      "sdk_identity": [heap, profile], "sdk_json_sha256": v.sha256(a.sdk / "sdk.json")},
            "platform": platform})
        print(f"--- xlang/{st} {rec} ({config}): {record['tally']}")
        if out_rows and all(r[1][0] == v.NO_READING for r in out_rows):
            status = 75
    return status


if __name__ == "__main__":
    sys.exit(main())
