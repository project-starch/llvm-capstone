#!/usr/bin/env python3
"""Run the SQLite C-API binding corpus on the virtual Capstone profile, all in one boot.

    run-virtual.py --arm virtual-malloc --virtual-kit <kit> --sdk <virtual SDK> --llvm-bin <dir with llvm-nm>
                   (or --board-kit <board kit>: the same steps in one vp2 board boot, tools/boardvm.py)
                   [--out DIR] [--raw DIR] [--only 1,2,16]

This runner REPORTS; tools/verdicts.py decides, as for every virtual corpus. It is
tools/run-virtual-cases.py's shape (the kit group, the C-81 gate, one boot, the configuration's
controls first) for a corpus whose cases are whole programs: each row's before.c against SQLite.

THE BUILD, the project lead's decision (a) of 2026-10-22, every input recorded in inputs.json:
  SQLite   3.53.3 (ports/sqlite/fetch-sqlite.sh, the host ASan arm's amalgamation), put through
           ports/sqlite/build-sqlite-capstone.sh's own 3.53.3 rewrite pass. The block is read out
           of that script at run time, so the six rewrites and their six asserts have one source;
           a script that no longer has that shape stops the build.
           Compiled -O1 -g -DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION (the host arm's flags)
           plus -DHAVE_MALLOC_H=1 -DHAVE_MALLOC_USABLE_SIZE=1. Without them sqlite3MemMalloc keeps
           an 8-byte size header, every SQLite object is only 8-aligned, and the first capability
           store faults with cause 6 (corpus-cell-probes, 2026-10-22). Lookaside is left at
           SQLite's default, as the host arm leaves it.
  a row    its before.c, -O0 -g, linked with that object and shared/virtual-main.c through
           -Wl,--wrap=main. A row with no before.c (row 15) is out of the denominator.

ONE Observation per row:
  reached     `CAPI BEGIN`, printed by the main wrapper at main's ENTRY. It is not a marker at the
              defective access, and every reading says so
  completed   the process exited 0, the row's own success path. A row that returns non-zero (its
              setup refused) is not completed and reads inconclusive, never MISSED
  fault       the launcher's domain fault line, the pc resolved from the row image's own symbols
  attribution `function` when that symbol is in the row's case.json fault_sites: the frame host
              ASan reported the access in (#0), or for the bad-free row the allocator's refusal

WHICH ROWS ARE SCORED, the lead's decision (c) of 2026-10-22, by the row's `oracle` file:
  heap-use-after-free, bad-free   judged by tools/verdicts.py (CAUGHT / MISSED / NO-READING)
  null-deref                      its own class, never a temporal catch: out-of-denominator, with
                                  the reading in notes and in inputs.json's `classes`
  none                            correct when it exits 0 (no false alarm): out-of-denominator,
                                  and a fault is recorded as a FALSE ALARM in notes and `classes`
The controls are the generic malloc controls (decision (b)): the configuration's, from
postgres/c-repros/shared/controls.c, in the same boot.

Exit: 0 judged, 75 nothing could be judged, 78 the C-81 SDK-builtins gate refused the SDK.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[1]
REPO = HERE.parents[5]
TOOLS = REPO / "capstone/bug-corpora/tools"
sys.path.insert(0, str(TOOLS))
import appvm  # noqa: E402
import boardvm  # noqa: E402
import virtualvm  # noqa: E402
import verdicts as v  # noqa: E402

CONTROLS_C = REPO / "capstone/bug-corpora/postgres/c-repros/shared/controls.c"
PORT_SCRIPT = REPO / "capstone/ports/sqlite/build-sqlite-capstone.sh"
FETCH = REPO / "capstone/ports/sqlite/fetch-sqlite.sh"
SQLITE_VERSION = "3.53.3"
SQLITE_FLAGS = ["-O1", "-g", "-DSQLITE_THREADSAFE=0", "-DSQLITE_OMIT_LOAD_EXTENSION",
                "-DHAVE_MALLOC_H=1", "-DHAVE_MALLOC_USABLE_SIZE=1"]
ROW_FLAGS = ["-O0", "-g"]
SCORED = {"heap-use-after-free", "bad-free"}
CLASSES = SCORED | {"null-deref", "none"}
PREDICTED = {"virtual-malloc": "heap-use-after-free and bad-free rows CAUGHT at their ASan frame"}


def rewrite_block():
    """build-sqlite-capstone.sh's 3.53.3 pass: from its `sed \\` to the `fi` that closes the else branch."""
    text = PORT_SCRIPT.read_text()
    start = text.index("\nsed \\\n")
    end = text.index("\nfi\n", start)
    block = text[start:end]
    if block.count("  -e '") != 6 or block.count("\ngrep -q ") != 6 \
            or '"$SQLITE_SRC_DIR/sqlite3.c" > "$PATCHED_SQLITE"' not in block:
        raise SystemExit(f"CONTROL-FAILED {PORT_SCRIPT}: the 3.53.3 rewrite pass no longer has six rewrites "
                         "and six asserts; refusing to guess at it")
    return block


def prefix_maps(bindir):
    """Path-independent images: the build directory and the checkout are mapped to fixed names, so the board's build
    of a row hashes equal to the QEMU reference's (without this, -g records both paths)."""
    return [f"-ffile-prefix-map={bindir}=/capi-build", f"-ffile-prefix-map={REPO}=/repo"]


def build(a, bindir):
    """(images by row name, build errors by row name, the build record). Everything into bindir."""
    cc = a.sdk / "capstone-cc"
    pm = prefix_maps(bindir)
    src = Path(subprocess.run(["bash", str(FETCH)], capture_output=True, text=True, check=True).stdout.split()[-1])
    if f'#define SQLITE_VERSION        "{SQLITE_VERSION}"' not in (src / "sqlite3.h").read_text():
        raise SystemExit(f"CONTROL-FAILED {src} is not SQLite {SQLITE_VERSION}")
    sq = bindir / "sqlite"
    sq.mkdir()
    (sq / "sqlite3.h").write_bytes((src / "sqlite3.h").read_bytes())
    block = rewrite_block()
    r = subprocess.run(["bash", "-euo", "pipefail", "-c", block], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "SQLITE_SRC_DIR": str(src), "PATCHED_SQLITE": str(sq / "sqlite3.c")})
    if r.returncode:
        raise SystemExit(f"CONTROL-FAILED the 3.53.3 rewrite pass failed: {r.stderr[:300]}")
    obj = bindir / "sqlite3.o"
    r = subprocess.run([str(cc), "-c", *SQLITE_FLAGS, *pm, f"-I{sq}", str(sq / "sqlite3.c"), "-o", str(obj)],
                       capture_output=True, text=True, cwd=bindir)
    if r.returncode:
        raise SystemExit(f"CONTROL-FAILED sqlite3.c on {a.sdk}: {r.stderr[-400:]}")
    wrap = bindir / "virtual-main.o"
    r = subprocess.run([str(cc), "-c", *ROW_FLAGS, *pm, str(HERE.with_name("virtual-main.c")), "-o", str(wrap)],
                       capture_output=True, text=True, cwd=bindir)
    if r.returncode:
        raise SystemExit(f"CONTROL-FAILED virtual-main.c: {r.stderr[-300:]}")
    images, errors = {}, {}
    for d in rows(a):
        if not (d / "before.c").is_file():
            continue
        img = bindir / f"{d.name}.dom"
        r = subprocess.run([str(cc), *ROW_FLAGS, *pm, f"-I{sq}", str(d / "before.c"), str(wrap), str(obj),
                            "-Wl,--wrap=main", "-o", str(img)], capture_output=True, text=True, cwd=bindir)
        if r.returncode:
            errors[d.name] = r.stderr[-300:]
        else:
            images[d.name] = img
    record = {"sqlite": SQLITE_VERSION, "amalgamation_sha256": v.sha256(src / "sqlite3.c"),
              "rewritten_sha256": v.sha256(sq / "sqlite3.c"), "rewrite_pass": str(PORT_SCRIPT.relative_to(REPO)),
              "rewrite_pass_sha256": v.sha256(PORT_SCRIPT), "sqlite_flags": SQLITE_FLAGS, "row_flags": ROW_FLAGS, "prefix_maps": ["<bindir>=/capi-build", "<repo>=/repo"],
              "sqlite3_o_sha256": v.sha256(obj), "link": ["-Wl,--wrap=main", "shared/virtual-main.c"],
              "sdk_identity": list(appvm.sdk_identity(a.sdk)), "sdk_json_sha256": v.sha256(a.sdk / "sdk.json")}
    return images, errors, record


def rows(a):
    found = sorted((d for d in CORPUS.glob("row[0-9]*") if d.is_dir()), key=lambda d: int(re.match(r"row(\d+)", d.name)[1]))
    if a.only:
        keep = set(a.only.split(","))
        found = [d for d in found if re.match(r"row(\d+)", d.name)[1] in keep]
    return found


def observe(name, oracle, sites, text, result, symbols, rec):
    """One Observation, facts only. The class rule (decision (c)) is applied here, not in the judge."""
    o = v.Observation(case=name, arm=rec, image_sha256=result.get("image_sha256"))
    if "[runner] TIMEOUT" in text:
        o.infra, o.notes = "infra", "runner timeout"
        return o, None
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    o.reached = "CAPI BEGIN" in text
    o.reach_evidence = "main's entry (CAPI BEGIN, the --wrap=main shim); no marker at the access" if o.reached else ""
    o.fault = fault
    exit0 = result.get("kind") == "exit" and result.get("value") == 0
    o.completed = exit0 and not fault
    seen = (f"fault cause={fault.cause} in {fault.symbol or 'an unsymbolized pc'}" if fault else
            f"exit {result.get('value')}" if result.get("kind") == "exit" else f"{result.get('kind')} {result.get('value')}")
    if oracle == "null-deref":
        o.infra = "out-of-denominator"
        o.notes = f"null-deref class (the lead's decision (c), 2026-10-22: not a temporal catch): {seen}"
        return o, ("null-deref", seen)
    if oracle == "none":
        o.infra = "out-of-denominator"
        verdict = "correct, no false alarm" if exit0 and not fault else f"FALSE ALARM: {seen}" if fault else f"not clean: {seen}"
        o.notes = f"none class (the lead's decision (c), 2026-10-22): {verdict}"
        return o, ("none", verdict)
    if not o.reached:
        o.notes = "no CAPI BEGIN: main never ran"
        if not fault:
            o.infra = "infra"
        return o, None
    if fault:
        if fault.symbol and fault.symbol in sites:
            o.attribution = "function"
            o.attribution_evidence = (f"{fault.symbol} is the row's declared site "
                                      + ("(the allocator refusing the free)" if oracle == "bad-free" else "(host ASan's frame #0)"))
        else:
            o.attribution_evidence = f"in {fault.symbol or 'no known function'}; declared {sorted(sites) or 'none'}"
    elif not exit0:
        o.notes = f"returned without completing: {seen}"
    return o, None


def control_seen(name, text, result):
    if f"CONTROL {name} mark" not in text:
        return "none"
    if v.domain_fault(result.get("fault") or "") or v.domain_fault(text):
        return "fault"
    return "complete" if f"CONTROL {name} RETURNED" in text else "none"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=sorted(PREDICTED))
    p.add_argument("--arm-name", help="the arm the bundle records (default: --arm; with --board-kit, vp2-<arm>)")
    kits = p.add_mutually_exclusive_group(required=True)
    kits.add_argument("--virtual-kit", type=Path, help="the QEMU kit (virtualvm)")
    kits.add_argument("--board-kit", type=Path, help="a vp2 board kit (boardvm): the same steps in one board boot")
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--llvm-bin", type=v.llvm_bin_dir, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--raw", type=Path)
    p.add_argument("--only", help="comma-separated row numbers")
    a = p.parse_args()
    # C-81: a builtins build can elide the allocations a case observes; refused before anything is built
    gate = subprocess.run([sys.executable, str(TOOLS / "check-sdk-builtins.py"), str(a.sdk)])
    if gate.returncode:
        print(f"CONTROL-FAILED {a.sdk}: SDK-BUILTINS gate exit {gate.returncode}", file=sys.stderr)
        return 78
    rec = a.arm_name or boardvm.record_arm(a.arm, a.board_kit)
    config = json.loads((CORPUS / "corpus.json").read_text())["arm_configurations"][a.arm]
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
    out = a.out or CORPUS / "results" / f"{stamp}-{'vp2' if a.board_kit else 'virtual'}" / rec
    raw = a.raw or Path("/tmp/capstone/virtual-cases-raw") / f"{stamp}-sqlite-capi-repros-{rec}"
    bindir = raw / "bin"
    bindir.mkdir(parents=True, exist_ok=False)

    images, errors, build_record = build(a, bindir)
    controls_img = bindir / "controls.dom"
    b = subprocess.run([str(a.sdk / "capstone-cc"), "-O0", *prefix_maps(bindir), str(CONTROLS_C), "-o", str(controls_img)],
                       capture_output=True, text=True, cwd=bindir)
    if b.returncode:
        print(f"CONTROL-FAILED controls build: {b.stderr[:300]}", file=sys.stderr)
        return 75
    plan = [(f"control-{n}", controls_img, [n]) for n in spec["controls"]]
    plan += [(name, img, []) for name, img in images.items()]

    # ---- one boot ----
    batch = Batch(raw / "stage")
    for name, image, argv in plan:
        batch.put(f"images/{image.name}", image)
        batch.step(name, f"./capstone-vexec ./images/{image.name} {' '.join(argv)}".rstrip())
    runs, serial, completed = batch.execute(kit, raw / "guest", timeout=300 + 60 * len(plan))
    print(f"  boot: {'completed' if completed else 'DID NOT COMPLETE'}; serial {serial}", flush=True)
    platform = kit_platform(kit, a.llvm_bin / "clang", HERE)
    if a.board_kit:
        boardvm.record_firmware(platform, raw / "guest")

    controls = []
    for name in spec["controls"]:
        if f"control-{name}" not in runs:
            controls.append(v.Control(name, "none", "the control did not run"))
            continue
        text, result = runs[f"control-{name}"]
        controls.append(v.Control(name, control_seen(name, text, result),
                                  (result.get("fault") or next(iter(text.strip().splitlines()[-1:]), ""))[:160]))
        print(f"  control {name:<14} {controls[-1].observed:<9} (expected {spec['controls'][name]})", flush=True)

    out_rows, classes = [], {}
    for d in rows(a):
        oracle = (d / "oracle").read_text().strip() if (d / "oracle").is_file() else ""
        case = json.loads((d / "case.json").read_text())
        sites = set(case.get("fault_sites") or ())
        cls = None
        if oracle not in CLASSES:
            o = v.Observation(case=d.name, arm=rec, infra="out-of-denominator", notes=f"oracle {oracle!r} is not a class this arm reads")
        elif not (d / "before.c").is_file():
            o = v.Observation(case=d.name, arm=rec, infra="out-of-denominator", notes="the row has no before.c program")
            cls = (oracle, "no program")
        elif d.name in errors:
            o = v.Observation(case=d.name, arm=rec, infra="build-failed", notes=errors[d.name])
        elif d.name not in runs:
            o = v.Observation(case=d.name, arm=rec, infra="infra", notes="the boot ended before this row ran")
        else:
            text, result = runs[d.name]
            o, cls = observe(d.name, oracle, sites, text, result, v.Symbols(a.llvm_bin, images[d.name]), rec)
            o.image_sha256 = v.sha256(images[d.name])
        o.case, o.arm, o.controls = d.name, rec, list(controls)
        if cls:
            classes.setdefault(cls[0], {})[d.name] = cls[1]
        verdict = v.judge(o, spec)
        out_rows.append((o, verdict))
        print(f"{d.name[:44]:<44} {oracle:<20} {verdict[0]}{':' + verdict[1] if verdict[1] else ''}  {verdict[2][:90]}",
              flush=True)
    record = v.write_bundle(out, "sqlite/capi-repros", rec, out_rows, {
        "configuration": config, "predicted": PREDICTED[a.arm], "classes": classes,
        "decisions": "the project lead, relayed by the runtime lane, 2026-10-22: (a) 3.53.3 + rewrites + usable size; "
                     "(b) generic malloc controls, no FIXED twin; (c) UAF cause 25 and the bad-free refusal CAUGHT, "
                     "null-deref its own class, none correct at exit 0",
        "build": build_record, "platform": platform})
    print(f"--- {rec} ({config}): {record['tally']}  classes {json.dumps(classes)}")
    return 75 if out_rows and all(r[1][0] == v.NO_READING for r in out_rows) else 0


if __name__ == "__main__":
    sys.exit(main())
