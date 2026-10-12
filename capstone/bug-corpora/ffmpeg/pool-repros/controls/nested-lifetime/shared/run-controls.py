#!/usr/bin/env python3
"""Run the nested-lifetime controls (91 parent, 92 sibling, 93 successor + stale) of
ffmpeg/pool-repros on the virtual Capstone profile, one arm per boot.

    run-controls.py boot --arm protected|release --build BUILD --sdk SDK --kit KIT --llvm-bin BIN
                         --raw DIR --out BOOT.json
    run-controls.py combine --manifest SNAPSHOT.json --out results.json BOOT.json...

A trimmed copy of tools/run-virtual-cases.py's hosted mode (one boot, virtualvm.Batch), because
the controls are not defect cases: they have no fixed/buggy pair and no verdict line, and every
access carries its own CTL-ACCESS line. Pre-registration:
capstone/docs/plans/2026-10-11-nested-lifetime-controls.md.

One boot runs, in order:
  platform-bounds-malloc, platform-uaf-malloc  the virtual heap's own controls (postgres/c-repros
      shared/controls.c): a write one past a malloc'd object, a read after free. Both must FAULT on
      both arms, or a column of completions could be a dead instrument.
  91-run, 92-run, 93-run  the permitted accesses
  93-stale                the positive control: the read through A's old pointer

This script REPORTS what each run did, then `combine` compares it with the predictions written in
the pre-registration. A missing run, a missing line, or a build that is not the arm it claims is an
error (exit 75), never a pass.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
CONTROLS = HERE.parents[1]
REPO = HERE.parents[7]
TOOLS = REPO / "capstone/bug-corpora/tools"
sys.path.insert(0, str(TOOLS))
import virtualvm  # noqa: E402
import verdicts as v  # noqa: E402

PLATFORM_C = REPO / "capstone/bug-corpora/postgres/c-repros/shared/controls.c"
STEPS = [("91-run", "91", "run"), ("92-run", "92", "run"), ("93-run", "93", "run"), ("93-stale", "93", "stale")]
PLATFORM = ("bounds-malloc", "uaf-malloc")
ARMS = {"protected": "ON", "release": "OFF"}


def repo_rel(path):
    """A source path as recorded: relative to the repo root of whichever snapshot built it."""
    m = re.search(r"(capstone/.*|compiler-rt/.*)$", path)
    return m.group(1) if m else path


def addr2line(llvm_bin, image, link_pc):
    out = subprocess.run([str(Path(llvm_bin) / "llvm-addr2line"), "-e", str(image), hex(link_pc)],
                         capture_output=True, text=True).stdout.strip()
    path, _, line = out.rpartition(":")
    return f"{repo_rel(path)}:{line.split()[0]}" if path else out


def symbol_address(llvm_bin, image, name):
    out = subprocess.run([str(Path(llvm_bin) / "llvm-nm"), "--defined-only", str(image)],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == name:
            return int(parts[0], 16)
    return None


def cache_value(cache, key):
    m = re.search(rf"^{key}:\w+=(.*)$", cache, re.M)
    return m.group(1) if m else None


def observe(text, result, image, llvm_bin):
    """What one run did, as facts."""
    rec = {"header": next((l.strip() for l in text.splitlines() if l.startswith("case=")), None),
           "pointers": {}, "accesses": [], "checks": [], "exit": result, "fault": None,
           # The run's result lines: the control's own output and the launcher's fault line.
           "lines": [l.strip() for l in text.splitlines()
                     if l.startswith(("case=", "CTL", "CONTROL", "capstone-exec:"))]}
    for m in re.finditer(r"^CTL-PTR (\S+) cursor=(0x[0-9a-f]+) base=(0x[0-9a-f]+) end=(0x[0-9a-f]+)$", text, re.M):
        rec["pointers"][m.group(1)] = {"cursor": m.group(2), "base": m.group(3), "end": m.group(4)}
    for line in text.splitlines():
        m = re.match(r"^CTL-ACCESS id=(\S+) op=(\S+) ref=(.*) site=(\S+)$", line)
        if m:
            rec["accesses"].append({"id": m.group(1), "op": m.group(2), "ref": m.group(3),
                                    "call_site": m.group(4), "done": None})
            continue
        m = re.match(r"^CTL-ACCESS-DONE id=(\S+) (.*)$", line)
        if m:
            hit = [a for a in rec["accesses"] if a["id"] == m.group(1) and a["done"] is None]
            if hit:
                hit[-1]["done"] = m.group(2)
        if line.startswith("CTL-CHECK") or line.startswith("CONTROL-FAILED"):
            rec["checks"].append(line.strip())
    symbols = v.Symbols(llvm_bin, image)
    m = v.FAULT_LINE.search(result.get("fault") or "") or v.FAULT_LINE.search(text)
    if m:
        cause, pc, addr, code = int(m.group(1)), int(m.group(2), 16), m.group(3), int(m.group(4), 16)
        link_pc = pc - code + symbols.base
        fn, off = symbols.lookup(pc, code)
        rec["fault"] = {"cause": cause, "pc": hex(pc), "address": addr, "link_pc": hex(link_pc),
                        "symbol": fn, "offset": off, "source": addr2line(llvm_bin, image, link_pc),
                        "line": (result.get("fault") or "")[:200]}
    in_flight = [a for a in rec["accesses"] if a["done"] is None]
    if "[runner] TIMEOUT" in text or result.get("kind") == "none":
        rec["outcome"] = "no-reading: the step never ended"
    elif rec["fault"]:
        rec["outcome"] = "fault"
        rec["fault"]["in_flight_access"] = in_flight[-1]["id"] if in_flight else None
    elif result.get("kind") == "exit" and result.get("value") == 75:
        rec["outcome"] = "no-reading: CONTROL-FAILED"
    elif (result.get("kind") == "exit" and result.get("value") == 0 and "CTL-RESULT completed" in text
          and not in_flight):
        rec["outcome"] = "completed"
    else:
        rec["outcome"] = f"no-reading: {result}"
    # Where each probe's accesses live in the source: the probe function's own location.
    for a in rec["accesses"]:
        probe = "stale_read_probe" if a["id"] == "stale-read" else ("write_probe" if a["op"] == "write" else "read_probe")
        addr = symbol_address(llvm_bin, image, probe)
        a["probe"] = probe
        a["probe_function_at"] = addr2line(llvm_bin, image, addr) if addr is not None else None
    return rec


def boot(a):
    build = a.build.resolve()
    cache = (build / "work/port/CMakeCache.txt").read_text()
    sub = (cache_value(cache, "FFPOOL_SUBLET") or "OFF").upper()
    sub = "ON" if sub in ("ON", "1", "TRUE", "YES") else "OFF"
    if sub != ARMS[a.arm]:
        sys.exit(f"CONTROL-FAILED {build} has FFPOOL_SUBLET={sub}; arm {a.arm} needs {ARMS[a.arm]}")
    sdk = cache_value(cache, "CAPSTONE_SDK")
    if not sdk or Path(sdk).resolve() != a.sdk.resolve():
        sys.exit(f"CONTROL-FAILED {build} was configured with CAPSTONE_SDK={sdk}, not {a.sdk}")
    images = {n: build / f"bin/ctl-{n}" for n in ("91", "92", "93")}
    missing = [str(p) for p in images.values() if not p.is_file()]
    if missing:
        sys.exit(f"CONTROL-FAILED missing {missing}")
    raw = a.raw.resolve()
    (raw / "bin").mkdir(parents=True, exist_ok=False)
    plat = raw / "bin/platform-controls.dom"
    b = subprocess.run([str(a.sdk / "capstone-cc"), "-O0", str(PLATFORM_C), "-o", str(plat)],
                       capture_output=True, text=True)
    if b.returncode:
        sys.exit(f"CONTROL-FAILED platform controls build: {b.stderr[:300]}")
    batch = virtualvm.Batch(raw / "stage")
    batch.put(f"images/{plat.name}", plat)
    for c in PLATFORM:
        batch.step(f"platform-{c}", f"./capstone-vexec ./images/{plat.name} {c}")
    for n, img in images.items():
        batch.put(f"images/{img.name}", img)
    for name, n, mode in STEPS:
        batch.step(name, f"./capstone-vexec ./images/ctl-{n} {mode} {n}")
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))
    runs, serial, completed = batch.execute(a.kit, raw / "guest", timeout=600)
    wall = round(time.time() - t0, 1)
    rec = {"arm": a.arm, "boot_started_utc": started, "boot_wall_s": wall, "boot_completed": completed,
           "configuration": {"FFPOOL_SUBLET": sub, "CTL_PROTECTED": 1 if a.arm == "protected" else 0,
                             "CMAKE_BUILD_TYPE": cache_value(cache, "CMAKE_BUILD_TYPE"),
                             "control_cflags": "-std=gnu11 -O1 -g -DFF2_SYSTEM_MEMORY",
                             "pool": "av_buffer_pool_init(64, NULL): entries from av_buffer_alloc"},
           "images": {f"ctl-{n}": v.sha256(p) for n, p in images.items()},
           "library_sha256": v.sha256(build / "work/port/libffmpeg-pool.a"),
           "platform_controls_sha256": v.sha256(plat),
           "platform": virtualvm.platform(a.kit, a.llvm_bin / "clang", HERE),
           "sdk": {"capstone-cc": v.sha256(a.sdk / "capstone-cc"),
                   "libapplication-runtime.a": v.sha256(a.sdk / "libapplication-runtime.a"),
                   "libcapstone-application-builtins.a": v.sha256(a.sdk / "libcapstone-application-builtins.a"),
                   "libc": v.sha256(json.loads((a.sdk / "sdk.json").read_text())["libc"])},
           "platform_controls": {}, "runs": {}}
    for c in PLATFORM:
        text, result = runs.get(f"platform-{c}", ("", {"kind": "none"}))
        seen = ("fault" if f"CONTROL {c} mark" in text and v.FAULT_LINE.search((result.get("fault") or "") + text)
                else "complete" if f"CONTROL {c} RETURNED" in text else "none")
        rec["platform_controls"][c] = {"observed": seen, "evidence": (result.get("fault") or "")[:160]}
    for name, n, mode in STEPS:
        if name not in runs:
            rec["runs"][name] = {"outcome": "no-reading: the step did not run"}
            continue
        text, result = runs[name]
        r = observe(text, result, images[n], a.llvm_bin)
        r["argv"] = [mode, n]
        r["image"] = f"ctl-{n}"
        want = f"case={n} control="
        if not (r["header"] or "").startswith(want) or f"arm={a.arm} mode={mode}" not in (r["header"] or ""):
            r["outcome"] = f"no-reading: header {r['header']!r} is not control {n} arm={a.arm} mode={mode}"
        rec["runs"][name] = r
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2) + "\n")
    for c, p in rec["platform_controls"].items():
        print(f"  {a.arm:<9} platform {c:<14} {p['observed']}")
    for name, r in rec["runs"].items():
        f = r.get("fault") or {}
        extra = f" cause={f.get('cause')} pc={f.get('pc')} {f.get('symbol')} {f.get('source')}" if f else ""
        print(f"  {a.arm:<9} {name:<9} {r['outcome']}{extra}")
    print(f"  boot {'completed' if completed else 'DID NOT COMPLETE'} in {wall}s; serial {serial}")
    return 0 if completed else 75


def access(run, aid):
    return next((x for x in run.get("accesses", []) if x["id"] == aid), None)


def predictions(rec):
    """The pre-registration's predictions for one boot, each as (claim, held, evidence)."""
    arm, runs, out = rec["arm"], rec["runs"], []
    for c in PLATFORM:
        p = rec["platform_controls"][c]
        out.append((f"platform control {c} faults (the instrument is live)", p["observed"] == "fault", p["evidence"]))
    r = runs["91-run"]
    ids = ["parent-read", "parent-write", "parent-readback"] + (
        ["lifetime-read", "lifetime-write", "lifetime-readback"] if arm == "protected" else [])
    ok = r["outcome"] == "completed" and all(access(r, i) and access(r, i)["done"] for i in ids)
    pr = access(r, "parent-read") or {}
    out.append(("control 1 (parent) completes; the read returns what the pool left there (A's 0x5A)",
                ok and "mismatches=0" in (pr.get("done") or ""), f"{r['outcome']}; parent-read {pr.get('done')}"))
    r = runs["92-run"]
    sr = access(r, "sibling-read") or {}
    out.append(("control 2 (sibling) completes, B intact",
                r["outcome"] == "completed" and "mismatches=0" in (sr.get("done") or ""),
                f"{r['outcome']}; sibling-read {sr.get('done')}"))
    r = runs["93-run"]
    eq = any("successor-address" in c and " equal" in c for c in r.get("checks", []))
    out.append(("control 3: C's start address equals A's; access through C completes",
                r["outcome"] == "completed" and eq, f"{r['outcome']}; {[c for c in r.get('checks', []) if 'address' in c]}"))
    r = runs["93-stale"]
    eq = any("successor-address" in c and " equal" in c for c in r.get("checks", []))
    if arm == "protected":
        f = r.get("fault") or {}
        held = (r["outcome"] == "fault" and f.get("cause") == 25 and f.get("symbol") == "stale_read_probe"
                and f.get("in_flight_access") == "stale-read" and eq)
        out.append(("stale read through A: rejected with cause 25 at the labelled probe", held,
                    f"{r['outcome']} cause={f.get('cause')} pc={f.get('pc')} in {f.get('symbol')} at {f.get('source')}"))
    else:
        st = access(r, "stale-read") or {}
        held = r["outcome"] == "completed" and "mismatches=0" in (st.get("done") or "") and eq
        out.append(("stale read through A: completes and reads C's data (0x77)", held,
                    f"{r['outcome']}; stale-read {st.get('done')}"))
    return out


def combine(a):
    boots = [json.loads(Path(p).read_text()) for p in a.boots]
    result = {"pre_registration": "capstone/docs/plans/2026-10-11-nested-lifetime-controls.md",
              "application": "FFmpeg 9.0.1",
              "nested_allocator": "AVBufferPool entries: av_buffer_pool_get; released by av_buffer_unref, "
                                  "which returns the entry to its pool (pool_release_buffer)",
              "protected_configuration": "buffer-pool port patch 0003 on 0001 (FFPOOL_SUBLET=ON)",
              "release_configuration": "the same source without 0003 (FFPOOL_SUBLET=OFF)",
              "revision": json.loads(Path(a.manifest).read_text()),
              "boots": boots, "predictions": []}
    all_held = True
    for b in boots:
        for claim, held, ev in predictions(b):
            result["predictions"].append({"arm": b["arm"], "boot_started_utc": b["boot_started_utc"],
                                          "claim": claim, "held": held, "evidence": ev})
            all_held &= held
    result["all_predictions_held"] = all_held
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    for p in result["predictions"]:
        print(f"{p['arm']:<9} {'HELD' if p['held'] else 'NOT HELD':<9} {p['claim']}  [{p['evidence'][:110]}]")
    print(f"all predictions held: {all_held}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    b = s.add_parser("boot")
    b.add_argument("--arm", choices=sorted(ARMS), required=True)
    b.add_argument("--build", type=Path, required=True)
    b.add_argument("--sdk", type=Path, required=True)
    b.add_argument("--kit", type=Path, required=True)
    b.add_argument("--llvm-bin", type=Path, required=True)
    b.add_argument("--raw", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    c = s.add_parser("combine")
    c.add_argument("--manifest", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("boots", nargs="+")
    a = p.parse_args()
    return boot(a) if a.cmd == "boot" else combine(a)


if __name__ == "__main__":
    sys.exit(main())
