#!/usr/bin/env python3
"""Run the memcached lifetime controls (91 parent, 92 sibling, 93 successor) on one virtual arm, in one boot.

    run-controls.py --arm virtual-nested-pools|virtual-malloc --build OUT --virtual-kit KIT --sdk SDK
                    --llvm-bin DIR --raw DIR --out FILE.json [--revision TEXT]

OUT is build-controls.sh's output (OUT/bin/ctl-NN, OUT/work/port/CMakeCache.txt). The build's
CMakeCache must be the arm's configuration (MCP_SUBLET / MCP_STOCK_MALLOC as tools/arms.json's
virtual-memcached-sublet / virtual-memcached-stock) on SDK, or nothing runs.

One boot per arm, staged by tools/virtualvm.py (runtime/virtual/run-staged.py: one hart, exact
bounds). Steps, in order: the virtual heap's own controls from postgres/c-repros (uaf-malloc,
bounds-malloc: both must fault, showing the launcher reports faults in this boot), then each control
as its own process. The stale read of control 93 is its own run, last.

This runner REPORTS what each run did -- completed, or the fault's cause and pc resolved to the
image's function and source line -- and checks each result against the pre-registered prediction
(docs/plans/2026-10-11-nested-lifetime-controls.md). Exit 0 when every run produced a reading
(whatever it was), 75 when any did not (infrastructure: no boot, a step that never ended, a
control-failed setup, a wrong build or kit).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[6]
TOOLS = REPO / "capstone/bug-corpora/tools"
sys.path.insert(0, str(TOOLS))
import virtualvm  # noqa: E402
import verdicts as v  # noqa: E402

CONTROLS_C = REPO / "capstone/bug-corpora/postgres/c-repros/shared/controls.c"
CACHE = {"virtual-nested-pools": {"MCP_SUBLET": "ON", "MCP_STOCK_MALLOC": "OFF"},
         "virtual-malloc": {"MCP_SUBLET": "OFF", "MCP_STOCK_MALLOC": "ON"}}
CONFIG = {"virtual-nested-pools": "virtual-memcached-sublet", "virtual-malloc": "virtual-memcached-stock"}
# The registered platform (2026-10-11-protected-same-answers.md): sha256 prefixes.
PLATFORM = {"qemu": "af37cc32", "launcher": "3975314a", "module": "a1c6cb6b"}
RUNS = {
    "virtual-nested-pools": [("91", "page"), ("91", "generation"), ("92", "sibling"), ("93", "through-c"),
                             ("93", "stale")],
    "virtual-malloc": [("91", "page"), ("92", "sibling"), ("93", "through-c"), ("93", "stale")],
}
KEYED = re.compile(r"(\w[\w.-]*)=(\S+)")
PATH = re.compile(r"/\S*?/(capstone/bug-corpora/[^\s:]+)")


def repo_paths(line):
    """Absolute paths in a program's output, as repository paths (normalised; no host directory)."""
    return PATH.sub(lambda m: os.path.normpath(m.group(1)), line)


def read_probe_line():
    """shared/corpus.h:N of read_probe's load on this target (the non-CHERI `return *p;`)."""
    lines = (HERE.parents[2] / "shared/corpus.h").read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if "read_probe(const volatile unsigned char *p)" in l)
    at = next(i for i in range(start, len(lines)) if lines[i].strip() == "return *p;")
    return f"capstone/bug-corpora/memcached/allocator-repros/shared/corpus.h:{at + 1}"


def keyed(line):
    return dict(KEYED.findall(line))


def cache_says(cache_file, want, sdk):
    text = cache_file.read_text() if cache_file.is_file() else ""
    bad = []
    for key, value in want.items():
        m = re.search(rf"^{key}:BOOL=(\w+)$", text, re.M)
        got = "ON" if (m and m.group(1).upper() in ("ON", "1", "TRUE", "YES")) else "OFF"
        if got != value:
            bad.append(f"{key}={got}, the arm needs {value}")
    m = re.search(r"^CAPSTONE_SDK:\w+=(.+)$", text, re.M)
    if not m or Path(m.group(1)).resolve() != sdk.resolve():
        bad.append(f"CAPSTONE_SDK={m.group(1) if m else 'unset'}, not {sdk}")
    if not re.search(r"^PORT_PLATFORM:\w+=capstone-application$", text, re.M):
        bad.append("PORT_PLATFORM is not capstone-application")
    return "; ".join(bad)


def source_line(llvm_bin, image, link_pc):
    """file:line of link_pc in image, from its DWARF, or None."""
    for tool, argv in (("llvm-symbolizer", ["--obj", str(image), "--functions=none", hex(link_pc)]),
                       ("llvm-addr2line", ["-e", str(image), hex(link_pc)])):
        exe = Path(llvm_bin) / tool
        if not exe.is_file():
            continue
        out = subprocess.run([str(exe), *argv], capture_output=True, text=True).stdout.strip().splitlines()
        if out and not out[0].startswith("??"):
            loc = out[0]
            return repo_paths(re.sub(r"^.*?/source/", "<source>/", loc))
    return None


def parse_run(text, result, symbols, llvm_bin, image):
    """Facts of one control run, from its console text and its result."""
    r = {"exit": result, "accesses": [], "chunks": {}, "checks": [], "lines": []}
    open_access = None
    for line in text.splitlines():
        line = repo_paths(line.rstrip())
        if not line:
            continue
        r["lines"].append(line)
        k = keyed(line)
        if line.startswith("ACCESS-BEGIN "):
            open_access = {key: k.get(key) for key in ("name", "kind", "via", "addr", "site", "probe")}
            open_access["done"] = False
            if open_access["probe"] and open_access["probe"].startswith("read_probe@"):
                open_access["probe"] = "read_probe@" + read_probe_line()
            r["accesses"].append(open_access)
        elif line.startswith("ACCESS-DONE ") and open_access and k.get("name") == open_access["name"]:
            open_access["done"] = True
            if "value" in k:
                open_access["value"] = k["value"]
            open_access = None
        elif line.startswith("CHUNK "):
            r["chunks"][k["name"]] = k["addr"]
        elif line.startswith("CHUNK-CAP "):
            r["chunks"][k["name"] + "-cap"] = f"[{k['base']},{k['end']})"
        elif line.startswith("PAGE "):
            r["page"] = {"slab_list0": k.get("slab_list0"), "span": k.get("span"),
                         "second_page_chunk": k.get("second_page_chunk")}
        elif line.startswith("PAGE-CAP "):
            r.setdefault("page", {})["cap"] = f"[{k['base']},{k['end']}) cursor={k['cursor']}"
        elif line.startswith("GENERATION "):
            r["generation"] = f"[{k['base']},{k['end']}) tag={k['tag']}"
        elif line.startswith("CLASS "):
            r["class"] = {key: int(k[key]) for key in ("id", "chunk_size", "perslab", "sizeof_item")}
        elif line.startswith("SUCCESSOR "):
            r["successor"] = {"A": k["A"], "C": k["C"], "equal": k["equal"] == "1",
                              "allocations": int(k["allocations"])}
        elif line.startswith("STALE-READ "):
            r["stale_read"] = {"value": k["value"], "reads": k["reads"]}
        elif line.startswith("CHECK ") or line.startswith("CHECK-FAILED "):
            r["checks"].append(line)
        elif line.startswith("CONTROL-FAILED"):
            r["control_failed"] = line
    r["completed"] = (any(l.startswith("CTL-COMPLETED ") for l in r["lines"])
                      and result.get("kind") == "exit" and result.get("value") == 0)
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if fault:
        m = v.FAULT_LINE.search(result.get("fault") or "") or v.FAULT_LINE.search(text)
        link_pc = int(m.group(2), 16) - int(m.group(4), 16) + symbols.base
        r["fault"] = {"cause": fault.cause, "pc": hex(fault.pc), "link_pc": hex(link_pc),
                      "symbol": fault.symbol, "offset": fault.offset,
                      "source": source_line(llvm_bin, image, link_pc),
                      "line": m.group(0)}
        r["faulted_access"] = next((a for a in r["accesses"] if not a["done"]), None)
    return r


def predict(arm, control, variant, r):
    """(holds, what was predicted, what was seen) against the pre-registration."""
    if "control_failed" in r or r["exit"].get("kind") == "none":
        return None, "", "no reading: " + r.get("control_failed", "the step never ended")
    seen = ("completed" if r["completed"] else
            f"fault cause={r['fault']['cause']} pc={r['fault']['pc']} in {r['fault']['symbol']}"
            f" at {r['fault']['source']} during {(r.get('faulted_access') or {}).get('name')}"
            if "fault" in r else f"neither completed nor faulted: {r['exit']}")
    if control == "93" and variant == "stale":
        if arm == "virtual-nested-pools":
            fa = r.get("faulted_access") or {}
            ok = ("fault" in r and r["fault"]["cause"] == 25 and r["fault"]["symbol"] == "read_probe"
                  and fa.get("name") == "stale-A-read" and r.get("successor", {}).get("equal") is True)
            return ok, "C's address equals A's; stale read rejected, cause 25 at read_probe", seen
        ok = (r["completed"] and r.get("successor", {}).get("equal") is True
              and r.get("stale_read", {}).get("reads") == "C-data")
        return ok, "C's address equals A's; stale read completes and reads C's data (0xc3)", seen + (
            f"; read {r.get('stale_read', {}).get('value')} ({r.get('stale_read', {}).get('reads')})"
            if r.get("stale_read") else "")
    if control == "93":
        ok = r["completed"] and r.get("successor", {}).get("equal") is True
        return ok, "C's address equals A's; access through C completes", seen
    if control == "92":
        return r["completed"], "completes, B's contents intact", seen
    return r["completed"], "completes, the read returns what the allocator left there", seen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=sorted(RUNS))
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--virtual-kit", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--revision", default="unrecorded")
    a = p.parse_args()

    why = cache_says(a.build / "work/port/CMakeCache.txt", CACHE[a.arm], a.sdk)
    if why:
        print(f"CONTROL-FAILED {a.build} is not the {a.arm} build: {why}", file=sys.stderr)
        return 75
    missing = [f for f in virtualvm.FILES if not (a.virtual_kit / f).is_file()]
    if missing:
        print(f"CONTROL-FAILED kit {a.virtual_kit} lacks {missing}", file=sys.stderr)
        return 75
    platform = virtualvm.platform(a.virtual_kit, a.llvm_bin / "clang", HERE)
    wrong = {k: platform[k] for k, pre in PLATFORM.items() if not platform[k].startswith(pre)}
    if wrong:
        print(f"CONTROL-FAILED the kit is not the registered platform: {wrong}", file=sys.stderr)
        return 75

    bindir = a.raw / "bin"
    bindir.mkdir(parents=True, exist_ok=False)
    heap_controls = bindir / "controls.dom"
    b = subprocess.run([str(a.sdk / "capstone-cc"), "-O0", str(CONTROLS_C), "-o", str(heap_controls)],
                       capture_output=True, text=True)
    if b.returncode:
        print(f"CONTROL-FAILED heap controls build: {b.stderr[:300]}", file=sys.stderr)
        return 75
    images = {n: a.build / f"bin/ctl-{n}" for n in ("91", "92", "93")}
    absent = [str(p) for p in images.values() if not p.is_file()]
    if absent:
        print(f"CONTROL-FAILED the build lacks {absent}", file=sys.stderr)
        return 75

    batch = virtualvm.Batch(a.raw / "stage")
    batch.put("images/controls.dom", heap_controls)
    for n, img in images.items():
        batch.put(f"images/ctl-{n}", img)
    plan = [("heap-uaf-malloc", "./capstone-vexec ./images/controls.dom uaf-malloc"),
            ("heap-bounds-malloc", "./capstone-vexec ./images/controls.dom bounds-malloc")]
    plan += [(f"ctl-{n}-{var}", f"./capstone-vexec ./images/ctl-{n} {n} {var}") for n, var in RUNS[a.arm]]
    for name, cmd in plan:
        batch.step(name, cmd)
    started = time.time()
    runs, serial, completed = batch.execute(a.virtual_kit, a.raw / "guest", timeout=300 + 60 * len(plan))
    wall = round(time.time() - started, 1)
    print(f"  virtual boot: {'completed' if completed else 'DID NOT COMPLETE'} in {wall}s; serial {serial}")

    record = {"arm": a.arm, "configuration": CONFIG[a.arm], "revision": a.revision,
              "boot_completed": completed, "boot_wall_seconds": wall,
              "build": {"cache": {k: CACHE[a.arm][k] for k in CACHE[a.arm]},
                        "flags": "-std=c11 -O1 -g -DNDEBUG -DCHUNK_ALIGN_BYTES=16"
                                 + (" -DMC_CARVE_BOUNDS -DMC_CAPSTONE_SUBLET" if a.arm == "virtual-nested-pools" else ""),
                        "images": {f"ctl-{n}": v.sha256(img) for n, img in images.items()},
                        "heap_controls": v.sha256(heap_controls)},
              "platform": platform, "heap_controls": {}, "runs": []}
    for c in ("uaf-malloc", "bounds-malloc"):
        text, result = runs.get(f"heap-{c}", ("", {"kind": "none"}))
        f = v.domain_fault(result.get("fault") or "") or v.domain_fault(text)
        record["heap_controls"][c] = {"observed": "fault" if f else "no fault", "cause": f.cause if f else None,
                                      "evidence": (result.get("fault") or "")[:160]}
        print(f"  heap control {c:<14} {record['heap_controls'][c]['observed']} (expected fault)")
    no_reading = False
    for n, var in RUNS[a.arm]:
        name = f"ctl-{n}-{var}"
        if name not in runs:
            r = {"exit": {"kind": "none"}, "control_failed": "the step did not run", "lines": []}
        else:
            text, result = runs[name]
            r = parse_run(text, result, v.Symbols(a.llvm_bin, images[n]), a.llvm_bin, images[n])
        holds, predicted, seen = predict(a.arm, n, var, r)
        no_reading |= holds is None
        r.update({"control": n, "variant": var, "image": f"ctl-{n}", "image_sha256": v.sha256(images[n]),
                  "argv": [n, var], "predicted": predicted, "observed": seen,
                  "prediction": "no reading" if holds is None else ("holds" if holds else "FAILS")})
        record["runs"].append(r)
        print(f"  {name:<22} {r['prediction']:<10} {seen}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"record: {a.out}")
    return 75 if (no_reading or not completed) else 0


if __name__ == "__main__":
    sys.exit(main())
