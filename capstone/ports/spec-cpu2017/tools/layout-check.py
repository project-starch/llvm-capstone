#!/usr/bin/env python3
"""Check that two arms' images of a benchmark have the same code layout outside the heap runtime.

  layout-check.py --rec REC --tree SPEC17 [--labels A B] [--bench B ...]
  layout-check.py --pair EXE_A EXE_B --sdk-a SDK --sdk-b SDK [--objects DIR ...]

--rec checks every executable pair of a tools/build.sh record (arms.json): same executable name, the two labels.
--pair checks one pair. SDK is the directory whose sdk.json the image was linked with (a layout SDK of
tools/layout-sdk.py or the arm's own).

Each image's symbols come from `llvm-nm --defined-only -S --format=sysv` (FUNC and OBJECT symbols; local labels and
NOTYPE symbols are skipped). A symbol is classed by the name:
- heap: defined by either SDK's runtime heap.c.obj. Not compared: its code is the arm. A heap name that another input
  defines too cannot be attributed from the image; it is excluded and listed as ambiguous.
- bench: defined by the benchmark's own objects (with --objects, or the record's build directories); without objects,
  every name that is neither heap nor SDK.
- sdk: defined by the runtime's other members, the libc archive or the builtins archive.
For every compared name the two images must hold the same sorted list of (address, size), and every allocated
output section must have the same name, address and size in both images. In a layout-controlled image (one with
tools/layout-sdk.py's region symbols) every heap.c symbol must also lie inside a heap region.

Exit status: 0 every pair matches; 1 a pair differs (each mismatch printed, the first 12 per class); 2 no data (a
missing image or SDK, a tool failure, or a pair with no benchmark function compared). Prints, per pair, how many
symbols of each class it compared, and where the heap's hot entry points and libc's mallocng sit in both images.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOT = ("main", "malloc", "free", "realloc", "calloc", "__capstone_mallocng_malloc", "__capstone_mallocng_free",
       "__capstone_mallocng_realloc")
KINDS = ("FUNC", "OBJECT", "TLS")


class NoData(Exception):
    pass


def tool(llvm_bin, name):
    p = Path(llvm_bin) / name
    if not p.exists():
        raise NoData(f"{p} does not exist")
    return str(p)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise NoData(f"{' '.join(cmd)}: exit {r.returncode}: {r.stderr.strip()[:400]}")
    return r.stdout


def symbols(nm, path):
    """[(name, kind, value, size, section)] of the defined FUNC/OBJECT/TLS symbols, local labels skipped."""
    out = []
    for line in run([nm, "--defined-only", "-S", "--format=sysv", str(path)]).splitlines():
        f = [x.strip() for x in line.split("|")]
        if len(f) != 7 or f[0] == "Name" and f[1] == "Value":
            continue
        name, value, _cls, kind, size, _line, section = f
        if kind not in KINDS or name.startswith(".L") or not name:
            continue
        out.append((name, kind, int(value, 16), int(size or "0", 16), section))
    return out


def defined_names(nm, path):
    """Names a file (object or archive, every member) defines."""
    names = set()
    for line in run([nm, "--defined-only", "--format=just-symbols", str(path)]).splitlines():
        line = line.strip()
        if line and not line.endswith(":") and not line.startswith(".L"):
            names.add(line)
    return names


def sections(readelf, path):
    """{name: (address, size)} of the allocated sections."""
    out = {}
    for line in run([readelf, "-S", "--wide", str(path)]).splitlines():
        line = line.strip()
        if not line.startswith("[") or "]" not in line:
            continue
        rest = line.split("]", 1)[1].split()
        if len(rest) < 9 or not rest[0].startswith("."):
            continue
        flags = rest[6] if len(rest) == 10 else ""
        if "A" in flags:
            out[rest[0]] = (int(rest[2], 16), int(rest[4], 16))
    return out


class Sdk:
    """The symbol name sets of one SDK: heap.c.obj's and the rest of the SDK's."""
    cache = {}

    @classmethod
    def get(cls, sdk_dir, nm):
        key = str(Path(sdk_dir).absolute())
        if key not in cls.cache:
            cls.cache[key] = cls(Path(key), nm)
        return cls.cache[key]

    def __init__(self, sdk_dir, nm):
        cfgf = sdk_dir / "sdk.json"
        if not cfgf.exists():
            raise NoData(f"{cfgf} does not exist")
        self.cfg = json.loads(cfgf.read_text())
        obj = _load_layout_sdk().archive_member(self.cfg["runtime"], "heap.c.obj")
        with tempfile.NamedTemporaryFile(suffix=".o") as f:
            f.write(obj)
            f.flush()
            self.heap = defined_names(nm, f.name)
        # The runtime's other members: every name of the archive, less the names only heap.c defines.
        # (A name heap.c shares with another member stays here, and so becomes ambiguous below.)
        runtime = self._members_without_heap(nm, self.cfg["runtime"])
        self.sdk = runtime | defined_names(nm, self.cfg["libc"]) | defined_names(nm, self.cfg["builtins"])
        self.script = self.cfg["script"]

    @staticmethod
    def _members_without_heap(nm, archive):
        names, current = set(), None
        for line in run([nm, "--defined-only", "--format=just-symbols", "--print-file-name", archive]).splitlines():
            # "archive:member: name" with --print-file-name in just-symbols format
            parts = line.rsplit(": ", 1)
            if len(parts) != 2:
                continue
            where, name = parts
            if where.endswith(":heap.c.obj") or name.startswith(".L"):
                continue
            names.add(name.strip())
        return names


_LS = None


def _load_layout_sdk():
    global _LS
    if _LS is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("layout_sdk", HERE / "layout-sdk.py")
        _LS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_LS)
    return _LS


def check_pair(exe_a, exe_b, sdk_a, sdk_b, objects, llvm_bin):
    """Returns (ok, report lines, counts)."""
    nm, readelf = tool(llvm_bin, "llvm-nm"), tool(llvm_bin, "llvm-readelf")
    for e in (exe_a, exe_b):
        if not Path(e).is_file():
            raise NoData(f"{e} does not exist")
    sa, sb = Sdk.get(sdk_a, nm), Sdk.get(sdk_b, nm)
    heap = sa.heap | sb.heap
    sdk = sa.sdk | sb.sdk
    bench = set()
    for o in objects:
        bench |= defined_names(nm, o)
    syms = {side: symbols(nm, e) for side, e in (("a", exe_a), ("b", exe_b))}
    ambiguous = sorted(n for n in heap if n in sdk or n in bench)

    def klass(name):
        if name in heap:
            return "ambiguous" if name in ambiguous else "heap"
        if name in bench:
            return "bench"
        if name in sdk:
            return "sdk"
        return "other" if objects else "bench"

    # A layout-controlled image carries tools/layout-sdk.py's region symbols. There the address decides: every
    # heap.c symbol must lie inside a heap region and nothing else may, and a name heap.c shares with libc is heap
    # inside a region and compared outside one.
    regions = {side: _regions(nm, e) for side, e in (("a", exe_a), ("b", exe_b))}
    controlled = all(regions.values())
    table = {side: defaultdict(list) for side in syms}
    region_problems = []
    for side, rows in syms.items():
        for name, kind, value, size, section in rows:
            k = klass(name)
            if controlled:
                inside = any(lo <= value and value + size <= hi for lo, hi in regions[side])
                if k == "ambiguous":
                    k = "heap" if inside else "sdk"
                elif k == "heap" and not inside:
                    region_problems.append(f"{side}: heap.c's {name} at {value:#x} lies outside every heap region")
                elif k != "heap" and any(lo <= value < hi for lo, hi in regions[side]):
                    region_problems.append(f"{side}: {name} ({k}) at {value:#x} lies inside a heap region")
            table[side][(k, "func" if kind == "FUNC" else "data", name)].append((value, size))
    counts, mismatches = defaultdict(int), defaultdict(list)
    for key in sorted(set(table["a"]) | set(table["b"])):
        k, kind, name = key
        va, vb = sorted(table["a"].get(key, [])), sorted(table["b"].get(key, []))
        counts[f"{k}_{kind}"] += max(len(va), len(vb))
        if k in ("heap", "ambiguous"):
            continue
        if va != vb:
            fmt = lambda v: ",".join(f"{x:#x}/{s}" for x, s in v) or "absent"
            # main first: it is the one a reader looks for
            mismatches[f"{k}_{kind}"].insert(0 if name == "main" else len(mismatches[f"{k}_{kind}"]),
                                             f"{name} {fmt(va)} vs {fmt(vb)}")
    # Sections.
    seca, secb = sections(readelf, exe_a), sections(readelf, exe_b)
    sec_mis = []
    for name in sorted(set(seca) | set(secb)):
        a, b = seca.get(name), secb.get(name)
        if a == b:
            continue
        sec_mis.append(f"{name} {a and f'{a[0]:#x}+{a[1]:#x}'} vs {b and f'{b[0]:#x}+{b[1]:#x}'}")
    counts["sections"] = len(set(seca) | set(secb))
    if sec_mis:
        mismatches["sections"] = sec_mis
    if region_problems:
        mismatches["heap_regions"] = region_problems
    hot = {}
    for name in HOT:
        hot[name] = [",".join(f"{v:#x}" for key, vals in table[s].items() if key[2] == name and key[1] == "func"
                              for v, _ in sorted(vals)) or "-" for s in ("a", "b")]
    ok = not mismatches and counts["bench_func"] > 0
    lines = [f"{'PASS' if ok else 'FAIL'} {Path(exe_a).name} vs {Path(exe_b).name}: "
             f"compared bench functions {counts['bench_func']}, bench data {counts['bench_data']}, "
             f"libc+runtime functions {counts['sdk_func']}, libc+runtime data {counts['sdk_data']}, "
             + (f"other symbols {counts['other_func'] + counts['other_data']}, " if objects else "")
             + f"sections {counts['sections']} ({'layout-controlled' if controlled else 'uncontrolled'}); "
             f"not compared: heap.c {counts['heap_func']} functions {counts['heap_data']} data"
             + (f", ambiguous names {counts['ambiguous_func'] + counts['ambiguous_data']} {ambiguous[:8]}"
                if counts['ambiguous_func'] + counts['ambiguous_data'] else "")]
    for k, v in mismatches.items():
        lines.append(f"   MISMATCH {k}: {len(v)}")
        lines += [f"      {x}" for x in v[:12]]
    lines.append("   hot: " + "; ".join(f"{n} {a} | {b}" for n, (a, b) in hot.items() if a != "-" or b != "-"))
    if counts["bench_func"] == 0:
        raise NoData(f"{exe_a}: no benchmark function to compare")
    return ok, lines, counts


def _regions(nm, path):
    """[(start, end)] of the heap regions tools/layout-sdk.py marks in an image; [] in an uncontrolled one."""
    marks = {}
    for line in run([nm, "--defined-only", str(path)]).splitlines():
        f = line.split()
        if len(f) == 3 and f[2].startswith("__layout_heap_"):
            marks[f[2]] = int(f[0], 16)
    return [(marks[f"__layout_heap_{c}_start"], marks[f"__layout_heap_{c}_end"])
            for c in ("text", "rodata", "data", "bss") if f"__layout_heap_{c}_start" in marks]


def record_pairs(rec, tree, labels, benches):
    arms = json.loads((rec / "arms.json").read_text())
    labels = labels or list(arms)[:2]
    if len(labels) != 2 or any(l not in arms for l in labels):
        raise NoData(f"{rec}/arms.json has arms {list(arms)}; need two of them, got {labels}")
    a, b = labels
    pairs = []
    for name_a in sorted(arms[a]["exe"]):
        name_b = name_a[: -len(a)] + b
        if name_b not in arms[b]["exe"]:
            raise NoData(f"{name_a} has no {b} twin in arms.json")
        hits = list(tree.glob(f"benchspec/CPU/*/exe/{name_a}"))
        if not hits:
            raise NoData(f"{name_a}: not under {tree}/benchspec/CPU/*/exe")
        bench = hits[0].parent.parent.name
        if benches and bench not in benches:
            continue
        objs = []
        for label in (a, b):
            dirs = sorted(tree.glob(f"benchspec/CPU/{bench}/build/build_base_{label}.[0-9][0-9][0-9][0-9]"))
            if dirs:
                objs += sorted(dirs[-1].rglob("*.o"))
        pairs.append((bench, hits[0], hits[0].parent / name_b, arms[a]["sdk"], arms[b]["sdk"], objs))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rec", type=Path)
    p.add_argument("--tree", type=Path)
    p.add_argument("--labels", nargs=2)
    p.add_argument("--bench", nargs="*", default=[])
    p.add_argument("--pair", nargs=2, type=Path)
    p.add_argument("--sdk-a", type=Path)
    p.add_argument("--sdk-b", type=Path)
    p.add_argument("--objects", nargs="*", type=Path, default=[])
    p.add_argument("--llvm-bin", help="default: CAPSTONE_LLVM_BIN, else the directory of sdk.json's ld")
    a = p.parse_args()
    try:
        if a.pair:
            if not (a.sdk_a and a.sdk_b):
                raise NoData("--pair needs --sdk-a and --sdk-b")
            pairs = [("pair", a.pair[0], a.pair[1], a.sdk_a, a.sdk_b, a.objects)]
        elif a.rec and a.tree:
            pairs = record_pairs(a.rec, a.tree, a.labels, set(a.bench))
        else:
            raise NoData("give --rec and --tree, or --pair with --sdk-a and --sdk-b")
        if not pairs:
            raise NoData("no executable pair selected")
        llvm_bin = a.llvm_bin or os.environ.get("CAPSTONE_LLVM_BIN") or str(
            Path(json.loads((Path(pairs[0][3]) / "sdk.json").read_text())["ld"]).parent)
        npass, totals = 0, defaultdict(int)
        for bench, ea, eb, sda, sdb, objs in pairs:
            ok, lines, counts = check_pair(ea, eb, sda, sdb, objs, llvm_bin)
            npass += ok
            for k, v in counts.items():
                totals[k] += v
            print(f"[{bench}] " + lines[0])
            for l in lines[1:]:
                print(l)
        print(f"layout-check: {npass} of {len(pairs)} pairs PASS; compared {totals['bench_func']} benchmark functions, "
              f"{totals['bench_data']} benchmark data, {totals['sdk_func']} libc+runtime functions, "
              f"{totals['sdk_data']} libc+runtime data, {totals['sections']} sections")
        return 0 if npass == len(pairs) else 1
    except NoData as e:
        print(f"layout-check: NO DATA: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
