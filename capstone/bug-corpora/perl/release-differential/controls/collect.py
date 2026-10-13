#!/usr/bin/env python3
"""Read the controls' serial logs into one results.json.

    collect.py OUT.json --llvm-bin BIN --arm NAME SERIAL IMAGE [--arm NAME SERIAL IMAGE] ...
                [--meta KEY=VALUE ...]

Per arm and mode: exit status, the RESULT line, the A/B/C/arena/parent addresses (PTR lines),
every announced probe (label, file:line), the last probe announced before the process ended,
and a domain fault's cause, pc, the function the pc lies in and the source line of the pc
(llvm-symbolizer on the image, which carries the harness's -g line table).

A run with no CTL_END line, or an image whose in-guest sha256 differs from IMAGE's, is an
error: the script exits 1 and says which, rather than writing a row.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "tools"))
import verdicts as v  # noqa: E402

BLOCK = re.compile(r"CTL_BEGIN:(?P<arm>[^:\r\n]+):(?P<mode>\w+)\r?\n(?P<out>.*?)CTL_STDERR:(?P=arm):(?P=mode)\r?\n"
                   r"(?P<err>.*?)CTL_END:(?P=arm):(?P=mode):(?P<rc>\d+):(?P<s>[\d.]+):(?P<e>[\d.]+)", re.S)
FAULT = re.compile(r"domain fault cause=(\d+) pc=(0x[0-9a-f]+) address=(0x[0-9a-f]+) entry=\S+ "
                   r"code=(0x[0-9a-f]+)-")


def source_line(llvm_bin, image, link_pc):
    out = subprocess.run([str(Path(llvm_bin) / "llvm-symbolizer"), f"--obj={image}", "--no-inlines",
                          hex(link_pc)], capture_output=True, text=True).stdout.split("\n")
    return out[1].strip() if len(out) > 1 else ""


def one_arm(name, serial, image, llvm_bin):
    text = Path(serial).read_bytes().decode("utf-8", "replace").replace("\r\n", "\n")
    sha = hashlib.sha256(Path(image).read_bytes()).hexdigest()
    m = re.search(rf"CTL_IMAGE:{re.escape(name)}:([0-9a-f]{{64}})", text)
    if not m or m.group(1) != sha:
        sys.exit(f"{name}: in-guest image hash {m.group(1) if m else 'missing'} != {sha}")
    syms = v.Symbols(llvm_bin, image)
    runs = {}
    for b in BLOCK.finditer(text):
        if b.group("arm") != name:
            continue
        out, err = b.group("out"), b.group("err")
        ptrs = {p.group(1): {"addr": p.group(2), "base": p.group(3), "end": p.group(4)}
                for p in re.finditer(r"^PTR (\S+) addr=(\S+) base=(\S+) end=(\S+)$", out, re.M)}
        probes = [{"label": p.group(1), "at": p.group(2)}
                  for p in re.finditer(r"^PROBE (\S+) (\S+)$", out, re.M)]
        res = re.search(r"^RESULT \S+ (PASS|FAIL) ?(.*)$", out, re.M)
        row = {"exit": int(b.group("rc")), "seconds": round(float(b.group("e")) - float(b.group("s")), 2),
               "result": res.group(1) if res else None, "result_detail": res.group(2) if res else None,
               "pointers": ptrs, "probes": probes,
               "last_probe": probes[-1] if probes else None,
               "lines": [l for l in out.splitlines() if l.split(" ", 1)[0] in
                         ("BUILD", "INTERP", "ARENA_HEAD", "PL_sv_root", "SAME_ARENA", "SUCCESSOR",
                          "READ", "FREE", "STALE", "RESULT", "DONE")]}
        f = FAULT.search(err)
        if f:
            cause, pc, code = int(f.group(1)), int(f.group(2), 16), int(f.group(4), 16)
            fn, off = syms.lookup(pc, code)
            link_pc = pc - code + syms.base
            row.update(outcome="fault", fault={
                "cause": cause, "pc": hex(pc), "address": f.group(3), "code_start": hex(code),
                "link_pc": hex(link_pc), "function": fn, "offset": off,
                "source": source_line(llvm_bin, image, link_pc)})
        else:
            row["outcome"] = "completed" if b.group("rc") == "0" else f"exit{b.group('rc')}"
        row["stderr"] = [l for l in err.splitlines() if l.strip()][:6]
        runs[b.group("mode")] = row
    missing = [m for m in ("parent", "sibling", "successor", "stale") if m not in runs]
    if missing:
        sys.exit(f"{name}: no CTL_END for {missing} in {serial}")
    return {"image": str(image), "image_sha256": sha, "serial_sha256": hashlib.sha256(
        Path(serial).read_bytes()).hexdigest(), "runs": runs}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--arm", nargs=3, action="append", required=True, metavar=("NAME", "SERIAL", "IMAGE"))
    p.add_argument("--meta", action="append", default=[], help="KEY=VALUE recorded as given")
    a = p.parse_args()
    doc = {"meta": dict(kv.split("=", 1) for kv in a.meta), "arms": {}}
    for name, serial, image in a.arm:
        doc["arms"][name] = one_arm(name, serial, image, a.llvm_bin)
    a.output.write_text(json.dumps(doc, indent=1) + "\n")
    for name, arm in doc["arms"].items():
        for mode, r in arm["runs"].items():
            f = r.get("fault") or {}
            print(f"{name:10} {mode:10} {r['outcome']:9} {str(r['result']):5} "
                  f"cause={f.get('cause', '-')} fn={f.get('function', '-')} src={f.get('source', '-')} "
                  f"last_probe={(r['last_probe'] or {}).get('at', '-')}")


if __name__ == "__main__":
    main()
