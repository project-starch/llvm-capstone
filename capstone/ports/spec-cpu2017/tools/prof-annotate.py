#!/usr/bin/env python3
"""Annotate one function's machine code with the attempts of a profile.sh run: per instruction, how
many times it was attempted, beside the disassembly. The loop that costs and the instruction in it.

usage: prof-annotate.py RUN FUNCTION [--window N] [--top K] [--objdump llvm-objdump]
  RUN       tools/profile.sh's OUT
  FUNCTION  a symbol of the program (the program's run executable, from the invocation record)
Prints: total attempts in the function, the instruction mix, and the instructions sorted by
attempts (--top, default 40), then the whole function in address order with its counts.
"""
import argparse, collections, json, os, re, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("function")
ap.add_argument("--window", type=int, default=0)
ap.add_argument("--top", type=int, default=40)
ap.add_argument("--objdump", default="llvm-objdump")
ap.add_argument("--nm", default="llvm-nm")
ap.add_argument("--mattr", default="", help="objdump --mattr, e.g. +f,+d for a Capstone image (its F/D encodings are not in the default decoder)")
A = ap.parse_args()

inv = None
for f in sorted(os.listdir(os.path.join(A.run, "invocations"))):
    if re.fullmatch(r"\d{4}\.json", f):
        d = json.load(open(os.path.join(A.run, "invocations", f)))
        if d.get("route") in ("guest", "guest-plain") and d.get("detail", {}).get("window", 0) == A.window:
            inv = d; break
if inv is None:
    sys.exit("no guest invocation with that window")
exe = inv["exe"]
# The program's slide: plain runs at its link address; a domain image at the launcher's entry.
slide = 0
rec = os.path.join(A.run, "image-records.txt")
if inv["route"] == "guest" and os.path.exists(rec):
    lines = [l for l in open(rec) if "entry=0x" in l]
    m = re.search(r"entry=0x([0-9a-f]+)", lines[A.window])
    nm = subprocess.run([A.nm, "-n", "--defined-only", exe], capture_output=True, text=True, check=True).stdout
    dm = [int(l.split()[0], 16) for l in nm.splitlines() if l.split()[-1] == "domain_main"]
    slide = int(m.group(1), 16) - dm[0]
# The function's range: its size from the symbol table (--format=sysv), else up to the next
# function symbol (local labels .L* and mapping symbols $x are not functions).
sysv = subprocess.run([A.nm, "--defined-only", "--format=sysv", exe], capture_output=True, text=True, check=True).stdout
lo = hi = None
for line in sysv.splitlines():
    f = [x.strip() for x in line.split("|")]
    if len(f) >= 5 and f[0] == A.function and f[2].upper() in ("T", "W") and f[1]:
        lo = int(f[1], 16); size = int(f[4] or "0", 16); hi = lo + size if size else None; break
if lo is None:
    sys.exit(f"{A.function}: no such function symbol in {exe}")
if hi is None:
    nm = subprocess.run([A.nm, "-n", "--defined-only", exe], capture_output=True, text=True, check=True).stdout
    after = [int(l.split()[0], 16) for l in nm.splitlines() if len(l.split()) == 3 and l.split()[1] in "TtWw"
             and not l.split()[2].startswith((".L", "$")) and int(l.split()[0], 16) > lo]
    hi = min(after) if after else lo + 0x100000
counts = {}
for line in open(os.path.join(A.run, "capprof", f"window-{A.window}.tsv")):
    if line.startswith("#"): continue
    va, ln, b, att, tr = line.split("\t")
    va = int(va, 16) - slide
    if lo <= va < hi and int(att):
        counts[va] = int(att)
total = sum(counts.values())
dis = subprocess.run([A.objdump, "-d", *([f"--mattr={A.mattr}"] if A.mattr else []), f"--start-address={lo:#x}", f"--stop-address={hi:#x}", exe],
                     capture_output=True, text=True, check=True).stdout
rows = []
for line in dis.splitlines():
    m = re.match(r"\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+|[0-9a-f]{4,8}\s+)\s*(.*)", line)
    if m:
        a = int(m.group(1), 16)
        text = m.group(3).strip() or "?"
        if "unknown" in text:   # an encoding objdump has no name for: keep the bytes
            text = f"<{m.group(2).strip()}>"
        rows.append((a, counts.get(a, 0), text))
mix = collections.Counter()
for a, c, text in rows:
    mix[text.split()[0] if text else "?"] += c
print(f"{A.function} in {os.path.basename(exe)}: {total:,} attempts over {len(rows)} instructions ({lo:#x}-{hi:#x}, slide {slide:#x})")
print("mnemonic mix:", ", ".join(f"{k} {v:,}" for k, v in mix.most_common(16)))
print(f"\ntop {A.top} instructions:")
for a, c, text in sorted(rows, key=lambda r: -r[1])[:A.top]:
    if c: print(f"  {a:#x}  {c:>12,}  {100 * c / total:5.1f}%  {text}")
print("\nwhole function:")
for a, c, text in rows:
    print(f"  {a:#x}  {c:>12,}  {text}")
