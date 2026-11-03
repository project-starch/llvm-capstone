#!/usr/bin/env python3
"""The verdict of one profile-php.sh run: OUT/result.json.

usage: judge.py OUT RC START END WALL SCRIPT [SCRIPT...]   (profile-php.sh)

PASS needs: exit 0, one guest invocation and one capprof window per SCRIPT, each window's records
summing to its footer, no invalid window, and in script K's output (OUT/stdout-K.txt) a line for
every test it times (each end_test name, PHP string escapes undone) and the Total line last.
"""
import glob, json, os, re, sys

if len(sys.argv) < 7:
    sys.exit(__doc__)
out, rc, start, end, wall = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
scripts = sys.argv[6:]
bad = []
if rc != 0:
    bad.append(f"exit {rc}" + (" (timeout)" if rc == 124 else ""))
inv = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "invocations", "[0-9]" * 4 + ".json")))]
guest = [d for d in inv if d.get("route") in ("guest", "guest-plain")]
if len(guest) != len(scripts):
    bad.append(f"{len(guest)} guest invocations, expected {len(scripts)}")
tsv = glob.glob(os.path.join(out, "capprof", "window-*.tsv"))
if len(tsv) != len(scripts):
    bad.append(f"{len(tsv)} windows, expected {len(scripts)}")
for p in tsv:
    lines = open(p).read().splitlines()
    if not lines or not lines[-1].startswith("# total "):
        bad.append(f"{p}: no footer"); continue
    if sum(int(l.split("\t")[3]) for l in lines if not l.startswith("#")) != int(lines[-1].split()[2]):
        bad.append(f"{p}: records do not sum to the footer")
if glob.glob(os.path.join(out, "capprof", "*.invalid")):
    bad.append("an invalid window")
# A PHP string literal, either quote, with backslash escapes ('$x = $GLOBALS[\'v\']').
tests = 0
for k, script in enumerate(scripts):
    names = [re.sub(r"\\(.)", r"\1", m.group(2))
             for m in re.finditer(r"end_test\(\$t, (['\"])((?:\\.|(?!\1).)*)\1", open(script).read())]
    tests += len(names)
    path = os.path.join(out, f"stdout-{k}.txt")
    printed = [l for l in open(path, errors="replace").read().splitlines() if l.strip()] if os.path.exists(path) else []
    missing = [n for n in names if not any(l.startswith(n) for l in printed)]
    tag = os.path.basename(script)
    if not names:
        bad.append(f"{tag}: no end_test names found in the script")
    if missing:
        bad.append(f"{tag}: missing test lines: {missing}")
    if not printed or not printed[-1].startswith("Total"):
        bad.append(f"{tag}: no Total line at the end")
r = {"verdict": "FAIL" if bad else "PASS", "reasons": bad, "scripts": len(scripts), "tests": tests, "rc": rc, "start": start,
     "end": end, "wall_s": wall, "guest_exits": [d.get("detail", {}).get("guest") for d in guest]}
json.dump(r, open(os.path.join(out, "result.json"), "w"), indent=1)
print(r["verdict"], bad, f"wall {wall}s")
