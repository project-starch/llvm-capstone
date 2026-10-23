#!/usr/bin/env python3
"""tier3-replay-board.py SESSION BOARDKIT OUT (2026-10-24): the compiler lane's r60-fmtfp/replay-stats2.py (sha256/16
02fe2756d7c0a686 at this copy's writing) ported to a BOARD kit: boardvm.BoardBatch (the dev worktree /tmp/capstone/wt-newarms,
bcf5f1724110) in place of virtualvm.Batch, the CAP_INSMOD_ARGS hook dropped (a board kit sets no module parameters here).
Every step, the staging and the readings are the original's. Its docstring follows.
replay-stats.py SESSION KITDIR OUT: replay an appvm session's collected calls (plan.jsonl, collect/images,
collect/share) in ONE boot on KITDIR, staged exactly as appvm-session.py's boot does (images, share, the /mnt/host
binding step, then `cd CWD && exec env ENV capstone-vexec IMAGE ARGS` per call), with a `capstone-vexec --stats` read after
every call. Each step's output is newline-terminated. Writes OUT/readings.json: per call, its exit, its fault line
(cause, pc, the function from the image's symbols) and the table counters after it. Exit 2 when nothing ran."""
import json, re, shlex, subprocess, sys
from pathlib import Path
sys.path.insert(0, "/tmp/capstone/wt-newarms/capstone/bug-corpora/tools")
import boardvm
import os

S, K, O = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
calls, seen = [], set()
for line in (S / "plan.jsonl").read_text().splitlines():
    c = json.loads(line)
    if c["name"] in seen: continue
    seen.add(c["name"]); calls.append(c)
b = boardvm.BoardBatch(O / "stage")
root = str(Path(b.put("images", S / "collect" / "images")).parent)
(S / "collect" / "share" / ".session-share").exists() or (S / "collect" / "share" / ".session-share").write_text("replay\n")
b.put("share", S / "collect" / "share")
b.step("session-share", f"if [ -L /mnt/host ] || [ ! -e /mnt/host ]; then mkdir -p /mnt && ln -sfn {root}/share /mnt/host; else mount -o bind {root}/share /mnt/host; fi; "
                        f"[ -f /mnt/host/.session-share ] && echo SESSION-SHARE-OK || echo 'SESSION: /mnt/host does not show the stage'")
wrap = lambda c: f"{c}; rc=$?; echo; exit $rc"
names = []
b.step("param", "cat /sys/module/capstone_vm/parameters/node_max_records; echo")
b.step("stats-0", wrap(f"{root}/capstone-vexec --stats"))
for i, c in enumerate(calls):
    n = f"c{i:02d}-" + re.sub(r"[^A-Za-z0-9_.+-]", "_", c["name"])[:60]; names.append((n, c))
    env = " ".join(shlex.quote(e) for e in c["env"])
    b.step(n, wrap(f"cd {shlex.quote(c['cwd'])} && env {env} {root}/capstone-vexec {root}/images/{c['image']} " + " ".join(shlex.quote(x) for x in c["args"])))
    b.step(f"stats-{i+1}", wrap(f"{root}/capstone-vexec --stats"))
timeout = 300 + sum(min(int(c.get("timeout", 120)), 600) for c in calls)
runs, serial, ok = b.execute(K, O / "guest", timeout=timeout)
def stats(name):
    t = runs.get(name, ("", {}))[0]
    return next((json.loads(l) for l in t.splitlines() if l.startswith("{")), None)
def symbol(image, pc, code):
    nm = subprocess.run([str(Path("/mnt/hdd") / Path.home().name / "capstone/t3-freeze/llvm/bin/llvm-nm"), "-n", str(S / "collect" / "images" / image)], capture_output=True, text=True).stdout
    ro = subprocess.run([str(Path("/mnt/hdd") / Path.home().name / "capstone/t3-freeze/llvm/bin/llvm-readelf"), "-l", str(S / "collect" / "images" / image)], capture_output=True, text=True).stdout
    base = next((int(l.split()[2], 16) for l in ro.splitlines() if l.strip().startswith("LOAD")), 0)
    va, best = base + pc - code, "?"
    for l in nm.splitlines():
        p = l.split()
        if len(p) == 3 and p[1] in "tT" and not p[2].startswith(".L"):
            if int(p[0], 16) <= va: best = p[2]
            else: break
    return best
out = {"kit": str(K), "completed": ok, "stats0": stats("stats-0"), "calls": []}
for i, (n, c) in enumerate(names):
    text, res = runs.get(n, ("", {}))
    f = res.get("fault") or ""
    m = re.search(r"cause=(\d+) pc=0x([0-9a-f]+).*?code=0x([0-9a-f]+)-", f)
    out["calls"].append({"name": c["name"], "result": res, "ran": n in runs,
                         "fault": m and {"cause": int(m.group(1)), "function": symbol(c["image"], int(m.group(2), 16), int(m.group(3), 16))},
                         "tail": [l for l in text.splitlines() if l.strip() and "VM_ADD_OK" not in l][-2:], "stats": stats(f"stats-{i+1}")})
(O / "readings.json").write_text(json.dumps(out, indent=1) + "\n")
ran = sum(x["ran"] for x in out["calls"])
print(f"calls {len(calls)} ran {ran} completed {ok}")
sys.exit(0 if ran else 2)
