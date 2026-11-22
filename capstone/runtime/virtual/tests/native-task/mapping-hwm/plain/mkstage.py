#!/usr/bin/env python3
"""mkstage.py <name> <source stage> <variant>: a twin stage perf/plainhwm/stages/<name>, a copy of the Plain al3 board
stage <source stage> with its one executable replaced by rl-<variant>/<exe>, plan.json's hash updated
(perf/fair/mkstages-al3.py's method). The ctl variant must reproduce the source's executable byte for byte."""
import hashlib, json, os, shutil, subprocess, sys
P = os.environ.get("CAPSTONE_WORK", f"/mnt/hdd/{os.environ.get('USER', '')}/capstone") + "/perf/plainhwm"
name, src, v = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
dst = f"{P}/stages/{name}"
assert not os.path.exists(dst), dst
os.makedirs(f"{P}/stages", exist_ok=True)
subprocess.run(["cp", "-a", src, dst], check=True)
plan = json.load(open(f"{dst}/plan.json")); bd = f"{dst}/stage/bin/plain-rv64"
names = os.listdir(bd); assert len(names) == 1, names
n = names[0]; exe = n.split("_base.")[0]
new = f"{P}/rl-{v}/{exe}"; assert os.path.exists(new), new
old = sha(f"{bd}/{n}"); assert plan["executables"]["plain-rv64"][n] == old, "plan disagrees with stage"
os.remove(f"{bd}/{n}"); shutil.copy2(new, f"{bd}/{n}")
h = sha(f"{bd}/{n}")
if v in ("ctl", "off"): assert h == old, (v, "does not reproduce the source executable")
else: assert h != old, (v, "unchanged")
plan["executables"]["plain-rv64"][n] = h
json.dump(plan, open(f"{dst}/plan.json", "w"), indent=1)
r = subprocess.run(["diff", "-rq", src, dst], capture_output=True, text=True).stdout.strip().splitlines()
assert all("plain-rv64/" in l or "plan.json" in l for l in r), r
print(name, v, n, h[:16])
