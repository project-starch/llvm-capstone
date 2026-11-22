#!/usr/bin/env python3
"""mkstage.py <name> <Full twin stage> <variant>: a twin stage perf/plainhwm/stages/<name>, a copy of the native-format
twin stage <Full twin stage> (perf/mrhwm/stages) with its one native executable replaced by the Plain image
rl-<variant>/<exe>, and plan.json's hash updated: the method of the RVAL twins (perf/alean/plain/st-gcc), since the
twin runner takes native-format stages only. The Plain binary has no Capstone sections and runs as an ordinary
process. The ctl and off variants must equal the Plain al3 board stage's executable byte for byte."""
import hashlib, json, os, shutil, subprocess, sys
W = os.environ.get("CAPSTONE_WORK", f"/mnt/hdd/{os.environ.get('USER', '')}/capstone")
P = W + "/perf/plainhwm"
BOARD = {"mcf_r": "2d48ad97ae6009c7", "cpugcc_r": "93bc8e196b036d07"}
name, src, v = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
dst = f"{P}/stages/{name}"
assert not os.path.exists(dst), dst
assert os.path.isdir(f"{P}/rl-{v}"), f"no relink rl-{v}: nothing is copied"   # a failed check after cp -a left a stage with the Full binary
os.makedirs(f"{P}/stages", exist_ok=True)
subprocess.run(["cp", "-a", src, dst], check=True)
plan = json.load(open(f"{dst}/plan.json")); bd = f"{dst}/stage/bin/native"
names = os.listdir(bd); assert len(names) == 1, names
n = names[0]; exe = n.split("_base.")[0]
new = f"{P}/rl-{v}/{exe}"; assert os.path.exists(new), new
os.remove(f"{bd}/{n}"); shutil.copy2(new, f"{bd}/{n}"); os.chmod(f"{bd}/{n}", 0o755)
h = sha(f"{bd}/{n}")
assert (h[:16] == BOARD[exe]) == (v in ("ctl", "off")), (v, h[:16])
(k, ex), = plan["executables"].items(); (kk, _), = ex.items(); ex[kk] = h
json.dump(plan, open(f"{dst}/plan.json", "w"), indent=1)
r = subprocess.run(["diff", "-rq", src, dst], capture_output=True, text=True).stdout.strip().splitlines()
assert len(r) == 2 and all("bin/native/" in l or "plan.json" in l for l in r), r
print(name, v, n, h[:16])
