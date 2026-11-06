#!/usr/bin/env python3
"""mk-stages.py: perlbench and x264 test stages in gcc train's best configuration (perf/ga2): the rest-ac stage with the
executable swapped for the c2ac12 build and ga2's module-parameter block (step budget 10 s, resume_quantum 1,
retire_exit_sweep 2, arena_pin 1, each read back, restored at the end)."""
import hashlib, json, os, shutil, sys
H = os.environ.get("CAPSTONE_WORK", "/mnt/hdd/" + os.environ["USER"] + "/capstone"); R = f"{H}/perf/rest/stages"; O = f"{H}/perf/best"
G = f"{H}/perf/ga2/stages/502.gcc_r-train01-capstone-lifetimes-devmk-c2ac7/stage/run.sh"
g = open(G).read().splitlines()
SET = [l for l in g if l.startswith("# perf/ga2") or l.startswith("sb_old=") or l.startswith("echo 10000000000 >")
       or l.startswith("echo 1 > /sys/module/capstone_vm/parameters/arena_pin") or l.startswith('echo "@@MX-SET')]
RES = [l for l in g if l.startswith("echo $sb_old") or l.startswith('echo "@@MX-RESTORED') or l == "echo 0 > /sys/module/capstone_vm/parameters/arena_pin"]
assert len(SET) == 5 and len(RES) == 3, (SET, RES)
SET[0] = SET[0].replace("# perf/ga2:", "# perf/best (gcc train's perf/ga2 set):")
for b, exe in (("500.perlbench_r", "perlbench_r"), ("525.x264_r", "x264_r")):
    src = f"{R}/{b}-test-capstone-lifetimes-devmk-ac"; dst = f"{O}/stages/{b}-test-capstone-lifetimes-devmk-c2ac12"
    if os.path.exists(dst): sys.exit(f"{dst} exists")
    shutil.copytree(src, dst, symlinks=True)
    e = f"{dst}/stage/bin/capstone-lifetimes/{exe}_base.capstone-lifetimes"
    assert os.path.isfile(e) and not os.path.islink(e)
    os.unlink(e); shutil.copy(f"{O}/{exe}.c2ac12", e); os.chmod(e, 0o755)
    p = json.load(open(f"{dst}/plan.json")); ex = p["executables"]["capstone-lifetimes"]
    assert f"{exe}_base.capstone-lifetimes" in ex
    ex[f"{exe}_base.capstone-lifetimes"] = hashlib.sha256(open(e, "rb").read()).hexdigest()
    json.dump(p, open(f"{dst}/plan.json", "w"), indent=1)
    r = open(f"{dst}/stage/run.sh").read().splitlines()
    i = next(k for k, l in enumerate(r) if l.startswith('echo "@@SPEC-STAGE'))
    j = next(k for k, l in enumerate(r) if l.startswith('echo "@@SPEC-DONE'))
    r = r[:i] + SET + r[i:j] + RES + r[j:]
    open(f"{dst}/stage/run.sh", "w").write("\n".join(r) + "\n")
    print(dst, ex[f"{exe}_base.capstone-lifetimes"][:16])
