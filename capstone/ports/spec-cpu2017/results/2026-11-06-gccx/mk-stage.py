#!/usr/bin/env python3
"""mk-stage.py: gcc train's Temporal stage (perf/gt) with resume_quantum 1, a 10 s step budget for
this stage only, each read back and restored at the end; the sweep stays at the kit's 3 (as results/2026-11-05-mcfx stage-x)."""
import os
H = os.environ["CAPSTONE_WORK"]; P = "/sys/module/capstone_vm/parameters"
def sub(t, a, b):
    assert t.count(a) == 1, (a, t.count(a)); return t.replace(a, b)
t = open(f"{H}/perf/gt/stages/502.gcc_r-train01-capstone-lifetimes-devmk/stage/run.sh").read()
SET = f"""# perf/gx bundle: resume_quantum on, step budget 10 s (the kit's retire_exit_sweep 3 kept: mode 2 sends in-run retires to mode 0), for this stage only
sb_old=$(cat {P}/step_budget_ns); rq_old=$(cat {P}/resume_quantum); sw_old=$(cat {P}/retire_exit_sweep)
echo 10000000000 > {P}/step_budget_ns; echo 1 > {P}/resume_quantum
echo "@@MX-SET step_budget_ns=$(cat {P}/step_budget_ns) resume_quantum=$(cat {P}/resume_quantum) retire_exit_sweep=$(cat {P}/retire_exit_sweep) was $sb_old,$rq_old,$sw_old"
"""
t = sub(t, "\nbench() {", "\n" + SET + "bench() {")
t = sub(t, 'echo "@@SPEC-DONE 502.gcc_r train"', f'echo $sb_old > {P}/step_budget_ns; echo $rq_old > {P}/resume_quantum; echo $sw_old > {P}/retire_exit_sweep\n'
        f'echo "@@MX-RESTORED step_budget_ns=$(cat {P}/step_budget_ns) resume_quantum=$(cat {P}/resume_quantum) retire_exit_sweep=$(cat {P}/retire_exit_sweep)"\n'
        'echo "@@SPEC-DONE 502.gcc_r train"')
open(f"{H}/perf/gx/stages/502.gcc_r-train01-capstone-lifetimes-devmk-x/stage/run.sh", "w").write(t); print("ok")
