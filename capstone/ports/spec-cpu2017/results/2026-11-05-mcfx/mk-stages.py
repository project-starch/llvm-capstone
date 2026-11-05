#!/usr/bin/env python3
"""mk-stages.py: write run.sh of the two mcf stages from their sources, asserting each anchor once."""
import os
H = os.environ["CAPSTONE_WORK"]; M = f"{H}/perf/mx/stages"
GTSNAP = ("gtsnap() { echo \"@@GT-SNAP$1 $2 hpm=$(cat /sys/module/capstone_vm/parameters/hpm_snapshot 2>/dev/null || echo NA) "
          "vm=$(grep -E '^(pgfault|pgmajfault|pgalloc_normal|pgfree|pgactivate|nr_free_pages) ' /proc/vmstat | tr ' \\n' ':,') "
          "st=$(grep -E '^(ctxt|processes|intr) ' /proc/stat | cut -d' ' -f1-2 | tr ' \\n' ':,') "
          "mem=$(grep -E '^(MemFree|MemAvailable):' /proc/meminfo | tr -s ' ' | tr ' \\n' ':,')\"; }")
def sub(t, a, b):
    assert t.count(a) == 1, (a, t.count(a)); return t.replace(a, b)
# Plain: rusage wrapper + machine-wide snapshots around the command (as results/2026-11-05-gcctrain stage-P)
t = open(f"{H}/spec/stages-plain-times/505.mcf_r-test-plain-rv64-times/stage/run.sh").read()
t = sub(t, "\nbench() {", "\n" + GTSNAP + "\nbench() {")
t = sub(t, '"$S/bin/plain-rv64/mcf_r_base.plain-rv64" inp.in', 'gtsnap 0 $n\n"$S/ru" "$S/bin/plain-rv64/mcf_r_base.plain-rv64" inp.in')
t = sub(t, "rc=$?\n", "rc=$?\ngtsnap 1 $n\necho \"@@GT-RU $n $(cat /tmp/ru.log 2>/dev/null)\"; rm -f /tmp/ru.log\n")
open(f"{M}/505.mcf_r-test-plain-rv64-hpm/stage/run.sh", "w").write(t)
# Temporal bundle: resume_quantum 1, step budget 10 s, exit sweep 2 (none), snapshots; all restored at the end
t = open(f"{H}/spec/stages-eval-b1/505.mcf_r-test-capstone-lifetimes-devmk/stage/run.sh").read()
SET = """# perf/mx bundle: resume_quantum on, step budget 10 s, no exit sweep (retire_exit_sweep 2), for this stage only;
# every value read back and restored at the end
PD=/sys/module/capstone_vm/parameters
sb_old=$(cat /sys/module/capstone_vm/parameters/step_budget_ns); rq_old=$(cat /sys/module/capstone_vm/parameters/resume_quantum); sw_old=$(cat /sys/module/capstone_vm/parameters/retire_exit_sweep)
echo 10000000000 > /sys/module/capstone_vm/parameters/step_budget_ns; echo 1 > /sys/module/capstone_vm/parameters/resume_quantum; echo 2 > /sys/module/capstone_vm/parameters/retire_exit_sweep
echo "@@MX-SET step_budget_ns=$(cat /sys/module/capstone_vm/parameters/step_budget_ns) resume_quantum=$(cat /sys/module/capstone_vm/parameters/resume_quantum) retire_exit_sweep=$(cat /sys/module/capstone_vm/parameters/retire_exit_sweep) was $sb_old,$rq_old,$sw_old"
"""
t = sub(t, "\nbench() {", "\n" + GTSNAP + "\n" + SET + "bench() {")
t = sub(t, 'u0=$(up)\n"$V"', 'u0=$(up)\ngtsnap 0 $n\n"$V"')
t = sub(t, "rc=$?\n", "rc=$?\ngtsnap 1 $n\n")
t = sub(t, 'echo "@@SPEC-DONE 505.mcf_r test"', 'echo $sb_old > /sys/module/capstone_vm/parameters/step_budget_ns; echo $rq_old > /sys/module/capstone_vm/parameters/resume_quantum; echo $sw_old > /sys/module/capstone_vm/parameters/retire_exit_sweep\n'
        'echo "@@MX-RESTORED step_budget_ns=$(cat /sys/module/capstone_vm/parameters/step_budget_ns) resume_quantum=$(cat /sys/module/capstone_vm/parameters/resume_quantum) retire_exit_sweep=$(cat /sys/module/capstone_vm/parameters/retire_exit_sweep)"\n'
        'echo "@@SPEC-DONE 505.mcf_r test"')
open(f"{M}/505.mcf_r-test-capstone-lifetimes-devmk-x/stage/run.sh", "w").write(t)
print("ok")
