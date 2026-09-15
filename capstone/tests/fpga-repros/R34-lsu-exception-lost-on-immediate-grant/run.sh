#!/usr/bin/env bash
# R-34: run the directed test in the capstone-ariane Verilator harness (see the rtl-sim skill).
# Usage: bash run.sh [/path/to/capstone-ariane]      (default: ../../../capstone-ariane)
# Copies src/ into the tree (untracked), deletes the previous run's artifacts, runs, prints the readings.
set -euo pipefail
A=${1:-$(cd "$(dirname "$0")/../../../capstone-ariane" && pwd)}
H=$(cd "$(dirname "$0")" && pwd)
cp -f "$H/src/lsu-mmode-gate.S" "$A/verif/tests/custom/capstone/"
cp -f "$H/src/testlist_lsu_mmode_gate.yaml" "$A/verif/tests/"
rm -f "$A"/verif/sim/out_*/veri-testharness_sim/lsu-mmode-gate* "$A"/verif/sim/out_*/directed_tests/lsu-mmode-gate*
cd "$A"
docker run --rm --memory 24g -v "$(pwd)":/workdir --user "$(id -u):$(id -g)" --entrypoint bash \
  -e HOME=/tmp -e RISCV=/toolchain -e CVA6_REPO_DIR=/workdir ${TRACE_FAST:+-e TRACE_FAST=1} \
  -e NUM_JOBS=16 -e VERILATOR_THREADS=1 cva6-build-rv -c '
set -e; cd /workdir
source verif/regress/install-verilator.sh >/dev/null 2>&1
source verif/regress/install-spike.sh     >/dev/null 2>&1
source verif/sim/setup-env.sh             >/dev/null 2>&1
cd verif/sim
python3 cva6.py --testlist=../tests/testlist_lsu_mmode_gate.yaml --test lsu-mmode-gate \
  --iss_yaml cva6.yaml --target capstone_cv64a6_imafdc_sv39 --iss=veri-testharness \
  --issrun_opts=+debug_disable=1+UVM_VERBOSITY=UVM_NONE --issrun_opts=+time_out=2000000'
f=$(ls -t verif/sim/out_*/veri-testharness_sim/lsu-mmode-gate*.log.iss | head -1)
python3 - "$f" <<'PY'
import sys, re
for l in open(sys.argv[1], errors="replace").read().splitlines():
    if re.search(r"SUCCESS|\[Cycle", l): print(l[:200])
PY
python3 - "${f%.iss}" <<'PY'
import sys, re
for l in open(sys.argv[1], errors="replace").read().splitlines():
    if re.search(r"mem 0x|exception @|dret", l): print(l[:160])
PY
