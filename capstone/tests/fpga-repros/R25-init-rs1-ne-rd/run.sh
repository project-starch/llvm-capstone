#!/usr/bin/env bash
# run.sh -- run one arm of this package in RTL simulation (capstone-ariane at fpga-testing-dev 66c4e7517 or later).
# usage: bash run.sh <capstone-ariane checkout> <test-name> [+define+S12_MEM_DELAY=40]
# The arms live in the checkout already (verif/tests/custom/capstone/, list verif/tests/testlist_r26.yaml);
# the copies under sim/ here are the frozen record. A timeout prints as "SUCCESS ... after <time_out+13>" --
# that is a HANG, not a pass. A memory-delay define is honoured only when passed on the run that builds the
# model; check work-ver/Variane_testharness__verFiles.dat for it before labelling a result with a latency.
set -eu
CO="$1"; T="$2"; DEF="${3:-}"; LIST=testlist_capstone.yaml
docker run --rm --cgroup-parent=docker.slice --memory=12g --memory-swap=12g -v "$CO":/workdir --user "$(id -u):$(id -g)" --entrypoint bash \
  -e HOME=/tmp -e ANVILC=/usr/local/bin/anvil -e RISCV=/toolchain -e CVA6_REPO_DIR=/workdir -e NUM_JOBS=8 -e VERILATOR_THREADS=1 cva6-build-rv -c '
set -e; cd /workdir; make -C core/anvil_build >/dev/null
source verif/regress/install-verilator.sh >/dev/null 2>&1; source verif/regress/install-spike.sh >/dev/null 2>&1
source verif/sim/setup-env.sh >/dev/null 2>&1; cd verif/sim; rm -f out_*/veri-testharness_sim/'"$T"'.*
python3 cva6.py --testlist=../tests/'"$LIST"' --test '"$T"' --iss_yaml cva6.yaml --target capstone_cv64a6_imafdc_sv39 \
  --iss=veri-testharness --issrun_opts=+debug_disable=1+UVM_VERBOSITY=UVM_NONE --issrun_opts=+time_out=6000 \
  --isscomp_opts="'"$DEF"'" >/dev/null 2>&1 || true
ISS=$(ls -t out_*/veri-testharness_sim/'"$T"'.*.log.iss 2>/dev/null | head -1)
[ -n "$ISS" ] || { echo "NO RESULT: the test did not build or run (check the testlist name)"; exit 2; }
grep -hE "\*\*\* (SUCCESS|FAILED)" "$ISS" || { echo "NO VERDICT LINE in $ISS"; exit 2; }
grep -qF "'"$DEF"'" ../../work-ver/Variane_testharness__verFiles.dat 2>/dev/null && echo "model carries '"$DEF"'" || echo "model does NOT carry the define '"$DEF"' (plain / delay-0 model)"'
