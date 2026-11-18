#!/usr/bin/env bash
# Not scored: case 11 from the scored build (builds/c19-driver-la) with a 1,200 s timeout, to see
# whether it faults or returns after the 300 s of the scored runs. Runs after c19-diag11.sh.
set -uo pipefail
P=$HOME/poisoncap-platform
K=$P/repo-19cases
cd "$P"
while [ ! -e logs/c19-diag11.done ]; do sleep 20; done
rm -rf runs/c19-long-la verdicts/c19-long-la
python3 "$K/capstone/ports/sqlite/poisoncap/corpus/run-cases.py" runs/c19-long-la --build builds/c19-driver-la \
  --expect published --port 10097 --timeout 1200 --cases 11 > logs/run-c19-long-la.out 2>&1
echo "rc=$?" >> logs/run-c19-long-la.out
echo done > logs/c19-long.done
