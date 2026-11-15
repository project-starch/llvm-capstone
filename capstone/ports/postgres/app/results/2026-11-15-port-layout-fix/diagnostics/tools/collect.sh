#!/usr/bin/env bash
# collect.sh: provenance.txt per run-6 PostgreSQL reuse run and run-6/reuse/postgres-reuse.json.
set -euo pipefail
F=$HOME/asr-work/fix-postgres TREE=$HOME/asr-work/fix-postgres/tree-38b1e0a R=$HOME/asr-work/eval/run-6/reuse
for arm in stock protected; do for rep in 1 2; do
  $F/provenance.sh $R/postgres-$arm-$rep $arm $F/build/reuse-$arm/image/postgres.dom
done; done
echo '"7d7d1e0fd3c (capstone/docs/plans/2026-11-15-postgres-port-layout-fix.md, branch fix/postgres-port-layout)"' > $F/prereg.json
python3 $TREE/capstone/experiments/study/reuse-hooks/cpython/collect-reuse.py --program "PostgreSQL 17.5" \
  --workload "pgbench 17.5 built-in tpcb-like script, scale 1, 4000 transactions, --random-seed=1, statements replayed in one single-user backend (make-pgbench-workload.sh, run-pgbench-reuse.py); run-6, memory-context patch 0003 with stock layout (fix/postgres-port-layout 38b1e0a3529)" \
  --out $R/postgres-reuse.json \
  --rep stock=$R/postgres-stock-1/rep.json --rep stock=$R/postgres-stock-2/rep.json \
  --rep protected=$R/postgres-protected-1/rep.json --rep protected=$R/postgres-protected-2/rep.json \
  --image stock=$F/build/reuse-stock/image/postgres.dom --image protected=$F/build/reuse-protected/image/postgres.dom \
  --extra pre_registration=$F/prereg.json
