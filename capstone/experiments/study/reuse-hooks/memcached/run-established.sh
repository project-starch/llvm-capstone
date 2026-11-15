#!/usr/bin/env bash
# The established-workload row (docs/plans/2026-11-15-memcached-established-workload.md): REPS servers per
# arm, one boot each, all boots in parallel, each under its own QEMU lock. Run from the repository root.
#   run-established.sh KIT IMAGES CLIENT OUT_ROOT [REPS]
#     KIT       a trap-route kit: qemu-system-riscv64, images/, adapter/, capstone-job
#     IMAGES    build-reuse.sh's output ({stock,protected}/memcached.dom)
#     CLIENT    build-load-client.sh's output (mc-crusher, mc-harness)
#     OUT_ROOT  NEW_ROOT/memcached-<arm>-<rep>/ per boot, NEW_ROOT/locks/
set -euo pipefail
[[ $# == 4 || $# == 5 ]] || { echo 'usage: run-established.sh KIT IMAGES CLIENT OUT_ROOT [REPS]' >&2; exit 2; }
KIT=$(realpath "$1") IMAGES=$(realpath "$2") CLIENT=$(realpath "$3") OUT=$(realpath -m "$4") REPS=${5:-3}
HOOKS=capstone/experiments/study/reuse-hooks/memcached
FLAGS='-l 127.0.0.1 -p 21299 -U 0 -m 32 -t 4 -A -o slab_automove=0'
CONF=$HOOKS/mc-crusher-established.conf
EXPECT=$(python3 - "$CONF" <<'PY'
import math, sys
total = 0
for line in open(sys.argv[1]):
    kv = dict(item.split('=', 1) for item in line.strip().split(','))
    assert kv['conns'] == '1' and kv['send'] in ('ascii_set', 'ascii_get'), line
    pipe = int(kv.get('pipelines', 1))
    total += math.ceil(int(kv['stop_after']) / pipe) * pipe
print(total)
PY
)
mkdir -p "$OUT/locks"
pids=()
for arm in stock protected; do
  for rep in $(seq 1 "$REPS"); do
    CAPSTONE_QEMU_LOCK=$OUT/locks/$arm-$rep python3 "$HOOKS/run-reuse.py" --arm $arm \
      --image "$IMAGES/$arm/memcached.dom" --work "$OUT/memcached-$arm-$rep" --qemu "$KIT/qemu-system-riscv64" \
      --images "$KIT/images" --adapter "$KIT/adapter" --job "$KIT/capstone-job" --harness "$CLIENT/mc-harness" \
      --run-staged capstone/runtime/virtual/run-staged.py --reps 1 --timeout 6000 --flags "$FLAGS" \
      --load-client "$CLIENT/mc-crusher" --load-conf "$CONF" --load-expect "$EXPECT" --load-poll 10 --load-stall 120 \
      > "$OUT/memcached-$arm-$rep.log" 2>&1 &
    pids+=($!)
  done
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "run-established: expect=$EXPECT reps=$REPS rc=$rc"
exit $rc
