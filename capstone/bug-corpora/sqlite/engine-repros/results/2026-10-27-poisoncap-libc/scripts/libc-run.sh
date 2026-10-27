#!/usr/bin/env bash
# PoisonCap on the corpus in its driver configuration (256 MiB heap from malloc, minimum 16, lookaside
# on), with the C library's revocation on for each measured process (run-cases.py --libc-revocation
# run), as in the speedtest1 reuse runs. The case binaries are those of 2026-10-26's driver-la, bit for
# bit; only the two new controls (libc_revoke, uaf_libc) are built here.
#   libc-run.sh build   build the controls, assemble builds/driver-libc-la, compare with driver-la
#   libc-run.sh smoke   the gate's controls and the evidence controls, once
#   libc-run.sh smoke-off   the same with the C library's revocation off (the negative control)
#   libc-run.sh full    every case, twice, then score both runs
set -uo pipefail
MODE=${1:?build, smoke or full}
P=$HOME/poisoncap-platform
K=$P/repo-heap256
cd "$P"
B=builds/driver-libc-la
PORT=10099

build() {
  rm -rf builds/libc-controls "$B"
  bash "$K/capstone/ports/sqlite/poisoncap/platform/run-in-container.sh" \
    python3 "$K/capstone/ports/sqlite/poisoncap/corpus/build-cases.py" --platform "$P" \
    --amalgamation sqlite/amalgamation-published --ext-src sqlite/sqlite-src \
    --out builds/libc-controls --heap-bytes 268435456 --heap-min 16 --heap-malloc --only none \
    > logs/build-libc-controls.out 2>&1
  echo "rc=$?" >> logs/build-libc-controls.out
  cp -a builds/driver-la "$B"
  cp -a builds/libc-controls/libc_revoke builds/libc-controls/uaf_libc "$B"/
  python3 - "$B" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
build = json.loads((out / 'build.json').read_text())
build['controls'] = sorted(set(build['controls']) | {'libc_revoke', 'uaf_libc'})
build['assembled'] = ('builds/driver-la of 2026-10-26, unchanged, plus libc_revoke and uaf_libc '
                      'from builds/libc-controls')
(out / 'build.json').write_text(json.dumps(build, indent=2) + '\n')
EOF
  (cd builds/driver-la && sha256sum $(python3 -c "import json;print(' '.join(json.load(open('build.json'))['built']))")) \
    > logs/driver-la.sha256
  (cd "$B" && sha256sum -c ../../logs/driver-la.sha256) > logs/driver-libc-la-compare.out 2>&1
  echo "rc=$?" >> logs/driver-libc-la-compare.out
}

pc() {  # $1 run name, $2 off or run, then extra run-cases.py arguments
  local run=$1 libc=$2; shift 2
  rm -rf "runs/$run" "verdicts/$run"
  python3 "$K/capstone/ports/sqlite/poisoncap/corpus/run-cases.py" "runs/$run" --build "$B" \
    --expect published --libc-revocation "$libc" --port "$PORT" "$@" > "logs/run-$run.out" 2>&1
  echo "rc=$?" >> "logs/run-$run.out"
}

score() {  # $1 run name
  bash "$K/capstone/ports/sqlite/poisoncap/platform/run-in-container.sh" \
    python3 "$K/capstone/ports/sqlite/poisoncap/corpus/score-cases.py" --published "runs/$1" \
    --nm work/sdk/bin/llvm-nm --rootfs work/output/rootfs-riscv64-purecap --out "verdicts/$1" \
    > "logs/score-$1.out" 2>&1
  echo "rc=$?" >> "logs/score-$1.out"
}

case "$MODE" in
  build) build ;;
  smoke) pc smoke-driver-libc-la run --controls-only ;;
  smoke-off) pc smoke-off-driver-libc-la off --controls-only ;;
  full) pc driver-libc-la run; score driver-libc-la; pc driver-libc-la-rep2 run; score driver-libc-la-rep2 ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo done > "logs/libc-run-$MODE.done"
