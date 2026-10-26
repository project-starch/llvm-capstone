#!/bin/bash
# One qualification attempt: from the SPEC archive to validated test and train runs of every
# benchmark in manifest.lock.json, with one toolchain and one platform.
# usage: qualify.sh KIT SPEC_ARCHIVE TOOLCHAIN PLATFORM OUT [FIRST_CPU]
#   OUT gets: tree/ (fresh extraction), build/, negative/, test/<bench>/, train/<bench>/,
#   qualification.json. All runs (every benchmark at both sizes) go in parallel, each in its own
#   guest pinned to its own host CPU from FIRST_CPU on (default 40).
set -euo pipefail
KIT=$(cd "$1" && pwd) ARCHIVE=$2 TC=$(cd "$3" && pwd) PL=$(cd "$4" && pwd) OUT=$5 CPU0=${6:-40}
[ -e "$OUT" ] && { echo "qualify: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
M=$KIT/manifest.lock.json
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/qualify.log"; }
got=$(sha256sum "$ARCHIVE" | cut -c1-64)
[ "$got" = "$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['spec']['archive_sha256'])" "$M")" ] \
  || { log "SPEC archive $got is not the manifest's"; exit 1; }
python3 "$KIT/tools/check-platform.py" "$M" "$PL" > "$OUT/platform-check.txt" || { log "platform check failed"; exit 1; }
log "extract"; python3 "$KIT/tools/spec-extract.py" "$ARCHIVE" "$OUT/tree" ALL > "$OUT/extract.txt"
S=$OUT/tree/SPEC17
log "build"; bash "$KIT/tools/build.sh" "$KIT" "$S" "$TC/sdk" "$OUT/build" capstone-virtual > "$OUT/build.out" 2>&1 || { log "build failed"; exit 1; }
log "negative controls"; bash "$KIT/tools/negative-controls.sh" "$KIT" "$S" "$TC" "$PL" "$OUT/build" "$OUT/negative" > "$OUT/negative.out" 2>&1 \
  || { log "negative controls failed"; exit 1; }
mapfile -t BENCH < <(python3 -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))['workloads']['benchmarks']))" "$M")
# Every benchmark at every size at once: each run has its own guest, run directory and host CPU.
log "test and train: ${#BENCH[@]} benchmarks each"
cpu=$CPU0
for size in test train; do
  for b in "${BENCH[@]}"; do
    taskset -c "$cpu" bash "$KIT/tools/run.sh" "$KIT" "$S" "$TC" "$PL" "$OUT/build" "$size" "$b" "$OUT/$size/$b" \
      > "$OUT/$size-$b.out" 2>&1 &
    cpu=$((cpu + 1))
  done
done
wait || true
for size in test train; do
  for b in "${BENCH[@]}"; do
    log "$size $b $(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['verdict'],d['start'],d['end'],'|',';'.join(d['reasons']))" "$OUT/$size/$b/result.json" 2>&1)"
  done
done
python3 - "$OUT" "$M" <<'PY' > "$OUT/qualification.json"
import json, os, sys
out, m = sys.argv[1], json.load(open(sys.argv[2]))
res = {}
for size in m['workloads']['sizes']:
    for b in m['workloads']['benchmarks']:
        p = os.path.join(out, size, b, 'result.json')
        res.setdefault(size, {})[b] = json.load(open(p)) if os.path.exists(p) else {"verdict": "MISSING"}
exe = [l.split()[1:3] for l in open(os.path.join(out, 'build', 'build-record.txt')) if l.startswith('exe ')]
allpass = all(r['verdict'] == 'PASS' for s in res.values() for r in s.values())
print(json.dumps({"verdict": "QUALIFIED" if allpass else "NOT QUALIFIED", "results": res,
                  "executables": exe, "toolchain_commit": m['toolchain']['commit']}, indent=2))
PY
log "$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['verdict'])" "$OUT/qualification.json")"
