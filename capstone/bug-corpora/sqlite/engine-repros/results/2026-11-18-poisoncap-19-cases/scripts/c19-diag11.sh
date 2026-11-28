#!/usr/bin/env bash
# DIAGNOSTIC ONLY, never a verdict: case 11 on PoisonCap's diagnostic amalgamation with the FTS3CMP
# hook (make-diag-amalgamation.py: the two terms fts3SegReaderCmp hands to memcmp, next to every
# free, flush and realloc), in the driver configuration. Run after c19-poisoncap.sh, one guest at a time.
set -uo pipefail
P=$HOME/poisoncap-platform
K=$P/repo-19cases
cd "$P"
while [ ! -e logs/c19-poisoncap.done ]; do sleep 20; done
rm -rf builds/c19-diag11-la runs/c19-diag11-la
bash "$K/capstone/ports/sqlite/poisoncap/platform/run-in-container.sh" \
  python3 "$K/capstone/ports/sqlite/poisoncap/corpus/build-cases.py" --platform "$P" \
  --amalgamation sqlite/amalgamation-published-diag-c19 --ext-src sqlite/sqlite-src \
  --out builds/c19-diag11-la --heap-bytes 268435456 --heap-min 16 --heap-malloc --only 11 > logs/build-c19-diag11-la.out 2>&1
echo "rc=$?" >> logs/build-c19-diag11-la.out
python3 "$K/capstone/ports/sqlite/poisoncap/corpus/run-cases.py" runs/c19-diag11-la --build builds/c19-diag11-la \
  --expect published --port 10097 --timeout 600 --cases 11 > logs/run-c19-diag11-la.out 2>&1
echo "rc=$?" >> logs/run-c19-diag11-la.out
echo done > logs/c19-diag11.done
