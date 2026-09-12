#!/usr/bin/env bash
# Does the Sublet port still keep its promise? The gate for the protected arm.
#
#   run-pg-sublet-gate.sh
#
# run-pg-gate.sh beside this asks whether the manager runs under capabilities
# at all. This one asks whether the discipline holds, and it checks the two
# things a change could break without any count noticing.
#
#   1  the level below alone, against every claim the design makes about it:
#      a reset is one revocation, the sub-pool works again afterwards, a
#      context that fills its sub-pool costs two and no more, and two thousand
#      creates and deletes leave nothing behind
#   2  the manager over it, replaying a generated trace, with the teardown cost
#      an identity the domain checks itself and every freed object's contents
#      read back
#
# It needs no recording, for the reason run-pg-gate.sh gives: the workload is
# tools/make-fixture-trace.py and the file is generated.
#
# Exit 0 if both hold, non-zero otherwise, as the nightly expects.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT=${OUT:-$CAPSTONE_TMP_ROOT/pg-mmgr-host}
GATE=${GATE:-$CAPSTONE_TMP_ROOT/pg-sublet-gate}
PYTHON=${PYTHON:-python3}

mkdir -p "$GATE"
TRACE=$GATE/fixture.a11

echo "== 1. the level below alone, nineteen claims"
SHARE_DIR="$GATE/share-subpool" LOG_FILE="$GATE/subpool.log" \
  bash "$SCRIPT_DIR/run-subpool-test.sh"

echo
echo "== the workload, generated so the gate needs no recording"
"$PYTHON" "$SCRIPT_DIR/tools/make-fixture-trace.py" "$TRACE"

echo
echo "== 2. the manager over the discipline, in a domain"
OUT="$OUT" SHARE_DIR="$GATE/share" LOG_FILE="$GATE/domain.log" \
  bash "$SCRIPT_DIR/run-pg-sublet.sh" "$TRACE"

# run-pg-sublet.sh prints what the domain said and judges the teardown
# identity; a gate has to judge the rest too.
for marker in __CAPSTONE_PG_SUBLET_ONE_EACH__ __CAPSTONE_PG_REPLAY_BALANCED__; do
  if ! grep -q "$marker" "$GATE/domain.log"; then
    echo "the domain did not report $marker; $GATE/domain.log has the table" >&2
    exit 1
  fi
done
if grep -q '__CAPSTONE_PG_REPLAY_FAILED__' "$GATE/domain.log"; then
  echo "the domain stopped early; $GATE/domain.log says where" >&2
  exit 1
fi

echo
echo "the level below keeps every claim, a teardown is one revocation, and"
echo "every object the manager freed held what was written into it"
