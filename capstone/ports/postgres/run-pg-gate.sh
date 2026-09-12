#!/usr/bin/env bash
# Does PostgreSQL's memory manager still run under capabilities? The gate.
#
#   run-pg-gate.sh
#
# The measurements replay a recording of a real backend, and that file is large
# and is not in this repository. A gate is a different thing. It has to run
# without one, be the same every time, and fail when something changes, so it
# generates its own trace and checks four things that a change would break.
#
#   1  the manager still compiles for capstone64, and the census still reports
#      the same two source lines and the same libc symbols
#   2  the host build asks the level below for exactly the blocks a known-good
#      build asked for, which pins the size-class ladder and the doubling
#   3  the domain build replays the same trace under capabilities, balanced
#   4  every object the trace frees holds what was written into it, on both
#      sides, which is the contract palloc makes
#
# Exit 0 if all four hold, non-zero otherwise, as the nightly expects.
# tools/make-fixture-trace.py holds the workload and the golden block counts.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT=${OUT:-$CAPSTONE_TMP_ROOT/pg-mmgr-host}
GATE=${GATE:-$CAPSTONE_TMP_ROOT/pg-gate}
PYTHON=${PYTHON:-python3}

mkdir -p "$GATE"
TRACE=$GATE/fixture.a11

echo "== the workload, generated so the gate needs no recording"
"$PYTHON" "$SCRIPT_DIR/tools/make-fixture-trace.py" "$TRACE"

echo
echo "== 1. the census: does the manager still compile for capstone64"
OUT="$OUT" bash "$SCRIPT_DIR/census-capstone.sh"

echo
echo "== 2 and 4. the host build, against the golden block counts"
# build-mmgr-host.sh exits non-zero when the manager asks for something else,
# and its replay checks every object's contents on the way.
OUT="$OUT" bash "$SCRIPT_DIR/build-mmgr-host.sh" "$TRACE"

echo
echo "== 3 and 4. the same trace in a Capstone domain"
OUT="$OUT" SHARE_DIR="$GATE/share" LOG_FILE="$GATE/domain.log" \
  bash "$SCRIPT_DIR/run-pg-replay.sh" "$TRACE"

# run-pg-replay.sh prints what the domain said but does not judge it, because
# a measurement run wants the numbers whatever they are. A gate has to judge.
if ! grep -q '__CAPSTONE_PG_REPLAY_BALANCED__' "$GATE/domain.log"; then
  echo "the domain did not give back what it took; $GATE/domain.log has the table" >&2
  exit 1
fi
if grep -q '__CAPSTONE_PG_REPLAY_FAILED__' "$GATE/domain.log"; then
  echo "the domain stopped early; $GATE/domain.log says where" >&2
  exit 1
fi

echo
echo "the manager compiles, asks for the same blocks, replays in a domain,"
echo "and every object it freed held what was written into it"
