#!/usr/bin/env bash
# Positive controls for run-llc-stress.py's classifier: every verdict class shown to
# fire on a known input before any fuzz result is read.  No QEMU.
#
#   OK      a module that compiles (o1-shapes.ll)
#   CRASH   a live crasher (F-02's reduced module, -O2), matched to its KNOWN entry.
#           C-20's cttz reproducer served until its fix landed on 2026-09-04; a
#           positive control has to be a bug that is still open.
#   ERROR   the C-17 wide-constant arm, matched to its KNOWN entry
#   HANG    a compiling module under a 1 ms timeout
#   UNKNOWN a crash NOT in known-signatures.txt makes the run exit 1
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
T="$CAPSTONE_REPO_ROOT/llvm/test/CodeGen/Capstone"
R="$CAPSTONE_REPO_ROOT/capstone/tests/compiler-repros"
W=$(mktemp -d "${CAPSTONE_TMP_ROOT:-/tmp}/fuzz-check.XXXXXX")
export CAPSTONE_LLVM_BIN

fail=0
expect() {  # label, expected rc, needle in stdout, args...
  local label=$1 want=$2 needle=$3; shift 3
  local out rc=0
  out=$(python3 "$SCRIPT_DIR/run-llc-stress.py" --out "$W/$label" "$@" 2>&1) || rc=$?
  if [[ "$rc" == "$want" && "$out" == *"$needle"* ]]; then echo "  ok   $label (rc=$rc, '$needle')"
  else echo "  FAIL $label: expected rc=$want with '$needle', got rc=$rc"; echo "$out" | sed 's/^/       /'; fail=1; fi
}

# The C-17 arm is a split-file test; extract the wide part into a plain module.
python3 - "$T/c17-wide-constant-arm.ll" "$W/wide.ll" <<'EOF'
import sys
src = open(sys.argv[1]).read()
part = src.split(";--- wide.ll\n")[1].split(";--- ")[0]
open(sys.argv[2], "w").write(part)
EOF

echo "== run-llc-stress positive controls"
expect "OK"      0 "OK, 0 skipped, 0 distinct"  --files "$T/o1-shapes.ll" --levels O0,O2
expect "CRASH"   0 "KNOWN F-02"   --files "$SCRIPT_DIR/findings/F02-vector-elt-load-recreated/reduced.ll" --levels O2
expect "ERROR"   0 "KNOWN C-17"   --files "$W/wide.ll" --levels O2
expect "HANG"    1 "HANG"         --files "$T/o1-shapes.ll" --levels O2 --timeout 0.001
expect "UNKNOWN" 1 "UNKNOWN"      --files "$SCRIPT_DIR/findings/F02-vector-elt-load-recreated/reduced.ll" --levels O2 --known /dev/null
expect "NOTHING" 2 "nothing was tested" --files --levels O0

rm -rf "$W"
if [[ "$fail" -eq 0 ]]; then echo "fuzz-check: ALL ARMS OK"; exit 0; fi
echo "fuzz-check: FAILED"; exit 1
