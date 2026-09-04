#!/usr/bin/env bash
# Positive controls for compare-twins.py -- every verdict class shown to fire, with the
# exact exit code, before any twin result is read through it.  No QEMU.
#
# Arms (rv8 format unless noted):
#   AGREE          identical summaries                          -> rc 0
#   B-ONLY-FAIL    one PASS flipped to FAIL on side B           -> rc 1
#   A-ONLY-FAIL    the same flip on side A                      -> rc 1
#   BOTH-FAIL      the same benchmark FAIL on both sides        -> rc 1
#   MISSING        a benchmark line dropped from side B         -> rc 2
#   EMPTY          side B's summary is empty (suite died)       -> rc 2
#   FLAKE (beebs)  a FLAKE line on side A                       -> rc 2
#   coremark       marker present vs absent                     -> rc 1
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W=$(mktemp -d "${CAPSTONE_TMP_ROOT:-/tmp}/twins-check.XXXXXX")
CT="$SCRIPT_DIR/compare-twins.py"

# FAIL lines carry the runner's real trailing log pointer (run-all-rv8.sh:37).
printf 'PASS  dhrystone\nPASS  qsort\nFAIL  sha512   (see /x/logs/sha512.attempt-*.log)\nPASS  aes\nSKIP  norx\nrun-all-rv8: 3 passed, 1 failed.\n' > "$W/a.txt"
cp "$W/a.txt" "$W/same.txt"
sed 's|^PASS  qsort$|FAIL  qsort   (see /x/logs/qsort.attempt-*.log)|' "$W/a.txt" > "$W/flip.txt"
grep -v '^PASS  aes$' "$W/a.txt" > "$W/drop.txt"
: > "$W/empty.txt"
printf 'run-all-beebs.sh: PASS bs (__BEEBS_BS_PASSED__, log=x)\nrun-all-beebs.sh: FLAKE crc32 (never booted after 3 attempts)\n' > "$W/bfl.txt"
printf 'run-all-beebs.sh: PASS bs (log=x)\nrun-all-beebs.sh: PASS (retried) crc32 (log=y)\n' > "$W/bok.txt"
printf 'run-coremark.sh: CoreMark CRC validated\n__COREMARK_PASSED__\n' > "$W/cm-ok.txt"
printf 'run-coremark.sh: boot failed\n' > "$W/cm-bad.txt"

fail=0
arm() {  # label, expected rc, expected verdict substring, args...
  local label=$1 want=$2 needle=$3; shift 3
  local out rc=0
  out=$(python3 "$CT" "$@" 2>&1) || rc=$?
  if [[ "$rc" == "$want" && "$out" == *"$needle"* ]]; then echo "  ok   $label (rc=$rc, $needle)"
  else echo "  FAIL $label: expected rc=$want containing '$needle', got rc=$rc"; echo "$out" | sed 's/^/       /'; fail=1; fi
}

echo "== compare-twins positive controls"
arm "AGREE with a shared BOTH-FAIL counted"  1 "BOTH-FAIL"    --suite rv8 --a "$W/a.txt" --b "$W/same.txt" --label-a O0 --label-b O2
printf 'PASS  dhrystone\nPASS  qsort\nPASS  aes\n' > "$W/bok-rv8.txt"
arm "all AGREE-PASS exits 0"                 0 "3/3 AGREE-PASS" --suite rv8 --a "$W/bok-rv8.txt" --b "$W/bok-rv8.txt" --label-a O0 --label-b O2
arm "B-ONLY-FAIL"                            1 "O2-ONLY-FAIL"  --suite rv8 --a "$W/a.txt" --b "$W/flip.txt" --label-a O0 --label-b O2
arm "A-ONLY-FAIL"                            1 "O0-ONLY-FAIL"  --suite rv8 --a "$W/flip.txt" --b "$W/a.txt" --label-a O0 --label-b O2
arm "MISSING"                                2 "MISSING"       --suite rv8 --a "$W/a.txt" --b "$W/drop.txt" --label-a O0 --label-b O2
arm "EMPTY summary is an ERROR"              2 "no verdicts"   --suite rv8 --a "$W/a.txt" --b "$W/empty.txt" --label-a O0 --label-b O2
arm "beebs FLAKE is an ERROR"                2 "FLAKE"         --suite beebs --a "$W/bfl.txt" --b "$W/bok.txt" --label-a O0 --label-b O2
arm "beebs retried PASS parses"              0 "2/2 AGREE-PASS" --suite beebs --a "$W/bok.txt" --b "$W/bok.txt" --label-a O0 --label-b O2
arm "coremark marker decides"                1 "O2-ONLY-FAIL"  --suite coremark --a "$W/cm-ok.txt" --b "$W/cm-bad.txt" --label-a O0 --label-b O2

rm -rf "$W"
if [[ "$fail" -eq 0 ]]; then echo "twins-check: ALL ARMS OK"; exit 0; fi
echo "twins-check: FAILED"; exit 1
