#!/usr/bin/env bash
# Negative test for the verdict logic in repro-lib.sh. Needs no board, no QEMU and no
# build: it runs check_row against a fixed results.tsv and requires it to REFUSE the
# wrong answers.
#
# A gate that has never rejected anything is not a passing gate, it is an unproven
# one, and this project has published wrong findings behind exactly that. Run this
# after touching repro-lib.sh.
set -uo pipefail
source "$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-lib.sh"
REPRO_OUT_DIR=$(git rev-parse --show-toplevel)/capstone/benchmarks/micropython/repro-selftest

fail=0
expect() {  # $1 = expected exit, $2 = label, rest = check_row args
  local want=$1 label=$2; shift 2
  ( check_row "$@" ) >/tmp/capstone/repro-selftest.out 2>&1
  local got=$?
  if [[ $got == "$want" ]]; then
    printf "  ok    %-42s exit %s\n" "$label" "$got"
  else
    printf "  FAIL  %-42s exit %s, wanted %s\n" "$label" "$got" "$want"
    sed 's/^/          /' /tmp/capstone/repro-selftest.out; fail=1
  fi
}

expect 0  "correct status and retval"       00_sanity.py  PASS  0x00077724
expect 0  "correct status, retval unchecked" 00_sanity.py PASS  -
expect 1  "wrong status is refused"          00_sanity.py FAULT -
expect 1  "wrong retval is refused"          00_sanity.py PASS  0xdeadbeef
expect 1  "a FAULT row claimed as PASS"      01_example.py PASS -
expect 75 "a missing row is VOID, not a pass" 99_absent.py PASS -

if [[ $fail == 0 ]]; then echo "repro-selftest: all six behaved"; else echo "repro-selftest: FAILED"; fi
exit $fail
