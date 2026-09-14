#!/usr/bin/env bash
# Does a pointer into a destroyed pool still work? Six runs, and the answer is the DIFFERENCE
# between two of them. A fault on its own would prove nothing, because a port can fault for any
# number of reasons, so the same driver runs over both levels below and they have to disagree.
#
#   stop 1  the pool is created, the object written and read back   both arms return C10000
#   stop 2  the pool is destroyed, nothing touched                  both arms return C20000
#   stop 3  the object is touched after the destroy                 plain returns C300A0,
#                                                                   the byte it was given
#                                                                   sublet FAULTS, and that is
#                                                                   the result
#
# WHY STAGE 6 EXPECTS THE SAME MARK ON BOTH ARMS, which is a result and not a gap. Revocation
# catches a DEREFERENCE. ngx_pfree walks the large list comparing the pointer it was given against
# each entry and never reads through it, so offering it a revoked capability is answered
# NGX_DECLINED on both arms. Stage 5 reads through the same pointer one line earlier and the
# protected arm halts. The pair is the bound on the claim: a stale free is caught when the
# allocator touches the object, and nginx's does not. An allocator whose free reads a header
# behind the pointer would be caught, and that is the vehicle this row still wants.
#
# The first version of stage 6 gave pool2 no large allocation, so the loop ran zero times and the
# pointer was never even compared. Both arms answered NGX_DECLINED then too, for a different and
# much weaker reason, and the stage proved nothing.

# The expectations are exact. "The domain came back" is not one of them: the question is which of
# three marks came back. And the fault has to be AT THE TOUCH, so its pc is mapped back to a
# function rather than counted.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
source "$REPO/capstone/tests/capstone-test-env.sh"
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/nginx-domain}

fail=0
run() {   # arm stop expectation
  local arm=$1 stop=$2 want=$3 sub=0 out rc
  [ "$arm" = sublet ] && sub=1
  if [ "$want" = FAULT ]; then
    out=$(NGX_DOMAIN=uaf NGX_SUBLET=$sub NGX_EXPECT_FAULT=1 DOM_NAME=ngx-uaf-$arm \
          EXTRA_CFLAGS="-DNGX_UAF_STOP=$stop" bash "$SCRIPT_DIR/run-nginx-domain.sh" 2>&1); rc=$?
  else
    out=$(NGX_DOMAIN=uaf NGX_SUBLET=$sub NGX_EXPECT_MARK=$((16#$want)) DOM_NAME=ngx-uaf-$arm \
          EXTRA_CFLAGS="-DNGX_UAF_STOP=$stop" bash "$SCRIPT_DIR/run-nginx-domain.sh" 2>&1); rc=$?
  fi
  printf "%-7s stop %d  expect %-7s  %s\n" "$arm" "$stop" "$want" \
      "$( [ $rc -eq 0 ] && echo ok || echo WRONG )"
  [ $rc -eq 0 ] || { printf '%s\n' "$out" | tail -4 | sed 's/^/      /'; fail=1; }
}

run plain  1 C10000
run plain  2 C20000
run plain  3 C300A0      # the byte survives the destroy, which is the blindspot this paper is about
run plain  4 C40001      # the address really came back, so the stages after this are not empty
run plain  5 C5005B      # the old pointer reads the NEW object's first byte, not its own 0xA0
run plain  6 C600FB      # -5, NGX_DECLINED, from a free offered the stale pointer
run sublet 4 C40001
run sublet 6 C600FB      # the SAME answer as the plain arm, and that is the result: see below
run sublet 1 C10000
run sublet 2 C20000
run sublet 3 FAULT
run sublet 5 FAULT       # last, so the locator below reads this cell's log

# Where the fault landed. A fault anywhere else would pass the line above and mean nothing, and
# working the address out by hand does not survive an image whose load base moved: a guessed base
# once read this very fault into ngx_palloc_block+0x3fd68, an offset no function has.
# fault-locate.py takes the base from the anchor rung and refuses rather than guessing.
loc=$("${PYTHON:-python3}" "$REPO/capstone/tests/runtime-qemu/fault-locate.py" \
      "$OUT_DIR/boot.log" "$OUT_DIR/ngx-uaf-sublet.dom" 2>&1)
printf '%s\n' "$loc"
printf '%s' "$loc" | grep -q 'domain_main' || { echo "the fault is not at the touch" >&2; fail=1; }

exit $fail
