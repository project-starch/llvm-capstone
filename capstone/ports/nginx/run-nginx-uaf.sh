#!/usr/bin/env bash
# Does a pointer into a destroyed pool still work? Sixteen runs, and the first answer is the DIFFERENCE
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

# WHY STAGE 7's MARK CARRIES THREE NIBBLES. The middle one is what the nested handle reports
# after its ancestor revoked, and on its own that is a number. The low one is what a slot known to
# hold nothing reports, measured in the same run. They agree at 7, which is outside the valid type
# range of 0 to 5, so the ancestor's revoke left the nested handle holding no capability. The high
# nibble is 2, CAP_TYPE_REV, which is what it was while it was alive.

# WHY STAGE 9 IS A FAULT AND WHAT IT NEEDS. Stage 8 leaves a handle that its ancestor already
# revoked, and a block that somebody else now owns and has written. Stage 9 offers that handle back,
# which is the operation a double free is made of: had it revoked, it would have taken the new
# owner's object with it. It faults instead, cause 24, in sublet_give_to.
#
# That cell needs capstone-qemu PR #4. Before it, helper_csrevoke ASSERTED on an untagged operand
# rather than raising, so the emulator aborted and the monitor never ran: a correctly caught double
# free and a dead emulator read exactly alike. The cell below says so by name when it meets that,
# instead of reporting a wrong answer, because an emulator that cannot deliver the exception is not
# a port that failed to raise it.
#
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
  # A GUEST THAT NEVER STARTED IS NOT A WRONG ANSWER, and the two must not read alike. This gate
  # boots sixteen times and the emulator occasionally does not reach the shell, which shows up as
  # "no return" with no domain output at all. That cell is retried ONCE and says so. A domain that
  # answered, and answered differently than expected, is never retried: that is the result.
  if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q 'no return'; then
    printf "%-7s stop %d  expect %-7s  the guest did not start, retrying once\n" "$arm" "$stop" "$want"
    if [ "$want" = FAULT ]; then
      out=$(NGX_DOMAIN=uaf NGX_SUBLET=$sub NGX_EXPECT_FAULT=1 DOM_NAME=ngx-uaf-$arm \
            EXTRA_CFLAGS="-DNGX_UAF_STOP=$stop" bash "$SCRIPT_DIR/run-nginx-domain.sh" 2>&1); rc=$?
    else
      out=$(NGX_DOMAIN=uaf NGX_SUBLET=$sub NGX_EXPECT_MARK=$((16#$want)) DOM_NAME=ngx-uaf-$arm \
            EXTRA_CFLAGS="-DNGX_UAF_STOP=$stop" bash "$SCRIPT_DIR/run-nginx-domain.sh" 2>&1); rc=$?
    fi
  fi

  # AN EMULATOR THAT CANNOT DELIVER THE EXCEPTION IS NOT A PORT THAT FAILED TO RAISE IT, and the
  # two would otherwise read alike here. capstone-qemu PR #4 makes REVOKE raise 24; without it the
  # helper asserts, which kills the machine before the monitor can report anything.
  if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q 'helper_csrevoke: Assertion'; then
    printf "%-7s stop %d  expect %-7s  the emulator aborts on REVOKE, needs capstone-qemu PR #4\n" \
        "$arm" "$stop" "$want"
    return
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
run plain  7 C7FF00      # no nested authority exists on this arm, and the mark says so
run sublet 4 C40001
run sublet 6 C600FB      # the SAME answer as the plain arm, and that is the result: see below
run sublet 7 C70277      # REV before, and afterwards what an EMPTY slot reports, both 7
run sublet 8 C80071      # the stale handle is empty AND the block has a new owner who is fine
run sublet 1 C10000
run sublet 2 C20000
run sublet 9 FAULT       # the stale handle offered back: see the paragraph above the table
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
