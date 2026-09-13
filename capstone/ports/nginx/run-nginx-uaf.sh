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
run sublet 1 C10000
run sublet 2 C20000
run sublet 3 FAULT

# Where the fault landed. A fault anywhere else would pass the line above and mean nothing.
pc=$(grep -m1 -oE 'pc = 0x[0-9a-f]+' "$OUT_DIR/boot.log" | grep -oE '0x[0-9a-f]+')
sym=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d --no-show-raw-insn "$OUT_DIR/ngx-uaf-sublet.dom" |
      "${PYTHON:-python3}" -c '
import re, sys
pc = int(sys.argv[1], 16)
link = 0x10000 + (pc - 0x101580000)
best = None
for line in sys.stdin:
    m = re.match(r"^([0-9a-f]+) <(.+)>:", line)
    if m and not m.group(2).startswith(".L"):
        a = int(m.group(1), 16)
        if a <= link: best = (a, m.group(2))
        else: break
print("%s+0x%x" % (best[1], link - best[0]) if best else "unknown")' "$pc")
printf "fault at %s in %s\n" "$pc" "$sym"
case "$sym" in domain_main*) ;; *) echo "the fault is not at the touch" >&2; fail=1 ;; esac

exit $fail
