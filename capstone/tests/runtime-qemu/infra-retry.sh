# Retry a guest run that failed for infrastructure reasons, and only for those.
#
# 75 is the shared infra-flake exit code: the guest never reached login, so
# nothing about the domain was measured. At the flake rate this setup actually
# has -- about 29% per boot, from the nightly logs -- a suite that boots once
# and gives up returns no result roughly every third run, which is how
# run-smoke.sh and run-coremark.sh have been behaving.
#
# Any other exit code is a RESULT and is returned unchanged. That is the whole
# contract: this must never be able to turn a failure into a pass, only to stop
# an infra flake from being mistaken for one.
#
# Source it; do not execute it.

capstone_retry_infra_flake() {  # "$@" = the command to run
  local rc attempt=0 max=${CAPSTONE_INFRA_RETRIES:-3} errexit=0
  # Restore the CALLER's errexit rather than forcing it on. Forcing it made a
  # non-zero return abort a caller that had deliberately turned it off, which is
  # the opposite of leaving a real result alone.
  case $- in *e*) errexit=1 ;; esac
  while :; do
    attempt=$((attempt + 1))
    set +e
    "$@"
    rc=$?
    [ "$errexit" -eq 1 ] && set -e
    if [ "$rc" -ne 75 ] || [ "$attempt" -ge "$max" ]; then
      return "$rc"
    fi
    echo "  ...infra flake (attempt $attempt of $max), retrying" >&2
  done
}

# Did the guest actually RUN? A suite may report a result only if it did.
#
# The absence of a success marker means nothing on its own: a guest that never
# reached login leaves exactly the same absence as a domain that ran and gave
# the wrong answer. Four suites have called the first one a FAIL, which is how a
# boot flake gets published as a capability defect.
#
# The discriminator is any line carrying the suite's own prefix. Domains print
# progress before their final marker, so one that ran and then faulted still
# leaves output; one that never booted leaves none.
capstone_domain_ran() {  # $1 = log file  $2 = the prefix the domain prints
  [ -f "$1" ] && grep -q -- "$2" "$1" 2>/dev/null
}

# Exhausted the retries: 75 if the guest never ran, otherwise the caller's code.
capstone_verdict_or_flake() {  # $1 = log  $2 = prefix  $3 = exit code for a real result
  if capstone_domain_ran "$1" "$2"; then
    return "$3"
  fi
  echo "  ...no domain output in $1 -- the guest never ran, so there is no verdict" >&2
  return 75
}
