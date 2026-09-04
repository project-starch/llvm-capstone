#!/usr/bin/env bash
# Does the verdict rule actually FIRE? The lint next door proves the helper is
# sourced; it cannot prove a suite calls it in the right place. This runs each
# suite twice against a stubbed guest and demands both answers:
#
#   guest never booted        -> 75 (FLAKE)   never 1, or an infra flake gets
#                                             published as a capability defect
#   domain ran, answered bad  ->  1 (FAIL)    never 75, or the suite can no
#                                             longer fail at all
#
# The second direction is the one worth having. Nine suites were changed to stop
# calling a flake a failure, and a change like that is one sign flip away from a
# suite that always passes.
#
# Slow (it runs every suite's build step twice). Run it when the rule changes,
# not in the nightly.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STUB=$(mktemp -d)
trap 'rm -rf "$STUB"' EXIT

cat > "$STUB/python3" <<'STUBEOF'
#!/bin/bash
# Intercept ONLY the guest run. Builds and log parsing go to the real python3.
is_smoke=0; logfile=""; prev=""
for a in "$@"; do
  case "$a" in *run-domain-smoke.py) is_smoke=1 ;; esac
  case "$prev" in --log-file) logfile="$a" ;; esac
  case "$a" in --log-file=*) logfile="${a#--log-file=}" ;; esac
  prev="$a"
done
if [ "$is_smoke" -eq 1 ]; then
  # STUB_RAN=1: leave the progress lines a domain that really ran would leave,
  # so the discriminator has something to find.
  if [ "${STUB_RAN:-0}" = 1 ] && [ -n "$logfile" ]; then
    mkdir -p "$(dirname "$logfile")"
    # The prefixes come from the suites themselves (STUB_PREFIXES). Maintaining
    # this list by hand already produced a false BROKEN: the list said
    # intra-domain-mrev-probe, the suite greps intra-domain-mrev-revoke-probe,
    # and the mismatch read exactly like a suite that could no longer fail.
    for p in $STUB_PREFIXES; do
      echo "${p} call retval = 0xbadbad00"
      echo "${p} progress"
    done > "$logfile"
    echo "Called dom (1-th time) retval = 1" >> "$logfile"
  fi
  exit "${STUB_RC:-75}"
fi
exec /usr/bin/python3 "$@"
STUBEOF
chmod +x "$STUB/python3"

# Every discriminator prefix any suite actually greps for. Derived, never typed.
mapfile -t PREFIX_LIST < <(
  {
    grep -rhoE 'capstone_domain_ran "[^"]*" "[^"]*"' "$SCRIPT_DIR" 2>/dev/null |
      sed -E 's/.*"([^"]*)"$/\1/'
    grep -rhoE 'capstone_verdict_or_flake "[^"]*" "[^"]*"' "$SCRIPT_DIR" 2>/dev/null |
      sed -E 's/.*"([^"]*)"$/\1/'
    grep -rhoE 'grep -q "[a-z][a-z0-9_-]*(-probe|-cb)?:' "$SCRIPT_DIR" 2>/dev/null |
      sed -E 's/^grep -q "//'
  } | sed -E 's/[[:space:]]+$//' | sort -u | grep -E ':$'
)
if [ "${#PREFIX_LIST[@]}" -eq 0 ]; then
  echo "no discriminator prefixes found -- the stub would prove nothing" >&2
  exit 2
fi
export STUB_PREFIXES="${PREFIX_LIST[*]}"

mapfile -t SUITES < <(grep -oE '\$(RUNTIME_DIR|SCRIPT_DIR|BENCH_DIR)/[a-zA-Z0-9/_.-]+\.sh' \
  "$SCRIPT_DIR/run-nightly.sh" | sed 's|.*/||' | sort -u)

one() { # $1=path  $2=STUB_RC  $3=STUB_RAN  -> prints the suite's exit code
  env -u CAPSTONE_REPO_ROOT PATH="$STUB:$PATH" \
      STUB_RC="$2" STUB_RAN="$3" STUB_PREFIXES="$STUB_PREFIXES" \
      CAPSTONE_INFRA_RETRIES=2 RETRIES=1 \
      timeout "${VERDICT_FIRE_TIMEOUT:-1200}" bash "$1" >/dev/null 2>&1
  echo $?
}

rc=0 checked=0
for s in "${SUITES[@]}"; do
  f=$(find "$SCRIPT_DIR" -name "$s" -print -quit 2>/dev/null)
  [ -n "$f" ] || continue
  grep -q "run-domain-smoke.py\|run-ladder-qemu.sh" "$f" 2>/dev/null || continue
  checked=$((checked + 1))
  flake=$(one "$f" 75 0)
  real=$(one "$f" 1 1)
  if [ "$flake" = 75 ] && [ "$real" = 1 ]; then
    printf '  ok        %-46s flake=75 fail=1\n' "$s"
  else
    rc=1
    printf '  BROKEN    %-46s flake=%s (want 75)  fail=%s (want 1)\n' "$s" "$flake" "$real"
  fi
done

echo
if [ "$checked" -eq 0 ]; then
  echo "no guest-booting suite found -- this check measured nothing" >&2
  exit 2   # never a silent pass: no data is an error, not a zero
fi
[ "$rc" -eq 0 ] && echo "$checked suite(s): the verdict rule fires in both directions." \
               || echo "$checked suite(s) checked; some cannot distinguish a flake from a failure." >&2
exit "$rc"
