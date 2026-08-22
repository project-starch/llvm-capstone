#!/usr/bin/env bash
# Which nightly suites use the shared verdict rule, and which do not?
#
# The rule: a suite may report a RESULT only if the guest actually ran. Absence
# of a success marker means nothing on its own -- a guest that never reached
# login leaves the same absence as a domain that ran and answered wrongly. Four
# suites got that wrong, and each published a boot flake as a capability defect.
#
# This is a LINT, and its limit is worth stating: it proves the helper is
# available to a suite, not that the suite calls it at the right place. It exists
# to catch the next suite that never hears about the rule, which is how all four
# earlier instances happened.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

mapfile -t SUITES < <(grep -oE '\$(RUNTIME_DIR|SCRIPT_DIR|BENCH_DIR)/[a-zA-Z0-9/_.-]+\.sh' \
  "$SCRIPT_DIR/run-nightly.sh" | sed 's|.*/||' | sort -u)

rc=0 uses=0 missing=0
for s in "${SUITES[@]}"; do
  f=$(find "$SCRIPT_DIR" -name "$s" -print -quit 2>/dev/null)
  [ -n "$f" ] || continue
  # A suite that never boots a guest cannot get this wrong.
  grep -q "run-domain-smoke.py\|run-ladder-qemu.sh" "$f" 2>/dev/null || continue
  if grep -q "infra-retry.sh" "$f" 2>/dev/null; then
    printf '  uses rule   %s\n' "$s"; uses=$((uses + 1))
  else
    printf '  NO RULE     %s\n' "$s"; missing=$((missing + 1)); rc=1
  fi
done

echo
echo "$uses suite(s) use the shared verdict rule, $missing do not."
[ "$rc" -eq 0 ] || echo "A suite without it can report a boot flake as a FAIL." >&2
exit "$rc"
