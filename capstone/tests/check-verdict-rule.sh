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

# Second check, same idea one level down: a `source` line whose path does not
# resolve. bash does not stop for it without -e, so the helper is simply absent
# and every call lands in its `||` branch -- which reads like "nothing selected"
# rather than like a broken script. That happened while adding select.sh: a regex
# escape leaked into the path and all seven rv8 benchmarks silently SKIPped.
badsrc=0
while IFS= read -r f; do
  dir=$(dirname "$f")
  while IFS= read -r ref; do
    resolved=${ref//\$SCRIPT_DIR/$dir}
    case "$resolved" in
      *'$'*) continue ;;   # another variable in the path: not resolvable here
      *.sh) ;;             # only judge things that name a shell file
      *) continue ;;       # not a source path (the grep pattern matches itself)
    esac
    if [ ! -f "$resolved" ]; then
      printf '  BAD SOURCE  %s -> %s\n' "${f##*/}" "$ref"
      badsrc=$((badsrc + 1)); rc=1
    fi
  done < <(grep -oE 'source "[^"]+"' "$f" 2>/dev/null | sed 's/^source "//; s/"$//')
done < <(find "$SCRIPT_DIR" "$SCRIPT_DIR/../benchmarks" -name '*.sh' -type f 2>/dev/null)

echo
[ "$badsrc" -eq 0 ] || echo "$badsrc source line(s) point at a file that does not exist." >&2
echo "$uses suite(s) use the shared verdict rule, $missing do not."
[ "$rc" -eq 0 ] || echo "A suite without it can report a boot flake as a FAIL." >&2
exit "$rc"
