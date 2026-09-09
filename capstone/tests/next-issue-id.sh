#!/usr/bin/env bash
# Allocate the next free issue ID -- I-02.
#
#   usage: bash capstone/tests/next-issue-id.sh <PREFIX>      e.g. R, C, S, Q, M, I, F
#          bash capstone/tests/next-issue-id.sh --audit       list every allocated-but-unregistered ID
#
# WHY THIS IS A SCRIPT AND NOT A CONVENTION IN A DOC. On 2026-09-04 a lane proposed reusing C-25 for
# a new defect. C-25 was already allocated -- by the external collaborator, on the c128 line, in two
# commits whose subjects say so -- and it appears NOWHERE in ISSUES.md. So the obvious method,
# "grep the registry and take the next free number", silently hands out a number that is already in
# use, and the collision was caught by a human review rather than by any check. Two defects sharing
# one ID survives into commit messages, handovers and papers, and is expensive precisely because
# both entries look correct on their own.
#
# A committed ledger file was the other candidate and was rejected: it is a third place to keep in
# sync, and the failure mode we are fixing is exactly that two places already disagree. This derives
# the answer from the two sources of truth every time, so it cannot drift.
#
# WHAT IT CHECKS -- both, always:
#   1. the registry, BOTH halves: docs/ref/ISSUES.md (open) and docs/ref/ISSUES-ARCHIVE.md (resolved)
#   2. the full commit history of every ref, git log --all --grep, which is what would have caught C-25
# It also scans the repro-folder names, because a folder can exist before its entry does.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2
R=capstone/docs/ref/ISSUES.md
A=capstone/docs/ref/ISSUES-ARCHIVE.md
[ -f "$R" ] && [ -f "$A" ] || { echo "ERROR: registry not found at $R / $A -- wrong tree?" >&2; exit 2; }

# Registry IDs: entry HEADINGS only, so a mere mention in prose does not reserve a number.
reg_ids() { grep -hoE '^#{2,3} [A-Z]+-[0-9]+' "$R" "$A" | grep -oE '[A-Z]+-[0-9]+' | sort -u; }
# Folder IDs: fpga-repros/<ID><digits>-slug, e.g. R29-…, S06-…  (folders drop the hyphen)
fold_ids() { ls capstone/tests/fpga-repros 2>/dev/null \
             | grep -oE '^[A-Z]+[0-9]+' \
             | sed -E 's/^([A-Z]+)0*([0-9]+)$/\1-\2/' | sort -u; }
# History IDs: any <PREFIX>-<n> in a commit subject or body on ANY ref -- but SCOPED TO OUR PATHS.
# This tree is an LLVM fork, so unscoped the upstream history matches things like "C-1", "F-2" and
# "S-2496" out of ordinary release prose and the audit becomes unreadable. Scoping to capstone/ keeps
# every commit this project has written, including the external collaborator's, and drops upstream's.
# The cost is over-conservatism in the other direction only: an ID we never touched under capstone/
# might be missed, which is why --audit exists and why the high-water number is the recommended one.
hist_ids() { git log --all --format='%s%n%b' -- capstone 2>/dev/null \
             | grep -oE '\b[A-Z]+-[0-9]+\b' | sort -u; }

norm() { sed -E 's/^([A-Z]+)-0*([0-9]+)$/\1-\2/'; }

if [ "${1:-}" = "--audit" ]; then
  echo "== IDs that appear in COMMIT HISTORY but have no entry heading in the registry =="
  echo "   (each is allocated: do not hand the number out again)"
  comm -23 <(hist_ids | norm | sort -u) <(reg_ids | norm | sort -u) \
    | grep -E '^(R|C|S|Q|M|I|F)-[0-9]+$' \
    | while read -r id; do
        # Only report a gap we can EVIDENCE. A row whose commit we cannot re-find is one the body
        # matched across a line break or in quoted upstream text; printing it with a blank citation
        # is worse than not printing it, because an unciteable gap reads exactly like a real one.
        ev=$(git log --all --format='%h %s' --grep="$id" -- capstone 2>/dev/null | tail -1 | cut -c1-96)
        [ -n "$ev" ] && printf '  %-8s first seen: %s\n' "$id" "$ev"
      done
  echo
  echo "== IDs that have a repro FOLDER but no entry heading =="
  comm -23 <(fold_ids | norm | sort -u) <(reg_ids | norm | sort -u) | sed 's/^/  /'
  exit 0
fi

P=${1:-}
case "$P" in
  [A-Z]) : ;;
  *) sed -n '2,8p' "$0"; exit 2 ;;
esac

used=$( { reg_ids; fold_ids; hist_ids; } | norm | grep -E "^$P-[0-9]+$" | grep -oE '[0-9]+$' | sort -n -u )
next=1
for n in $used; do [ "$n" -eq "$next" ] && next=$((next+1)); done
# next is the lowest free number; also report the high-water mark, which is what most people want.
high=$(echo "$used" | tail -1); high=${high:-0}

echo "prefix $P"
echo "  allocated (registry + folders + commit history): $(echo "$used" | tr '\n' ' ')"
echo "  lowest FREE number : $P-$next"
echo "  next after the high-water mark : $P-$((high+1))    <- prefer this one"
echo
echo "  Take the high-water number unless you are deliberately filling a gap: a gap usually means an"
echo "  ID was allocated and its entry never written (run --audit), not that the number is spare."
