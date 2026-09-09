#!/usr/bin/env bash
# Run the reproductions against a STOCK MicroPython host build at our pinned commit.
#
# Why the host and not the domain: this answers "is the defect still in the source
# we compile", which is a prerequisite for anything on Capstone and costs seconds
# rather than a firmware rebuild. A row that does not reproduce here cannot
# reproduce on silicon, and finding that out on the host is free.
#
# Build the binary first, from a CLEAN worktree, so the Capstone portability
# patches do not change what is being measured:
#
#   git -C $CAPSTONE_TMP_ROOT/micropython worktree add /tmp/capstone/mpy-stock-pin <pin>
#   make -C /tmp/capstone/mpy-stock-pin/mpy-cross -j16
#   make -C /tmp/capstone/mpy-stock-pin/ports/unix -j16
set -uo pipefail

MP=${MP:-/tmp/capstone/mpy-stock-pin/ports/unix/build-standard/micropython}
HERE=$(cd -- "$(dirname -- "$0")" && pwd)

if [[ ! -x "$MP" ]]; then
    echo "no stock binary at $MP -- see the header of this script" >&2
    exit 2
fi

echo "interpreter: $("$MP" -c 'import sys; print(sys.version)')"
echo

rc_overall=0
for f in "$HERE"/t*.py; do
    name=$(basename "$f" .py)
    out=$(timeout 30 "$MP" "$f" 2>&1)
    rc=$?
    if [[ $rc -ge 128 ]]; then
        verdict="CRASH (signal $((rc - 128)))"
    elif [[ $rc -eq 124 ]]; then
        verdict="TIMEOUT"
    else
        verdict="no crash (exit $rc): $(tail -1 <<<"$out" | cut -c1-58)"
    fi
    printf '  %-34s %s\n' "$name" "$verdict"
done

# A "no crash" result is not the same as "defect absent". This probe is the
# positive control for that distinction: it must report DANGLING, and if it ever
# reports IN-PLACE then MPY-T09 has genuinely been fixed and the row is stale.
echo
echo "silent-defect probe:"
proof=$(timeout 30 "$MP" "$HERE/stale-view-proof.py" 2>&1)
echo "  $proof"
if ! grep -q DANGLING <<<"$proof"; then
    echo "  ^ expected DANGLING; MPY-T09 may have been fixed upstream, re-check the row" >&2
    rc_overall=1
fi
exit $rc_overall
