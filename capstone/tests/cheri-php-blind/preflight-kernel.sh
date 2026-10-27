#!/usr/bin/env bash
# Verify, BEFORE building a disk image or booting anything, that the CheriBSD
# release we are about to use actually supports capability revocation. Without
# it the `temporal` and `eager` arms measure nothing and every MISS they report
# is an artefact of the kernel, not a property of CHERI.
#
# provision-cheri-vehicle.sh carries a comment saying the knobs were verified --
# but that was verified against a different release, so this re-checks the
# kernel actually on disk.
#
# Checked by reading the kernel's string table. Note: the sysctl path components
# are stored separately, so look for the LEAF names, not "security.cheri.*".
set -uo pipefail
CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri}
KTXZ=${KTXZ:-$CHERI_ROOT/dist/kernel.txz}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/kpreflight-XXXX")
trap 'rm -rf "$WORK"' EXIT

[ -f "$KTXZ" ] || { echo "MISSING kernel set: $KTXZ"; exit 2; }
tar -xJf "$KTXZ" -C "$WORK" 2>/dev/null
K=$(find "$WORK" -name kernel -type f | head -1)
[ -n "$K" ] || { echo "no kernel found inside $KTXZ"; exit 2; }

# Materialise the string table ONCE. Do not pipe into `grep -q`: under
# `set -o pipefail` grep -q exits on the first match, the writer dies of SIGPIPE
# (141), and pipefail reports the pipeline as FAILED -- so a present symbol
# reads as absent. That is exactly how this check first reported all three
# revocation knobs missing when all three were present.
STR="$WORK/strings.txt"
strings -a "$K" > "$STR"
echo "[*] kernel: $(basename "$KTXZ")  strings=$(wc -l < "$STR")"
echo "[*] ident:  $(grep -m1 -E '^FreeBSD [0-9]' "$STR")"
echo "[*] config: $(grep -m1 -xE 'CHERI-[A-Z-]+' "$STR")"

fail=0
need() {  # leaf-name
  local n
  n=$(grep -cxF -- "$1" "$STR")
  if [ "$n" -ge 1 ]; then
    printf '  PRESENT  %s\n' "$1"
  else
    printf '  MISSING  %s\n' "$1"; fail=1
  fi
}
echo "[*] revocation sysctl leaves:"
need runtime_revocation_default
need runtime_revocation_every_free_default
need runtime_revocation_async

# Positive control for the check itself: three sysctl leaves that must exist in
# ANY FreeBSD kernel. If these come back MISSING the check is broken, not the
# kernel, and the result above means nothing.
echo "[*] positive control (must all be PRESENT, else this check is broken):"
ctl=0
for s in maxproc osrelease consmsgbuf_size; do
  if [ "$(grep -cxF -- "$s" "$STR")" -ge 1 ]; then printf '  PRESENT  %s\n' "$s"
  else printf '  MISSING  %s\n' "$s"; ctl=1; fi
done
[ "$ctl" -eq 0 ] || { echo "PREFLIGHT BROKEN: control symbols absent -- do not trust the result above"; exit 3; }

if [ "$fail" -eq 0 ]; then
  echo "PREFLIGHT: kernel supports revocation -- temporal and eager arms are meaningful"
else
  echo "PREFLIGHT: BLOCKED -- this kernel lacks revocation knobs; only the spatial arm would be valid"
  exit 1
fi
