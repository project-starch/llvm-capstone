#!/bin/bash
# run-cheribsd-revocation.sh BUILD OUT: the twenty pymalloc defects under the
# platform's own libc revocation, with pymalloc stock.
#
# WHAT THIS ARM IS AND WHAT IT IS EXPECTED TO SHOW: CheriBSD's libc revocation
# exactly as the platform ships it, which is the arm the allocator-boundary
# corpus uses, so the two corpora are comparable on it.
#
# The expected result is that NOTHING is caught, and that is the point. libc
# revocation acts on libc's own free; pymalloc does not return a freed block to
# libc, it links it onto its own pool free list, so there is no free for the
# platform to revoke. A 0/20 here is the measurement that says the nested
# allocator makes the platform's defence inert. If instead something IS caught,
# that is a finding and the reason has to be established before the number is
# used.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORPUS=$(cd -- "$HERE/../.." && pwd)
BUILD=${1:?usage: run-cheribsd-revocation.sh BUILD OUT}
OUT=${2:?usage: run-cheribsd-revocation.sh BUILD OUT}
PORT_G=${GUEST_PORT:-10086}
SICODE=${SICODE:-/root/sicode.so}
BUDGET=${CASE_BUDGET:-120}
K="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=120"
G() { ssh $K -p "$PORT_G" root@localhost "$@" 2>/dev/null; }
P() { scp -q $K -P "$PORT_G" "$1" root@localhost:"$2" 2>/dev/null; }

n=$(ls "$BUILD"/bin/defect-?? 2>/dev/null | wc -l)
[ "$n" -gt 0 ] || { echo "no $BUILD/bin/defect-NN -- build first:" >&2
  echo "  bash $CORPUS/shared/build-cases.sh cheribsd $BUILD" >&2
  exit 2; }
mkdir -p "$OUT"

G true || { echo "guest on port $PORT_G is not reachable" >&2; exit 2; }
G "sysctl security.cheri 2>/dev/null" > "$OUT/sysctl-before.txt"
rev=$(awk -F': ' '/runtime_revocation_default/{print $2}' "$OUT/sysctl-before.txt")
[ "$rev" = 1 ] || { echo "REFUSING: runtime_revocation_default is '$rev', not 1." >&2
  echo "  This arm IS the platform's revocation; scoring it with revocation off" >&2
  echo "  would record the platform default as a result about the defects." >&2; exit 2; }
ef=$(awk -F': ' '/runtime_revocation_every_free_default/{print $2}' "$OUT/sysctl-before.txt")
echo "revocation_default=$rev every_free_default=${ef:-unrecorded}"
printf 'arm\tcheribsd-revocation\nrevocation_default\t%s\nevery_free_default\t%s\nposioncap\toff\nstarted_utc\t%s\n' \
  "$rev" "${ef:-unrecorded}" "$(date -u +%FT%TZ)" > "$OUT/run.meta"

G "test -f $SICODE" || { echo "no $SICODE in the guest -- faults would be unclassifiable" >&2; exit 2; }
G "rm -rf /root/pymrepro && mkdir -p /root/pymrepro"
for b in "$BUILD"/bin/defect-??; do
  P "$b" "/root/pymrepro/$(basename "$b")"
  G "chmod 755 /root/pymrepro/$(basename "$b")"
  G "test -x /root/pymrepro/$(basename "$b")" || { echo "$(basename "$b"): staging failed" >&2; exit 2; }
done
echo "staged $n binaries"

printf 'case\tarm\tverdict\trc\tsi_code\tlast\n' > "$OUT/verdicts.tsv"
for b in "$BUILD"/bin/defect-??; do
  c=$(basename "$b" | sed 's/^defect-//')
  out=$(G "cd /root/pymrepro && env LD_PRELOAD=$SICODE timeout $BUDGET ./defect-$c 2>&1; echo RC=\$?")
  rc=$(printf '%s' "$out" | grep -oE 'RC=[0-9]+' | tail -1 | cut -d= -f2)
  sic=$(printf '%s' "$out" | grep -oE 'si_code=[0-9]+ \([A-Z_]+\)' | head -1)
  last=$(printf '%s' "$out" | grep -v '^RC=' | tail -2 | tr '\n' ' ')
  if [ -n "$sic" ];      then v=DETECTED
  elif [ "$rc" = 124 ];  then v=TIMEOUT
  elif [ "$rc" = 0 ];    then v=SILENT
  else                        v="OTHER-rc$rc"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$c" cheribsd-revocation "$v" "${rc:-?}" "${sic:-}" "$last" \
    >> "$OUT/verdicts.tsv"
  printf '  defect-%-4s %-10s rc=%-5s %s\n' "$c" "$v" "${rc:-?}" "${sic:-}"
done
G "sysctl security.cheri 2>/dev/null" > "$OUT/sysctl-after.txt"
diff -q <(sed 's/^ *//' "$OUT/sysctl-before.txt") <(sed 's/^ *//' "$OUT/sysctl-after.txt") >/dev/null \
  && echo "security.cheri unchanged across the run" \
  || { echo "WARNING: security.cheri changed during the run" >&2
       diff "$OUT/sysctl-before.txt" "$OUT/sysctl-after.txt" >&2; }
printf 'ended_utc\t%s\ncases\t%s\n' "$(date -u +%FT%TZ)" "$n" >> "$OUT/run.meta"
echo "rows: $(($(wc -l < "$OUT/verdicts.tsv")-1)) / $n"
echo "out:  $OUT"
