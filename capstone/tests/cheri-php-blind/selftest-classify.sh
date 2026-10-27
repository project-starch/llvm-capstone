#!/usr/bin/env bash
# Positive control for classify.py: a synthetic serial log that exercises every
# verdict branch. A classifier that has never produced a CATCH is not a
# classifier that found a MISS -- it is an unproven instrument. Run this before
# believing any real result.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
T=$(mktemp -d "${TMPDIR:-/tmp}/phpblind-selftest-XXXX")
LOG="$T/serial.log"
TSV="$T/rows.tsv"

cat > "$TSV" <<'EOF'
# synthetic
t_miss	m	23	temporal	MISS	MISS	x
t_sync	m	6	spatial	BLOCKED-SYNC	BLOCKED-SYNC	x
t_sweep_eager	m	23	temporal	BLOCKED-SWEEP	BLOCKED-SWEEP	x
t_late	m	-	control	LATE@SHUT	LATE@SHUT	x
t_abort	m	-	control	BLOCKED-ABORT	BLOCKED-ABORT	x
EOF

emit() { # row arm cfg rc last
  echo "ROW $1 arm=$2 cfg=$3 rc=$4 last=$5 out=[synthetic]"
}
{
for cfg in spatial temporal eager; do
  echo "===PHP-BLIND-BEGIN cfg=$cfg==="
  echo "KNOB runtime_revocation_default=$([ $cfg = spatial ] && echo 0 || echo 1)"
  echo "KNOB runtime_revocation_every_free_default=$([ $cfg = eager ] && echo 1 || echo 0)"
  echo "STATUS REVOKE_ENABLED=$([ $cfg = spatial ] && echo 0 || echo 1)"
  echo "--- runs (cfg=$cfg) ---"
  # survives everything, both arms
  emit t_miss stockalloc $cfg 0 DONE
  emit t_miss patched    $cfg 0 DONE
  # faults at the access with no revocation
  emit t_sync stockalloc $cfg 162 USE-BEGIN
  emit t_sync patched    $cfg 162 USE-BEGIN
  # caught at the access only under revoke-on-every-free
  if [ "$cfg" = eager ]; then
    emit t_sweep_eager stockalloc $cfg 162 USE-BEGIN
    emit t_sweep_eager patched    $cfg 162 USE-BEGIN
  else
    emit t_sweep_eager stockalloc $cfg 0 DONE
    emit t_sweep_eager patched    $cfg 0 DONE
  fi
  # faults only during the shutdown drain
  emit t_late stockalloc $cfg 162 SHUT-BEGIN
  emit t_late patched    $cfg 162 SHUT-BEGIN
  # allocator abort
  emit t_abort stockalloc $cfg 134 SHUT-BEGIN
  emit t_abort patched    $cfg 134 USE-BEGIN
  echo "===PHP-BLIND-END cfg=$cfg==="
done
} > "$LOG"

echo "=== classifier output on the synthetic log ==="
python3 -I "$SCRIPT_DIR/classify.py" "$LOG" "$TSV"
rc=$?
echo
echo "=== checks ==="
out=$(python3 -I "$SCRIPT_DIR/classify.py" "$LOG" "$TSV" 2>/dev/null)
fail=0
# NB: grep via a HERESTRING, never `cmd | grep -q`. Under `set -o pipefail`,
# grep -q exits on the first match, the writer dies of SIGPIPE (141), and
# pipefail reports the PIPELINE as failed -- so a successful match is read as a
# miss. That exact pattern made a kernel check report three revocation sysctls
# as absent when all three were present.
chk() { # description expected-substring
  if grep -q -- "$2" <<<"$out"; then echo "  PASS  $1"; else echo "  FAIL  $1 (expected to see: $2)"; fail=1; fi
}
chk "MISS is produced"                  "t_miss .*stockalloc.* MISS"
chk "BLOCKED-SYNC is produced"          "t_sync .*stockalloc.* BLOCKED-SYNC"
chk "BLOCKED-SWEEP(eager) is produced"  "t_sweep_eager .*stockalloc.* BLOCKED-SWEEP(eager)"
chk "LATE@SHUT is produced"             "t_late .*stockalloc.* LATE@SHUT"
chk "BLOCKED-ABORT is produced"         "t_abort .*stockalloc.* BLOCKED-ABORT"
chk "a shutdown-only fault is NOT called a catch at the access" "t_late .*LATE@SHUT"
chk "prediction agreement is reported"  "every row matched its pre-registered prediction"

# negative: an empty log must be an ERROR, not a table of zeros
: > "$T/empty.log"
python3 -I "$SCRIPT_DIR/classify.py" "$T/empty.log" "$TSV" >/dev/null 2>&1
if [ $? -ne 0 ]; then echo "  PASS  an empty log exits non-zero (no data is not a result)";
else echo "  FAIL  an empty log was accepted"; fail=1; fi

# negative: a wrong prediction must be flagged, not absorbed
sed -i 's/^t_miss\tm\t23\ttemporal\tMISS/t_miss\tm\t23\ttemporal\tBLOCKED-SYNC/' "$TSV"
flagged=$(python3 -I "$SCRIPT_DIR/classify.py" "$LOG" "$TSV" 2>/dev/null)
if grep -q "PREDICTION MISSES" <<<"$flagged"; then
  echo "  PASS  a wrong pre-registered prediction is flagged"
else
  echo "  FAIL  a wrong prediction was silently absorbed"; fail=1
fi

rm -rf "$T"
[ "$fail" -eq 0 ] && echo "SELFTEST: ALL PASS" || { echo "SELFTEST: FAILURES"; exit 1; }
