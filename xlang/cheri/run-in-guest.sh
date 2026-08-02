#!/bin/sh
# Runs INSIDE CheriBSD purecap (as root) under CHERI-QEMU. For each in-scope
# xlang row it runs the purecap ELF and prints one structured result line; the
# host driver (classify.py, reused from capstone/tests/cheri-baseline/) classifies
# BLOCKED-SYNC / BLOCKED-SWEEP / MISS from the exit status (128+signum on a
# capability fault) and the stderr text.
#
# arg1 = config: spatial | temporal | eager
#   spatial  : revocation OFF                      -> CHERI spatial safety only
#   temporal : revocation ON, async quarantine     -> realistic CHERI temporal
#   eager    : revocation ON, revoke on every free -> aggressive synchronous UB
DIR=/root/cheri-baseline-xlang
CFG="${1:-spatial}"
cd "$DIR" || exit 2

case "$CFG" in
  spatial)  REV=0; EVERY=0 ;;
  temporal) REV=1; EVERY=0 ;;
  eager)    REV=1; EVERY=1 ;;
  *) echo "unknown cfg $CFG"; exit 2 ;;
esac

echo "===CHERI-BASELINE-BEGIN cfg=$CFG==="
sysctl security.cheri.runtime_revocation_default=$REV >/dev/null 2>&1 \
  && echo "KNOB runtime_revocation_default=$(sysctl -n security.cheri.runtime_revocation_default 2>/dev/null)" \
  || echo "KNOB runtime_revocation_default=ABSENT"
sysctl security.cheri.runtime_revocation_every_free_default=$EVERY >/dev/null 2>&1 \
  && echo "KNOB runtime_revocation_every_free_default=$(sysctl -n security.cheri.runtime_revocation_every_free_default 2>/dev/null)" \
  || echo "KNOB runtime_revocation_every_free_default=ABSENT"

# Confirm what the process runtime actually does under this policy, rather than
# assuming the sysctl took effect.
if [ -x ./cheri_status ]; then
  echo "STATUS $(./cheri_status 2>&1 | tr '\n' ' ')"
fi

# Sanity: the lifecycle harness itself must run clean purecap, else every row's
# verdict would be about the harness rather than the defect.
if [ "$CFG" = spatial ]; then
  for s in sanity_mock; do
    [ -x "./$s" ] || continue
    o=$(timeout 20 ./"$s" 2>&1); rc=$?
    echo "SANITY $s rc=$rc out=[$(printf '%s' "$o" | tr '\n\t' ';;' | cut -c1-60)]"
  done
fi

echo "--- runs (cfg=$CFG) ---"
run_bin() {  # $1=label $2=binary
  [ -x "./$2" ] || { echo "ROW $1 BIN-MISSING"; return; }
  out=$(timeout 20 ./"$2" 2>&1); rc=$?
  # rc 66 = the harness refused to measure (realloc grew the stack in place),
  # which is a shim-validity failure, NOT a CHERI MISS. It must never be
  # silently folded into the table.
  echo "ROW $1 cfg=$CFG rc=$rc out=[$(printf '%s' "$out" | tr '\n\t' ';;' | cut -c1-100)]"
}
# rows.tsv column 1 is the corpus/paper table key, column 2 the shim source.
# The binary is named after the defect (the shim's stem); the emitted line
# keeps "ROW <key>" because the shared classify.py parses that.
while IFS='	' read -r key shim rest; do
  case "$key" in ''|\#*) continue;; esac
  bin=$(basename "$shim" .c)
  run_bin "$key" "$bin"
done < rows.tsv
echo "===CHERI-BASELINE-END cfg=$CFG==="
