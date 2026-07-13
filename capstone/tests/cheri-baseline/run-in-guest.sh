#!/bin/sh
# Runs INSIDE CheriBSD purecap (as root) under CHERI-QEMU. For each in-scope
# corpus row it runs the purecap ELF and prints one structured result line; the
# host driver classifies BLOCKED-SYNC / BLOCKED-SWEEP / MISS from the exit status
# (128+signum on a capability fault) and stderr text.
#
# arg1 = config: spatial | temporal | eager
#   spatial  : revocation OFF                      -> CHERI spatial safety only
#   temporal : revocation ON, async quarantine     -> realistic CHERI temporal
#   eager    : revocation ON, revoke on every free -> aggressive synchronous UB
DIR=/root/cheri-baseline
CFG="${1:-spatial}"
cd "$DIR" || exit 2

case "$CFG" in
  spatial)  REV=0; EVERY=0 ;;
  temporal) REV=1; EVERY=0 ;;
  eager)    REV=1; EVERY=1 ;;
  *) echo "unknown cfg $CFG"; exit 2 ;;
esac

echo "===CHERI-BASELINE-BEGIN cfg=$CFG==="
# Set the default revocation policy for processes we spawn below.
sysctl security.cheri.runtime_revocation_default=$REV >/dev/null 2>&1 \
  && echo "KNOB runtime_revocation_default=$(sysctl -n security.cheri.runtime_revocation_default 2>/dev/null)" \
  || echo "KNOB runtime_revocation_default=ABSENT"
sysctl security.cheri.runtime_revocation_every_free_default=$EVERY >/dev/null 2>&1 \
  && echo "KNOB runtime_revocation_every_free_default=$(sysctl -n security.cheri.runtime_revocation_every_free_default 2>/dev/null)" \
  || echo "KNOB runtime_revocation_every_free_default=ABSENT"

# Confirm what the process runtime actually does under this policy.
if [ -x ./cheri_status ]; then
  echo "STATUS $(./cheri_status 2>&1 | tr '\n' ' ')"
fi

# Sanity: a defect-free SQLite exercise must run to completion under purecap,
# else the corpus results would be measuring an unclean amalgamation, not the
# injected defect. Run once (config-independent) on the first pass.
if [ "$CFG" = spatial ]; then
  # sanity_vanilla / sanity_clean use the REAL upstream / patched amalgamation
  # (document that neither runs purecap); sanity_mock is the lifecycle harness
  # the shims link against (must run clean). timeout guards the THREADSAFE=1 hang.
  for s in sanity_vanilla sanity_clean sanity_mock; do
    [ -x "./$s" ] || continue
    o=$(timeout 20 ./"$s" 2>&1); rc=$?
    echo "SANITY $s rc=$rc out=[$(printf '%s' "$o" | tr '\n\t' ';;' | cut -c1-60)]"
  done
fi

echo "--- runs (cfg=$CFG) ---"
run_bin() {  # $1=label $2=binary
  [ -x "./$2" ] || { echo "ROW $1 BIN-MISSING"; return; }
  out=$(timeout 20 ./"$2" 2>&1); rc=$?
  echo "ROW $1 cfg=$CFG rc=$rc out=[$(printf '%s' "$out" | tr '\n\t' ';;' | cut -c1-100)]"
}
while IFS='	' read -r newrow dir oracle klass predA predB; do
  case "$newrow" in ''|\#*) continue;; esac
  run_bin "$newrow" "${newrow}_${dir}"
done < rows.tsv
# faithful reuse-not-free variant of row 3 (the stale-but-allocated headline)
run_bin "3r" "3r_row3_reuse"
echo "===CHERI-BASELINE-END cfg=$CFG==="
