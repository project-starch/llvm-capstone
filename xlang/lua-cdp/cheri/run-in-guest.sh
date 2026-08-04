#!/bin/sh
# Runs INSIDE CheriBSD purecap (as root) under CHERI-QEMU. For each Lua-CDP row
# it runs the purecap ELF and prints one structured result line; the host driver
# (classify.py, reused from capstone/tests/cheri-baseline/) classifies
# BLOCKED-SYNC / BLOCKED-SWEEP / MISS from the exit status (128+signum on a
# capability fault).
#
# Identical to ../../cheri/run-in-guest.sh except DIR — the SAME three configs,
# the SAME sysctl knobs, so a Lua row's verdict is produced the same way as an
# mruby row's.
#
# arg1 = config: spatial | temporal | eager
#   spatial  : revocation OFF                      -> CHERI spatial safety only
#   temporal : revocation ON, async quarantine     -> realistic CHERI temporal (default)
#   eager    : revocation ON, revoke on every free -> aggressive synchronous
DIR=/root/lua-cdp-cheri
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

# Confirm what the process runtime actually does under this policy.
if [ -x ./cheri_status ]; then
  echo "STATUS $(./cheri_status 2>&1 | tr '\n' ' ')"
fi

echo "--- runs (cfg=$CFG) ---"
run_bin() {  # $1=key $2=binary
  [ -x "./$2" ] || { echo "ROW $1 BIN-MISSING"; return; }
  out=$(timeout 20 ./"$2" 2>&1); rc=$?
  echo "ROW $1 cfg=$CFG rc=$rc out=[$(printf '%s' "$out" | tr '\n\t' ';;' | cut -c1-100)]"
}
# rows.tsv col 1 = numeric key, col 2 = shim .c whose basename is the binary.
while IFS='	' read -r key shim rest; do
  case "$key" in ''|\#*) continue;; esac
  bin=$(basename "$shim" .c)
  run_bin "$key" "$bin"
done < rows.tsv
echo "===CHERI-BASELINE-END cfg=$CFG==="
