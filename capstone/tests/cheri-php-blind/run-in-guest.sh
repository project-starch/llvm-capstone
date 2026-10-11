#!/bin/sh
# Runs INSIDE CheriBSD purecap (as root) under CHERI-QEMU. For each row and each
# allocator arm it runs the purecap ELF and prints one structured result line.
#
# arg1 = config: spatial | temporal | eager
#   spatial  : revocation OFF                      -> CHERI spatial safety only
#   temporal : revocation ON, async quarantine     -> realistic CHERI deployment
#   eager    : revocation ON, revoke on every free -> aggressive upper bound
#
# The host classifier needs BOTH the exit status and the last phase marker the
# row printed: the status says whether a capability fault happened, the marker
# says WHERE. PHP's shutdown_memory_manager drains the size-class cache and
# free()s every block, so a fault during shutdown is a late catch and NOT a
# catch of the defect at its use site.
DIR=${DIR:-/root/cheri-php-blind}
CFG="${1:-spatial}"
cd "$DIR" || exit 2

case "$CFG" in
  spatial)  REV=0; EVERY=0 ;;
  temporal) REV=1; EVERY=0 ;;
  eager)    REV=1; EVERY=1 ;;
  *) echo "unknown cfg $CFG"; exit 2 ;;
esac

echo "===PHP-BLIND-BEGIN cfg=$CFG==="
sysctl security.cheri.runtime_revocation_default=$REV >/dev/null 2>&1 \
  && echo "KNOB runtime_revocation_default=$(sysctl -n security.cheri.runtime_revocation_default 2>/dev/null)" \
  || echo "KNOB runtime_revocation_default=ABSENT"
sysctl security.cheri.runtime_revocation_every_free_default=$EVERY >/dev/null 2>&1 \
  && echo "KNOB runtime_revocation_every_free_default=$(sysctl -n security.cheri.runtime_revocation_every_free_default 2>/dev/null)" \
  || echo "KNOB runtime_revocation_every_free_default=ABSENT"

if [ -x ./cheri_status ]; then
  echo "STATUS $(./cheri_status 2>&1 | tr '\n' ' ')"
fi

echo "--- runs (cfg=$CFG) ---"
run_one() {  # $1=arm $2=row
  bin="$1__$2"
  if [ ! -x "./$bin" ]; then
    echo "ROW $2 arm=$1 cfg=$CFG BIN-MISSING"
    return
  fi
  out=$(timeout 30 ./"$bin" 2>&1); rc=$?
  # last phase marker reached, for the where-did-it-fault question
  last=$(printf '%s\n' "$out" | sed -n 's/^\[\(.*\)\]$/\1/p' | tail -1)
  [ -n "$last" ] || last=NONE
  echo "ROW $2 arm=$1 cfg=$CFG rc=$rc last=$last out=[$(printf '%s' "$out" | tr '\n\t' ';;' | cut -c1-160)]"
}
while IFS='	' read -r row models ncases klass es ep note; do
  case "$row" in ''|\#*) continue;; esac
  run_one stockalloc "$row"
  run_one patched    "$row"
done < rows.tsv
echo "===PHP-BLIND-END cfg=$CFG==="
