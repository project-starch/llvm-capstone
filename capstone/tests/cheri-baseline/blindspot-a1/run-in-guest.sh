#!/bin/sh
# Runs INSIDE CheriBSD purecap. One blind-spot case, three revocation configs.
#
# A1 is mruby issue 6339: mrb_ary_delete keeps the removed element in a local the
# GC does not know about, the object is swept mid-delete, and its slot comes back
# off the PAGE FREE LIST as a String. Nothing reaches malloc or free, so purecap
# sees an in-bounds access through a tagged capability and revocation has nothing
# to revoke. The oracle is therefore the ANSWER, not a crash.
#
# The verdict is a PRINTED MARKER, because this build has no Kernel#exit and an
# exit-code oracle reported NoMethodError as rc=1, which reads exactly like a
# correct answer:
#   A1RESULT=1  correct        A1RESULT=2  WRONG ANSWER, a MISS
#   no marker and rc>=128      CHERI caught it on a signal
#   SANITY_OK=7                the interpreter and the channel both work
DIR=/root/a1-6339
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
  && echo "KNOB revocation=$(sysctl -n security.cheri.runtime_revocation_default 2>/dev/null)" \
  || echo "KNOB revocation=ABSENT"
sysctl security.cheri.runtime_revocation_every_free_default=$EVERY >/dev/null 2>&1 \
  && echo "KNOB every_free=$(sysctl -n security.cheri.runtime_revocation_every_free_default 2>/dev/null)" \
  || echo "KNOB every_free=ABSENT"

o=$(timeout 60 ./mruby sanity.rb 2>&1); rc=$?
echo "SANITY rc=$rc out=[$(printf '%s' "$o" | tr '\n\t' ';;' | cut -c1-80)]"

o=$(timeout 60 ./mruby a1.rb 2>&1); rc=$?
echo "ROW A1 cfg=$CFG rc=$rc out=[$(printf '%s' "$o" | tr '\n\t' ';;' | cut -c1-100)]"
echo "===CHERI-BASELINE-END cfg=$CFG==="
