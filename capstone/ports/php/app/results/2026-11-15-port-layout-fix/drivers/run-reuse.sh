#!/usr/bin/env bash
# run-reuse.sh ARM REP: one boot of the PHP reuse study at the run-5 settings (plan
# docs/plans/2026-11-15-php-port-layout-fix.md) into run-6/reuse/php-ARM-REP, with provenance.txt.
# ARM: stock, protected, exact (the reuse images), or trace-stock, trace-protected, trace-exact
# (the layout trace images, into run-6/layout/php-ARM).
E=$HOME/asr-work/eval F=$HOME/asr-work/fix-php
ARM=$1 REP=${2:-1}
case $ARM in
  stock) IMG=$F/prebuilt/php-stock/php.dom MODE= OUT=$E/run-6/reuse/php-$ARM-$REP ;;
  protected) IMG=$F/prebuilt/php-protected/php.dom MODE="PHP_CAP_MODE=6 " OUT=$E/run-6/reuse/php-$ARM-$REP ;;
  exact) IMG=$F/prebuilt/php-protected/php.dom MODE="PHP_CAP_MODE=7 " OUT=$E/run-6/reuse/php-$ARM-$REP ;;
  trace-stock) IMG=$F/prebuilt/trace-stock/trace.dom MODE= OUT=$E/run-6/layout/php-$ARM-$REP ;;
  stock-env) STEP=php-stock IMG=$F/prebuilt/php-stock/php.dom MODE="PHP_CAP_MODE=6 " OUT=$E/run-6/reuse/php-$ARM-$REP ;;
  trace-stock-env) STEP=php-stock IMG=$F/prebuilt/trace-stock/trace.dom MODE="PHP_CAP_MODE=6 " OUT=$E/run-6/layout/php-$ARM-$REP ;;
  trace-protected) IMG=$F/prebuilt/trace-protected/trace.dom MODE="PHP_CAP_MODE=6 " OUT=$E/run-6/layout/php-$ARM-$REP ;;
  trace-exact) IMG=$F/prebuilt/trace-protected/trace.dom MODE="PHP_CAP_MODE=7 " OUT=$E/run-6/layout/php-$ARM-$REP ;;
  *) echo "arm?" >&2; exit 2 ;;
esac
BENCH=$E/prebuilt/reuse/php-inputs/bench.php KIT=$E/kit-run-3
cd $F/tree && source capstone/runtime/virtual/tests/trap-platform/eval-p13.env
export CAPSTONE_QEMU_DOMAIN_LOCK=$F/lock-$ARM-$REP CAPSTONE_TMP_ROOT=$F/tmp PATH=$HOME/lanes/gatefix/llvm/bin:/usr/local/bin:/usr/bin:/bin
mkdir -p $(dirname $OUT)
t0=$(date +%s); s0=$(date -u +%FT%TZ)
python3 capstone/runtime/virtual/trap_platform.py run --kit $KIT --work $OUT --step-timeout ${STEP_TIMEOUT:-14400} \
  --file bench.php=$BENCH --file php.dom=$IMG \
  --step "${STEP:-php-${ARM#trace-}}=cd /mnt/vm; ${MODE}./capstone-vexec ./php.dom -n bench.php"
rc=$?
{
  echo "run $ARM-$REP: plan docs/plans/2026-11-15-php-port-layout-fix.md, tree fix/php-port-layout $(cat $F/tree.commit 2>/dev/null)"
  echo "command: cd /mnt/vm; ${MODE}./capstone-vexec ./php.dom -n bench.php (trap_platform.py run, kit-run-3, step timeout ${STEP_TIMEOUT:-14400})"
  echo "start $s0 end $(date -u +%FT%TZ) wall $(( $(date +%s) - t0 )) s, trap_platform rc=$rc"
  sha256sum $IMG $BENCH
  (cd $KIT && sha256sum adapter/capstone-vexec images/Image images/fw_jump.elf images/initramfs.cpio.gz qemu-system-riscv64)
} > $OUT/provenance.txt
echo "RUN php-$ARM-$REP rc=$rc wall=$(( $(date +%s) - t0 ))s $(date -u +%T)"
