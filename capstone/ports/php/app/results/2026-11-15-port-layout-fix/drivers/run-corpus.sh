#!/usr/bin/env bash
# run-corpus.sh WHICH: one PHP corpus on the fixed corpus image, with the run-4 runner and kit,
# into run-6/corpora/WHICH. WHICH: php-engine (asan-blind), php-engine-tav (temporal-asan-visible),
# php-plain-temporal, php-zend-alloc.
E=$HOME/asr-work/eval F=$HOME/asr-work/fix-php
W=$1 KIT=$E/kit-run-4 IMG=$F/prebuilt/whole-php-protected/realphp/php-real-nested.dom PROBE=$F/prebuilt/whole-php-protected/probe-nested.dom
OUT=$E/run-6/corpora/$W
cd $F/tree && source capstone/runtime/virtual/tests/trap-platform/eval-p13.env
export CAPSTONE_QEMU_DOMAIN_LOCK=$F/lock-corpus-$W CAPSTONE_TMP_ROOT=$F/tmp-$W PATH=$HOME/lanes/gatefix/llvm/bin:/usr/local/bin:/usr/bin:/bin
mkdir -p $E/run-6/corpora $CAPSTONE_TMP_ROOT
COMMON=(--qemu $KIT/qemu-system-riscv64 --images $KIT/images --adapter $KIT/adapter --image $IMG --probe $PROBE --llvm-bin $LLVM_BIN --out $OUT)
t0=$(date +%s); s0=$(date -u +%FT%TZ)
case $W in
  php-engine) python3 capstone/ports/php/app/run-realphp.py --corpus capstone/bug-corpora/php/engine-repros --population asan-blind "${COMMON[@]}" ;;
  php-engine-tav) python3 capstone/ports/php/app/run-realphp.py --corpus capstone/bug-corpora/php/engine-repros --population temporal-asan-visible "${COMMON[@]}" ;;
  php-plain-temporal) python3 capstone/bug-corpora/php/tools/run-php-corpus.py --corpus capstone/bug-corpora/php/plain-temporal-repros --arm virtual-malloc --arm virtual-nested-pools "${COMMON[@]}" ;;
  php-zend-alloc) python3 capstone/bug-corpora/php/tools/run-php-corpus.py --corpus capstone/bug-corpora/php/zend-alloc-repros --arm virtual-malloc --arm virtual-nested-pools "${COMMON[@]}" ;;
  *) echo "which?" >&2; exit 2 ;;
esac > $E/run-6/corpora/$W.log 2>&1
rc=$?
{
  echo "corpus $W: plan docs/plans/2026-11-15-php-port-layout-fix.md, tree fix/php-port-layout $(cat $F/tree.commit 2>/dev/null)"
  echo "start $s0 end $(date -u +%FT%TZ) wall $(( $(date +%s) - t0 )) s, runner rc=$rc, kit-run-4"
  sha256sum $IMG $PROBE
  (cd $KIT && sha256sum adapter/capstone-vexec images/Image images/fw_jump.elf images/initramfs.cpio.gz qemu-system-riscv64)
} > $OUT/provenance.txt
echo "CORPUS $W rc=$rc wall=$(( $(date +%s) - t0 ))s $(date -u +%T)"
