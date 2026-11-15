#!/usr/bin/env bash
# One boot of the PHP reuse study (docs/plans/2026-11-15-php-reuse-distance.md):
#
#   run-one.sh stock|protected|exact REP
#
# Environment: KIT (the trap-route kit, kit-run-3 on p13), PREBUILT (holds php-stock/php.dom and
# php-protected/php.dom from build.sh, and php-inputs/bench.php from make-bench.py), OUT (the
# results root; the bundle goes to OUT/ARM-REP), STEP_TIMEOUT (seconds, default 14400). Run from
# the repository root. The exact arm is the protected image with PHP_CAP_MODE=7; protected is 6;
# stock names no mode (its zend_alloc.c has no nested layer).
set -euo pipefail
ARM=$1 REP=$2
case $ARM in
  stock) IMG=php-stock MODE= ;;
  protected) IMG=php-protected MODE="PHP_CAP_MODE=6 " ;;
  exact) IMG=php-protected MODE="PHP_CAP_MODE=7 " ;;
  *) echo "arm: stock, protected or exact" >&2; exit 2 ;;
esac
: "${KIT:?}" "${PREBUILT:?}" "${OUT:?}"
mkdir -p "$OUT"
python3 capstone/runtime/virtual/trap_platform.py run --kit "$KIT" --work "$OUT/$ARM-$REP" \
  --step-timeout "${STEP_TIMEOUT:-14400}" \
  --file bench.php="$PREBUILT/php-inputs/bench.php" --file "$IMG.dom=$PREBUILT/$IMG/php.dom" \
  --step "php-$ARM=cd /mnt/vm; ${MODE}./capstone-vexec ./$IMG.dom -n bench.php"
