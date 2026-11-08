#!/bin/bash
# dry-run harness for e0-gate.sh: every board call stubbed. usage: dry.sh <case> <control-set> [VAR=value ...]
# A fresh output directory per case, so no count ever includes a previous run's log.
G=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}/perf/e5cgate; D=$G/dry; c=$1; src=$2; shift 2
rm -rf "$D/$c" "$D/$c.out"
env E0_STORE_DIR=$D/store E0_BITSHA_FILE=$D/store/e5n-bit.sha E0_O=$D/$c E0_BUSY=${E0_BUSY:-true} E0_BOOK="echo bk-$c" E0_FLASH="${E0_FLASH:-echo FLASHSTUB}" \
    E0_READBACK="echo READBACKSTUB" E0_STOREHAS=false E0_RUN="$D/run-stub.sh" E0_DRY_SRC=$G/ctl/$src "$@" \
    bash $G/e5c-gate.sh 3 > $D/$c.out 2>&1
rc=$?
echo "$c rc=$rc flashes: $(grep -c 'FLASH BEGIN' $D/$c/e0-gate.log 2>/dev/null) [$(grep -o 'FLASH BEGIN [a-z0-9_]*' $D/$c/e0-gate.log 2>/dev/null | awk '{print $3}' | tr '\n' ' ')] images $(grep -c 'IMAGE .* END' $D/$c/e0-gate.log 2>/dev/null) last: $(tail -1 $D/$c/e0-gate.log 2>/dev/null | cut -c27-)"
