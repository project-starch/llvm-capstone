#!/bin/bash
# boot.sh LOCKNAME OUTDIR IMAGE REPS [case-batch.py options]: one boot under its own lock file, marker file at the end
source $HOME/crenew-step4/corpus/work-fix-php/env.sh
lock=$1; out=$2; image=$3; reps=$4; shift 4
export CAPSTONE_QEMU_LOCK=$W/$lock.lock
mkdir -p "$out"; rm -f "$out/.done"
cd $W
python3 $W/scripts/case-batch.py "$image" "$out" "$reps" "$@" > "$out/boot.log" 2>&1
echo "rc=$?" >> "$out/boot.log"
touch "$out/.done"
