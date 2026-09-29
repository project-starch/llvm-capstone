#!/usr/bin/env bash
# Runtime regressions use the same application SDK and runner as tshark.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APP=$(cd -- "$HERE/../.." && pwd)
source "$APP/deps/env.sh" > /dev/null
CASE=${1:?c64 | c65 | i11}
case $CASE in c64) SRC=ctors.c ;; c65) SRC=cond.c ;; i11) SRC=closefd1.c ;; *) exit 2 ;; esac
W=$TS_WORK/runtime-gaps/$CASE
mkdir -p "$W"
"$CC" -O1 "$HERE/$SRC" -o "$W/prog.dom"
cc -O1 -o "$W/native" "$HERE/$SRC" -lpthread
"$W/native" > "$W/native.stdout" 2> "$W/native.stderr"
python3 "$APP/../../common/application/run.py" \
 --state "${CAPSTONE_VM_STATE:?running VM state required}" --result "$W/result.json" \
 "$W/prog.dom" > "$W/domain.stdout" 2> "$W/domain.stderr"
cmp "$W/native.stdout" "$W/domain.stdout"
