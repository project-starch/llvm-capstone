#!/bin/sh
S=$(cd "$(dirname "$0")" && pwd)
V=${SPEC_VEXEC:?set SPEC_VEXEC}
run() { echo "@@PROBE-BEGIN $*"; b=$1; shift; "$V" "$S/bin/$b" "$@"; echo "@@PROBE-END $b rc=$?"; dmesg | grep "domain fault" | tail -1; }
run pair-T1 32720 32768 16
run pair-T3 32720 32768 16
run pair-T1 261888 262144 16
run pair-T3 261888 262144 16
run pair-T1 32704 32768 16
echo "@@PROBE-DONE"
