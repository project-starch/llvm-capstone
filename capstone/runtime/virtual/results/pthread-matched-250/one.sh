#!/bin/bash
# one.sh ARM N : one run-pthreads.py run on platform-ARM (ARM = sdev: dev 4985df88 control; sreg: #250 head), as drive32.sh/run-arm-kit2.sh ran it
set -u
A=$1 N=$2 B=$HOME/spec-cap/gates-1018
export LD_LIBRARY_PATH=$HOME/vkit/lib CAPSTONE_QEMU_LOCK=$HOME/hosted-ab/qemu.lock CAPSTONE_QEMU_CPU_EXTRA=svadu=false,x-capstone-supervisor-slots=32
case $A in sdev) T=$B/tree-dev6;; sreg) T=$B/tree-reg6;; esac
P=$B/platform-$A V=$T/capstone/runtime/virtual
s=$(date +%s.%N)
python3 $V/run-pthreads.py --qemu $P/qemu-system-riscv64 --images $P/images --adapter $P/adapter --application $B/out-$A/gates/img/pthread.dom --work $B/PT50-250/run-$A-$N > $B/PT50-250/run-$A-$N.log 2>&1
rc=$?
e=$(date +%s.%N)
f=$(python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print(sorted(k for k,v in r.get(\"tests\",{}).items() if v is not True))" $B/PT50-250/run-$A-$N/result.json 2>/dev/null || echo "NO_RESULT")
echo "pt $A $N rc=$rc wall=$(echo "$e - $s" | bc -l | cut -c1-6)s failed=$f"
