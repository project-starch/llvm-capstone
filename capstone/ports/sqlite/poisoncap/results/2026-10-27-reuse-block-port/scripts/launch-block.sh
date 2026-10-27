#!/usr/bin/env bash
# launch-block.sh ARM DOMAIN WORK: reuse/launch-sublet.sh for the block-port arms (reuse-capstone-block,
# built by kit repo-heap256 0cfe025 with sdk-v6), same platform, module arguments and limits.
cd ~/poisoncap-platform/repo-heap256
source capstone/tests/capstone-test-env.sh > /dev/null 2>&1
export CAPSTONE_QEMU_CPU_EXTRA=svadu=false,x-capstone-supervisor-slots=32,x-capstone-test-csrs=true
B=~/poisoncap-platform/reuse-capstone-block; W=~/poisoncap-platform/reuse-capstone-runs
capstone_with_qemu_domain "$2" python3 capstone/ports/sqlite/poisoncap/reuse/run-sublet.py \
  --platform ~/poisoncap-platform/platform-paged --module-args "supervisor_slots=32" \
  --work "$W/$3" --timeout 50400 --per-process 46800 --run "$1-size100=$B/$1/speedtest1.dom"
echo "EXIT $?"
