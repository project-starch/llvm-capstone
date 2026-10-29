#!/usr/bin/env bash
# launch-sublet.sh ARM DOMAIN WORK: one Sublet arm's speedtest1 (no arguments) in one staged boot,
# on the PAGED virtual platform: v7 QEMU with x-capstone-test-csrs=true and the default (paged, growing)
# capstone_vm build. platform-sdev's flat module (the vp2 silicon twin, 65,536 records) cannot hold this run.
cd ~/poisoncap-platform/repo
source capstone/tests/capstone-test-env.sh > /dev/null 2>&1
export CAPSTONE_QEMU_CPU_EXTRA=svadu=false,x-capstone-supervisor-slots=32,x-capstone-test-csrs=true
B=~/poisoncap-platform/reuse-capstone; W=~/poisoncap-platform/reuse-capstone-runs
capstone_with_qemu_domain "$2" python3 capstone/ports/sqlite/poisoncap/reuse/run-sublet.py \
  --platform ~/poisoncap-platform/platform-paged --module-args "supervisor_slots=32" \
  --work "$W/$3" --timeout 50400 --per-process 46800 --run "$1-size100=$B/$1/speedtest1.dom"
echo "EXIT $?"
