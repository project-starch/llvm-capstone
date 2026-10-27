#!/usr/bin/env bash
# launch-qual-block.sh WORK DOMAIN: launch-qual-paged.sh for the two block-port arms.
cd ~/poisoncap-platform/repo-heap256
source capstone/tests/capstone-test-env.sh > /dev/null 2>&1
export CAPSTONE_QEMU_CPU_EXTRA=svadu=false,x-capstone-supervisor-slots=32,x-capstone-test-csrs=true
B=~/poisoncap-platform/reuse-capstone-block; W=~/poisoncap-platform/reuse-capstone-runs
args=()
for arm in block-port block-port-spatial; do
  args+=(--run "$arm-test-wrapped=$B/$arm/test-wrapped.dom" --run "$arm-test-traced=$B/$arm/test-traced.dom"
         --run "$arm-test-plain=$B/$arm/test-plain.dom" --run "$arm-st-size1=$B/$arm/speedtest1.dom:--size 1"
         --run "$arm-st-plain-size1=$B/$arm/speedtest1-plain.dom:--size 1")
done
capstone_with_qemu_domain "$2" python3 capstone/ports/sqlite/poisoncap/reuse/run-sublet.py \
  --platform ~/poisoncap-platform/platform-paged --module-args "supervisor_slots=32" \
  --work "$W/$1" --timeout 5400 --per-process 1500 "${args[@]}"
echo "EXIT $?"
