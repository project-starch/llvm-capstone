#!/usr/bin/env bash
# build-wm.sh: wmem's six units of the fixed and of the original patch 0001, with the port's GLib
# shim and inject.h, linked with wm-test.c through the trap SDK.
set -euo pipefail
L=$HOME/asr-work/life-c T=$L/test O=$L/wm-test SDK=$HOME/asr-work/eval/prebuilt/sdk
RT=$HOME/asr-work/eval/tree/capstone/runtime/include
mkdir -p "$O"
for v in fixed orig; do
  W=$L/wm-src-$v/wsutil/wmem
  F="-O1 -w -DWM_SUBLET_Q15=1 -I$L/wmshim/shim -I$L/wmshim -I$W -I$RT"
  objs=()
  for u in wmem_core wmem_user_cb wmem_allocator_simple wmem_allocator_block wmem_allocator_block_fast wmem_allocator_strict; do
    $SDK/capstone-cc -c $F -include "$T/inject.h" -o "$O/$u-$v.o" "$W/$u.c"; objs+=("$O/$u-$v.o")
  done
  for s in system compat; do $SDK/capstone-cc -c $F -o "$O/$s.o" "$L/wmshim/$s.c"; done
  $SDK/capstone-cc -c $F -o "$O/wm-test.o" "$T/wm-test.c"
  $SDK/capstone-cc -o "$O/wm-test-$v.dom" "$O/wm-test.o" "${objs[@]}" "$O/system.o" "$O/compat.o"
  echo "built $O/wm-test-$v.dom"
done
