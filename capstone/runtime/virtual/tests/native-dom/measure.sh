#!/bin/bash
# p12 lane artifact layout. Builds are separate; one guest per arm, run serially.
set -euo pipefail
W=${LANE_ROOT:-$HOME/lanes/laneAG2}
P=${NATIVE_PLATFORM:-$W/platform-native}
R=${MEASURE_ROOT:-$W}
mkdir -p "$R"
cd "$W/src"
export CAPSTONE_TOOLCHAIN_CHECKED=1 CAPSTONE_TMP_ROOT=$W/tmp CAPSTONE_QEMU_LOCK=$W/measure.lock
source capstone/tests/capstone-test-env.sh
export PYTHONDONTWRITEBYTECODE=1 CAPSTONE_SPEC_VM_STATES=$W/states SHARE=$W
export MICRO=$W/micro KIT=$HOME/prof-dev/kit4 VMHOST=$W/src/capstone/runtime/host SLOTS=${NATIVE_SLOTS:-2}
export DOM_TRAMPOLINE=$W/trampoline.bin
export DOM_CHILD=$W/micro/lqmicro-cap DOM_PROFILER=$W/contracts/profile.py
export ORIGINAL_PROF_REPORT=$KIT/tools/prof-report.py NM=$HOME/prof-dev/cap4/build/bin/llvm-nm
python3 - "$P" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);d=json.loads((p/'platform.json').read_text())
for f in d['files']: d['files'][f]=hashlib.sha256((p/f).read_bytes()).hexdigest()
(p/'platform.json').write_text(json.dumps(d,indent=1)+'\n')
PY
for DOM_ARM in plain launcher native; do
 export DOM_ARM
 child=/tmp/child-cap; mod=dom_binfmt=0
 [ "$DOM_ARM" != plain ] || child=/tmp/lqmicro
 [ "$DOM_ARM" != native ] || mod="dom_binfmt=1 ${NATIVE_MODULE_ARGS:-}"
 list=$W/micro/exec-$DOM_ARM.list
 for n in 10 100 10 100 10 100; do echo "-- exec $n $child"; done > "$list"
 df -h "$W"
 bash "$W/contracts/run-micro.sh" "$R/measure-$DOM_ARM" "$P" plain 8G "$list" "$mod"
done
