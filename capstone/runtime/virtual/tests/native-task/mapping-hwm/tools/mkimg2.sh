#!/bin/bash
# mkimg2.sh <name> <malloc-mapping object> <bench>: FRCDKMSa3's own objects for <bench>, relinked by its recorded link.txt
# through a copy of its SDK whose runtime archive has one object replaced (the nested lane's mcfr0 recipe, generalised).
# Output $R/<name>/<exe>.strip.dom for each exe; prints name, exe and sha256.
: "${CAPSTONE_WORK:=/mnt/hdd/$USER/capstone}" "${CAPSTONE_LLVM:=$HOME/dev/llvm-capstone}"
set -euo pipefail
R=${CAPSTONE_WORK}/perf/mrhwm
A=${CAPSTONE_WORK}/perf/cclever/arms/FRCDKMSa3
RT=${CAPSTONE_WORK}/perf/alean/sdkm13b/sdk/sdk/libapplication-runtime.a
AR=${CAPSTONE_LLVM}/llvm/cmake-build-debug/bin/llvm-ar
OC=${CAPSTONE_WORK}/llvm-c4/bin/llvm-objcopy
name=$1 obj=$2 bench=$3
d=$R/$name; rm -rf "$d"; mkdir -p "$d"/{lib,m,sdk,build}
cp "$RT" "$d/lib/"
cp "$obj" "$d/m/malloc-mapping.c.obj"
(cd "$d/m" && "$AR" r ../lib/libapplication-runtime.a malloc-mapping.c.obj)
cmp <("$AR" t "$RT") <("$AR" t "$d/lib/libapplication-runtime.a")
cp "$A/sdk/capstone-cc" "$d/sdk/"
python3 - "$A/sdk/sdk.json" "$d/sdk/sdk.json" "$d/lib/libapplication-runtime.a" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1])); assert "runtime" in j; j["runtime"] = sys.argv[3]
json.dump(j, open(sys.argv[2], "w"), indent=2)
EOF
S=$A/spec/$bench
(cd "$S" && find . -name '*.o' -print0) | (cd "$S" && xargs -0 cp --parents -t "$d/build")
python3 - "$S/link.txt" "$A/sdk/capstone-cc" "$d/sdk/capstone-cc" > "$d/link.txt" <<'EOF'
import sys
old, new = sys.argv[2], sys.argv[3]
for l in open(sys.argv[1]).read().splitlines():
    assert l.startswith(old + " "), l[:80]
    print(new + l[len(old):])
EOF
(cd "$d/build" && env -u CAPSTONE_SDK bash -e "$d/link.txt" > "$d/link.log" 2>&1)
for e in $(cat "$S/exes.txt"); do
  "$OC" --strip-all --keep-section=.capstone_virtual --keep-section=.capstone_application --keep-section=.capstone_domreq \
    "$d/build/$e" "$d/$e.strip.dom"
  echo "$name $e $(sha256sum < "$d/$e.strip.dom" | cut -c1-16)"
done
