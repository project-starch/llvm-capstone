#!/usr/bin/env bash
# build-cpy.sh: obmalloc.o of the fixed and of the original patch 0014 with inject.h, each linked with
# cpy-test.c into the prebuilt protected CPython (reuse arm) in place of Programs/python.o.
set -euo pipefail
L=$HOME/asr-work/life-c T=$L/test O=$L/cpy-test
PB=$HOME/asr-work/eval/prebuilt/reuse/cpython-protected
B=$PB/source/build
mkdir -p "$O"
source "$B/capstone-env.sh"
cc_line=$(grep -m1 'obmalloc.c$' "$B/survey-make.log")
cc_line=${cc_line/-DEXP_REUSE_GAP/}
compile() {   # source object
  local cmd=${cc_line% -o *}
  ( cd "$B" && eval "$cmd -include $T/inject.h -o $2 $1" )
}
compile "$L/cpy/Python-3.13.7/Objects/obmalloc.c" "$O/obmalloc-fixed.o"
compile "$O/obmalloc-orig.c" "$O/obmalloc-orig.o"
( cd "$B" && eval "${cc_line% -o *} -o $O/cpy-test.o $T/cpy-test.c" )
for v in fixed orig; do
  python3 - "$PB/image/commands.json" "$O/obmalloc-$v.o" "$O/cpy-test.o" "$O/cpy-test-$v.dom" "$B/" <<'PY'
import json, subprocess, sys
cmds, obm, test, out, B = sys.argv[1:]
link = json.load(open(cmds))[2]
args = []
for a in link:
    if a == B + 'Objects/obmalloc.o': a = obm
    elif a == B + 'Programs/python.o': a = test
    args.append(a)
i = args.index('-o'); args[i + 1] = out
assert obm in args and test in args
subprocess.run(args, check=True)
PY
  echo "built $O/cpy-test-$v.dom"
done
