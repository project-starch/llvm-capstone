#!/usr/bin/env bash
# build-cpy-blocks.sh: both reuse arms of CPython with the current port patches (stock: all but
# 0014; protected: all) and the arm's reuse-gap hook patch. Only obmalloc.c differs from the
# prebuilt arm, so its obmalloc.o is rebuilt with the arm's own compile line and the arm's link
# command is rerun with it, into $L/cpy-blocks/cpython-ARM.dom.
set -euo pipefail
L=$HOME/asr-work/life-c O=$L/cpy-blocks
mkdir -p "$O"
for arm in stock protected; do
  PB=$HOME/asr-work/eval/prebuilt/reuse/cpython-$arm B=$PB/source/build
  S=$O/src-$arm; rm -rf "$S"; mkdir -p "$S"
  tar xzf "$HOME/pg-virtual-kit/inputs/Python-3.13.7.tgz" -C "$S"
  ( cd "$S/Python-3.13.7"
    for p in "$L"/patches/cpython-app/cpython-3.13.7-*.patch; do
      [[ $arm == stock && $p == *-0014-* ]] && continue
      patch --batch --forward --fuzz=0 -p1 < "$p" > /dev/null || { echo "FAIL $p"; exit 1; }
    done
    patch --batch --forward --fuzz=0 -p1 < "$L/patches/obmalloc-reuse-gap-$arm.patch" > /dev/null \
      || { echo "FAIL hook $arm"; exit 1; } )
  cp "$HOME/asr-work/eval/tree/capstone/experiments/study/reuse-gap-observer.h" "$S/Python-3.13.7/Objects/"
  source "$B/capstone-env.sh"
  cc_line=$(grep -m1 'obmalloc.c$' "$B/survey-make.log")
  ( cd "$B" && eval "${cc_line% -o *} -o $O/obmalloc-$arm.o $S/Python-3.13.7/Objects/obmalloc.c" ) 2> "$O/cc-$arm.log"
  python3 - "$PB/image/commands.json" "$O/obmalloc-$arm.o" "$O/cpython-$arm.dom" "$B/" <<'PY'
import json, subprocess, sys
cmds, obm, out, B = sys.argv[1:]
args = [obm if a == B + 'Objects/obmalloc.o' else a for a in json.load(open(cmds))[2]]
i = args.index('-o'); args[i + 1] = out
assert obm in args
subprocess.run(args, check=True)
PY
  echo "built $O/cpython-$arm.dom"
done
