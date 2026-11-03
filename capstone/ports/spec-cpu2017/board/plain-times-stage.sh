#!/bin/bash
# mk.sh <plain test stage>: a copy whose run.sh prints the shell's `times` (user and system CPU time of its waited-for
# children, kernel tick accounting at HZ=100) right before and right after every benchmark command:
#   @@SPEC-TIMES0 <n> / @@SPEC-TIMES1 <n>, each followed by the two lines `times` prints (shell; children).
# bin/ and in/ are hard links; plan.json and the commands are unchanged.
set -eu
src=$1; dst=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}/spec/stages-plain-times/$(basename $src)-times
[ -e "$dst" ] && { echo "exists: $dst"; exit 1; }
mkdir -p "$dst/stage"; cp "$src/plan.json" "$dst/"
for e in "$src"/stage/*; do b=$(basename "$e"); [ "$b" = run.sh ] && continue; cp -al "$e" "$dst/stage/$b"; done
python3 - "$src/stage/run.sh" "$dst/stage/run.sh" <<'PY'
import sys
L=open(sys.argv[1]).read().split('\n'); out=[]; n=0
for i,l in enumerate(L):
    if l.startswith('rc=$?'): out+= [l, 'echo "@@SPEC-TIMES1 $n"; times']; n+=1; continue
    if i+1<len(L) and L[i+1].startswith('rc=$?'): out+=['echo "@@SPEC-TIMES0 $n"; times']
    out.append(l)
assert n>0
open(sys.argv[2],'w').write('\n'.join(out)); print('times pairs',n)
PY
chmod +x "$dst/stage/run.sh"; echo "$dst"
