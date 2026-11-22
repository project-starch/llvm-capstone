#!/bin/bash
# relink.sh <variant> <bench dir> <exe>: the plain-rv64al SPEC objects of <bench>, linked by their recorded make.out
# line with plain-cc-<variant> in place of plain-cc-alean, into rl-<variant>/<exe>.
set -euo pipefail
P=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}/perf/plainhwm v=$1 b=$2 e=$3
B=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}/spec/tree-plain/SPEC17/benchspec/CPU/$b/build/build_base_plain-rv64al.0000
mkdir -p $P/rl-$v
line=$(grep -h 'plain-cc-alean' $B/make.out | grep -v ' -c ' | grep -E -- "-o $e[[:space:]]*\$" | tail -1)
[ -n "$line" ] || { echo "no link line for $e"; exit 2; }
python3 - "$line" "$P/plain-cc-$v" "$e" "$P/rl-$v/$e.tmp" > $P/rl-$v/$e.cmd <<'PY'
import re, shlex, sys
line, cc, e, out = sys.argv[1:]
w = shlex.split(line); assert w[0].endswith("/plain-cc-alean"), w[0]; w[0] = cc
i = [k for k, x in enumerate(w) if x == "-o"]; assert len(i) == 1 and w[i[0] + 1] == e, w[-3:]; w[i[0] + 1] = out
print(shlex.join(w))
PY
(cd $B && bash $P/rl-$v/$e.cmd) 2> $P/rl-$v/$e.log
mv $P/rl-$v/$e.tmp $P/rl-$v/$e
echo "$v $e $(sha256sum < $P/rl-$v/$e | cut -c1-16)"
