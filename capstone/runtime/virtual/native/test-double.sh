#!/bin/bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../tests/capstone-test-env.sh"
out=${1:?output directory}
mkdir -p "$out"
cc -std=c11 -O2 -Wall -Wextra -Werror "$here/grant-double.c" -o "$out/grant-double"
"$out/grant-double" > "$out/normal.log" 2>&1
cat "$out/normal.log"
for mutant in grant-check retire-first unmap-first skip-retire sleep-revoke namespace; do
    rc=0
    "$out/grant-double" "--mutant=$mutant" > "$out/$mutant.log" 2>&1 || rc=$?
    [[ $rc == 1 ]] || { echo "mutant $mutant did not fail: rc=$rc"; exit 1; }
    tail -1 "$out/$mutant.log"
done
echo 'grant-double: normal and all negative controls PASS'
