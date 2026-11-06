#!/bin/bash
# Consume explicit peer snapshots; build only under fresh OUT.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
source "$repo/capstone/tests/capstone-test-env.sh"
fault=${1:?fault peer source root}
tag=${2:?page lifecycle double directory}
out=${3:?fresh output directory}
[[ ! -e $out ]] || { echo 'output already exists' >&2; exit 2; }
mkdir -p "$out"
free_kb=$(df -Pk "$out" | awk 'NR==2 {print $4}')
((free_kb >= 12582912)) || { echo 'less than 12 GiB free' >&2; exit 2; }
fault=$fault/capstone/runtime/virtual
sha256sum "$fault/native-grant.h" "$fault/native/grant-double.c" \
    "$fault/native/context-fault-probe.c" "$tag/page_double.py" "$tag/test_page_double.py" \
    "$here/reference.c" "$repo/capstone/runtime/include/capstone/native-context.h" > "$out/inputs.sha256"
cc=${CC:-cc}
flags=(-std=c11 -O2 -Wall -Wextra -Werror)
"$cc" "${flags[@]}" "$fault/native/grant-double.c" -o "$out/grant-double"
"$out/grant-double" > "$out/grant-normal.log" 2>&1
cat "$out/grant-normal.log"
for mutant in grant-check retire-first unmap-first skip-retire sleep-revoke namespace; do
    rc=0
    "$out/grant-double" "--mutant=$mutant" > "$out/grant-$mutant.log" 2>&1 || rc=$?
    [[ $rc == 1 ]] || { echo "grant mutant $mutant did not fail: rc=$rc"; exit 1; }
    tail -1 "$out/grant-$mutant.log"
done
for kind in normal mutant; do
    mutation=(); [[ $kind != mutant ]] || mutation+=(-DCV_BRIDGE_CAUSE_ONLY)
    "$cc" "${flags[@]}" "${mutation[@]}" -I"$repo/capstone/runtime/include" \
        "$fault/native/context-fault-probe.c" "$here/reference.c" -o "$out/fault-$kind"
    rc=0
    (ulimit -c 0; "$out/fault-$kind") > "$out/fault-$kind.log" 2>&1 || rc=$?
    if [[ $kind == normal ]]; then [[ $rc == 0 ]]; else [[ $rc == 134 ]]; fi
    echo "context-fault peer $kind rc=$rc"
done
PYTHONDONTWRITEBYTECODE=1 python3 "$tag/test_page_double.py" -v > "$out/tag.log" 2>&1
cat "$out/tag.log"
echo 'PASS peer doubles; see inputs.sha256 for exact snapshots'
