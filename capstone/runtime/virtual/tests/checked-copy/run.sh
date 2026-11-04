#!/bin/bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
source "$repo/capstone/tests/capstone-test-env.sh"
out=${1:?fresh output directory}
[[ ! -e $out ]] || exit 2
mkdir -p "$out"
flags=(-std=c11 -O1 -g -Wall -Wextra -Werror -pthread -I"$repo/capstone/runtime/include")
sources=("$here/../../checked-copy.c" "$here/../native-context/reference.c" "$here/test.c")
cc "${flags[@]}" -fsanitize=address,undefined -fno-omit-frame-pointer "${sources[@]}" -o "$out/check"
"$out/check" | tee "$out/check.log"
for mutation in SKIP_CHECK NO_EXCLUSION DUMMY_STORE; do
    cc "${flags[@]}" -D"CU_MUTATE_$mutation" "${sources[@]}" -o "$out/$mutation"
    rc=0
    "$out/$mutation" > "$out/$mutation.log" 2>&1 || rc=$?
    [[ $rc == 1 ]] && grep -q '^FAIL line' "$out/$mutation.log" || exit 1
    echo "PASS negative control $mutation"
done
# Losing both guards permits actual MAP_FIXED reuse and stale copy corruption.
cc "${flags[@]}" -DCU_MUTATE_SKIP_CHECK -DCU_MUTATE_NO_HOLD "${sources[@]}" -o "$out/NO_HOLD"
rc=0
"$out/NO_HOLD" > "$out/NO_HOLD.log" 2>&1 || rc=$?
[[ $rc == 1 ]] && grep -q 'reuse mode=2 .*unauthorized=4096' "$out/NO_HOLD.log"
echo 'PASS negative control NO_HOLD+SKIP_CHECK'
sha256sum "${sources[@]}" "$repo/capstone/runtime/include/capstone/checked-copy.h" > "$out/sources.sha256"
