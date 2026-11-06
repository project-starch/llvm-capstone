#!/bin/bash
# Run on a build host; OUT must be a fresh lane-owned directory.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
source "$repo/capstone/tests/capstone-test-env.sh"
out=${1:?fresh output directory}
[[ ! -e $out ]] || { echo 'output already exists' >&2; exit 2; }
mkdir -p "$out"
free_kb=$(df -Pk "$out" | awk 'NR==2 {print $4}')
((free_kb >= 12582912)) || { echo 'less than 12 GiB free' >&2; exit 2; }
cc=${CC:-cc}
flags=(-std=c11 -O1 -g -Wall -Wextra -Werror -I"$repo/capstone/runtime/include")
"$cc" "${flags[@]}" -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer \
    "$here/reference.c" "$here/test.c" -o "$out/reference"
ASAN_OPTIONS=detect_leaks=1 "$out/reference" > "$out/reference.log" 2>&1
cat "$out/reference.log"
for mutation in SKIP_NAMESPACE SKIP_ROOT SKIP_FP DROP_TAGS SCALAR_TAG REPLAY; do
    "$cc" "${flags[@]}" -D"CN_MUTATE_$mutation" "$here/reference.c" "$here/test.c" -o "$out/$mutation"
    set +e
    "$out/$mutation" > "$out/$mutation.log" 2>&1
    rc=$?
    set -e
    [[ $rc == 1 ]] && grep -q '^FAIL line ' "$out/$mutation.log" || {
        echo "FAIL negative control $mutation rc=$rc" >&2; exit 1;
    }
    echo "PASS negative control $mutation: checker rejected broken variant"
done
"$cc" "${flags[@]}" -c "$here/reference.c" -o "$out/reference.o"
ar rcs "$out/libnative-context-reference.a" "$out/reference.o"
sha256sum "$here/reference.c" "$here/test.c" "$repo/capstone/runtime/include/capstone/native-context.h" > "$out/sources.sha256"
