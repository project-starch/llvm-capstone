#!/bin/bash
# Build host only; output must be fresh and lane-owned.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
source "$repo/capstone/tests/capstone-test-env.sh"
out=${1:?fresh output directory}
[[ ! -e $out ]] || { echo 'output already exists' >&2; exit 2; }
free_kb=$(df -Pk "$(dirname "$out")" | awk 'NR==2 {print $4}')
((free_kb >= 12582912)) || { echo 'less than 12 GiB free' >&2; exit 2; }
mkdir -p "$out/build"
cc=${CC:-cc}
flags=(-std=c11 -O1 -g -Wall -Wextra -Werror -I"$repo/capstone/runtime/include")
common=("$repo/capstone/runtime/common/native-syscall-table.c" "$repo/capstone/runtime/common/delegate.c")
source_files=("${common[@]}" "$here/../native-context/reference.c" "$here/reference.c" "$here/test.c")
"$cc" "${flags[@]}" "${common[@]}" "$here/table-test.c" -o "$out/build/table-test"
"$out/build/table-test" | tee "$out/table.log"
"$cc" "${flags[@]}" -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer \
    "${source_files[@]}" -o "$out/build/reference"
ASAN_OPTIONS=detect_leaks=1 "$out/build/reference" > "$out/reference.log" 2>&1
cat "$out/reference.log"
mutants=(PROC INHERITED REPLACEMENT OVERLAP HIDDEN RECHECK TAGS IOV RESTART)
labels=('proc-memory and alias open refused' 'inherited fd refused' 'descriptor replacement refused'
        'overlapping argument snapshot' 'hidden pointer command refused'
        'recycled sentinel unchanged' 'scalar output clears tags'
        'iovec snapshot used for copyout' 'never restart after effects')
for i in "${!mutants[@]}"; do
    mutation=${mutants[i]}
    "$cc" "${flags[@]}" -D"NS_MUTATE_$mutation" "${source_files[@]}" -o "$out/build/$mutation"
    rc=0
    "$out/build/$mutation" > "$out/$mutation.log" 2>&1 || rc=$?
    [[ $rc == 1 ]] && grep -q "^FAIL ${labels[i]} line " "$out/$mutation.log" || {
        echo "FAIL negative control $mutation rc=$rc" >&2; exit 1;
    }
    echo "PASS negative control $mutation: required oracle rejected broken variant"
done
"$cc" "${flags[@]}" -c "$here/reference.c" -o "$out/build/native-syscall-reference.o"
"$cc" "${flags[@]}" -c "${common[0]}" -o "$out/build/native-syscall-table.o"
ar rcs "$out/build/libnative-syscall-reference.a" "$out/build/native-syscall-reference.o" "$out/build/native-syscall-table.o"
sha256sum "${source_files[@]}" "$here/table-test.c" "$here/run.sh" \
    "$repo/capstone/runtime/include/capstone/native-syscall.h" \
    "$repo/capstone/runtime/include/capstone/native-syscall-abi.h" \
    "$repo/capstone/runtime/include/capstone/native-context.h" > "$out/sources.sha256"
