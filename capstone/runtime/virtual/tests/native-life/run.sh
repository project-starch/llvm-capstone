#!/bin/bash
# Lane L lifecycle controls. OUT must be a fresh directory.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
out=${1:?fresh output directory}
[[ ! -e $out ]] || { echo 'output already exists' >&2; exit 2; }
mkdir -p "$out"
free_kb=$(/usr/bin/df -Pk "$out" | awk 'NR==2 {print $4}')
((free_kb >= 2097152)) || { echo 'less than 2 GiB free' >&2; exit 2; }
cc=${CC:-cc}
ref=$here/../native-context/reference.c
flags=(-std=gnu11 -O1 -g -Wall -Wextra -Werror -I"$repo/capstone/runtime/include")
"$cc" "${flags[@]}" -fsanitize=address,undefined -fno-omit-frame-pointer "$ref" "$here/life.c" "$here/test.c" -o "$out/life"
ASAN_OPTIONS=detect_leaks=1 "$out/life" > "$out/life.log" 2>&1
cat "$out/life.log"
mutants=(
  NL_MUTATE_SHARE_ROOT NL_MUTATE_NO_GATE NL_MUTATE_COW_DROP_TAGS
  NL_MUTATE_EXIT_NO_DRAIN NL_MUTATE_EXIT_ARENA_LOOP NL_MUTATE_EXIT_NO_INVALIDATE NL_MUTATE_EXIT_NO_STOP
  NL_MUTATE_SIG_TRUST_USER NL_MUTATE_SIGACT_SCALAR CN_MUTATE_REPLAY CN_MUTATE_SKIP_NAMESPACE
  NL_MUTATE_PTRACE_DOMAIN NL_MUTATE_PROCMEM_DOMAIN NL_MUTATE_POKE_KEEP_TAG NL_MUTATE_POKE_NO_COW
  NL_MUTATE_PTRACE_LEAK_NONCE CN_MUTATE_SCALAR_TAG
  NL_MUTATE_TLS_SCALAR NL_MUTATE_FUTEX_KEY_VA NL_MUTATE_ROBUST_RAW NL_MUTATE_ROBUST_UNBOUNDED
  NL_MUTATE_CLEARTID_RAW
)
bad=0
for m in "${mutants[@]}"; do
    "$cc" "${flags[@]}" -D"$m" "$ref" "$here/life.c" "$here/test.c" -o "$out/$m"
    set +e
    "$out/$m" > "$out/$m.log" 2>&1
    rc=$?
    set -e
    if [[ $rc == 1 ]] && grep -q '^FAIL line ' "$out/$m.log"; then
        echo "PASS negative control $m: $(grep '^FAIL line ' "$out/$m.log" | head -1)"
    else
        echo "SURVIVED $m rc=$rc" >&2; bad=1
    fi
done
sha256sum "$here/life.c" "$here/test.c" "$repo/capstone/runtime/include/capstone/native-life.h" > "$out/sources.sha256"
exit $bad
