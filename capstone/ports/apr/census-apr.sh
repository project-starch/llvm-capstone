#!/usr/bin/env bash
# What APR's pools would cost to bring under the discipline, counted from the source rather than
# recalled. Run before any porting, so the estimate and the thing estimated are the same file.
# The same shape as census-nginx.sh, so that one target can be read against the other.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC=$(bash "$SCRIPT_DIR/fetch-apr.sh" | tail -1)
C=$SRC/memory/unix/apr_pools.c
H=$SRC/include/apr_pools.h
A=$SRC/include/apr_allocator.h

printf "%-46s %s\n" "the allocator, lines" "$(wc -l < "$C") + $(wc -l < "$H") + $(wc -l < "$A") header"

# APR_POOL_DEBUG is a whole second implementation. A port does not touch it, so the estimate must
# not count it. The tracker follows #if nesting rather than grepping, because #else of a debug
# block is the arm a port DOES touch.
"${PYTHON:-python3}" - "$C" <<'PY'
import sys, re
lines = open(sys.argv[1]).read().splitlines()
stack, dbg, real = [], 0, 0
for l in lines:
    s = l.strip()
    m = re.match(r'#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)', s)
    if m:
        kind, rest = m.group(1), m.group(2)
        if kind in ("if", "ifdef", "ifndef"):
            neg = kind == "ifndef" or re.search(r'!\s*APR_POOL_DEBUG', rest)
            stack.append({"dbg": ("APR_POOL_DEBUG" in rest) and not neg})
        elif kind == "endif":
            if stack: stack.pop()
        continue
    if not s or s.startswith(("/*", "*", "//")):
        continue
    if any(f["dbg"] for f in stack): dbg += 1
    else: real += 1
print("%-46s %d" % ("  code lines a port would touch", real))
print("%-46s %d" % ("  code lines inside APR_POOL_DEBUG", dbg))
PY

printf "%-46s %s public, %s public\n" "the two levels it already has" \
    "$(grep -cE '^APR_DECLARE\(.*\) apr_allocator_' "$C")" \
    "$(grep -cE '^APR_DECLARE\(.*\) apr_pool_' "$C")"

echo "what stands between it and a freestanding build:"
for pair in "APR_ALLOCATOR_USES_MMAP:mmap, munmap, sysconf" "APR_HAS_THREADS:apr_thread_mutex, apr_atomic"; do
  flag=${pair%%:*}; what=${pair##*:}
  printf "  %-28s %s, behind %s\n" "$what" "$(grep -c "$flag" "$C") guards" "$flag"
done
printf "  %-28s %s\n" "malloc and free" "$(grep -cE '\b(malloc|free)\(' "$C") calls, the level below"

echo "the return path, which is what decides the discipline:"
printf "  %s\n" "the level below is free[MAX_INDEX], one list per node size in BOUNDARY_SIZE multiples"
printf "  %s\n" "a node goes back whole and is handed out whole, and nothing merges"
printf "  %s\n" "the smallest node is MIN_ALLOC, two boundaries, so $(grep -oE '#define BOUNDARY_INDEX [0-9]+' "$C" | grep -oE '[0-9]+' | head -1) bit pages times two"
printf "  %s\n" "a pool carves objects forward out of a node and never frees one on its own"
printf "  %s\n" "apr_pool_clear and apr_pool_destroy give every node back at once"
printf "  %-46s %s\n" "  references to the parent and child chain" "$(grep -cE 'pool->(parent|child|sibling)' "$C")"
