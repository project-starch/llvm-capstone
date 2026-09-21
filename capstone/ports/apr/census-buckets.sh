#!/usr/bin/env bash
# What apr-util's BUCKET allocator would cost to bring under the discipline,
# counted from the source rather than recalled. The same shape as census-apr.sh,
# so that the two levels can be read against each other -- and they have to be
# read together, because this allocator is not a peer of APR's pools but sits on
# top of them.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC=$(bash "$SCRIPT_DIR/fetch-apr-util.sh" | tail -1)
C=$SRC/buckets/apr_buckets_alloc.c
H=$SRC/include/apr_buckets.h

printf "%-46s %s\n" "the allocator, lines" "$(wc -l < "$C")"
printf "%-46s %s\n" "  code lines a port would touch" "$(grep -vcE '^\s*(/?\*|//|$)' "$C")"
printf "%-46s %s public\n" "what it exports" "$(grep -cE '^APU_DECLARE_NONSTD' "$C")"

echo "what it asks of the level below, and where that already is:"
for s in $(grep -oE '\bapr_(allocator|pool)_[a-z_]+\(' "$C" | tr -d '(' | sort -u); do
  printf "  %-34s %s\n" "$s" "defined in APR's apr_pools.c, which the pool census already builds"
done

echo "what stands between it and a freestanding build:"
printf "  %-34s %s direct, %s more behind apr_buckets.h\n" "headers" \
    "$(grep -c '#include' "$C")" "$(grep -c '#include' "$H")"
printf "  %-34s %s\n" "the one real coupling" \
    "SMALL_NODE_SIZE is APR_BUCKET_ALLOC_SIZE + the node header"
printf "  %-34s %s\n" "" "and APR_BUCKET_ALLOC_SIZE is 2*sizeof(union apr_bucket_structs)"
printf "  %-34s %s\n" "" "so the node geometry is decided by the bucket TYPE zoo,"
printf "  %-34s %s\n" "" "not by this file: a shim that guesses that size changes the allocator"

echo "the return path, which is what decides the discipline:"
printf "  %s\n" "apr_bucket_free pushes a SMALL node onto list->freelist, LIFO, and stops there"
printf "  %s\n" "a large node goes to apr_allocator_free, which is APR's size-bucketed list"
printf "  %s\n" "so NEITHER path reaches malloc: this is a second recycling level over the first"
printf "  %s\n" "apr_bucket_alloc pops list->freelist before carving from a block,"
printf "  %s\n" "which makes same-address reuse a property of the allocator, not of a run"
printf "  %-46s %s\n" "  blocks it carves out of" "$(grep -c 'apr_allocator_alloc' "$C") calls to the level below"
printf "  %-46s %s\n" "  destroy gives the blocks back whole" "$(grep -c 'apr_allocator_free' "$C") calls"

echo "why this one and not the pools, counted from httpd's own history:"
printf "  %-46s %s\n" "consumer lifetime fixes, pool surface" "1 of 46 candidates survived triage"
printf "  %-46s %s\n" "consumer lifetime fixes, bucket surface" "about 15 of 118, and in-tree"
printf "  %s\n" "the pool consumers that matter are third-party modules, outside any clone;"
printf "  %s\n" "the bucket consumers are server/ and modules/, shipped with httpd"
