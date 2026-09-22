#!/usr/bin/env bash
# What memcached 1.6.45 has that allocates, counted from the source rather than recalled, and
# what each would cost to bring under the discipline. Run before any porting, so the estimate
# and the thing estimated are the same file. The same shape as census-apr.sh; the numbers in
# README.md beside this script come from here and from nowhere else.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC=$(bash "$SCRIPT_DIR/fetch-memcached.sh" | tail -1)
cd "$SRC"

sites() { grep -cE '(^|[^a-z_.>])(malloc|calloc|realloc|free)\(' "$1" 2>/dev/null || true; }
lines() { wc -l < "$1"; }

echo "THE ALLOCATORS -- storage owned, reused without free()"
printf "  %-22s %5s lines  %2s malloc sites  %s\n" "slabs.c" "$(lines slabs.c)" "$(sites slabs.c)" \
  "size classes over 1 MiB pages; per-class LIFO free list (slots); $(grep -c 'mem_base' slabs.c) mem_base refs"
printf "  %-22s %5s lines  %2s malloc sites  %s\n" "slabs_mover.c" "$(lines slabs_mover.c)" "$(sites slabs_mover.c)" \
  "the rebalancer: pages move between classes, items in them are relocated or evicted"
printf "  %-22s %5s lines  %2s malloc sites  %s\n" "slab_automove*.c" "$(( $(lines slab_automove.c) + $(lines slab_automove_extstore.c) ))" \
  "$(( $(sites slab_automove.c) + $(sites slab_automove_extstore.c) ))" "the policy that drives the mover"
printf "  %-22s %5s lines  %2s malloc sites  %s\n" "cache.c" "$(lines cache.c)" "$(sites cache.c)" \
  "generic object cache, STAILQ of freed objects handed back uncleared; instances:"
grep -n 'cache_create(' thread.c | sed -E 's/.*cache_create\("([a-z]+)", ([^,]+),.*/      "\1"  objects of \2/' | sed 's/^/  /'
printf "  %-22s %5s lines  %2s malloc sites  %s\n" "mc_resp bundles" "$(( $(grep -n '^void resp_free' memcached.c | cut -d: -f1) - $(grep -n '^static mc_resp\* resp_allocate' memcached.c | cut -d: -f1) ))" "-" \
  "per-thread response objects in bundles sized to a READ_BUFFER_SIZE block (MAX_RESP_PER_BUNDLE is computed, not a literal); slot reuse by refcount (memcached.c)"
printf "  %-22s %5s lines  %2s malloc sites  %s\n" "bipbuffer.c" "$(lines bipbuffer.c)" "$(sites bipbuffer.c)" \
  "bipartite ring; users: $(grep -l 'bipbuf_' *.c | grep -v bipbuffer.c | tr '\n' ' ')"
printf "  %-22s %5s lines  %2s malloc sites  %s\n" "extstore.c" "$(lines extstore.c)" "$(sites extstore.c)" \
  "its own page allocator over a file: free_page_buckets, page versions ($(grep -c 'page_version' extstore.c) refs); optional"

echo
echo "ALLOCATOR-LIKE LIFETIMES -- a slot or a table reissued, malloc underneath"
printf "  %-22s %s\n" "conn per fd" "conns[sfd] is created once and reused for every later connection on that fd; conn_free is called from $(grep -c 'conn_free(c)' memcached.c) site(s)"
printf "  %-22s %s\n" "assoc.c" "$(lines assoc.c) lines; primary_hashtable rebuilt by a maintenance thread, old table freed after migration"
printf "  %-22s %s\n" "restart.c" "$(lines restart.c) lines; an mmap arena handed to slabs_init as mem_base_external and reused across process restarts ($(grep -c 'restart_fixup' restart.c memcached.c slabs.c | awk -F: '{s+=$2} END{print s}') fixup refs)"

echo
echo "CONSUMERS, NOT ALLOCATORS"
printf "  %-22s %5s lines  %s\n" "items.c" "$(lines items.c)" "LRU over slabs; chunked items chain do_item_alloc_chunk from the largest class"
printf "  %-22s %5s lines  %s\n" "storage.c" "$(lines storage.c)" "over extstore ($(grep -c 'extstore_' storage.c) calls)"
printf "  %-22s %5s lines  %s\n" "logger.c" "$(lines logger.c)" "over bipbuffer ($(grep -c 'bipbuf_' logger.c) calls)"
printf "  %-22s %5s lines  %s\n" "crawler.c" "$(lines crawler.c)" "LRU crawler"
printf "  %-22s %5s files  %s\n" "proxy" "$(ls proxy_*.c proto_proxy.c | wc -l)" "Lua-driven; no allocator of its own, $(for f in proxy_*.c proto_proxy.c; do sites $f; done | awk '{s+=$1} END{print s}') libc sites; optional"

echo
echo "THE LEVEL BELOW: malloc/calloc/realloc/free call sites"
core=0; proxy=0
for f in *.c; do n=$(sites "$f"); case "$f" in proxy_*|proto_proxy.c) proxy=$((proxy+n));; testapp.c) ;; *) core=$((core+n));; esac; done
printf "  %-22s %s\n" "core (no testapp)" "$core"
printf "  %-22s %s\n" "proxy" "$proxy"

echo
echo "WHAT A FREESTANDING slabs.c WOULD HAVE TO BE GIVEN (the shim estimate)"
printf "  %-46s %s\n" "memcached.h, which slabs.c includes whole" "$(lines memcached.h) lines"
for pair in "item\b:the item struct" "settings\.:the settings struct" "stats\.:global stats" "pthread_mutex:pthread mutexes -- threads are ON, unlike APR's census" "ADD_STAT:the stats callback type" "slab_stats_automove:the automove stats struct"; do
  pat=${pair%%:*}; what=${pair#*:}
  printf "  %-46s %s refs\n" "$what" "$(grep -cE "$pat" slabs.c)"
done
printf "  %-46s %s\n" "system headers slabs.c includes" "$(grep -c '^#include <' slabs.c)"

echo
echo "THE REUSE PATHS, which is what decides the discipline"
printf "  %s\n" "slabs: do_slabs_free pushes an item on p->slots; slabs_alloc pops it; a chunked item's chunks return one by one through do_slabs_free_chunked"
printf "  %s\n" "slabs: a page leaves a class only through the mover, never through free(); mem_base pages never return at all"
printf "  %s\n" "cache: do_cache_alloc returns STAILQ_FIRST of the freed list, uncleared; malloc only on a miss under the limit"
printf "  %s\n" "resp: a bundle slot is reused when its refcount drops; a bundle is freed only when empty and not the open one"
printf "  %s\n" "conn: the struct behind an fd is never freed on close; the next accept on that fd gets the same struct"
printf "  %s\n" "extstore: a page is recycled through free_page_buckets and its page_version bumped -- a stale reference is detected by version, not by address"
