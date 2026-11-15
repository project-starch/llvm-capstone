/* The static layout check of docs/plans/2026-11-15-memcached-port-layout-fix.md. build-reuse.sh
 * compiles this file in each arm's source tree with the flags make uses for memcached's own sources.
 * Each array's size is one layout fact, and llvm-nm -S prints it. Both arms must print the same. */
#include "memcached.h"
#include <stddef.h>

char mc_layout_sizeof_item[sizeof(item)] = {0};
char mc_layout_offsetof_item_data[offsetof(item, data)] = {0};
char mc_layout_sizeof_item_chunk[sizeof(item_chunk)] = {0};
char mc_layout_offsetof_item_chunk_data[offsetof(item_chunk, data)] = {0};
