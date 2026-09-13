#!/usr/bin/env bash
# What ngx_palloc would cost to bring under the discipline, counted from the source rather than
# recalled. Run before any porting, so the estimate and the thing estimated are the same file.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/fetch-nginx.sh" >/dev/null
NGX_SRC_DIR=${NGX_SRC_DIR:-/tmp/capstone/nginx-${NGX_VERSION:-1.28.0}}
C=$NGX_SRC_DIR/src/core/ngx_palloc.c
H=$NGX_SRC_DIR/src/core/ngx_palloc.h

printf "%-42s %s\n" "the allocator, lines" "$(wc -l < "$C") + $(wc -l < "$H") header"
printf "%-42s %s\n" "the level below it calls, lines" \
    "$(wc -l < "$NGX_SRC_DIR/src/os/unix/ngx_alloc.c")"

# What the allocator needs from the rest of nginx. Its own entry points are excluded: they are
# what it provides, not what it wants.
echo "external symbols it calls:"
grep -oE '\bngx_[a-z_]+\(' "$C" | sort -u \
  | grep -vE 'ngx_(palloc|pnalloc|pcalloc|pmemalign|pfree|create_pool|destroy_pool|reset_pool|pool_cleanup|pool_run|pool_delete)' \
  | sed 's/($//; s/(//' | sed 's/^/  /'

# The property that decides whether the discipline fits: how does freed space return.
echo "the return path:"
if grep -q 'ngx_pfree' "$C" && ! grep -qE 'free_list|freelist' "$C"; then
  echo "  no free list anywhere in the allocator"
fi
printf "  %s\n" "$(grep -c 'ngx_free' "$C") calls to the level below's free, all of them in the large path"
printf "  %s\n" "small objects: bump only, released with the pool"
