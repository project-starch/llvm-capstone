#!/usr/bin/env bash
# Does apr-util's bucket allocator build freestanding on top of APR's pools, and
# does the return path do what the census says? The pool census answers one
# compile; this one answers three questions, because the bucket allocator's risk
# is not whether it compiles.
#
#   1. is the transcribed node geometry FAITHFUL to upstream's header?
#   2. does APR_HAS_MMAP move APR_BUCKET_ALLOC_SIZE, as the shim claims it does not?
#   3. do both recycling levels reissue storage, and does anything reach malloc?
#
# The third is measured by interposing free(), not quoted from the source.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APU=$(bash "$SCRIPT_DIR/fetch-apr-util.sh" | tail -1)
APR=$(bash "$SCRIPT_DIR/fetch-apr.sh" | tail -1)
OUT=${OUT_DIR:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/apr-buckets-census}; mkdir -p "$OUT"
CC=${CC:-cc}

# Upstream byte for byte, plus APR's own two headers and both shims.
cp "$APU/buckets/apr_buckets_alloc.c" "$OUT/"
cp "$APR/memory/unix/apr_pools.c" "$OUT/"
cp "$APR/include/apr_pools.h" "$APR/include/apr_allocator.h" "$OUT/"
cp "$SCRIPT_DIR/adapted/apr_shim.h" "$SCRIPT_DIR/adapted/apr_bucket_shim.h" "$OUT/"
cp "$SCRIPT_DIR/probes/"*.c "$OUT/"

"${PYTHON:-python3}" - "$OUT" "$APR" <<'PY'
import pathlib, re, sys
out, apr = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
# The same fourteen the pool census drops, kept in step with it by hand rather
# than globbed, because a port keeps this list.
drop = ["apr.h", "apr_private.h", "apr_atomic.h", "apr_portable.h", "apr_strings.h",
        "apr_general.h", "apr_lib.h", "apr_thread_mutex.h", "apr_hash.h", "apr_time.h",
        "apr_support.h", "apr_want.h", "apr_env.h", "apr_errno.h"]
for name, shim in (("apr_pools.c", "apr_shim.h"), ("apr_pools.h", "apr_shim.h"),
                   ("apr_allocator.h", "apr_shim.h")):
    p = out / name; s = p.read_text()
    for d in drop:
        s = s.replace('#include "%s"' % d, "").replace("#include <%s>" % d, "")
    p.write_text('#include "%s"\n' % shim + s)
# The bucket allocator's includes all become ONE shim: its five, and the ten behind
# apr_buckets.h that a port would otherwise inherit.
p = out / "apr_buckets_alloc.c"; s = p.read_text()
s = re.sub(r'#include\s+"apr[a-z_]*\.h"', '#include "apr_bucket_shim.h"', s)
p.write_text('#include "apr_bucket_shim.h"\n' + s)
PY

echo "1. is the transcribed geometry faithful to upstream?"
"${PYTHON:-python3}" - "$APU/include/apr_buckets.h" "$OUT/apr_bucket_shim.h" <<'PY'
import re, sys, pathlib
def bodies(text, names):
    out = {}
    for n in names:
        m = re.search(r'\b(struct|union)\s+%s\s*\{(.*?)\n\};' % re.escape(n), text, re.S)
        if m:
            b = [re.sub(r'/\*.*?\*/', '', l).strip() for l in m.group(2).splitlines()]
            out[n] = [re.sub(r'\s+', ' ', l) for l in b
                      if l.strip() and not l.strip().startswith(('/*', '*'))]
    return out
names = ["apr_bucket", "apr_bucket_refcount", "apr_bucket_heap", "apr_bucket_pool",
         "apr_bucket_mmap", "apr_bucket_file", "apr_bucket_structs"]
u = bodies(pathlib.Path(sys.argv[1]).read_text(), names)
m = bodies(pathlib.Path(sys.argv[2]).read_text(), names)
bad = [n for n in names if u.get(n) != m.get(n)]
for n in names:
    print("   %-22s %s" % (n, "differs" if n in bad else "identical, %d fields" % len(u[n])))
sys.exit(1 if bad else 0)
PY

echo "2. does APR_HAS_MMAP move the allocation size?"
for m in 1 0; do
  "$CC" -O1 -DAPR_HAS_MMAP=$m -o "$OUT/geo-$m" "$OUT/bucket-geometry.c" -I"$OUT"
  "$OUT/geo-$m" | tail -2 | sed 's/^/   /'
done

echo "3. do both levels recycle, and does anything reach malloc?"
"$CC" -O1 -g -o "$OUT/two-levels" "$OUT/two-levels.c" "$OUT/apr_buckets_alloc.c" \
  "$OUT/apr_pools.c" "$OUT/apr-stubs.c" "$OUT/count-free.c" -I"$OUT"
"$OUT/two-levels" | sed 's/^/   /'
