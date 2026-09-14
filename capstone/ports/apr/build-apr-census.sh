#!/usr/bin/env bash
# Does apr_pools.c build freestanding for capstone64 at all, and what does it then want?
# The census names one risk, the fourteen headers it includes, and this answers it rather than
# estimating around it. No domain, no run: one compile. The same shape as the nginx census build.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC=$(bash "$SCRIPT_DIR/fetch-apr.sh" | tail -1)
OUT=${OUT_DIR:-/tmp/capstone/apr-census}; mkdir -p "$OUT"
CLANG=${CAPSTONE_CLANG:-${CAPSTONE_LLVM_BUILD_DIR:-/home/biecho/llvm-capstone/llvm/build-rel}/bin/clang}
NM=${CAPSTONE_LLVM_NM:-$(dirname "$CLANG")/llvm-nm}

# The allocator and the two headers that carry its types, with every APR include replaced by the
# shim beside this script. The allocator itself is upstream, byte for byte.
cp "$SRC/memory/unix/apr_pools.c" "$OUT/"
cp "$SRC/include/apr_pools.h" "$SRC/include/apr_allocator.h" "$OUT/"
cp "$SCRIPT_DIR/adapted/apr_shim.h" "$OUT/"

# One sed, and it is the whole seam: every APR header becomes the shim, and the shim is included
# once. A port keeps this list, so it is spelled out rather than globbed.
"${PYTHON:-python3}" - "$OUT" <<'PY'
import pathlib, re, sys
out = pathlib.Path(sys.argv[1])
drop = ["apr.h", "apr_private.h", "apr_atomic.h", "apr_portable.h", "apr_strings.h",
        "apr_general.h", "apr_lib.h", "apr_thread_mutex.h", "apr_hash.h", "apr_time.h",
        "apr_support.h", "apr_want.h", "apr_env.h", "apr_errno.h"]
for name in ("apr_pools.c", "apr_pools.h", "apr_allocator.h"):
    p = out / name
    s = p.read_text()
    for h in drop:
        # the trailing comment on apr_portable.h is why this does not anchor on the line end
        s = re.sub(r'#\s*include\s+["<]%s[">][^\n]*\n' % re.escape(h), "", s)
    if name == "apr_pools.c":
        s = '#include "apr_shim.h"\n' + s
    else:
        s = '#include "apr_shim.h"\n' + s
    p.write_text(s)
print("  the seam: %d APR headers replaced by one shim" % len(drop))
PY

rc=0
for o in 0 1 2; do
  if "$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
      -mllvm -capstone-gp-captable -ffreestanding -fno-jump-tables -std=c99 -w \
      -I"$OUT" -O$o -c -o "$OUT/apr_pools-O$o.o" "$OUT/apr_pools.c" 2>"$OUT/err-O$o.txt"; then
    printf "  -O%s   ok, %s bytes\n" "$o" "$(stat -c %s "$OUT/apr_pools-O$o.o")"
  else
    printf "  -O%s   FAILED, %s errors\n" "$o" "$(grep -c 'error:' "$OUT/err-O$o.txt" || true)"
    rc=1
  fi
done
[ $rc -eq 0 ] || { echo "  first errors:"; grep 'error:' "$OUT/err-O0.txt" | head -12 | sed 's/^/    /'; exit 1; }

# The undefined symbols, classified, because the count alone says nothing about the work. A
# level-below symbol is what the discipline replaces. A peripheral one belongs to a service that
# happens to share this translation unit and that a port never touches.
echo "what it still wants, and from whom:"
"$NM" --undefined-only "$OUT/apr_pools-O0.o" | awk '{print $2}' | while read -r sym; do
  case "$sym" in
    malloc|free|memcpy|memset)
        printf "  %-24s the level below, which the discipline replaces\n" "$sym" ;;
    apr_hash_*)
        printf "  %-24s userdata, a service sharing this file\n" "$sym" ;;
    apr_proc_*|apr_sleep)
        printf "  %-24s the subprocess chain, a service sharing this file\n" "$sym" ;;
    apr_vformatter|apr_pstrdup)
        printf "  %-24s apr_psprintf, a service sharing this file\n" "$sym" ;;
    apr_atomic_init)
        printf "  %-24s one call, in apr_pool_initialize\n" "$sym" ;;
    __gpfree_globals_base)
        printf "  %-24s the capstone gp-captable runtime, as in every port\n" "$sym" ;;
    *)  printf "  %-24s UNCLASSIFIED, look at it\n" "$sym" ;;
  esac
done
echo "the shim that replaced those fourteen headers: $(wc -l < "$SCRIPT_DIR/adapted/apr_shim.h") lines"
