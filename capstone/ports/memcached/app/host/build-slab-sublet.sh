#!/usr/bin/env bash
# The slabsublet arm: memcached 1.6.45 with Sublet inside its own slab and cache allocators (patch
# 0006, -DMC_CAPSTONE_SLAB_SUBLET), on the Sublet runtime heap at HEAP_LOG 27, whose pool lends the
# 64 MiB payload LINEAR. The component port's adapter (ports/memcached/allocators: leases.c,
# metadata.c, authority.c) is compiled unchanged and linked in an archive with the glue
# (src/slab-sublet/mcapp-slab-sublet.c). Two images:
#   $MC_WORK/slabsublet/memcached-slabsublet.dom          the oracle's
#   $MC_WORK/slabsublet/memcached-safety-slabsublet.dom   with patch 0005's fixture hook as well
# The mode is the run's: MC_SLAB_SUBLET_MODE=0 (spatial) or 1 (sublet), required.
# Gates, each fatal:
#   DRIFT   patch 0006's hooks are the component's patch 0002's, name for name and call for call
#           (--drift-selftest runs the check against a copy missing one hook, which must fail)
#   HEADERS the adapter was compiled against capstone/runtime/include/sublet/sublet.h and the app's
#           mc_slabs_shim.h, read from the compiler's own dependency lists
#   ARM     both images carry the adapter and the hooks' wrappers; no other arm's image does; both
#           differ from the plain sublet arm's image
#   LINK    nothing undefined but _DYNAMIC, no undefined weak symbol (as build-domain.sh)
set -euo pipefail
APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ALLOC=$(cd -- "$APP/../allocators" && pwd)
COMP_PATCH=$ALLOC/patches/memcached-1.6.45-0002-lifetime-hooks.patch
APP_PATCH=$APP/patches/memcached-1.6.45-0006-slab-sublet-hooks.patch

drift() {   # component patch, app patch: the hook calls each adds, compared
  python3 - "$1" "$2" <<'EOF'
import collections, re, sys
def added(path, prefix):
    lines = [l for l in open(path).read().splitlines() if l.startswith('+') and not l.startswith('+++')]
    return collections.Counter(re.findall(r'\b' + prefix + r'(\w+)\(', '\n'.join(lines)))
comp, app = added(sys.argv[1], 'mcp_'), added(sys.argv[2], 'mcs_')
app.pop('init', None)
if comp != app or not comp:
    print('DRIFT GATE: patch 0006 hooks differ from the component patch 0002:')
    for k in sorted(set(comp) | set(app)):
        if comp[k] != app[k]:
            print(f'  {k}: component {comp[k]}, app {app[k]}')
    sys.exit(1)
print(f'drift: patch 0006 carries the component\'s {sum(comp.values())} hook calls over {len(comp)} hooks')
EOF
}

if [ "${1:-}" = --drift-selftest ]; then
  t=$(mktemp); grep -v 'mcs_chunk_issue(it)' "$APP_PATCH" > "$t"
  if drift "$COMP_PATCH" "$t"; then rm -f "$t"; echo "DRIFT SELFTEST FAILED: a patch missing a hook passed"; exit 1; fi
  rm -f "$t"; echo "drift selftest: a patch missing one hook is refused"; exit 0
fi
drift "$COMP_PATCH" "$APP_PATCH"

source "$APP/deps/env.sh"
[ -f "$MC_DEPS_PREFIX/lib/libevent_core.a" ] || { echo "no libevent in $MC_DEPS_PREFIX: run deps/build-libevent.sh" >&2; exit 2; }
RT_INC=$CAPSTONE_REPO_ROOT/capstone/runtime/include
SDK=$MC_WORK/sdk-sublet27
if [ ! -x "$SDK/capstone-cc" ]; then
  rm -rf "$SDK"
  bash "$CAPSTONE_REPO_ROOT/capstone/ports/common/application/build-sdk.sh" "$SDK" "$MC_MUSL" "$MC_LIBC_ARCHIVE" \
    -DCAPSTONE_APPLICATION_HEAP=sublet -DCAPSTONE_APPLICATION_HEAP_LOG=27 > "$MC_WORK/sdk-sublet27.log" 2>&1 \
    || { echo "SDK sublet27 FAILED"; tail -3 "$MC_WORK/sdk-sublet27.log"; exit 1; }
fi
grep -q 'CAPSTONE_APPLICATION_HEAP_LOG:STRING=27' "$SDK/CMakeCache.txt" || { echo "SDK $SDK is not HEAP_LOG 27"; exit 1; }
MSRC=$(CAPSTONE_TMP_ROOT=$MC_WORK bash "$APP/../fetch-memcached.sh" | tail -1)
OUT=$MC_WORK/slabsublet; rm -rf "$OUT"; mkdir -p "$OUT/logs"

build() {   # name, extra CPPFLAGS
  local name=$1 defs=$2 X=$OUT/src-$1 L=$OUT/logs/$1
  mkdir -p "$L"; cp -a "$MSRC" "$X"
  for p in "$APP"/patches/*.patch; do
    ( cd "$X" && patch --batch --forward --fuzz=0 -p1 < "$p" > "$L/patch-$(basename "$p").log" ) || { echo "patch $p FAILED" >&2; exit 1; }
  done
  cp "$APP/src/slab-sublet/mcapp-slab-sublet.h" "$APP/src/mcapp-safety.c" "$X/"
  ( cd "$X" && ./configure --host=riscv64-unknown-linux-musl --with-libevent="$MC_DEPS_PREFIX" \
      --disable-extstore --disable-proxy --disable-tls --disable-sasl --disable-docs \
      CPPFLAGS="$defs" > "$L/configure.log" 2>&1 )
  # the adapter and the glue, one archive; -MD keeps each object's header list for the HEADERS gate
  local inc=(-DHAVE_CONFIG_H -I"$APP/src/slab-sublet" -I"$X" -I"$ALLOC/src/shared" -I"$RT_INC" -I"$MC_DEPS_PREFIX/include")
  for f in "$ALLOC/src/shared/leases.c" "$ALLOC/src/shared/metadata.c" "$ALLOC/src/allocators/sublet/authority.c" \
           "$APP/src/slab-sublet/mcapp-slab-sublet.c"; do
    o=$X/mcss-$(basename "$f" .c).o
    "$CC" -O2 -g -Wall "${inc[@]}" -MD -MF "$o.d" -c "$f" -o "$o" > "$L/mcss-$(basename "$f" .c).log" 2>&1 \
      || { echo "compile $f FAILED"; cat "$L/mcss-$(basename "$f" .c).log" | head; exit 1; }
  done
  "$AR" rcs "$X/libmcss.a" "$X"/mcss-*.o
  python3 - "$X" "$RT_INC" "$APP/src/slab-sublet" <<'EOF' || exit 1
import os, sys
x, rt, glue = sys.argv[1:4]
def deps(name):
    text = open(os.path.join(x, f'mcss-{name}.o.d')).read().replace('\\\n', ' ')
    return [os.path.realpath(t) for t in text.split(':', 1)[1].split()]
want_sublet = os.path.realpath(os.path.join(rt, 'sublet', 'sublet.h'))
want_shim = os.path.realpath(os.path.join(glue, 'mc_slabs_shim.h'))
a, l = deps('authority'), deps('leases')
bad = [p for p in a if p.endswith('/sublet.h') and p != want_sublet]
if want_sublet not in a or bad:
    sys.exit(f'HEADERS GATE: authority.c read {[p for p in a if p.endswith("sublet.h")]}, not {want_sublet}')
if want_shim not in l or not any(p.endswith('/memcached.h') and p.startswith(os.path.realpath(x)) for p in l):
    sys.exit("HEADERS GATE: leases.c did not take the item layout from the app's memcached.h")
print('headers: authority.c read runtime/include/sublet/sublet.h; leases.c read the app memcached.h')
EOF
  local libs; libs=$(sed -n 's/^LIBS = //p' "$X/Makefile")
  ( cd "$X" && MC_RUNTIME_DIR=$SDK CAPSTONE_SDK=$SDK make -j"${JOBS:-8}" memcached LIBS="$libs $X/libmcss.a" > "$L/build.log" 2>&1 ) \
    || { echo "build $name FAILED"; grep -E 'error' "$L/build.log" | head -5; exit 1; }
  cp "$X/memcached" "$OUT/memcached-$name.dom"
}
build slabsublet "-DMC_CAPSTONE_SLAB_SUBLET"
build safety-slabsublet "-DMC_CAPSTONE_SLAB_SUBLET -DMC_CAPSTONE_SAFETY_FIXTURES"

nm_count() { "$CAPSTONE_LLVM_BIN/llvm-nm" "$1" | grep -cE "$2" || true; }
adapter=' [Tt] (mcp_chunk_issue|mcp_chunk_release|mcp_page_carve|mcp_authority_init|mcs_init|mcs_chunk_issue)$'
for f in "$OUT/memcached-slabsublet.dom" "$OUT/memcached-safety-slabsublet.dom"; do
  [ "$(nm_count "$f" "$adapter")" = 6 ] || { echo "ARM GATE: $f lacks the adapter or the wrappers"; exit 1; }
  [ "$(nm_count "$f" ' [Tt] (sh_free|sh_carve_block)$')" = 2 ] || { echo "ARM GATE: $f lacks the Sublet heap"; exit 1; }
  undef=$("$CAPSTONE_LLVM_BIN/llvm-nm" -u "$f" | grep -v ' w ' | grep -vE ' _DYNAMIC$' || true)
  weak=$("$CAPSTONE_LLVM_BIN/llvm-nm" "$f" | awk '$1 == "w" || ($2 == "w")' || true)
  [ -z "$undef" ] || { echo "LINK GATE: $f undefined:"; echo "$undef" | head; exit 1; }
  [ -z "$weak" ] || { echo "LINK GATE: $f undefined weak:"; echo "$weak" | head; exit 1; }
done
for f in "$MC_WORK"/domain/memcached-*.dom "$MC_WORK"/safety/memcached-safety-*.dom "$MC_WORK/marker/memcached.dom"; do
  [ -f "$f" ] || continue
  [ "$(nm_count "$f" "$adapter")" = 0 ] || { echo "ARM GATE: the other arm's image $f carries the adapter"; exit 1; }
done
if [ -f "$MC_WORK/domain/memcached-sublet.dom" ]; then
  for f in "$OUT/memcached-slabsublet.dom" "$OUT/memcached-safety-slabsublet.dom"; do
    ! cmp -s "$f" "$MC_WORK/domain/memcached-sublet.dom" || { echo "ARM GATE: $f equals the plain sublet image"; exit 1; }
  done
fi
for f in "$OUT/memcached-slabsublet.dom" "$OUT/memcached-safety-slabsublet.dom"; do
  echo "$(basename "$f") $(stat -c %s "$f") bytes sha256 $(sha256sum < "$f" | cut -c1-16)"
done
echo "slabsublet: both images carry the adapter on the HEAP_LOG 27 Sublet heap, no other arm does, links clean"
