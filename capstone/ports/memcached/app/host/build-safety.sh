#!/usr/bin/env bash
# The Safety milestone's images: memcached 1.6.45 with -DMC_CAPSTONE_SAFETY_FIXTURES (patch 0005, and
# src/mcapp-safety.c beside proto_text.c), one image per heap arm, from the same objects:
#   shrink (the deps SDK), level0 and sublet (build-domain.sh's arm SDKs, which must exist).
# A hidden command, `mc_capstone_fixture <n>`, runs fixture n on the worker that took the connection.
# Output: $MC_WORK/safety/memcached-safety-<arm>.dom. Both directions are checked: the safety images
# carry the hook's command string, and none of the oracle's or the marker's images do.
set -euo pipefail
APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$APP/deps/env.sh"
if [[ ${CAPSTONE_APPLICATION_PROFILE:-physical} == virtual ]]; then
  arms=(); default=virtual
else
  arms=(level0 sublet); default=shrink
fi
for arm in "${arms[@]}"; do
  [ -x "$MC_WORK/sdk-$arm/capstone-cc" ] || { echo "no $MC_WORK/sdk-$arm: run host/build-domain.sh first" >&2; exit 2; }
done
MSRC=$(CAPSTONE_TMP_ROOT=$MC_WORK bash "$APP/../fetch-memcached.sh" | tail -1)
OUT=$MC_WORK/safety; X=$OUT/src; LOG=$OUT/logs; rm -rf "$OUT"; mkdir -p "$LOG"; cp -a "$MSRC" "$X"
for p in "$APP"/patches/*.patch; do
  ( cd "$X" && patch --batch --forward --fuzz=0 -p1 < "$p" > "$LOG/patch-$(basename "$p").log" ) || { echo "patch $p FAILED" >&2; exit 1; }
done
cp "$APP/src/mcapp-safety.c" "$X/mcapp-safety.c"
( cd "$X" && ./configure --host=riscv64-unknown-linux-musl --with-libevent="$MC_DEPS_PREFIX" \
    --disable-extstore --disable-proxy --disable-tls --disable-sasl --disable-docs \
    CPPFLAGS=-DMC_CAPSTONE_SAFETY_FIXTURES > "$LOG/configure.log" 2>&1 )
( cd "$X" && make -j"${JOBS:-8}" memcached > "$LOG/build.log" 2>&1 ) || { echo "build FAILED"; grep -E 'error' "$LOG/build.log" | head -5; exit 1; }
cp "$X/memcached" "$OUT/memcached-safety-$default.dom"
for arm in "${arms[@]}"; do
  SDK=$MC_WORK/sdk-$arm; rm -f "$X/memcached"
  # both variables, as in build-domain.sh: capstone-cc reads CAPSTONE_SDK first
  ( cd "$X" && MC_RUNTIME_DIR=$SDK CAPSTONE_SDK=$SDK make memcached > "$LOG/link-$arm.log" 2>&1 ) || { echo "link $arm FAILED"; exit 1; }
  cp "$X/memcached" "$OUT/memcached-safety-$arm.dom"
done
nm_count() { "$CAPSTONE_LLVM_BIN/llvm-nm" "$1" | grep -cE "$2" || true; }
has_hook() { python3 -c 'import sys; sys.exit(0 if b"mc_capstone_fixture " in open(sys.argv[1],"rb").read() else 1)' "$1"; }
for arm in "$default" "${arms[@]}"; do
  f=$OUT/memcached-safety-$arm.dom
  has_hook "$f" || { echo "SAFETY GATE: $f lacks the hook: the define did not reach proto_text.c"; exit 1; }
  echo "safety $arm: $(sha256sum < "$f" | cut -c1-16) sublet-heap symbols: $(nm_count "$f" ' [Tt] (sh_free|sh_carve_block)$')" \
       "fixture symbols: $(nm_count "$f" ' [Tt] mcapp_fix_(touch|poke)$')"
done
if [[ $default != virtual ]]; then
[ "$(for a in level0 shrink sublet; do sha256sum < "$OUT/memcached-safety-$a.dom"; done | sort -u | wc -l)" = 3 ] \
  || { echo "ARM GATE: the three safety images are not pairwise distinct"; exit 1; }
[ "$(nm_count "$OUT/memcached-safety-sublet.dom" ' [Tt] (sh_free|sh_carve_block)$')" = 2 ] && \
[ "$(nm_count "$OUT/memcached-safety-shrink.dom" ' [Tt] (sh_free|sh_carve_block)$')" = 0 ] && \
[ "$(nm_count "$OUT/memcached-safety-level0.dom" ' [Tt] (sh_free|sh_carve_block)$')" = 0 ] \
  || { echo "ARM GATE: the Sublet heap is not in the sublet image only"; exit 1; }
fi
for f in "$MC_WORK"/domain/memcached-*.dom "$MC_WORK/marker/memcached.dom" "$MC_WORK/native/bin/memcached"; do
  [ -f "$f" ] || continue
  ! has_hook "$f" || { echo "SAFETY GATE: the non-safety image $f carries the hook"; exit 1; }
done
echo "safety: $default profile carries the hook; non-safety images do not"
