#!/usr/bin/env bash
# memcached 1.6.45 as a capstone64 domain: the pinned source, patches/ applied, configured with the
# deps' capstone-cc against the libevent deps/build-libevent.sh installed, the pointer-cast census on.
# Output: $MC_WORK/domain/memcached.dom, plus the cross config.h read against the native one.
set -euo pipefail
APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$APP/deps/env.sh"
[ -f "$MC_DEPS_PREFIX/lib/libevent_core.a" ] || { echo "no libevent in $MC_DEPS_PREFIX: run deps/build-libevent.sh" >&2; exit 2; }
MSRC=$(CAPSTONE_TMP_ROOT=$MC_WORK bash "$APP/../fetch-memcached.sh" | tail -1)
OUT=$MC_WORK/domain; X=$OUT/src; LOG=$OUT/logs; rm -rf "$OUT"; mkdir -p "$LOG"; cp -a "$MSRC" "$X"
for p in "$APP"/patches/*.patch; do
  ( cd "$X" && patch --batch --forward --fuzz=0 -p1 < "$p" > "$LOG/patch-$(basename "$p").log" ) || { echo "patch $p FAILED" >&2; exit 1; }
  echo "applied $(basename "$p")"
done
( cd "$X" && ./configure --host=riscv64-unknown-linux-musl --with-libevent="$MC_DEPS_PREFIX" \
    --disable-extstore --disable-proxy --disable-tls --disable-sasl --disable-docs > "$LOG/configure.log" 2>&1 )
NCFG=$MC_WORK/native/memcached/config.h
if [ -f "$NCFG" ]; then
  diff <(grep -E '^#define ' "$NCFG" | sort) <(grep -E '^#define ' "$X/config.h" | sort) > "$LOG/config-h.diff" || true
  echo "config.h native vs cross: $(grep -c '^[<>]' "$LOG/config-h.diff") differing lines ($LOG/config-h.diff)"
fi
( cd "$X" && MC_CENSUS=1 MC_CENSUS_LOG="$LOG/cast-log.txt" make -j"${JOBS:-8}" memcached > "$LOG/build.log" 2>&1 ) \
  || { echo "build FAILED"; grep -E 'error' "$LOG/build.log" | head -5; exit 1; }
touch "$LOG/cast-log.txt"; sort -u "$LOG/cast-log.txt" > "$LOG/cast-sites.txt"
cp "$X/memcached" "$OUT/memcached.dom"
# The three heap arms, from the same objects: only the runtime the image links differs.
#   shrink: the SDK's default level0, each allocation bounded (memcached.dom above);
#   level0: level0 built with CAPSTONE_LEVEL0_OBJECT_BOUNDS=0, no heap safety;
#   sublet: HEAP=sublet, revocation on free.
cp "$OUT/memcached.dom" "$OUT/memcached-shrink.dom"
for arm in level0 sublet; do
  case $arm in
    level0) extra=("-DCMAKE_C_FLAGS_RELEASE=-O1 -DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0") ;;
    sublet) extra=(-DCAPSTONE_APPLICATION_HEAP=sublet -DCAPSTONE_APPLICATION_HEAP_LOG=26) ;;
  esac
  SDK=$MC_WORK/sdk-$arm; rm -rf "$SDK"
  bash "$CAPSTONE_REPO_ROOT/capstone/ports/common/application/build-sdk.sh" "$SDK" "$MC_MUSL" "$MC_LIBC_ARCHIVE" "${extra[@]}" \
    > "$LOG/sdk-$arm.log" 2>&1 || { echo "SDK $arm FAILED"; tail -3 "$LOG/sdk-$arm.log"; exit 1; }
  rm -f "$X/memcached"
  # Both variables: the SDK's capstone-cc takes its configuration from CAPSTONE_SDK when it is set,
  # and env.sh sets it to the deps SDK, so MC_RUNTIME_DIR alone relinks every arm against that one.
  ( cd "$X" && MC_RUNTIME_DIR=$SDK CAPSTONE_SDK=$SDK make memcached > "$LOG/link-$arm.log" 2>&1 ) || { echo "link $arm FAILED"; exit 1; }
  cp "$X/memcached" "$OUT/memcached-$arm.dom"
done
for arm in level0 shrink sublet; do
  sdk=$MC_WORK/sdk-$arm; [ $arm = shrink ] && sdk=$MC_RUNTIME_DIR
  echo "arm $arm: $(sha256sum < "$OUT/memcached-$arm.dom" | cut -c1-16) sublet-heap symbols (sh_free, sh_carve_block): $("$CAPSTONE_LLVM_BIN/llvm-nm" "$OUT/memcached-$arm.dom" | grep -cE ' [Tt] (sh_free|sh_carve_block)$'); SDK $(grep -oE 'CAPSTONE_APPLICATION_HEAP:STRING=[a-z0-9]+' "$sdk/CMakeCache.txt") $(grep -oE 'CMAKE_C_FLAGS_RELEASE:STRING=.*' "$sdk/CMakeCache.txt")"
done
# The arms must be three different images: an arm that silently linked another arm's runtime (as the
# first version of this script did) gives the same hash, and this stops it.
[ "$(for a in level0 shrink sublet; do sha256sum < "$OUT/memcached-$a.dom"; done | sort -u | wc -l)" = 3 ] \
  || { echo "ARM GATE: the three arm images are not pairwise distinct"; exit 1; }
[ "$("$CAPSTONE_LLVM_BIN/llvm-nm" "$OUT/memcached-sublet.dom" | grep -cE ' [Tt] (sh_free|sh_carve_block)$')" = 2 ] && \
[ "$("$CAPSTONE_LLVM_BIN/llvm-nm" "$OUT/memcached-shrink.dom" | grep -cE ' [Tt] (sh_free|sh_carve_block)$')" = 0 ] \
  || { echo "ARM GATE: the sublet image lacks the Sublet heap, or the level0 image has it"; exit 1; }
echo "arms: three distinct images; the Sublet heap is in the sublet image only"
"$CAPSTONE_LLVM_BIN/llvm-readelf" -h "$OUT/memcached.dom" > /dev/null
# _DYNAMIC is the one undefined name every SDK image carries (the startup's reference; the threads
# probe and every tshark arm have it and run): anything else undefined fails the gate.
undef=$("$CAPSTONE_LLVM_BIN/llvm-nm" -u "$OUT/memcached.dom" | grep -v ' w ' | grep -vE ' _DYNAMIC$' || true)
weak=$("$CAPSTONE_LLVM_BIN/llvm-nm" "$OUT/memcached.dom" | awk '$1 == "w" || ($2 == "w")' || true)
[ -z "$undef" ] || { echo "LINK GATE: undefined symbols:"; echo "$undef" | head; exit 1; }
[ -z "$weak" ] || { echo "LINK GATE: undefined weak symbols:"; echo "$weak" | head; exit 1; }
# deps/build-libevent.sh gates libevent's own suite except evdns, evhttp and evrpc, on the claim that
# memcached links none of them. Proven here, with a control that the same pattern finds them in libevent.a.
# Functions only: evutil.c keeps two static hook pointers named evdns_getaddrinfo_impl and
# evdns_getaddrinfo_cancel_impl (bss, NULL until evdns_base_new registers itself), which are core, not
# DNS code; the claim is that no code of evdns.c, http.c or evrpc.c is linked.
scope='^[0-9a-f ]+ [Tt] (evdns|evhttp|evrpc)_'
[ "$("$CAPSTONE_LLVM_BIN/llvm-nm" "$MC_DEPS_PREFIX/lib/libevent.a" 2>/dev/null | grep -cE "$scope")" -gt 0 ] \
  || { echo "SCOPE CONTROL FAILED: the pattern finds no evdns/evhttp/evrpc symbol in libevent.a"; exit 1; }
n=$("$CAPSTONE_LLVM_BIN/llvm-nm" "$OUT/memcached.dom" | grep -cE "$scope" || true)
[ "$n" = 0 ] || { echo "SCOPE GATE: memcached.dom links $n evdns/evhttp/evrpc symbols, which libevent's gate did not cover"; exit 1; }
echo "scope: no evdns/evhttp/evrpc symbol in the image (the pattern finds them in libevent.a)"
echo "memcached.dom $(stat -c %s "$OUT/memcached.dom") bytes sha256 $(sha256sum < "$OUT/memcached.dom" | cut -c1-16); cast sites: $(wc -l < "$LOG/cast-sites.txt")"
