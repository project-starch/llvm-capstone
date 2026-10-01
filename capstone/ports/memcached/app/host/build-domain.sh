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
